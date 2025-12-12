import random
import math
from typing import Dict, List, Sequence, Optional, Tuple

import numpy as np

# Import DSL utilities for AST-based operations
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from model.dsl import Node, tokens_to_ast, ast_to_tokens, set_library
from model.dsl.fragments import Library
from model.program_induction import discover_fragments, apply_fragments, fragment_usage
from model.eval import ib_loss, ib_complexity, ib_accuracy


# Global configuration and example DSL


# Example tower programs from the tutorial (Notebook 2)
MANUAL_TOWER_PROGRAMS: Dict[str, str] = {
    "CL": "h l_1 v v r_1 h r_12 h l_4 h l_1 v v",
    "CPi": "h l_1 v v r_1 h r_6 v r_6 v l_5 h r_4 h",
    "PiC": "v r_6 v l_5 h r_4 h r_7 h l_1 v v r_1 h",
    "LPi": "h l_4 h l_1 v v r_9 v r_6 v l_5 h r_4 h",
    "LC": "h l_4 h l_1 v v r_12 h l_1 v v r_1 h",
    "PiL": "v r_6 v l_5 h r_4 h r_9 h l_4 h l_1 v v",
}

# Derive a simple DSL as the set of tokens used in the manual tower programs
DEFAULT_DSL: List[str] = sorted(
    {token for program in MANUAL_TOWER_PROGRAMS.values() for token in program.split()}
)

# Default parameters for stochastic editing of programs
DEFAULT_NOISE_PARAMS: Dict[str, float] = {
    "delete": 0.05,      # probability of deleting a token
    "insert": 0.05,      # probability of inserting a random token after a token
    "substitute": 0.05,  # probability of substituting a token with another token
}

# Default parameters for compression behavior
DEFAULT_COMPRESSION_BIAS: Dict[str, float] = {
    "merge_repeats": 0.6,  # probability of compressing a run of repeated tokens
    "min_repeat": 2.0,     # minimum run length eligible for compression
}



# Helper functions


def expand_program(program: Sequence[str]) -> List[str]:
    """Expand macro tokens of the form 'token*k' into k repetitions of 'token'."""
    expanded: List[str] = []
    for tok in program:
        if "*" in tok:
            base, count_str = tok.split("*", 1)
            try:
                count = int(count_str)
            except ValueError:
                expanded.append(tok)
                continue
            expanded.extend([base] * max(count, 0))
        else:
            expanded.append(tok)
    return expanded


def expanded_length(program: Sequence[str]) -> int:
    """Return the length of a program after expanding any macro tokens."""
    return len(expand_program(program))


def sequence_accuracy(true_program: Sequence[str], observed_program: Sequence[str]) -> float:
    """
    Compute a simple accuracy score between two programs.

    The score is the proportion of positions in the true program that match
    the observed program after expanding both, aligned from the start.
    """
    true_expanded = expand_program(true_program)
    obs_expanded = expand_program(observed_program)
    if not true_expanded:
        return 0.0
    limit = min(len(true_expanded), len(obs_expanded))
    matches = 0
    for i in range(limit):
        if true_expanded[i] == obs_expanded[i]:
            matches += 1
    return matches / float(len(true_expanded))


def visual_overlap_score(true_program: Sequence[str], observed_program: Sequence[str]) -> float:
    """
    A coarse \"visual\" overlap proxy based on token content overlap.

    We expand both programs and compute the Jaccard overlap between the sets
    of tokens, ignoring exact positions. This is a lightweight proxy for how
    similar two programs are in terms of the building primitives they use.
    """
    true_expanded = expand_program(true_program)
    obs_expanded = expand_program(observed_program)

    if not true_expanded and not obs_expanded:
        return 0.0
    set_true = set(true_expanded)
    set_obs = set(obs_expanded)
    union = set_true | set_obs
    if not union:
        return 0.0
    return float(len(set_true & set_obs)) / float(len(union))



# Transmission chain model


class TransmissionChain:
    """
    A simple transmission chain model over block-building programs.

    Programs are represented as sequences of DSL tokens, e.g.:
        ["h", "l_1", "v", "v", "r_1", "h"]

    The transmit_once method applies stochastic edits (noise) and optional
    compression of repeated tokens. The run_chain method iterates this process
    over a specified number of generations.
    """

    def __init__(
        self,
        dsl: Optional[Sequence[str]] = None,
        noise_params: Optional[Dict[str, float]] = None,
        compression_bias: Optional[Dict[str, float]] = None,
        random_seed: Optional[int] = None,
    ) -> None:
        self.dsl: List[str] = list(dsl) if dsl is not None else list(DEFAULT_DSL)
        self.noise_params: Dict[str, float] = dict(DEFAULT_NOISE_PARAMS)
        if noise_params is not None:
            self.noise_params.update(noise_params)
        self.compression_bias: Dict[str, float] = dict(DEFAULT_COMPRESSION_BIAS)
        if compression_bias is not None:
            self.compression_bias.update(compression_bias)
        self.random: random.Random = random.Random(random_seed)

    def transmit_once(self, program: Sequence[str]) -> List[str]:
        """
        Apply one noisy, potentially compressive transmission step.

        Returns a new program (list of tokens), without modifying the input.
        """
        base_tokens = expand_program(program)
        edited: List[str] = []

        for tok in base_tokens:
            # Deletion noise
            if self.random.random() < self.noise_params["delete"]:
                continue

            new_tok = tok

            # Substitution noise
            if self.random.random() < self.noise_params["substitute"] and self.dsl:
                new_tok = self.random.choice(self.dsl)

            edited.append(new_tok)

            # Insertion noise
            if self.random.random() < self.noise_params["insert"] and self.dsl:
                inserted = self.random.choice(self.dsl)
                edited.append(inserted)

        # Compression step: optionally merge runs of repeated tokens
        compressed = self._compress_repeats(edited)
        return compressed

    def _compress_repeats(self, tokens: Sequence[str]) -> List[str]:
        """Compress runs of repeated tokens into 'tok*k' macros."""
        if not tokens:
            return []

        min_repeat = int(self.compression_bias.get("min_repeat", 2.0))
        merge_prob = self.compression_bias.get("merge_repeats", 0.0)

        if merge_prob <= 0.0 or min_repeat <= 1:
            return list(tokens)

        out: List[str] = []
        i = 0
        n = len(tokens)

        while i < n:
            j = i + 1
            while j < n and tokens[j] == tokens[i]:
                j += 1
            run_len = j - i
            if run_len >= min_repeat and self.random.random() < merge_prob:
                out.append(f"{tokens[i]}*{run_len}")
            else:
                out.extend(tokens[i:j])
            i = j

        return out

    def run_chain(self, initial_program: Sequence[str], num_generations: int) -> List[List[str]]:
        """
        Run a transmission chain starting from an initial program.

        Returns a list of programs, including the initial program as generation 0.
        """
        chain: List[List[str]] = [list(initial_program)]
        current: List[str] = list(initial_program)
        for _ in range(num_generations):
            current = self.transmit_once(current)
            chain.append(current)
        return chain


def run_chain_with_selection(
    model: TransmissionChain,
    true_program: Sequence[str],
    num_generations: int,
    lambda_compression: float = 0.3,
    num_candidates: int = 5,
    temperature: float = 0.0,
) -> List[List[str]]:
    """
    Run a transmission chain with simple selection over noisy candidates.

    At each generation, we:
    - Generate several candidate variants from the current program using
      the model's transmit_once (plus the identity candidate).
    - Score each candidate with a loss that trades off sequence accuracy
      against normalized length using relative compression ratio.
    - Choose the next program as either the argmin-loss candidate
      (temperature=0) or via a softmax over negative loss (temperature>0).
    
    The loss function (for consistency with AST-level IB loss) is:
        loss = compression_penalty - lambda * accuracy
    
    where compression_penalty is normalized relative to the true program length:
        compression_penalty = max(0, (L - base_length) / base_length)
    
    Note: This is mathematically equivalent to utility = accuracy - lambda * penalty,
    but using loss makes the concept consistent with IB loss (lower is better).
    """
    if num_candidates < 1:
        raise ValueError("num_candidates must be at least 1.")

    base_true = expand_program(true_program)
    base_length = len(base_true) if base_true else 1

    chain: List[List[str]] = [list(true_program)]
    current: List[str] = list(true_program)

    for _ in range(num_generations):
        candidates: List[List[str]] = [list(current)]
        # Generate additional noisy candidates
        for _ in range(num_candidates - 1):
            candidates.append(model.transmit_once(current))

        losses: List[float] = []
        # Compute normalized lengths for all candidates (for dynamic normalization)
        candidate_lengths = [expanded_length(cand) for cand in candidates]
        max_length = max(candidate_lengths) if candidate_lengths else base_length
        min_length = min(candidate_lengths) if candidate_lengths else base_length
        
        for cand in candidates:
            acc = sequence_accuracy(true_program, cand)
            L = expanded_length(cand)
            
            # Relative compression penalty (relative to base_length)
            # Penalty = 0 if L <= base_length, increases if L > base_length
            compression_penalty = max(0.0, (L - base_length) / float(base_length))
            
            # Loss = penalty - lambda * accuracy (lower is better)
            # This is equivalent to utility = accuracy - lambda * penalty,
            # but using loss makes it consistent with IB loss concept
            loss = compression_penalty - lambda_compression * acc
            losses.append(loss)

        # Selection: argmin loss or softmax over negative loss
        if temperature <= 0.0:
            best_idx = min(range(len(candidates)), key=lambda i: losses[i])
        else:
            # Softmax over negative loss (convert loss to probability)
            # Lower loss -> higher probability
            neg_losses = np.array([-l for l in losses], dtype=float) / float(temperature)
            # Numerical stability
            neg_losses -= np.max(neg_losses)
            probs = np.exp(neg_losses)
            probs /= probs.sum()
            r = model.random.random()
            cum = 0.0
            best_idx = 0
            for i, p in enumerate(probs):
                cum += float(p)
                if r <= cum:
                    best_idx = i
                    break

        current = list(candidates[best_idx])
        chain.append(current)

    return chain



# Tiny Bayesian lexicon model (inspired by notebook 3)



class TinyLexicon:
    """
    Simple lexicon mapping meanings to words.

    meanings: list of meaning identifiers (e.g., scene keys like 'CL')
    words: list of word strings (e.g., ['w1', 'w2', 'w3'])
    mapping: dict from meaning -> word
    """

    def __init__(self, meanings: Sequence[str], words: Sequence[str], mapping: Dict[str, str]):
        self.meanings = list(meanings)
        self.words = list(words)
        self.mapping = dict(mapping)

    def word_for(self, meaning: str) -> str:
        return self.mapping[meaning]

    def meaning_for(self, word: str) -> Optional[str]:
        for m, w in self.mapping.items():
            if w == word:
                return m
        return None


def make_lexicon_space(meanings: Sequence[str], words: Sequence[str]) -> List[TinyLexicon]:
    """
    Generate all one-to-one lexicons mapping meanings to words.
    """
    meanings = list(meanings)
    words = list(words)
    if len(words) < len(meanings):
        raise ValueError("Need at least as many words as meanings.")
    out: List[TinyLexicon] = []
    from itertools import permutations

    for perm in permutations(words, len(meanings)):
        mapping = {m: w for m, w in zip(meanings, perm)}
        out.append(TinyLexicon(meanings, words, mapping))
    return out


def update_lexicon_posterior(
    lexicons: Sequence[TinyLexicon],
    prior: np.ndarray,
    meaning: str,
    utterance: str,
    eps: float = 0.01,
) -> np.ndarray:
    """
    One-step Bayes update over lexicon hypotheses given (meaning, utterance).

    Likelihood:
      - high (1-eps) if lexicon maps meaning to utterance
      - low (eps) otherwise
    """
    pri = np.array(prior, dtype=float)
    if pri.shape[0] != len(lexicons):
        raise ValueError("Prior length must match number of lexicons.")

    like = np.zeros(len(lexicons), dtype=float)
    for i, L in enumerate(lexicons):
        like[i] = 1.0 - eps if L.word_for(meaning) == utterance else eps

    post_unnorm = pri * like
    Z = post_unnorm.sum()
    if Z <= 0.0:
        # fallback to uniform
        return np.ones_like(post_unnorm) / float(len(post_unnorm))
    return post_unnorm / Z


def run_bayesian_lexicon_chain(
    num_generations: int = 5,
    trials_per_generation: int = 20,
    eps: float = 0.01,
) -> List[Dict[str, object]]:
    """
    Run a small chain of dyads that learn a lexicon via Bayesian updating.

    We reuse a subset of MANUAL_TOWER_PROGRAMS keys as meanings and use
    abstract word labels as lexemes. Each generation:
      - Starts with the previous generation's posterior over lexicons
      - Simulates several Architect-Builder interactions
      - Updates the shared posterior after each trial
      - Records communicative success and posterior entropy

    Returns a list of dicts with keys:
      - 'generation'
      - 'success_rate'
      - 'posterior_entropy'
      - 'map_lexicon' (meaning -> word mapping for MAP hypothesis)
    """
    # Choose a small set of meanings from the available scenes
    meanings = ["CL", "CPi", "LC"]
    words = ["w1", "w2", "w3"]

    lexicons = make_lexicon_space(meanings, words)
    n_lex = len(lexicons)

    # Prior over lexicons: uniform
    prior = np.ones(n_lex, dtype=float) / float(n_lex)

    results: List[Dict[str, object]] = []
    rng = np.random.default_rng(0)

    for gen in range(num_generations + 1):
        posterior = np.array(prior, dtype=float)
        successes: List[float] = []

        for _ in range(trials_per_generation):
            # Sample a meaning uniformly
            meaning = rng.choice(meanings)

            # Architect samples a lexicon from posterior and speaks deterministically
            idx_arch = rng.choice(np.arange(n_lex), p=posterior)
            L_arch = lexicons[int(idx_arch)]
            utt = L_arch.word_for(meaning)

            # Builder infers meaning from utterance using the same posterior
            # P(m | utt) ∝ sum_{L: L(m) = utt} P(L)
            meaning_scores = np.zeros(len(meanings), dtype=float)
            for i_m, m in enumerate(meanings):
                mask = np.array(
                    [1.0 if L.word_for(m) == utt else 0.0 for L in lexicons],
                    dtype=float,
                )
                meaning_scores[i_m] = float((posterior * mask).sum())
            if meaning_scores.sum() <= 0.0:
                # fall back to uniform guess
                meaning_probs = np.ones_like(meaning_scores) / float(len(meaning_scores))
            else:
                meaning_probs = meaning_scores / meaning_scores.sum()
            idx_guess = int(np.argmax(meaning_probs))
            guessed_meaning = meanings[idx_guess]

            successes.append(1.0 if guessed_meaning == meaning else 0.0)

            # Update posterior on lexicons based on observed (meaning, utt)
            posterior = update_lexicon_posterior(
                lexicons=lexicons,
                prior=posterior,
                meaning=meaning,
                utterance=utt,
                eps=eps,
            )

        # Record generation summary
        success_rate = float(np.mean(successes)) if successes else 0.0
        # Entropy of posterior over lexicons
        nonzero_mask = posterior > 0.0
        if np.any(nonzero_mask):
            nonzero_probs = np.array([float(p) for p in posterior[nonzero_mask]])
            log_probs = np.array([np.log2(p) if p > 0 else 0.0 for p in nonzero_probs])
            posterior_entropy = float(-sum(nonzero_probs * log_probs))
        else:
            posterior_entropy = 0.0
        map_idx = int(np.argmax(posterior))
        map_lex = lexicons[map_idx].mapping

        results.append(
            {
                "generation": gen,
                "success_rate": success_rate,
                "posterior_entropy": posterior_entropy,
                "map_lexicon": dict(map_lex),
            }
        )

        # Next generation starts from this posterior
        prior = posterior.copy()

    return results


def run_chain_ast(
    initial_program: Sequence[str],
    num_generations: int,
    dsl: Optional[Sequence[str]] = None,
    lambda_c: float = 1.0,
    lambda_m: float = 1.0,
    num_candidates: int = 5,
    fragment_discovery_freq: int = 3,
    random_seed: Optional[int] = None,
) -> List[List[str]]:
    """
    Run a transmission chain using AST-level reconstruction with fragments.
    
    At each generation:
    - Discover fragments from accumulated programs (every fragment_discovery_freq steps)
    - Generate candidate AST edits from the current program
    - Score candidates using reconstruction_cost (complexity + mismatch)
    - Apply fragments to compress the selected AST
    - Convert back to tokens for compatibility
    
    Returns a list of token programs over generations.
    """
    from .selection import (
        generate_ast_candidates,
        reconstruction_cost,
        complexity,
    )
    
    dsl_tokens = list(dsl) if dsl is not None else list(DEFAULT_DSL)
    rng = random.Random(random_seed)
    
    library = Library()
    all_programs = []
    chain: List[List[str]] = [list(initial_program)]
    all_programs.append(list(initial_program))
    current_tokens = list(initial_program)
    
    for gen in range(num_generations):
        # Discover fragments more frequently to improve early compression
        # Discover on generation 1 (first generation after initial) and then periodically
        # Note: We need at least 2 programs for fragment discovery (min_freq=2)
        should_discover = (
            gen == 1  # Discover on first generation after initial (when we have 2 programs)
            or (gen > 1 and gen % fragment_discovery_freq == 0 and len(all_programs) > 1)
        )
        
        if should_discover and len(all_programs) >= 2:
            library = discover_fragments(
                programs=all_programs,
                min_length=3,
                max_length=5,
                min_freq=2,
            )
            set_library(library)
        
        # Convert current to AST
        current_ast = tokens_to_ast(current_tokens)
        
        # Generate candidate ASTs (pass library for fragment-based edits)
        candidates = generate_ast_candidates(current_ast, num_candidates, rng, library=library)
        
        # Score each candidate
        costs = []
        for cand in candidates:
            cost = reconstruction_cost(
                candidate=cand,
                perceived=current_ast,
                lambda_c=lambda_c,
                lambda_m=lambda_m
            )
            costs.append(cost)
        
        # Select best (argmin cost)
        best_idx = min(range(len(candidates)), key=lambda i: costs[i])
        selected_ast = candidates[best_idx]
        
        # Apply fragments to compress
        if len(library) > 0:
            selected_ast, _ = apply_fragments(selected_ast, library)
        
        # Convert back to tokens (temporarily clear library so fragment calls are preserved as #FX tokens)
        from model.dsl.parser import set_library as parser_set_library, get_library as parser_get_library
        old_library = parser_get_library()
        parser_set_library(None)  # Clear library so CALL nodes become #FX tokens
        next_tokens = ast_to_tokens(selected_ast)
        parser_set_library(old_library)  # Restore library
        
        chain.append(next_tokens)
        all_programs.append(next_tokens)
        current_tokens = next_tokens
    
    return chain


def run_chain_ib(
    initial_program: Sequence[str],
    true_program: Sequence[str],
    num_generations: int,
    beta: float = 1.0,
    num_candidates: int = 5,
    fragment_discovery_freq: int = 3,
    temperature: float = 0.0,
    noise_params: Optional[Dict[str, float]] = None,
    use_true_target: bool = False,
    random_seed: Optional[int] = None,
) -> Tuple[List[List[str]], Library]:
    """
    Run a transmission chain using explicit IB-style loss.
    
    At each generation:
    - Discover fragments periodically
    - Generate candidate AST edits
    - Score with IB loss: L = C - beta * A
      where C = description_length, A = accuracy vs true_program
    - Select argmin loss
    - Apply fragments and return tokens
    
    Returns (chain, library) tuple where chain is list of token programs
    and library is the fragment library discovered during this chain.
    """
    from .selection import generate_ast_candidates
    
    rng = random.Random(random_seed)
    true_ast = tokens_to_ast(true_program)
    dsl_tokens = list(DEFAULT_DSL)

    eff_noise = dict(DEFAULT_NOISE_PARAMS)
    if noise_params is not None:
        eff_noise.update(noise_params)

    def _apply_token_noise(tokens: Sequence[str]) -> List[str]:
        out: List[str] = []
        for tok in list(tokens):
            if rng.random() < float(eff_noise.get("delete", 0.0)):
                continue
            new_tok = tok
            if rng.random() < float(eff_noise.get("substitute", 0.0)) and dsl_tokens:
                new_tok = rng.choice(dsl_tokens)
            out.append(new_tok)
            if rng.random() < float(eff_noise.get("insert", 0.0)) and dsl_tokens:
                out.append(rng.choice(dsl_tokens))
        return out
    
    library = Library()
    # Store expanded (primitive) token programs for fragment discovery stability.
    expanded_history: List[List[str]] = []
    chain: List[List[str]] = [list(initial_program)]
    expanded_history.append(expand_program(list(initial_program)))
    current_tokens = list(initial_program)
    
    for gen in range(num_generations):
        # Discover fragments more frequently to improve early compression
        # Discover on generation 1 (first generation after initial) and then periodically
        # Note: We need at least 2 programs for fragment discovery (min_freq=2)
        should_discover = (
            gen == 1
            or (gen > 1 and gen % fragment_discovery_freq == 0 and len(expanded_history) > 1)
        )
        
        if should_discover and len(expanded_history) >= 2:
            expanded_history_seq: List[Sequence[str]] = list(expanded_history)
            library = discover_fragments(
                programs=expanded_history_seq,
                min_length=3,
                max_length=5,
                min_freq=2,
            )
            set_library(library)
        
        # Noisy perception of the previous generation's program
        perceived_tokens = _apply_token_noise(current_tokens)
        perceived_ast = tokens_to_ast(perceived_tokens)
        
        # Generate candidates from perceived program (pass library for fragment-based edits)
        candidates = generate_ast_candidates(perceived_ast, num_candidates, rng, library=library)

        target_ast = true_ast if use_true_target else perceived_ast
        
        # Score with IB loss using dynamic normalization (based on candidate set)
        losses = []
        for cand in candidates:
            loss = ib_loss(
                cand, target_ast,
                beta=beta, 
                library=library,
                normalize_mode='dynamic',
                candidates=candidates
            )
            losses.append(loss)
        
        # Select next program
        if temperature is None or float(temperature) <= 0.0:
            best_idx = min(range(len(candidates)), key=lambda i: losses[i])
            selected_ast = candidates[best_idx]
        else:
            t = float(temperature)
            scaled = [-(l / t) for l in losses]
            m = max(scaled)
            exps = [math.exp(x - m) for x in scaled]
            s = sum(exps) if exps else 1.0
            r = rng.random()
            acc = 0.0
            chosen = 0
            for i, e in enumerate(exps):
                acc += e / s
                if r <= acc:
                    chosen = i
                    break
            selected_ast = candidates[chosen]
        
        # Apply fragments
        if len(library) > 0:
            selected_ast, _ = apply_fragments(selected_ast, library)
        
        # Convert to tokens:
        # - expanded_next: primitive tokens used for fragment discovery stability
        # - next_tokens: compressed tokens with #FX to track fragment usage over generations
        from model.dsl.parser import set_library as parser_set_library, get_library as parser_get_library
        old_library = parser_get_library()

        parser_set_library(library)
        expanded_next = ast_to_tokens(selected_ast)

        parser_set_library(None)
        next_tokens = ast_to_tokens(selected_ast)

        parser_set_library(old_library)
        
        chain.append(next_tokens)
        expanded_history.append(expanded_next)
        current_tokens = next_tokens
    
    return chain, library


def run_chain_bayes(
    initial_program: Sequence[str],
    true_program: Sequence[str],
    num_generations: int,
    alpha: float = 10.0,
    lambda_len: float = 0.1,
    num_candidates: int = 10,
    fragment_discovery_freq: int = 3,
    temperature: float = 0.0,
    noise_params: Optional[Dict[str, float]] = None,
    random_seed: Optional[int] = None,
) -> Tuple[List[List[str]], Library]:
    """
    Run a Bayesian listener transmission chain.
    
    At each generation:
    - Apply perception noise to current program
    - Generate candidate reconstructions
    - Score with Bayesian posterior: log p(cand | obs) = log_prior + log_likelihood
      where prior favors short programs and likelihood favors match to observation
    - Select via MAP (temperature=0) or soft sampling (temperature>0)
    - Apply fragments and emit tokens
    - Periodically discover fragments from expanded token history
    
    Args:
        initial_program: Starting program tokens
        true_program: True target program (for evaluation only, not used in reconstruction)
        num_generations: Number of generations to run
        alpha: Likelihood scaling (higher = stronger fit to observation)
        lambda_len: Prior scaling (higher = stronger preference for compression)
        num_candidates: Number of candidate programs to generate
        fragment_discovery_freq: Discover fragments every N generations
        temperature: Selection temperature (0 = greedy MAP, >0 = soft sampling)
        noise_params: Noise parameters for perception (delete, insert, substitute)
        random_seed: Random seed
    
    Returns:
        (chain, library) tuple where chain is list of token programs
        and library is the fragment library discovered during this chain
    """
    from .selection import generate_ast_candidates, select_bayesian_candidate
    
    rng = random.Random(random_seed)
    
    # Initialize noise model
    if noise_params is None:
        noise_params = {'delete': 0.02, 'insert': 0.02, 'substitute': 0.02}
    noise_model = TransmissionChain(
        dsl=DEFAULT_DSL,
        noise_params=noise_params,
        random_seed=random_seed
    )
    
    library = Library()
    all_programs = []
    chain: List[List[str]] = [list(initial_program)]
    all_programs.append(list(initial_program))
    current_tokens = list(initial_program)
    
    for gen in range(num_generations):
        # Fragment discovery (periodic)
        should_discover = (
            gen == 1
            or (gen > 1 and gen % fragment_discovery_freq == 0 and len(all_programs) > 1)
        )
        
        if should_discover and len(all_programs) >= 2:
            # Discover fragments from expanded tokens (not from #F macros)
            expanded_programs: List[Sequence[str]] = [expand_program(p) for p in all_programs]
            library = discover_fragments(
                programs=expanded_programs,
                min_length=3,
                max_length=5,
                min_freq=2,
            )
            set_library(library)
        
        # Apply perception noise to current program
        obs_tokens = noise_model.transmit_once(current_tokens)
        # Expand any compressed tokens (e.g., 'r_4*2') before converting to AST
        obs_tokens_expanded = expand_program(obs_tokens)
        obs_ast = tokens_to_ast(obs_tokens_expanded)
        
        # Generate candidate reconstructions
        # Include both: (1) observation and its variants, (2) current program and its variants
        # This allows listener to either trust observation or stick with current belief
        current_ast = tokens_to_ast(current_tokens)
        
        # Half candidates from observation (what was heard)
        obs_candidates = generate_ast_candidates(obs_ast, num_candidates // 2, rng, library=library, max_edits_per_candidate=1)
        # Half candidates from current (maintain continuity)
        current_candidates = generate_ast_candidates(current_ast, num_candidates - len(obs_candidates), rng, library=library, max_edits_per_candidate=1)
        
        candidates = obs_candidates + current_candidates
        
        # Bayesian selection: posterior = prior + likelihood
        selected_ast = select_bayesian_candidate(
            candidates=candidates,
            obs_ast=obs_ast,
            library=library,
            alpha=alpha,
            lambda_len=lambda_len,
            temperature=temperature,
            rng=rng,
        )
        
        # Apply fragments to compress
        if len(library) > 0:
            selected_ast, _ = apply_fragments(selected_ast, library)
        
        # Convert to tokens (preserve #F macros for tracking)
        from model.dsl.parser import set_library as parser_set_library, get_library as parser_get_library
        old_library = parser_get_library()
        parser_set_library(None)  # Clear so CALL nodes become #FX tokens
        next_tokens = ast_to_tokens(selected_ast)
        parser_set_library(old_library)
        
        chain.append(next_tokens)
        all_programs.append(next_tokens)
        current_tokens = next_tokens
    
    return chain, library
