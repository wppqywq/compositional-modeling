import math
import os
import random
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

# from model.dsl import ast_to_tokens, tokens_to_ast, set_library
# from model.dsl.fragments import Library
# from model.program_induction import discover_fragments, apply_fragments
# from model.eval import ib_loss


MANUAL_TOWER_PROGRAMS: Dict[str, str] = {
    "CL": "h l_1 v v r_1 h r_12 h l_4 h l_1 v v",
    "CPi": "h l_1 v v r_1 h r_6 v r_6 v l_5 h r_4 h",
    "PiC": "v r_6 v l_5 h r_4 h r_7 h l_1 v v r_1 h",
    "LPi": "h l_4 h l_1 v v r_9 v r_6 v l_5 h r_4 h",
    "LC": "h l_4 h l_1 v v r_12 h l_1 v v r_1 h",
    "PiL": "v r_6 v l_5 h r_4 h r_9 h l_4 h l_1 v v",
}

DEFAULT_DSL: List[str] = sorted(
    {token for program in MANUAL_TOWER_PROGRAMS.values() for token in program.split()}
)

DEFAULT_NOISE_PARAMS: Dict[str, float] = {
    "delete": 0.05,
    "insert": 0.05,
    "substitute": 0.05,
}

DEFAULT_COMPRESSION_BIAS: Dict[str, float] = {
    "merge_repeats": 0.6,
    "min_repeat": 2.0,
}

@lru_cache(maxsize=32)
def _load_trials_cached(path: str) -> Any:
    import json
    import pandas as pd

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return pd.DataFrame(raw)



def expand_program(program: Sequence[str]) -> List[str]:
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
    return len(expand_program(program))


def sequence_accuracy(true_program: Sequence[str], observed_program: Sequence[str]) -> float:
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



class TransmissionChain:
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
        base_tokens = expand_program(program)
        edited: List[str] = []

        for tok in base_tokens:
            if self.random.random() < self.noise_params["delete"]:
                continue

            new_tok = tok

            if self.random.random() < self.noise_params["substitute"] and self.dsl:
                new_tok = self.random.choice(self.dsl)

            edited.append(new_tok)

            if self.random.random() < self.noise_params["insert"] and self.dsl:
                inserted = self.random.choice(self.dsl)
                edited.append(inserted)

        compressed = self._compress_repeats(edited)
        return compressed

    def _compress_repeats(self, tokens: Sequence[str]) -> List[str]:
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

        if temperature <= 0.0:
            best_idx = min(range(len(candidates)), key=lambda i: losses[i])
        else:
            neg_losses = np.array([-l for l in losses], dtype=float) / float(temperature)
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
) -> Tuple[List[List[str]], Any]:
    from .selection import generate_ast_candidates
    from model.dsl import ast_to_tokens, tokens_to_ast, set_library
    from model.dsl.fragments import Library
    from model.program_induction import discover_fragments, apply_fragments
    from model.eval import ib_loss
    
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
    expanded_history: List[List[str]] = []
    chain: List[List[str]] = [list(initial_program)]
    expanded_history.append(expand_program(list(initial_program)))
    current_tokens = list(initial_program)
    
    for gen in range(num_generations):
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
        
        perceived_tokens = _apply_token_noise(current_tokens)
        perceived_ast = tokens_to_ast(perceived_tokens)
        
        candidates = generate_ast_candidates(perceived_ast, num_candidates, rng, library=library)

        target_ast = true_ast if use_true_target else perceived_ast
        
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
        
        if len(library) > 0:
            selected_ast, _ = apply_fragments(selected_ast, library)
        
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
) -> Tuple[List[List[str]], Any]:
    from .selection import generate_ast_candidates, select_bayesian_candidate
    from model.dsl import ast_to_tokens, tokens_to_ast, set_library
    from model.dsl.fragments import Library
    from model.program_induction import discover_fragments, apply_fragments
    
    rng = random.Random(random_seed)
    
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
        should_discover = (
            gen == 1
            or (gen > 1 and gen % fragment_discovery_freq == 0 and len(all_programs) > 1)
        )
        
        if should_discover and len(all_programs) >= 2:
            expanded_programs: List[Sequence[str]] = [expand_program(p) for p in all_programs]
            library = discover_fragments(
                programs=expanded_programs,
                min_length=3,
                max_length=5,
                min_freq=2,
            )
            set_library(library)
        
        obs_tokens = noise_model.transmit_once(current_tokens)
        obs_tokens_expanded = expand_program(obs_tokens)
        obs_ast = tokens_to_ast(obs_tokens_expanded)
        
        current_ast = tokens_to_ast(current_tokens)
        
        obs_candidates = generate_ast_candidates(obs_ast, num_candidates // 2, rng, library=library, max_edits_per_candidate=1)
        current_candidates = generate_ast_candidates(current_ast, num_candidates - len(obs_candidates), rng, library=library, max_edits_per_candidate=1)
        
        candidates = obs_candidates + current_candidates
        
        selected_ast = select_bayesian_candidate(
            candidates=candidates,
            obs_ast=obs_ast,
            library=library,
            alpha=alpha,
            lambda_len=lambda_len,
            temperature=temperature,
            rng=rng,
        )
        
        if len(library) > 0:
            selected_ast, _ = apply_fragments(selected_ast, library)
        
        from model.dsl.parser import set_library as parser_set_library, get_library as parser_get_library
        old_library = parser_get_library()
        parser_set_library(None)
        next_tokens = ast_to_tokens(selected_ast)
        parser_set_library(old_library)
        
        chain.append(next_tokens)
        all_programs.append(next_tokens)
        current_tokens = next_tokens
    
    return chain, library


def _softmax_np(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    t = float(temperature) if temperature is not None else 1.0
    t = max(t, 1e-9)
    z = x / t
    z = z - np.max(z)
    e = np.exp(z)
    s = float(np.sum(e))
    return e / s if s > 0 else np.ones_like(e) / float(len(e))


def _entropy_np(probs: np.ndarray) -> float:
    p = np.array(probs, dtype=float)
    p = p[p > 0]
    if len(p) == 0:
        return 0.0
    return float(-np.sum(p * np.log(p)))


def compute_lexeme_mapping_entropy(
    beliefs: Any,
    utterances: Sequence[str],
    actions: Sequence[str],
    epsilon: float = 0.01,
) -> float:
    if not utterances or not actions:
        return 0.0
    action_list = list(actions)
    entropies: List[float] = []
    for utt in utterances:
        probs = np.zeros(len(action_list), dtype=float)
        for lex in beliefs.support():
            pL = float(beliefs.score(lex))
            if pL <= 0.0:
                continue
            intended = lex.language_to_dsl(utt)
            for i, a in enumerate(action_list):
                if intended == a:
                    probs[i] += pL * 1.0
                else:
                    probs[i] += pL * float(epsilon)
        s = float(np.sum(probs))
        if s > 0:
            probs = probs / s
        entropies.append(_entropy_np(probs))
    return float(np.mean(entropies)) if entropies else 0.0


def load_empirical_trials(
    ppt_id: int,
    base_dir: str,
    source_subdir: str = "programs_for_you",
) -> Any:
    path = os.path.join(base_dir, source_subdir, f"programs_ppt_{ppt_id}.json")
    return _load_trials_cached(path).copy()


def _expected_inf(
    beliefs: Any,
    utterance: str,
    intention: str,
    actions: Sequence[str],
    epsilon: float,
) -> float:
    total = 0.0
    for lex in beliefs.support():
        pL = float(beliefs.score(lex))
        if pL <= 0.0:
            continue
        intended = lex.language_to_dsl(utterance)
        p = 1.0 if intended == intention else float(epsilon)
        total += pL * float(np.log(p))
    return float(total)


def _choose_utterance(
    beliefs: Any,
    intention: str,
    utterances: Sequence[str],
    actions: Sequence[str],
    speaker_alpha: float,
    epsilon: float,
    rng: np.random.Generator,
    inf_cache: Optional[Dict[Tuple[str, str], float]] = None,
) -> str:
    utils = np.array(
        [
            (
                float(inf_cache[(str(u), str(intention))])
                if inf_cache is not None and (str(u), str(intention)) in inf_cache
                else _expected_inf(beliefs, u, intention, actions=actions, epsilon=epsilon)
            )
            for u in utterances
        ],
        dtype=float,
    )
    probs = _softmax_np(speaker_alpha * utils, temperature=1.0)
    idx = int(rng.choice(len(utterances), p=probs))
    return str(list(utterances)[idx])


def _program_utility(
    beliefs: Any,
    program_steps: Sequence[str],
    utterances: Sequence[str],
    actions: Sequence[str],
    speaker_alpha: float,
    speaker_beta_cost: float,
    epsilon: float,
    inf_cache: Optional[Dict[Tuple[str, str], float]] = None,
) -> float:
    if len(program_steps) == 0:
        return -1e9

    step_vals: List[float] = []
    for step in program_steps:
        utt_utils = np.array(
            [
                (
                    float(inf_cache[(str(u), str(step))])
                    if inf_cache is not None and (str(u), str(step)) in inf_cache
                    else _expected_inf(beliefs, u, step, actions=actions, epsilon=epsilon)
                )
                for u in utterances
            ],
            dtype=float,
        )
        utt_probs = _softmax_np(speaker_alpha * utt_utils, temperature=1.0)
        step_vals.append(float(np.sum(utt_utils * utt_probs)))

    mean_inf = float(np.mean(step_vals))
    beta = float(speaker_beta_cost)
    return (1.0 - beta) * mean_inf - beta * float(len(program_steps))


def _choose_program_representation(
    beliefs: Any,
    programs_with_length: Dict[str, int],
    utterances: Sequence[str],
    actions: Sequence[str],
    speaker_alpha_prog: float,
    speaker_alpha_utt: float,
    speaker_beta_cost: float,
    epsilon: float,
    rng: np.random.Generator,
    return_entropy: bool = False,
    inf_cache: Optional[Dict[Tuple[str, str], float]] = None,
) -> Any:
    programs = list(programs_with_length.keys())
    utils = np.array(
        [
            _program_utility(
                beliefs,
                program_steps=p.split(" "),
                utterances=utterances,
                actions=actions,
                speaker_alpha=speaker_alpha_utt,
                speaker_beta_cost=speaker_beta_cost,
                epsilon=epsilon,
                inf_cache=inf_cache,
            )
            for p in programs
        ],
        dtype=float,
    )
    probs = _softmax_np(speaker_alpha_prog * utils, temperature=1.0)
    idx = int(rng.choice(len(programs), p=probs))
    chosen = str(programs[idx])
    if return_entropy:
        entropy = _entropy_np(probs)
        return chosen, entropy
    return chosen


def _choose_program_fixed(programs_with_length: Dict[str, int], mode: str = "max") -> str:
    programs = list(programs_with_length.keys())
    if len(programs) == 0:
        return ""
    if mode == "min":
        return min(programs, key=lambda p: int(programs_with_length.get(p, len(p.split(" ")))))
    return max(programs, key=lambda p: int(programs_with_length.get(p, len(p.split(" ")))))


def _extract_chunks(steps: Sequence[str]) -> Set[str]:
    return {s for s in steps if isinstance(s, str) and s.startswith("chunk")}


@dataclass
class ChunkRegistry:
    all_chunks: Set[str]
    min_attempts: int = 5
    min_success_rate: float = 0.4
    p_innovate: float = 0.10
    pre_promote_chunk_correct: float = 0.44
    max_promote_per_gen: int = 1
    active_chunks: Set[str] = field(default_factory=set)
    candidate_attempts: Dict[str, int] = field(default_factory=dict)
    candidate_successes: Dict[str, int] = field(default_factory=dict)

    def choose_candidate(self, rng: np.random.Generator) -> Optional[str]:
        if float(self.p_innovate) <= 0.0:
            return None
        if float(rng.random()) >= float(self.p_innovate):
            return None
        pool = sorted(list(self.all_chunks - set(self.active_chunks)))
        if not pool:
            return None
        idx = int(rng.integers(low=0, high=len(pool)))
        return str(pool[idx])

    def allowed_chunks_for_episode(self, candidate: Optional[str]) -> Set[str]:
        allowed = set(self.active_chunks)
        if candidate is not None:
            allowed.add(candidate)
        return allowed

    def update_candidate(self, chunk: str, attempts: int, successes: int) -> None:
        self.candidate_attempts[chunk] = int(self.candidate_attempts.get(chunk, 0)) + int(attempts)
        self.candidate_successes[chunk] = int(self.candidate_successes.get(chunk, 0)) + int(successes)

    def max_candidate_attempts(self) -> int:
        return int(max(self.candidate_attempts.values(), default=0))

    def max_candidate_success_rate(self) -> float:
        best = 0.0
        for c, a in self.candidate_attempts.items():
            if c in self.active_chunks:
                continue
            aa = int(a)
            if aa <= 0:
                continue
            ss = int(self.candidate_successes.get(c, 0))
            best = max(best, float(ss) / float(aa))
        return float(best)

    def promote(self) -> List[str]:
        eligible = []
        for c, a in list(self.candidate_attempts.items()):
            if c in self.active_chunks:
                continue
            aa = int(a)
            if aa < int(self.min_attempts):
                continue
            ss = int(self.candidate_successes.get(c, 0))
            rate = float(ss) / float(max(1, aa))
            if rate >= float(self.min_success_rate):
                eligible.append((c, rate, aa))
        eligible.sort(key=lambda x: (-x[1], -x[2]))
        promoted: List[str] = []
        k = int(self.max_promote_per_gen)
        if k <= 0:
            return promoted
        for c, _, _ in eligible[:k]:
            self.active_chunks.add(c)
            promoted.append(c)
        return promoted


def _literal_builder_act(
    beliefs: Any,
    utterance: str,
    actions: Sequence[str],
    epsilon: float,
    rng: np.random.Generator,
) -> str:
    scores = []
    for a in actions:
        p = 0.0
        for lex in beliefs.support():
            pL = float(beliefs.score(lex))
            intended = lex.language_to_dsl(utterance)
            p += pL * (1.0 if intended == a else float(epsilon))
        scores.append(p)
    probs = np.array(scores, dtype=float)
    s = float(np.sum(probs))
    if s <= 0.0:
        probs = np.ones_like(probs) / float(len(probs))
    else:
        probs = probs / s
    idx = int(rng.choice(len(actions), p=probs))
    return str(list(actions)[idx])


def _update_posterior(
    role: str,
    prior: Any,
    observations_df: Any,
    actions: Sequence[str],
    utterances: Sequence[str],
    epsilon: float,
) -> Any:
    import pandas as pd
    from model.convention_formation.distribution import EmptyDistribution

    if observations_df is None or len(observations_df) == 0:
        return prior.copy()

    posterior = EmptyDistribution()
    posterior.to_logspace()

    for lex in prior.support():
        p0 = float(prior.score(lex))
        prior_term = float(np.log(max(p0, 1e-12)))
        ll = 0.0

        for _, row in observations_df.iterrows():
            utt = str(row["utterance"])
            intent = str(row["intention"])
            resp = str(row["response"])

            if role == "architect":
                intended = lex.language_to_dsl(utt)
                p = 1.0 if intended == resp else float(epsilon)
                ll += float(np.log(p))
            else:
                expected_utt = lex.dsl_to_language(intent)
                p = 1.0 if expected_utt == utt else float(epsilon)
                ll += float(np.log(p))

        posterior.update({lex: prior_term + ll})

    posterior.renormalize()
    posterior.from_logspace()
    return posterior


def simulate_generation(
    trials: Any,
    arch_prior: Any,
    build_prior: Any,
    speaker_mode: str,
    all_utterances: Sequence[str],
    registry: Optional[ChunkRegistry],
    speaker_alpha_prog: float,
    speaker_alpha_utt: float,
    speaker_beta_cost: float,
    epsilon: float,
    rng: np.random.Generator,
) -> Tuple[Any, Dict[str, float], Dict[str, Any]]:
    import pandas as pd

    if speaker_mode not in {"random_utt", "literal_step", "rsa_step", "rsa_program"}:
        raise ValueError(f"Unknown speaker_mode: {speaker_mode}")

    obs_steps: List[Dict[str, Any]] = []
    cand_events: List[Dict[str, Any]] = []
    prog_choice_entropies: List[float] = []
    for _, trial in trials.iterrows():
        actions = list(trial["dsl"])
        programs_with_length = dict(trial["programs_with_length"])

        inf_cache: Dict[Tuple[str, str], float] = {}
        intentions_set: Set[str] = set(str(a) for a in actions)
        for p in programs_with_length.keys():
            steps_p = str(p).split(" ") if p else []
            for s in steps_p:
                intentions_set.add(str(s))
        for intent in intentions_set:
            for u in all_utterances:
                inf_cache[(str(u), str(intent))] = _expected_inf(
                    arch_prior, str(u), str(intent), actions=actions, epsilon=epsilon
                )

        allowed_candidate_chunk: Optional[str] = None
        allowed_chunks: Optional[Set[str]] = None
        if speaker_mode == "rsa_program" and registry is not None:
            episode_chunks: Set[str] = set()
            for p in programs_with_length.keys():
                steps_p = str(p).split(" ") if p else []
                episode_chunks |= _extract_chunks(steps_p)

            allowed_candidate_chunk = None
            if float(registry.p_innovate) > 0.0 and float(rng.random()) < float(registry.p_innovate):
                pool = sorted(list((registry.all_chunks - set(registry.active_chunks)) & set(episode_chunks)))
                if pool:
                    allowed_candidate_chunk = str(pool[int(rng.integers(low=0, high=len(pool)))])

            allowed_chunks = registry.allowed_chunks_for_episode(allowed_candidate_chunk)
            filtered: Dict[str, int] = {}
            for p, L in programs_with_length.items():
                steps_p = str(p).split(" ") if p else []
                chunks_p = _extract_chunks(steps_p)
                if chunks_p.issubset(set(allowed_chunks)):
                    filtered[str(p)] = int(L)
            if filtered:
                programs_with_length = filtered

        if speaker_mode == "rsa_program":
            chosen_program, prog_ent = _choose_program_representation(
                beliefs=arch_prior,
                programs_with_length=programs_with_length,
                utterances=all_utterances,
                actions=actions,
                speaker_alpha_prog=speaker_alpha_prog,
                speaker_alpha_utt=speaker_alpha_utt,
                speaker_beta_cost=speaker_beta_cost,
                epsilon=epsilon,
                rng=rng,
                return_entropy=True,
                inf_cache=inf_cache,
            )
            prog_choice_entropies.append(float(prog_ent))
        else:
            chosen_program = _choose_program_fixed(programs_with_length, mode="max")

        steps = str(chosen_program).split(" ") if chosen_program else []
        program_len = int(len(steps))
        lengths = [int(v) for v in list(programs_with_length.values())] if programs_with_length else []
        if not lengths:
            L_min, L_max = program_len, program_len
        else:
            L_min, L_max = int(min(lengths)), int(max(lengths))
        denom = float(max(1, L_max - L_min))
        program_level = float(L_max - program_len) / denom if denom > 0 else 0.0

        cand_attempts = 0
        cand_successes = 0
        for step in steps:
            is_chunk = bool(isinstance(step, str) and step.startswith("chunk"))
            is_active_chunk = bool(registry is not None and is_chunk and step in registry.active_chunks)
            if speaker_mode == "random_utt":
                utt = str(rng.choice(list(all_utterances)))
            elif speaker_mode == "literal_step":
                utt_dist = arch_prior.marginalize(lambda L: L.dsl_to_language(step))
                utt = str(utt_dist.sample())
            else:
                utt = _choose_utterance(
                    beliefs=arch_prior,
                    intention=step,
                    utterances=all_utterances,
                    actions=actions,
                    speaker_alpha=speaker_alpha_utt,
                    epsilon=epsilon,
                    rng=rng,
                    inf_cache=inf_cache,
                )

            resp = _literal_builder_act(
                beliefs=build_prior,
                utterance=utt,
                actions=actions,
                epsilon=epsilon,
                rng=rng,
            )

            if registry is not None and is_chunk and step not in registry.active_chunks:
                p_ok = float(registry.pre_promote_chunk_correct)
                ok = bool(float(rng.random()) < p_ok)
                if ok:
                    resp = step
                else:
                    pool = [a for a in actions if a != step]
                    if pool:
                        resp = str(pool[int(rng.integers(low=0, high=len(pool)))])
                if allowed_candidate_chunk is not None and step == allowed_candidate_chunk:
                    cand_attempts += 1
                    cand_successes += 1 if ok else 0

            acc = 1.0 if resp == step else 0.0
            obs_steps.append(
                {
                    "trial": int(trial["trial_num"]),
                    "utterance": utt,
                    "response": resp,
                    "intention": step,
                    "target_program": chosen_program,
                    "acc": acc,
                    "is_chunk": 1.0 if is_chunk else 0.0,
                    "is_active_chunk": 1.0 if is_active_chunk else 0.0,
                    "program_len": float(program_len),
                    "program_level": float(program_level),
                }
            )

        if allowed_candidate_chunk is not None:
            cand_events.append(
                {
                    "candidate": allowed_candidate_chunk,
                    "attempts": int(cand_attempts),
                    "successes": int(cand_successes),
                }
            )

    gen_df = pd.DataFrame(obs_steps)
    acc_comm = float(gen_df["acc"].mean()) if len(gen_df) > 0 else 0.0
    if len(gen_df) > 0:
        msg_len = float(
            gen_df.groupby("trial")["target_program"]
            .apply(lambda s: len(str(s.iloc[0]).split(" ")))
            .mean()
        )
        frag_rate = float(gen_df["is_chunk"].mean())
        num_chunk_steps = float(gen_df["is_chunk"].sum())
        active_chunk_rate = float(gen_df["is_active_chunk"].sum()) / float(max(1.0, num_chunk_steps)) if num_chunk_steps > 0 else 0.0
        program_level_mean = float(gen_df.groupby("trial")["program_level"].first().mean())
    else:
        msg_len = 0.0
        frag_rate = 0.0
        active_chunk_rate = 0.0
        program_level_mean = 0.0

    prog_choice_ent_mean = float(np.mean(prog_choice_entropies)) if prog_choice_entropies else 0.0
    extra = {"cand_events": cand_events}
    return gen_df, {"acc_comm": acc_comm, "msg_len": msg_len, "frag_rate": frag_rate, "active_chunk_rate": active_chunk_rate, "program_level": program_level_mean, "program_choice_entropy": prog_choice_ent_mean}, extra

def run_comm_chain_bayes_rsa(
    ppt_id: int,
    data_model_dir: str,
    num_generations: int = 20,
    lexemes: Optional[Sequence[str]] = None,
    speaker_mode: str = "rsa_program",
    min_attempts: int = 5,
    min_success_rate: float = 0.4,
    p_innovate: float = 0.10,
    pre_promote_chunk_correct: float = 0.55,
    max_promote_per_gen: int = 1,
    speaker_alpha_prog: float = 2.0,
    speaker_alpha_utt: float = 2.0,
    speaker_beta_cost: float = 0.3,
    epsilon: float = 0.01,
    random_seed: int = 0,
    source_subdir: str = "programs_for_you",
    trace_gens: Optional[Sequence[int]] = None,
    return_traces: bool = False,
    init_arch_prior: Optional[Any] = None,
    init_build_prior: Optional[Any] = None,
    init_active_chunks: Optional[Set[str]] = None,
    freeze_updates: bool = False,
    return_final_state: bool = False,
) -> Any:
    import pandas as pd
    from model.convention_formation.distribution import LexiconPrior, Distribution
    from model.convention_formation.lexicon import BlockLexicon

    if lexemes is None:
        lexemes = ["blah", "blab", "bloop", "bleep", "floop"]

    trials = load_empirical_trials(
        ppt_id=ppt_id,
        base_dir=data_model_dir,
        source_subdir=source_subdir,
    ).copy()

    full_dsl = sorted(set().union(*[set(x) for x in trials["dsl"].tolist()]))

    utt_lex = BlockLexicon(full_dsl, list(lexemes))
    all_utterances = sorted(list(utt_lex.utterances))

    prior0 = LexiconPrior(full_dsl, list(lexemes))
    arch_prior: Distribution = init_arch_prior if init_arch_prior is not None else prior0
    build_prior: Distribution = init_build_prior if init_build_prior is not None else prior0

    all_chunks: Set[str] = set()
    for _, tr in trials.iterrows():
        for a in list(tr["dsl"]):
            if isinstance(a, str) and a.startswith("chunk"):
                all_chunks.add(a)
        for p in dict(tr["programs_with_length"]).keys():
            steps_p = str(p).split(" ") if p else []
            all_chunks |= _extract_chunks(steps_p)

    registry = ChunkRegistry(
        all_chunks=set(all_chunks),
        min_attempts=int(min_attempts),
        min_success_rate=float(min_success_rate),
        p_innovate=float(p_innovate) if not freeze_updates else 0.0,
        pre_promote_chunk_correct=float(pre_promote_chunk_correct),
        max_promote_per_gen=int(max_promote_per_gen) if not freeze_updates else 0,
    )
    if init_active_chunks is not None:
        registry.active_chunks = set(init_active_chunks)

    rng = np.random.default_rng(int(random_seed))
    summaries: List[Dict[str, Any]] = []
    traces: Dict[int, Any] = {}
    trace_set: Set[int] = set(int(x) for x in list(trace_gens)) if trace_gens is not None else set()
    seen_chunks_used: Set[str] = set()

    for gen in range(int(num_generations)):
        gen_df, summary, extra = simulate_generation(
            trials=trials,
            arch_prior=arch_prior,
            build_prior=build_prior,
            speaker_mode=speaker_mode,
            all_utterances=all_utterances,
            registry=registry,
            speaker_alpha_prog=speaker_alpha_prog,
            speaker_alpha_utt=speaker_alpha_utt,
            speaker_beta_cost=speaker_beta_cost,
            epsilon=epsilon,
            rng=rng,
        )

        if int(gen) in trace_set:
            traces[int(gen)] = gen_df.copy()

        for ev in list(extra.get("cand_events", [])):
            c = str(ev.get("candidate", ""))
            attempts = int(ev.get("attempts", 0))
            successes = int(ev.get("successes", 0))
            if c and attempts > 0:
                registry.update_candidate(c, attempts=attempts, successes=successes)
        promoted = registry.promote()

        chunks_used_set: Set[str] = set()
        if gen_df is not None and len(gen_df) > 0:
            chunks_used_set = set(
                str(x) for x in gen_df.loc[gen_df["is_chunk"] > 0.0, "intention"].astype(str).tolist()
            )
        new_chunks_used = set(chunks_used_set - set(seen_chunks_used))
        reuse_chunk_rate = (
            float(len(chunks_used_set & set(seen_chunks_used))) / float(max(1, len(chunks_used_set)))
            if len(chunks_used_set) > 0
            else 0.0
        )
        seen_chunks_used |= set(chunks_used_set)

        arch_lex_ent = compute_lexeme_mapping_entropy(arch_prior, all_utterances, full_dsl, epsilon)
        build_lex_ent = compute_lexeme_mapping_entropy(build_prior, all_utterances, full_dsl, epsilon)
        lexeme_mapping_ent = float(arch_lex_ent + build_lex_ent) / 2.0

        summaries.append(
            {
                "generation": float(gen),
                "acc_comm": float(summary["acc_comm"]),
                "msg_len": float(summary["msg_len"]),
                "frag_rate": float(summary["frag_rate"]),
                "active_chunk_rate": float(summary.get("active_chunk_rate", 0.0)),
                "program_level": float(summary.get("program_level", 0.0)),
                "program_choice_entropy": float(summary.get("program_choice_entropy", 0.0)),
                "lexeme_mapping_entropy": float(lexeme_mapping_ent),
                "ppt_id": float(ppt_id),
                "speaker_mode": str(speaker_mode),
                "num_active_chunks": float(len(registry.active_chunks)),
                "max_candidate_attempts": float(registry.max_candidate_attempts()),
                "max_candidate_success_rate": float(registry.max_candidate_success_rate()),
                "num_chunks_used": float(len(chunks_used_set)),
                "num_new_chunks_used": float(len(new_chunks_used)),
                "reuse_chunk_rate": float(reuse_chunk_rate),
                "chunks_used": "|".join(sorted(list(chunks_used_set))),
                "num_promoted_this_gen": float(len(promoted)),
            }
        )

        if not freeze_updates:
            arch_prior = _update_posterior(
                role="architect",
                prior=arch_prior,
                observations_df=gen_df,
                actions=full_dsl,
                utterances=all_utterances,
                epsilon=epsilon,
            )
            build_prior = _update_posterior(
                role="builder",
                prior=build_prior,
                observations_df=gen_df,
                actions=full_dsl,
                utterances=all_utterances,
                epsilon=epsilon,
            )

    out_df = pd.DataFrame(summaries)
    final_state = {
        "arch_prior": arch_prior,
        "build_prior": build_prior,
        "active_chunks": set(registry.active_chunks),
    }
    if bool(return_final_state) and bool(return_traces):
        return out_df, traces, final_state
    if bool(return_final_state):
        return out_df, final_state
    if bool(return_traces):
        return out_df, traces
    return out_df


def _run_comm_worker(args: Dict[str, Any]) -> Any:
    import pandas as pd

    res = run_comm_chain_bayes_rsa(
        ppt_id=int(args["ppt_id"]),
        data_model_dir=str(args["data_model_dir"]),
        num_generations=int(args["num_generations"]),
        lexemes=args.get("lexemes"),
        speaker_mode=str(args["speaker_mode"]),
        min_attempts=int(args.get("min_attempts", 5)),
        min_success_rate=float(args.get("min_success_rate", 0.30)),
        p_innovate=float(args.get("p_innovate", 0.10)),
        pre_promote_chunk_correct=float(args.get("pre_promote_chunk_correct", 0.60)),
        max_promote_per_gen=int(args.get("max_promote_per_gen", 1)),
        speaker_alpha_prog=float(args.get("speaker_alpha_prog", 2.0)),
        speaker_alpha_utt=float(args.get("speaker_alpha_utt", 2.0)),
        speaker_beta_cost=float(args.get("speaker_beta_cost", 0.3)),
        epsilon=float(args.get("epsilon", 0.01)),
        random_seed=int(args.get("random_seed", 0)),
        source_subdir=str(args.get("source_subdir", "programs_for_you")),
    ).copy()
    df = res[0] if isinstance(res, tuple) else res
    df = df.copy()

    df["model"] = str(args["model_label"]) if "model_label" in args else str(args["speaker_mode"])
    if "variant" in args:
        df["variant"] = str(args["variant"])
    return pd.DataFrame(df)