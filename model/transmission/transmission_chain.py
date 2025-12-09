import random
from typing import Dict, List, Sequence, Optional

import numpy as np


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


