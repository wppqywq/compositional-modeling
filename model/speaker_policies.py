"""
Speaker policies for program and utterance selection.

Contains:
- RSA-based program/utterance choice
- Pact-Aware policy (Brennan & Clark conceptual pacts)
- Drift-Biased policy (Bentley neutral copying)
- Belief update utilities
"""

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from .convention_formation.distribution import EmptyDistribution
from .convention_formation.lexicon import BlockLexicon


def softmax_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    x = x - float(np.max(x))
    ex = np.exp(x)
    s = float(ex.sum())
    if s <= 0.0:
        return np.ones_like(ex) / float(max(1, len(ex)))
    return ex / s


def entropy_np(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    p = p[p > 0.0]
    if p.size == 0:
        return 0.0
    val = float(np.sum(p * np.log(p)))
    return -val


def as_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def extract_chunks(steps: List[str]) -> Set[str]:
    return {s for s in steps if isinstance(s, str) and s.startswith("chunk")}


# RSA primitives

def B0(utterance: str, lex: BlockLexicon, actions: List[str], epsilon: float) -> EmptyDistribution:
    d = EmptyDistribution()
    intended = lex.language_to_dsl(str(utterance))
    for a in actions:
        d.update({str(a): 1.0 if str(a) == str(intended) else float(epsilon)})
    d.renormalize()
    return d


def A0(intention: str, lex: BlockLexicon, utterances: List[str], epsilon: float) -> EmptyDistribution:
    d = EmptyDistribution()
    expected = lex.dsl_to_language(str(intention))
    for u in utterances:
        d.update({str(u): 1.0 if str(u) == str(expected) else float(epsilon)})
    d.renormalize()
    return d


def update_beliefs(
    role: str,
    prior: Any,
    observations_df: pd.DataFrame,
    utterances: List[str],
    epsilon: float,
) -> Any:
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
            row_actions_obj = row["actions"] if "actions" in row else []
            row_actions = [str(x) for x in list(row_actions_obj)]
            if role == "architect":
                ll += float(np.log(B0(utt, lex, actions=row_actions, epsilon=epsilon).score(resp)))
            else:
                ll += float(np.log(A0(intent, lex, utterances=utterances, epsilon=epsilon).score(utt)))
        posterior.update({lex: prior_term + ll})

    posterior.renormalize()
    posterior.from_logspace()
    return posterior


def expected_inf(
    beliefs: Any,
    utterance: str,
    intention: str,
    actions: List[str],
    epsilon: float,
) -> float:
    total = 0.0
    for lex in beliefs.support():
        pL = float(beliefs.score(lex))
        if pL <= 0.0:
            continue
        total += pL * float(np.log(B0(utterance, lex, actions=actions, epsilon=epsilon).score(intention)))
    return float(total)


def choose_utterance_rsa(
    beliefs: Any,
    utterances: List[str],
    intention: str,
    actions: List[str],
    alpha_utt: float,
    epsilon: float,
    rng: np.random.Generator,
) -> str:
    utt_utils = np.array(
        [expected_inf(beliefs, u, intention, actions=actions, epsilon=epsilon) for u in utterances],
        dtype=float,
    )
    probs = softmax_np(float(alpha_utt) * utt_utils)
    idx = int(rng.choice(len(utterances), p=probs))
    return str(utterances[idx])


def compute_program_base_utility(
    beliefs: Any,
    program: str,
    utterances: List[str],
    actions: List[str],
    alpha_utt: float,
    beta_cost: float,
    epsilon: float,
) -> float:
    steps = str(program).split(" ") if program else []
    if not steps:
        return -1e9
    step_vals: List[float] = []
    for step in steps:
        utt_utils = np.array(
            [expected_inf(beliefs, u, step, actions=actions, epsilon=epsilon) for u in utterances],
            dtype=float,
        )
        utt_probs = softmax_np(float(alpha_utt) * utt_utils)
        step_vals.append(float(np.sum(utt_utils * utt_probs)))
    mean_inf = float(np.mean(step_vals)) if step_vals else -1e9
    util = (1.0 - float(beta_cost)) * mean_inf - float(beta_cost) * float(len(steps))
    return float(util)


def choose_program_rsa(
    beliefs: Any,
    programs_with_length: Dict[str, int],
    utterances: List[str],
    actions: List[str],
    alpha_prog: float,
    alpha_utt: float,
    beta_cost: float,
    epsilon: float,
    rng: np.random.Generator,
) -> Tuple[str, float]:
    programs = list(programs_with_length.keys())
    if not programs:
        return "", 0.0

    utils = np.array([
        compute_program_base_utility(beliefs, p, utterances, actions, alpha_utt, beta_cost, epsilon)
        for p in programs
    ], dtype=float)

    probs = softmax_np(float(alpha_prog) * utils)
    idx = int(rng.choice(len(programs), p=probs))
    return str(programs[idx]), entropy_np(probs)


def compute_lexeme_mapping_entropy(
    beliefs: Any,
    utterances: List[str],
    actions: List[str],
    epsilon: float,
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
            intended = str(lex.language_to_dsl(str(utt)))
            for i, a in enumerate(action_list):
                probs[i] += pL * (1.0 if intended == str(a) else float(epsilon))
        s = float(np.sum(probs))
        if s > 0.0:
            probs = probs / s
        entropies.append(entropy_np(probs))
    return float(np.mean(entropies)) if entropies else 0.0


# Pact Memory for Brennan & Clark style conceptual pacts (partner-specific)

class PactMemory:
    def __init__(self) -> None:
        self.n_k: Dict[str, int] = {}   # usage count per key
        self.r_k: Dict[str, int] = {}   # ratified count per key
        self.q_k: Dict[str, int] = {}   # repair count per key
        self.k_star: Optional[str] = None  # active pact key (last ratified)

    def get_strength(self, k: str) -> float:
        r = int(self.r_k.get(k, 0))
        q = int(self.q_k.get(k, 0))
        return float(np.log(1.0 + r) - np.log(1.0 + q))

    def compute_switching_cost(self, k: str, eta: float) -> float:
        if self.k_star is None or k == self.k_star:
            return 0.0
        return float(eta) * self.get_strength(self.k_star)

    def record_proposal(self, k: str) -> None:
        self.n_k[k] = int(self.n_k.get(k, 0)) + 1

    def record_ratify(self, k: str) -> None:
        self.r_k[k] = int(self.r_k.get(k, 0)) + 1
        self.k_star = k

    def record_repair(self, k: str) -> None:
        self.q_k[k] = int(self.q_k.get(k, 0)) + 1


def choose_program_pact(
    beliefs: Any,
    programs_with_length: Dict[str, int],
    utterances: List[str],
    actions: List[str],
    alpha_prog: float,
    alpha_utt: float,
    beta_cost: float,
    epsilon: float,
    pact_memory: PactMemory,
    pact_gamma: float,
    pact_eta: float,
    rng: np.random.Generator,
) -> Tuple[str, float]:
    programs = list(programs_with_length.keys())
    if not programs:
        return "", 0.0

    utils: List[float] = []
    for p in programs:
        base_util = compute_program_base_utility(
            beliefs, p, utterances, actions, alpha_utt, beta_cost, epsilon
        )
        switch_cost = pact_memory.compute_switching_cost(p, float(pact_eta))
        util = base_util - float(pact_gamma) * switch_cost
        utils.append(float(util))

    probs = softmax_np(float(alpha_prog) * np.array(utils, dtype=float))
    idx = int(rng.choice(len(programs), p=probs))
    return str(programs[idx]), entropy_np(probs)


# Drift Pool for Bentley-style random copying (mixed: previous-gen + within-ppt)

class DriftPool:
    def __init__(self) -> None:
        self.h_prev_gen: Dict[str, int] = {}      # keys from previous generation
        self.h_within_ppt: Dict[int, Dict[str, int]] = {}  # keys per ppt_id

    def record_key_prev_gen(self, k: str) -> None:
        self.h_prev_gen[k] = int(self.h_prev_gen.get(k, 0)) + 1

    def record_key_within_ppt(self, ppt_id: int, k: str) -> None:
        if ppt_id not in self.h_within_ppt:
            self.h_within_ppt[ppt_id] = {}
        self.h_within_ppt[ppt_id][k] = int(self.h_within_ppt[ppt_id].get(k, 0)) + 1

    def reset_prev_gen(self) -> None:
        self.h_prev_gen = {}

    def get_mixed_distribution(
        self,
        ppt_id: int,
        candidate_keys: List[str],
        weight_prev_gen: float,
        tau: float,
    ) -> Dict[str, float]:
        h_ppt = self.h_within_ppt.get(ppt_id, {})
        mixed: Dict[str, float] = {}
        for k in candidate_keys:
            cnt_prev = float(self.h_prev_gen.get(k, 0))
            cnt_ppt = float(h_ppt.get(k, 0))
            mixed[k] = float(weight_prev_gen) * cnt_prev + (1.0 - float(weight_prev_gen)) * cnt_ppt
        total = sum(v ** float(tau) for v in mixed.values())
        if total <= 0.0:
            return {}
        return {k: (v ** float(tau)) / total for k, v in mixed.items()}

    def sample_key(
        self,
        ppt_id: int,
        candidate_keys: List[str],
        weight_prev_gen: float,
        tau: float,
        rng: np.random.Generator,
    ) -> Optional[str]:
        dist = self.get_mixed_distribution(ppt_id, candidate_keys, weight_prev_gen, tau)
        if not dist:
            return None
        keys = list(dist.keys())
        probs = np.array([dist[k] for k in keys], dtype=float)
        s = float(np.sum(probs))
        if s <= 0.0:
            return None
        probs = probs / s
        idx = int(rng.choice(len(keys), p=probs))
        return str(keys[idx])


def choose_program_drift(
    beliefs: Any,
    programs_with_length: Dict[str, int],
    utterances: List[str],
    actions: List[str],
    alpha_prog: float,
    alpha_utt: float,
    beta_cost: float,
    epsilon: float,
    drift_pool: DriftPool,
    ppt_id: int,
    drift_epsilon: float,
    drift_tau: float,
    drift_mu: float,
    drift_weight_prev_gen: float,
    rng: np.random.Generator,
) -> Tuple[str, float]:
    programs = list(programs_with_length.keys())
    if not programs:
        return "", 0.0

    # Channel decision: copying vs rational
    if float(rng.random()) < float(drift_epsilon):
        # Copying channel
        if float(rng.random()) < float(drift_mu):
            # Innovation: pick a random program (possibly not in pool)
            chosen = str(programs[int(rng.integers(low=0, high=len(programs)))])
            return chosen, 0.0
        else:
            # Sample from mixed pool
            sampled_key = drift_pool.sample_key(
                ppt_id=int(ppt_id),
                candidate_keys=programs,
                weight_prev_gen=float(drift_weight_prev_gen),
                tau=float(drift_tau),
                rng=rng,
            )
            if sampled_key is not None and sampled_key in programs:
                return str(sampled_key), 0.0
            # Fallback to RSA if sampling fails
            return choose_program_rsa(
                beliefs, programs_with_length, utterances, actions,
                alpha_prog, alpha_utt, beta_cost, epsilon, rng
            )
    else:
        # Rational RSA channel
        return choose_program_rsa(
            beliefs, programs_with_length, utterances, actions,
            alpha_prog, alpha_utt, beta_cost, epsilon, rng
        )

