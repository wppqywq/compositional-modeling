"""AST-level reconstruction and selection for transmission chains."""

import random
from typing import List, Sequence, Optional

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from model.dsl import Node, tokens_to_ast, ast_to_tokens
from model.dsl.fragments import Library


PRIMITIVE_OPS = [
    ('PLACE', {'shape': 'h'}),
    ('PLACE', {'shape': 'v'}),
    ('MOVE', {'direction': 'left', 'distance': 1}),
    ('MOVE', {'direction': 'right', 'distance': 1}),
]


def generate_ast_candidates(
    ast: Node,
    num_candidates: int,
    rng: random.Random,
    library: Optional[Library] = None,
    max_edits_per_candidate: int = 3,
) -> List[Node]:
    """
    Generate candidate AST variants via local edits and fragment-based transformations.
    
    Operations:
    - Delete a child node
    - Insert a random primitive action
    - Replace a child node with a random primitive
    - Replace a subtree with a fragment call (if library available)
    - Expand a fragment call (if library available)
    
    Args:
        ast: Base AST to generate candidates from
        num_candidates: Number of candidates to generate (including identity)
        rng: Random number generator
        library: Optional fragment library for fragment-based edits
    """
    if ast.op != 'SEQ':
        return [ast.copy()]
    
    candidates = [ast.copy()]  # Always include identity
    max_attempts = num_candidates * 3  # Try harder to get diverse candidates
    
    # Determine edit types based on whether library is available
    edit_types = ['delete', 'insert', 'replace']
    if library is not None and len(library.fragments) > 0:
        edit_types.extend(['replace_with_fragment', 'expand_fragment'])
    
    attempts = 0
    while len(candidates) < num_candidates and attempts < max_attempts:
        attempts += 1
        new_ast = ast.copy()
        if len(new_ast.children) == 0:
            continue
        
        # Apply 1 to max_edits_per_candidate edits
        num_edits = rng.randint(1, max_edits_per_candidate)
        edited = False
        
        for _ in range(num_edits):
            if len(new_ast.children) == 0:
                break
            
            # Choose random edit type
            edit_type = rng.choice(edit_types)
            
            if edit_type == 'delete' and len(new_ast.children) > 1:
                idx = rng.randint(0, len(new_ast.children) - 1)
                new_ast.children.pop(idx)
                edited = True
            
            elif edit_type == 'insert':
                idx = rng.randint(0, len(new_ast.children))
                op, args = rng.choice(PRIMITIVE_OPS)
                new_ast.children.insert(idx, Node(op, args.copy()))
                edited = True
            
            elif edit_type == 'replace' and len(new_ast.children) > 0:
                idx = rng.randint(0, len(new_ast.children) - 1)
                op, args = rng.choice(PRIMITIVE_OPS)
                new_ast.children[idx] = Node(op, args.copy())
                edited = True
            
            elif edit_type == 'replace_with_fragment' and library is not None and len(new_ast.children) > 0:
                # Try to replace a random subtree with a fragment call
                non_call_indices = [i for i in range(len(new_ast.children)) if new_ast.children[i].op != 'CALL']
                if non_call_indices:
                    idx = rng.choice(non_call_indices)
                    frag = rng.choice(list(library.fragments.values()))
                    new_ast.children[idx] = Node('CALL', {'name': frag.name})
                    edited = True
            
            elif edit_type == 'expand_fragment' and library is not None:
                # Try to expand a random fragment call
                call_indices = [i for i in range(len(new_ast.children)) if new_ast.children[i].op == 'CALL']
                if call_indices:
                    idx = rng.choice(call_indices)
                    call_node = new_ast.children[idx]
                    frag_name = call_node.args.get('name')
                    frag = library.get(frag_name) if frag_name else None
                    if frag is not None:
                        new_ast.children[idx] = frag.body.copy()
                        edited = True
        
        if edited:
            candidates.append(new_ast)
    
    return candidates


def complexity(ast: Node) -> float:
    """Simple complexity measure: number of nodes in AST."""
    return float(ast.num_nodes())


def structural_mismatch(ast1: Node, ast2: Node) -> float:
    """
    Simple structural mismatch: absolute difference in node count.
    
    A more sophisticated version would use tree edit distance.
    """
    return abs(ast1.num_nodes() - ast2.num_nodes())


def reconstruction_cost(
    candidate: Node,
    perceived: Node,
    lambda_c: float = 1.0,
    lambda_m: float = 1.0
) -> float:
    """
    Compute reconstruction cost for a candidate AST.
    
    J(ast) = lambda_c * complexity(ast) + lambda_m * mismatch(ast, perceived)
    """
    C = complexity(candidate)
    M = structural_mismatch(candidate, perceived)
    return lambda_c * C + lambda_m * M


# Bayesian listener scoring functions

def log_prior(candidate_ast: Node, library: Optional[Library] = None, lambda_len: float = 1.0) -> float:
    """
    Compute log prior for a candidate program.
    
    log p(candidate) = -lambda_len * description_length(candidate)
    
    This favors shorter programs (abstraction via fragments).
    """
    from model.eval.metrics import description_length
    length = description_length(candidate_ast, library)
    return -lambda_len * float(length)


def log_likelihood(candidate_ast: Node, obs_ast: Node, library: Optional[Library] = None, alpha: float = 1.0) -> float:
    """
    Compute log likelihood of observing obs_ast given candidate_ast.
    
    log p(obs | candidate) = -alpha * normalized_tree_distance(candidate, obs)
    
    This measures how similar the candidate is to the observation after
    expanding fragment calls (semantic similarity).
    
    Args:
        candidate_ast: Reconstructed program candidate
        obs_ast: Observed/noisy program
        library: Fragment library for expanding CALL nodes
        alpha: Scaling factor for distance penalty
    """
    from model.eval.metrics import normalized_tree_distance
    from model.dsl.parser import set_library as parser_set_library, get_library as parser_get_library
    
    # Temporarily set library for correct expansion during distance computation
    old_lib = parser_get_library()
    if library is not None:
        parser_set_library(library)
    
    try:
        # Distance already expands CALL nodes via expand_calls in normalized_tree_distance
        dist = normalized_tree_distance(candidate_ast, obs_ast)
        log_lik = -alpha * dist
    finally:
        parser_set_library(old_lib)
    
    return log_lik


def select_bayesian_candidate(
    candidates: List[Node],
    obs_ast: Node,
    library: Optional[Library] = None,
    alpha: float = 10.0,
    lambda_len: float = 0.1,
    temperature: float = 0.0,
    rng: Optional[random.Random] = None,
) -> Node:
    """
    Select a candidate using Bayesian posterior (MAP or soft-max sampling).
    
    Posterior: log p(candidate | obs) = log_likelihood + log_prior
    
    Args:
        candidates: List of candidate ASTs
        obs_ast: Observed/noisy program AST
        library: Fragment library
        alpha: Likelihood scaling (higher = stronger fit to observation)
        lambda_len: Prior scaling (higher = stronger preference for short programs)
        temperature: Selection temperature (0 = greedy MAP, >0 = soft sampling)
        rng: Random number generator for sampling
    
    Returns:
        Selected candidate AST
    """
    if len(candidates) == 0:
        raise ValueError("Cannot select from empty candidate list")
    
    if len(candidates) == 1:
        return candidates[0]
    
    # Compute log posterior for each candidate
    log_posts = []
    for cand in candidates:
        log_pri = log_prior(cand, library, lambda_len)
        log_lik = log_likelihood(cand, obs_ast, library, alpha)
        log_post = log_pri + log_lik
        log_posts.append(log_post)
    
    # Select candidate
    if temperature <= 0.0:
        # Greedy MAP: argmax posterior
        best_idx = max(range(len(candidates)), key=lambda i: log_posts[i])
        return candidates[best_idx]
    else:
        # Soft sampling via softmax
        import numpy as np
        # Shift for numerical stability
        max_log_post = max(log_posts)
        shifted = [(lp - max_log_post) / temperature for lp in log_posts]
        exp_vals = [np.exp(s) for s in shifted]
        total = sum(exp_vals)
        probs = [e / total for e in exp_vals]
        
        if rng is None:
            rng = random.Random()
        
        chosen_idx = rng.choices(range(len(candidates)), weights=probs)[0]
        return candidates[chosen_idx]

