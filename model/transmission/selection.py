import random
from typing import List, Optional

from model.dsl import Node
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
    if ast.op != 'SEQ':
        return [ast.copy()]
    
    candidates = [ast.copy()]
    max_attempts = num_candidates * 3
    
    edit_types = ['delete', 'insert', 'replace']
    if library is not None and len(library.fragments) > 0:
        edit_types.extend(['replace_with_fragment', 'expand_fragment'])
    
    attempts = 0
    while len(candidates) < num_candidates and attempts < max_attempts:
        attempts += 1
        new_ast = ast.copy()
        if len(new_ast.children) == 0:
            continue
        
        num_edits = rng.randint(1, max_edits_per_candidate)
        edited = False
        
        for _ in range(num_edits):
            if len(new_ast.children) == 0:
                break
            
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
                non_call_indices = [i for i in range(len(new_ast.children)) if new_ast.children[i].op != 'CALL']
                if non_call_indices:
                    idx = rng.choice(non_call_indices)
                    frag = rng.choice(list(library.fragments.values()))
                    new_ast.children[idx] = Node('CALL', {'name': frag.name})
                    edited = True
            
            elif edit_type == 'expand_fragment' and library is not None:
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
    return float(ast.num_nodes())


def structural_mismatch(ast1: Node, ast2: Node) -> float:
    return abs(ast1.num_nodes() - ast2.num_nodes())


def reconstruction_cost(
    candidate: Node,
    perceived: Node,
    lambda_c: float = 1.0,
    lambda_m: float = 1.0
) -> float:
    C = complexity(candidate)
    M = structural_mismatch(candidate, perceived)
    return lambda_c * C + lambda_m * M


# Bayesian listener scoring functions

def log_prior(candidate_ast: Node, library: Optional[Library] = None, lambda_len: float = 1.0) -> float:
    from model.eval.metrics import description_length
    length = description_length(candidate_ast, library)
    return -lambda_len * float(length)


def log_likelihood(candidate_ast: Node, obs_ast: Node, library: Optional[Library] = None, alpha: float = 1.0) -> float:
    from model.eval.metrics import normalized_tree_distance
    from model.dsl.parser import set_library as parser_set_library, get_library as parser_get_library
    
    old_lib = parser_get_library()
    if library is not None:
        parser_set_library(library)
    
    try:
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
        best_idx = max(range(len(candidates)), key=lambda i: log_posts[i])
        return candidates[best_idx]
    else:
        import numpy as np
        max_log_post = max(log_posts)
        shifted = [(lp - max_log_post) / temperature for lp in log_posts]
        exp_vals = [np.exp(s) for s in shifted]
        total = sum(exp_vals)
        probs = [e / total for e in exp_vals]
        
        if rng is None:
            rng = random.Random()
        
        chosen_idx = rng.choices(range(len(candidates)), weights=probs)[0]
        return candidates[chosen_idx]

