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
    
    # Determine edit types based on whether library is available
    edit_types = ['delete', 'insert', 'replace']
    if library is not None and len(library.fragments) > 0:
        edit_types.extend(['replace_with_fragment', 'expand_fragment'])
    
    for _ in range(num_candidates - 1):
        new_ast = ast.copy()
        if len(new_ast.children) == 0:
            candidates.append(new_ast)
            continue
        
        # Choose random edit type
        edit_type = rng.choice(edit_types)
        
        if edit_type == 'delete' and len(new_ast.children) > 1:
            idx = rng.randint(0, len(new_ast.children) - 1)
            new_ast.children.pop(idx)
        
        elif edit_type == 'insert':
            idx = rng.randint(0, len(new_ast.children))
            op, args = rng.choice(PRIMITIVE_OPS)
            new_ast.children.insert(idx, Node(op, args.copy()))
        
        elif edit_type == 'replace' and len(new_ast.children) > 0:
            idx = rng.randint(0, len(new_ast.children) - 1)
            op, args = rng.choice(PRIMITIVE_OPS)
            new_ast.children[idx] = Node(op, args.copy())
        
        elif edit_type == 'replace_with_fragment' and library is not None and len(new_ast.children) > 0:
            # Try to replace a random subtree with a fragment call
            # Find a random non-CALL child
            non_call_indices = [i for i in range(len(new_ast.children)) if new_ast.children[i].op != 'CALL']
            if non_call_indices:
                idx = rng.choice(non_call_indices)
                # Choose a random fragment
                frag = rng.choice(list(library.fragments.values()))
                new_ast.children[idx] = Node('CALL', {'name': frag.name})
        
        elif edit_type == 'expand_fragment' and library is not None:
            # Try to expand a random fragment call
            call_indices = [i for i in range(len(new_ast.children)) if new_ast.children[i].op == 'CALL']
            if call_indices:
                idx = rng.choice(call_indices)
                call_node = new_ast.children[idx]
                frag_name = call_node.args.get('name')
                frag = library.get(frag_name) if frag_name else None
                if frag is not None:
                    # Replace CALL with fragment body
                    new_ast.children[idx] = frag.body.copy()
        
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

