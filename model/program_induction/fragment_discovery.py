"""Fragment discovery from programs."""

from typing import List, Sequence, Tuple
from collections import Counter

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from model.dsl import Node, tokens_to_ast, ast_to_tokens
from model.dsl.fragments import Fragment, Library


MIN_FRAGMENT_LENGTH = 2
MAX_FRAGMENT_LENGTH = 5
MIN_FREQUENCY = 2


def discover_fragments(
    programs: List[Sequence[str]],
    min_length: int = MIN_FRAGMENT_LENGTH,
    max_length: int = MAX_FRAGMENT_LENGTH,
    min_freq: int = MIN_FREQUENCY,
) -> Library:
    """
    Discover frequent subsequences across programs and create a fragment library.
    
    Simple n-gram approach:
    - Extract all subsequences of length [min_length, max_length]
    - Count occurrences across all programs
    - Keep those with frequency >= min_freq
    """
    library = Library()
    
    # Collect all subsequences
    subseq_counts = Counter()
    for prog in programs:
        tokens = list(prog)
        for length in range(min_length, min(max_length + 1, len(tokens) + 1)):
            for i in range(len(tokens) - length + 1):
                subseq = tuple(tokens[i:i + length])
                subseq_counts[subseq] += 1
    
    # Create fragments from frequent subsequences
    fragment_id = 1
    for subseq, freq in subseq_counts.items():
        if freq >= min_freq:
            # Convert subsequence to AST
            frag_ast = tokens_to_ast(list(subseq))
            frag = Fragment(name=f'F{fragment_id}', body=frag_ast, frequency=freq)
            library.add(frag)
            fragment_id += 1
    
    return library


def _ast_equals(ast1: Node, ast2: Node) -> bool:
    """
    Check if two ASTs are structurally equivalent.
    
    This is a simple structural equality check based on op, args, and children.
    """
    if ast1.op != ast2.op:
        return False
    if ast1.args != ast2.args:
        return False
    if len(ast1.children) != len(ast2.children):
        return False
    return all(_ast_equals(c1, c2) for c1, c2 in zip(ast1.children, ast2.children))


def apply_fragments(ast: Node, library: Library) -> Tuple[Node, int]:
    """
    Replace occurrences of fragment bodies in the AST with CALL nodes.
    
    This function matches fragments at the AST level (not token level) to avoid
    issues with token count vs AST node count mismatches.
    
    Returns:
        (modified_ast, num_replacements)
    """
    if ast.op != 'SEQ':
        return ast.copy(), 0
    
    if len(ast.children) == 0:
        return ast.copy(), 0
    
    num_replacements = 0
    
    # Try to replace with fragments (greedy left-to-right)
    # Match at AST level, trying longest fragments first
    new_children = []
    i = 0
    while i < len(ast.children):
        replaced = False
        
        # Sort fragments by size (longest first) to maximize compression
        sorted_fragments = sorted(
            library.fragments.values(), 
            key=lambda f: f.body.num_nodes(), 
            reverse=True
        )
        
        for frag in sorted_fragments:
            frag_ast = frag.body
            frag_size = len(frag_ast.children) if frag_ast.op == 'SEQ' else 1
            
            # Check if fragment can match at current position
            if i + frag_size <= len(ast.children):
                # Create candidate AST from children slice
                if frag_ast.op == 'SEQ':
                    # Fragment is a sequence, check if children match
                    candidate_children = ast.children[i:i + frag_size]
                    candidate_ast = Node('SEQ', children=candidate_children)
                    
                    # Check structural match
                    if _ast_equals(frag_ast, candidate_ast):
                        # Replace with CALL node
                        new_children.append(Node('CALL', {'name': frag.name}))
                        i += frag_size
                        num_replacements += 1
                        replaced = True
                        break
                else:
                    # Fragment is a single node, check direct match
                    if _ast_equals(frag_ast, ast.children[i]):
                        new_children.append(Node('CALL', {'name': frag.name}))
                        i += 1
                        num_replacements += 1
                        replaced = True
                        break
        
        if not replaced:
            # Recursively apply fragments to this child if it's a SEQ node
            child = ast.children[i]
            if child.op == 'SEQ' and len(child.children) > 0:
                child_modified, child_replacements = apply_fragments(child, library)
                new_children.append(child_modified)
                num_replacements += child_replacements
            else:
                new_children.append(child.copy())
            i += 1
    
    return Node('SEQ', children=new_children), num_replacements


def fragment_usage(ast: Node) -> float:
    """
    Compute the fraction of nodes that are fragment calls.
    
    Returns fragment_calls / total_nodes.
    """
    total = ast.num_nodes()
    if total == 0:
        return 0.0
    
    def count_calls(node):
        count = 1 if node.op == 'CALL' else 0
        for child in node.children:
            count += count_calls(child)
        return count
    
    calls = count_calls(ast)
    return calls / float(total)

