"""Evaluation metrics for AST-based programs."""

from typing import Sequence

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from model.dsl import Node


def tree_edit_distance(ast1: Node, ast2: Node) -> int:
    """
    Compute a simple tree edit distance between two ASTs.
    
    This is a minimal recursive implementation that counts:
    - Node replacements (different op or args)
    - Insertions/deletions (different number of children)
    
    For simplicity, we only consider ordered children.
    """
    # If ops differ, must replace this node plus recursively handle children
    if ast1.op != ast2.op or ast1.args != ast2.args:
        # Cost of replacement + costs of making children match
        cost = 1
        # Recursively align children (simplified)
        max_children = max(len(ast1.children), len(ast2.children))
        for i in range(max_children):
            if i < len(ast1.children) and i < len(ast2.children):
                cost += tree_edit_distance(ast1.children[i], ast2.children[i])
            else:
                # One tree has extra child, count as insertion/deletion
                extra = ast1.children[i] if i < len(ast1.children) else ast2.children[i]
                cost += extra.num_nodes()
        return cost
    
    # Ops match, recursively align children
    cost = 0
    max_children = max(len(ast1.children), len(ast2.children))
    for i in range(max_children):
        if i < len(ast1.children) and i < len(ast2.children):
            cost += tree_edit_distance(ast1.children[i], ast2.children[i])
        else:
            # Extra child in one tree
            extra = ast1.children[i] if i < len(ast1.children) else ast2.children[i]
            cost += extra.num_nodes()
    
    return cost


def normalized_tree_distance(ast1: Node, ast2: Node) -> float:
    """
    Normalized tree edit distance in [0, 1].
    
    Divides the raw edit distance by the sum of node counts.
    """
    dist = tree_edit_distance(ast1, ast2)
    total_nodes = ast1.num_nodes() + ast2.num_nodes()
    if total_nodes == 0:
        return 0.0
    return dist / float(total_nodes)


def description_length(ast: Node, library=None) -> int:
    """
    Compute description length of an AST given a fragment library.
    
    For now: count nodes, but give a small discount for CALL nodes
    to reflect that fragments are reused knowledge.
    """
    total = 0
    def count_nodes(node):
        if node.op == 'CALL':
            return 1  # Fragment call costs less than its full expansion
        return 1 + sum(count_nodes(c) for c in node.children)
    return count_nodes(ast)

