"""Evaluation metrics for AST-based programs."""

from typing import Optional

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from model.dsl import Node
from model.dsl.fragments import Library
from model.dsl.parser import get_library as parser_get_library


def expand_calls(ast: Node, library: Optional[Library] = None) -> Node:
    """
    Expand CALL nodes using the provided fragment library.

    This ensures "accuracy" metrics reflect the decoded (semantic) program,
    not the compressed representation.
    """
    lib = library if library is not None else parser_get_library()
    if lib is None:
        return ast.copy()

    def _expand_as_nodes(node: Node) -> list[Node]:
        if node.op == 'CALL':
            name = node.args.get('name')
            frag = lib.get(name) if name is not None else None
            if frag is None:
                return [node.copy()]
            body = frag.body
            # Fragments are typically SEQ nodes produced by tokens_to_ast; inline them.
            if body.op == 'SEQ':
                out: list[Node] = []
                for c in body.children:
                    out.extend(_expand_as_nodes(c))
                return out
            return _expand_as_nodes(body)

        if node.op == 'SEQ':
            out_children: list[Node] = []
            for c in node.children:
                out_children.extend(_expand_as_nodes(c))
            return [Node('SEQ', node.args.copy(), out_children)]

        return [Node(node.op, node.args.copy(), [_expand_as_nodes(c)[0] for c in node.children])]

    expanded = _expand_as_nodes(ast)
    if len(expanded) == 1 and expanded[0].op == 'SEQ':
        return expanded[0]
    return Node('SEQ', children=expanded)


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
    a1 = expand_calls(ast1)
    a2 = expand_calls(ast2)
    dist = tree_edit_distance(a1, a2)
    total_nodes = a1.num_nodes() + a2.num_nodes()
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

