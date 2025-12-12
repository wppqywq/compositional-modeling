"""Minimal AST representation for block-building programs."""

from typing import List, Optional, Any


class Node:
    """
    Minimal AST node for compositional programs.
    
    op: operation type (e.g., 'PLACE', 'MOVE', 'SEQ', 'CALL')
    args: optional arguments (e.g., shape, direction, distance)
    children: optional child nodes for compositional structures
    """
    def __init__(self, op: str, args: Optional[Any] = None, children: Optional[List['Node']] = None):
        self.op = op
        self.args = args if args is not None else {}
        self.children = children if children is not None else []
    
    def __repr__(self):
        if self.children:
            children_repr = ', '.join(repr(c) for c in self.children)
            return f"Node({self.op}, {self.args}, [{children_repr}])"
        elif self.args:
            return f"Node({self.op}, {self.args})"
        return f"Node({self.op})"
    
    def num_nodes(self) -> int:
        """Count total nodes in this subtree."""
        return 1 + sum(c.num_nodes() for c in self.children)
    
    def copy(self) -> 'Node':
        """Deep copy of this node and its subtree."""
        return Node(
            op=self.op,
            args=self.args.copy() if isinstance(self.args, dict) else self.args,
            children=[c.copy() for c in self.children]
        )

