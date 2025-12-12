"""Fragment and library definitions for program abstraction."""

from typing import Dict, Optional
from .ast import Node


class Fragment:
    """
    A reusable program fragment (subroutine).
    
    name: identifier (e.g., 'F1', 'chunk_C')
    body: AST subtree representing the fragment
    frequency: how many times this fragment has been observed
    """
    def __init__(self, name: str, body: Node, frequency: int = 1):
        self.name = name
        self.body = body
        self.frequency = frequency
    
    def __repr__(self):
        return f"Fragment({self.name}, freq={self.frequency}, nodes={self.body.num_nodes()})"


class Library:
    """
    A collection of learned fragments.
    """
    def __init__(self):
        self.fragments: Dict[str, Fragment] = {}
    
    def add(self, fragment: Fragment):
        """Add or update a fragment in the library."""
        if fragment.name in self.fragments:
            self.fragments[fragment.name].frequency += fragment.frequency
        else:
            self.fragments[fragment.name] = fragment
    
    def get(self, name: str) -> Optional[Fragment]:
        """Retrieve a fragment by name."""
        return self.fragments.get(name)
    
    def __len__(self):
        return len(self.fragments)
    
    def __repr__(self):
        return f"Library({len(self.fragments)} fragments)"

