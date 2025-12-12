"""Token-AST conversion for the block-building DSL."""

from typing import List, Sequence, Optional
from .ast import Node

# Will be set by fragment module when needed
_GLOBAL_LIBRARY = None


def set_library(library):
    """
    Set the global library for expanding fragment calls during ast_to_tokens.
    
    Note: This uses global state for convenience, but consider passing library
    as a parameter to ast_to_tokens() in future versions for better thread safety.
    """
    global _GLOBAL_LIBRARY
    _GLOBAL_LIBRARY = library


def get_library():
    """
    Get the current global library.
    
    Returns None if no library has been set.
    """
    return _GLOBAL_LIBRARY


def tokens_to_ast(tokens: Sequence[str]) -> Node:
    """
    Convert a flat token sequence to an AST.
    
    For now, we use a simple mapping:
    - The entire sequence becomes a SEQ node
    - Each token becomes a primitive action node:
      - 'h' or 'v' -> PLACE(shape)
      - 'l_X' or 'r_X' -> MOVE(direction, distance)
    """
    children = []
    for tok in tokens:
        if tok == 'h':
            children.append(Node('PLACE', {'shape': 'h'}))
        elif tok == 'v':
            children.append(Node('PLACE', {'shape': 'v'}))
        elif tok.startswith('l_'):
            dist = int(tok[2:])
            children.append(Node('MOVE', {'direction': 'left', 'distance': dist}))
        elif tok.startswith('r_'):
            dist = int(tok[2:])
            children.append(Node('MOVE', {'direction': 'right', 'distance': dist}))
        else:
            # Unknown token: wrap as generic action
            children.append(Node('ACTION', {'token': tok}))
    
    return Node('SEQ', children=children)


def ast_to_tokens(ast: Node) -> List[str]:
    """
    Convert an AST back to a flat token sequence.
    
    This unpacks the structure recursively:
    - SEQ nodes: concatenate children's tokens
    - PLACE nodes: emit 'h' or 'v'
    - MOVE nodes: emit 'l_X' or 'r_X'
    - CALL nodes (fragments): expand their body if available
    """
    if ast.op == 'SEQ':
        tokens = []
        for child in ast.children:
            tokens.extend(ast_to_tokens(child))
        return tokens
    
    elif ast.op == 'PLACE':
        shape = ast.args.get('shape', 'h')
        return [shape]
    
    elif ast.op == 'MOVE':
        direction = ast.args.get('direction', 'left')
        distance = ast.args.get('distance', 1)
        prefix = 'l' if direction == 'left' else 'r'
        return [f'{prefix}_{distance}']
    
    elif ast.op == 'ACTION':
        return [ast.args.get('token', '')]
    
    elif ast.op == 'CALL':
        # Fragment call: expand from library if available
        frag_name = ast.args.get('name', 'F')
        if _GLOBAL_LIBRARY is not None:
            frag = _GLOBAL_LIBRARY.get(frag_name)
            if frag is not None:
                return ast_to_tokens(frag.body)
        # Fallback: return placeholder token
        return [f'#{frag_name}']
    
    else:
        # Unknown op: return empty
        return []


def tokens_roundtrip_test(tokens: Sequence[str]) -> bool:
    """Test that tokens->AST->tokens preserves the original sequence."""
    ast = tokens_to_ast(tokens)
    recovered = ast_to_tokens(ast)
    return list(tokens) == recovered

