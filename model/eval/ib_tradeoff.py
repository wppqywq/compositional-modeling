"""Information Bottleneck-style loss for program transmission."""

import sys
import os
from typing import Optional, List, Tuple
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from model.dsl import Node
from .metrics import description_length, normalized_tree_distance


def _normalize_complexity(
    C_raw: float,
    C_ref: Optional[float] = None,
    candidates: Optional[List[Node]] = None,
    library=None
) -> float:
    """
    Normalize complexity to [0, 1] scale.
    
    Args:
        C_raw: Raw description length
        C_ref: Reference complexity for fixed normalization (e.g., true_program's complexity)
        candidates: List of candidate ASTs for dynamic normalization (min-max scaling)
        library: Fragment library for computing description lengths
    
    Returns:
        Normalized complexity in [0, 1]
    """
    if C_ref is not None:
        # Fixed normalization: use a reasonable upper bound
        # Allow candidates to be up to 1.5x the reference complexity before clamping
        # This preserves some distinction between complex candidates
        C_upper_bound = max(C_ref * 1.5, C_raw, 10.0)  # At least 10 to avoid division issues
        C_norm = C_raw / float(C_upper_bound)
        return max(0.0, min(1.0, C_norm))
    
    elif candidates is not None and len(candidates) > 0:
        # Dynamic normalization: min-max scaling over candidates
        C_values = [description_length(cand, library) for cand in candidates]
        C_min = min(C_values)
        C_max = max(C_values)
        
        if C_max == C_min:
            # All candidates have same complexity: use fixed normalization if possible
            # Otherwise return a reasonable normalized value based on absolute size
            if C_min == 0:
                return 0.0
            # Use a heuristic: normalize by a reasonable upper bound
            C_upper_bound = max(C_min * 1.2, 20.0)
            return min(1.0, C_raw / float(C_upper_bound))
        
        C_norm = (C_raw - C_min) / float(C_max - C_min)
        return max(0.0, min(1.0, C_norm))  # Clamp to [0, 1]
    
    else:
        # Fallback: use a heuristic upper bound (assume max complexity ~50 nodes)
        # This is not ideal but provides reasonable normalization
        C_upper_bound = 50.0
        return min(1.0, C_raw / C_upper_bound)


def ib_loss(
    candidate_ast: Node,
    true_ast: Node,
    beta: float = 1.0,
    library=None,
    normalize_mode: str = 'fixed',
    candidates: Optional[List[Node]] = None,
) -> float:
    """
    Compute IB-style loss for a candidate program with normalized complexity.
    
    L(ast) = C_norm(ast) - beta * A(ast)
    
    where:
    - C_norm = normalized complexity proxy in [0, 1]
    - A = accuracy proxy (1 - normalized tree distance) in [0, 1]
    - beta = trade-off parameter
        * beta = 1.0: 1 unit complexity ≈ 1 unit accuracy
        * beta > 1.0: favor accuracy over compression
        * beta < 1.0: favor compression over accuracy
    
    Args:
        candidate_ast: Candidate program AST
        true_ast: True target program AST
        beta: Trade-off parameter (default 1.0 means equal weight)
        library: Fragment library for computing description length
        normalize_mode: 'fixed' (use true_ast as reference) or 'dynamic' (use candidates)
        candidates: List of candidate ASTs for dynamic normalization
    
    Returns:
        Loss value (lower is better). With normalized C and A in [0,1],
        loss range is approximately [-beta, 1].
    """
    C_raw = description_length(candidate_ast, library)
    
    # Normalize complexity based on mode
    if normalize_mode == 'fixed':
        C_ref = description_length(true_ast, library)
        C_norm = _normalize_complexity(C_raw, C_ref=C_ref, library=library)
    elif normalize_mode == 'dynamic':
        if candidates is None:
            # Fallback to fixed if no candidates provided
            C_ref = description_length(true_ast, library)
            C_norm = _normalize_complexity(C_raw, C_ref=C_ref, library=library)
        else:
            C_norm = _normalize_complexity(C_raw, candidates=candidates, library=library)
    else:
        raise ValueError(f"Unknown normalize_mode: {normalize_mode}")
    
    # Accuracy is already normalized [0, 1]
    tree_dist = normalized_tree_distance(candidate_ast, true_ast)
    A = 1.0 - tree_dist  # Accuracy: higher when distance is smaller
    
    # Loss: lower is better
    loss = C_norm - beta * A
    
    return loss


def ib_complexity(ast: Node, library=None, normalize: bool = False, C_ref: Optional[float] = None) -> float:
    """
    Complexity proxy for IB analysis.
    
    Args:
        ast: Program AST
        library: Fragment library
        normalize: If True, return normalized complexity [0, 1]
        C_ref: Reference complexity for normalization
    
    Returns:
        Complexity (raw if normalize=False, normalized if normalize=True)
    """
    C_raw = float(description_length(ast, library))
    if normalize:
        if C_ref is None:
            # Use heuristic upper bound
            C_ref = 50.0
        return _normalize_complexity(C_raw, C_ref=C_ref, library=library)
    return C_raw


def ib_accuracy(candidate_ast: Node, true_ast: Node) -> float:
    """
    Accuracy proxy for IB analysis.
    
    Returns:
        Accuracy in [0, 1], where 1 = identical, 0 = completely different
    """
    tree_dist = normalized_tree_distance(candidate_ast, true_ast)
    return 1.0 - tree_dist


def ib_loss_raw_components(
    candidate_ast: Node,
    true_ast: Node,
    library=None
) -> Tuple[float, float]:
    """
    Return raw (unnormalized) complexity and accuracy components.
    
    Useful for analysis and plotting complexity-accuracy frontiers.
    
    Returns:
        (complexity_raw, accuracy) tuple
    """
    C_raw = description_length(candidate_ast, library)
    A = ib_accuracy(candidate_ast, true_ast)
    return (float(C_raw), A)

