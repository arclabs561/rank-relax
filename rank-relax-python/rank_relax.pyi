"""Type stubs for rank-relax Python bindings."""

def soft_rank(values: list[float], regularization_strength: float) -> list[float]:
    """
    Compute soft ranks for a vector of values.
    
    Uses a smooth relaxation of the discrete ranking operation, enabling
    gradient flow through the ranking.
    
    Args:
        values: Input values to rank
        regularization_strength: Temperature parameter controlling sharpness
            (higher = sharper, more discrete-like behavior)
    
    Returns:
        List of soft ranks (continuous approximations of integer ranks)
    """
    ...

def soft_sort(values: list[float], regularization_strength: float) -> list[float]:
    """
    Compute soft sorted values for a vector.
    
    Uses a smooth relaxation of the discrete sorting operation, enabling
    gradient flow through the sorting.
    
    Args:
        values: Input values to sort
        regularization_strength: Temperature parameter controlling sharpness
    
    Returns:
        List of soft sorted values (continuous approximations)
    """
    ...

def spearman_loss(predictions: list[float], targets: list[float], regularization_strength: float) -> float:
    """
    Compute Spearman correlation loss between predictions and targets.
    
    Uses soft ranking to compute differentiable Spearman correlation.
    Loss = 1 - Spearman correlation (so lower is better, Spearman higher is better).
    
    Args:
        predictions: Model predictions
        targets: Ground truth values
        regularization_strength: Temperature parameter for soft ranking
    
    Returns:
        Loss value (1 - Spearman correlation)
    """
    ...

