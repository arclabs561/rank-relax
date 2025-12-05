"""Python bindings for rank-relax — differentiable ranking operations for ML training."""

__version__ = "0.1.0"

# Import the compiled Rust extension
try:
    from . import rank_relax as _rust_module
    
    # Re-export all functions from the Rust module (they have _py suffix in Rust)
    from .rank_relax import (
        soft_rank_py as soft_rank,
        soft_sort_py as soft_sort,
        spearman_loss_py as spearman_loss,
        soft_rank_gradient_py as soft_rank_gradient,
        spearman_loss_gradient_py as spearman_loss_gradient,
        soft_rank_with_method_py as soft_rank_with_method,
    )
    
    __all__ = [
        "soft_rank",
        "soft_sort",
        "spearman_loss",
        "soft_rank_gradient",
        "spearman_loss_gradient",
        "soft_rank_with_method",
    ]
except ImportError as e:
    import warnings
    warnings.warn(f"Failed to import rank_relax Rust module: {e}")
    __all__ = []

