"""
Integration tests for rank-relax Python bindings.

Tests all methods, gradients, and batch processing.
"""

import pytest
import rank_relax
import numpy as np


def test_soft_rank_basic():
    """Test basic soft ranking."""
    values = [5.0, 1.0, 2.0, 4.0, 3.0]
    ranks = rank_relax.soft_rank(values, regularization_strength=1.0)
    
    assert len(ranks) == len(values)
    assert all(0.0 <= r <= 4.0 for r in ranks)
    
    # Should preserve ordering
    assert ranks[1] < ranks[2]  # value[1] (1.0) < value[2] (2.0)
    assert ranks[2] < ranks[4]  # value[2] (2.0) < value[4] (3.0)
    assert ranks[4] < ranks[3]  # value[4] (3.0) < value[3] (4.0)
    assert ranks[3] < ranks[0]  # value[3] (4.0) < value[0] (5.0)


def test_soft_rank_methods():
    """Test all ranking methods."""
    values = [5.0, 1.0, 2.0, 4.0, 3.0]
    
    methods = ["sigmoid", "neural_sort", "probabilistic", "smooth_i"]
    
    for method in methods:
        ranks = rank_relax.soft_rank_with_method(
            values, regularization_strength=1.0, method=method
        )
        
        assert len(ranks) == len(values)
        # All methods should preserve ordering
        assert ranks[1] < ranks[2]
        assert ranks[2] < ranks[4]
        assert ranks[4] < ranks[3]
        assert ranks[3] < ranks[0]


def test_spearman_loss():
    """Test Spearman correlation loss."""
    predictions = [0.1, 0.9, 0.3, 0.7, 0.5]
    targets = [0.0, 1.0, 0.2, 0.8, 0.4]
    
    loss = rank_relax.spearman_loss(
        predictions, targets, regularization_strength=1.0
    )
    
    assert 0.0 <= loss <= 2.0
    assert np.isfinite(loss)


def test_gradient_computation():
    """Test analytical gradient computation."""
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    ranks = rank_relax.soft_rank(values, regularization_strength=1.0)
    
    # Compute gradient
    grad = rank_relax.soft_rank_gradient(
        values, ranks, regularization_strength=1.0
    )
    
    assert len(grad) == len(values)
    assert all(len(row) == len(values) for row in grad)
    
    # Diagonal elements should be positive
    for i in range(len(values)):
        assert grad[i][i] > 0.0


def test_spearman_loss_gradient():
    """Test Spearman loss gradient."""
    predictions = [0.1, 0.9, 0.3, 0.7, 0.5]
    targets = [0.0, 1.0, 0.2, 0.8, 0.4]
    
    pred_ranks = rank_relax.soft_rank(predictions, regularization_strength=1.0)
    target_ranks = rank_relax.soft_rank(targets, regularization_strength=1.0)
    
    grad = rank_relax.spearman_loss_gradient(
        predictions, targets, pred_ranks, target_ranks, regularization_strength=1.0
    )
    
    assert len(grad) == len(predictions)
    assert all(np.isfinite(g) for g in grad)


def test_edge_cases():
    """Test edge cases."""
    # Empty input
    ranks = rank_relax.soft_rank([], regularization_strength=1.0)
    assert ranks == []
    
    # Single element
    ranks = rank_relax.soft_rank([1.0], regularization_strength=1.0)
    assert ranks == [0.0]
    
    # All equal
    ranks = rank_relax.soft_rank([1.0, 1.0, 1.0], regularization_strength=1.0)
    assert len(ranks) == 3
    assert all(0.0 <= r <= 2.0 for r in ranks)


def test_regularization_effects():
    """Test that regularization strength affects sharpness."""
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    
    ranks_low = rank_relax.soft_rank(values, regularization_strength=0.1)
    ranks_high = rank_relax.soft_rank(values, regularization_strength=10.0)
    
    # High regularization should produce ranks closer to discrete
    # Check variance from expected integer ranks
    expected_ranks = [0.0, 1.0, 2.0, 3.0, 4.0]
    
    var_low = sum((r - e)**2 for r, e in zip(ranks_low, expected_ranks))
    var_high = sum((r - e)**2 for r, e in zip(ranks_high, expected_ranks))
    
    assert var_high < var_low, "High regularization should be sharper"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

