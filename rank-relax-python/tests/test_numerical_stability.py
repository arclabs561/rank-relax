"""
Test numerical stability across parameter ranges and edge cases.

Based on MCP research: numerical stability is the #1 production blocker.
"""

import pytest
import numpy as np
import rank_relax


def test_stability_across_regularization_range():
    """Test that operations remain stable across wide regularization range."""
    values = [5.0, 1.0, 2.0, 4.0, 3.0]
    
    # Test very low to very high regularization
    for reg in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]:
        ranks = rank_relax.soft_rank(values, reg)
        
        # Check for NaN/Inf
        assert not any(np.isnan(r) for r in ranks), f"NaN at reg={reg}"
        assert not any(np.isinf(r) for r in ranks), f"Inf at reg={reg}"
        
        # Check for reasonable values
        assert all(-10.0 <= r <= 10.0 for r in ranks), f"Unreasonable ranks at reg={reg}"


def test_stability_with_extreme_values():
    """Test stability with very large/small input values."""
    # Very large values
    values_large = [1e10, 2e10, 3e10]
    ranks_large = rank_relax.soft_rank(values_large, 1.0)
    assert not any(np.isnan(r) or np.isinf(r) for r in ranks_large)
    
    # Very small values
    values_small = [1e-10, 2e-10, 3e-10]
    ranks_small = rank_relax.soft_rank(values_small, 1.0)
    assert not any(np.isnan(r) or np.isinf(r) for r in ranks_small)
    
    # Mixed scales (challenging case)
    values_mixed = [0.001, 1000.0, 0.1]
    ranks_mixed = rank_relax.soft_rank(values_mixed, 1.0)
    assert not any(np.isnan(r) or np.isinf(r) for r in ranks_mixed)


def test_gradient_stability():
    """Test that gradients remain stable and don't explode/vanish."""
    values = [5.0, 1.0, 2.0, 4.0, 3.0]
    
    for reg in [0.1, 1.0, 10.0]:
        ranks = rank_relax.soft_rank(values, reg)
        grad = rank_relax.soft_rank_gradient(values, ranks, reg)
        
        # Flatten gradient matrix
        grad_flat = [g for row in grad for g in row]
        
        # Check for NaN/Inf
        assert not any(np.isnan(g) for g in grad_flat), f"NaN in gradient at reg={reg}"
        assert not any(np.isinf(g) for g in grad_flat), f"Inf in gradient at reg={reg}"
        
        # Check for reasonable magnitude (not exploding)
        max_grad = max(abs(g) for g in grad_flat)
        assert max_grad < 1e6, f"Gradient explosion at reg={reg}: max={max_grad}"
        
        # Check for reasonable magnitude (not vanishing)
        # At least some gradients should be non-zero
        non_zero = sum(1 for g in grad_flat if abs(g) > 1e-10)
        assert non_zero > 0, f"All gradients vanished at reg={reg}"


def test_stability_with_ties():
    """Test stability when values are tied (equal)."""
    # All equal
    values_equal = [1.0, 1.0, 1.0, 1.0]
    ranks_equal = rank_relax.soft_rank(values_equal, 1.0)
    assert not any(np.isnan(r) or np.isinf(r) for r in ranks_equal)
    
    # Some ties
    values_some_ties = [1.0, 1.0, 2.0, 2.0, 3.0]
    ranks_some_ties = rank_relax.soft_rank(values_some_ties, 1.0)
    assert not any(np.isnan(r) or np.isinf(r) for r in ranks_some_ties)


def test_stability_with_large_inputs():
    """Test stability with larger input sizes."""
    # Larger input (stress test)
    n = 100
    values_large = list(range(n))
    ranks_large = rank_relax.soft_rank(values_large, 1.0)
    
    assert len(ranks_large) == n
    assert not any(np.isnan(r) or np.isinf(r) for r in ranks_large)
    
    # Check ordering is preserved (larger values should have larger ranks)
    for i in range(n - 1):
        assert ranks_large[i] < ranks_large[i + 1] + 0.1, "Ordering should be preserved"


def test_spearman_loss_stability():
    """Test Spearman loss stability across parameter ranges."""
    predictions = [0.1, 0.9, 0.3, 0.7, 0.5]
    targets = [0.0, 1.0, 0.2, 0.8, 0.4]
    
    for reg in [0.1, 1.0, 10.0]:
        loss = rank_relax.spearman_loss(predictions, targets, reg)
        
        # Check for NaN/Inf
        assert not np.isnan(loss), f"NaN in loss at reg={reg}"
        assert not np.isinf(loss), f"Inf in loss at reg={reg}"
        
        # Check for reasonable range (loss should be in [0, 2])
        assert 0.0 <= loss <= 2.0, f"Loss out of range at reg={reg}: {loss}"


def test_warning_for_extreme_regularization():
    """Test that we can detect problematic regularization values."""
    values = [5.0, 1.0, 2.0, 4.0, 3.0]
    
    # Very low regularization (may cause vanishing gradients)
    ranks_low = rank_relax.soft_rank(values, 0.001)
    grad_low = rank_relax.soft_rank_gradient(values, ranks_low, 0.001)
    grad_max_low = max(abs(g) for row in grad_low for g in row)
    
    # Very high regularization (may cause exploding gradients)
    ranks_high = rank_relax.soft_rank(values, 1000.0)
    grad_high = rank_relax.soft_rank_gradient(values, ranks_high, 1000.0)
    grad_max_high = max(abs(g) for row in grad_high for g in row)
    
    # Both should still be finite, but magnitudes may differ significantly
    assert not np.isnan(grad_max_low) and not np.isinf(grad_max_low)
    assert not np.isnan(grad_max_high) and not np.isinf(grad_max_high)

