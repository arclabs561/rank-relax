"""
Test gradient correctness for analytical gradients.

Compares analytical gradients against numerical gradients using
finite differences to ensure correctness.
"""

import pytest
import numpy as np
import rank_relax


def numerical_gradient_soft_rank(values, regularization_strength, epsilon=1e-5):
    """Compute numerical gradient using finite differences."""
    n = len(values)
    grad = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            # Forward difference
            values_plus = values.copy()
            values_plus[j] += epsilon
            ranks_plus = rank_relax.soft_rank(values_plus.tolist(), regularization_strength)
            
            # Backward difference
            values_minus = values.copy()
            values_minus[j] -= epsilon
            ranks_minus = rank_relax.soft_rank(values_minus.tolist(), regularization_strength)
            
            # Central difference
            grad[i, j] = (ranks_plus[i] - ranks_minus[i]) / (2 * epsilon)
    
    return grad


def numerical_gradient_spearman_loss(predictions, targets, regularization_strength, epsilon=1e-5):
    """Compute numerical gradient of Spearman loss using finite differences."""
    n = len(predictions)
    grad = np.zeros(n)
    
    for i in range(n):
        # Forward difference
        pred_plus = predictions.copy()
        pred_plus[i] += epsilon
        loss_plus = rank_relax.spearman_loss(
            pred_plus.tolist(), targets.tolist(), regularization_strength
        )
        
        # Backward difference
        pred_minus = predictions.copy()
        pred_minus[i] -= epsilon
        loss_minus = rank_relax.spearman_loss(
            pred_minus.tolist(), targets.tolist(), regularization_strength
        )
        
        # Central difference
        grad[i] = (loss_plus - loss_minus) / (2 * epsilon)
    
    return grad


def test_soft_rank_gradient_correctness():
    """Test that analytical gradient matches numerical gradient."""
    values = np.array([5.0, 1.0, 2.0, 4.0, 3.0])
    regularization_strength = 1.0
    
    # Compute analytical gradient
    ranks = rank_relax.soft_rank(values.tolist(), regularization_strength)
    grad_analytical = rank_relax.soft_rank_gradient(
        values.tolist(), ranks, regularization_strength
    )
    grad_analytical = np.array(grad_analytical)
    
    # Compute numerical gradient
    grad_numerical = numerical_gradient_soft_rank(values, regularization_strength)
    
    # Compare (allow small numerical errors)
    np.testing.assert_allclose(
        grad_analytical, grad_numerical, rtol=1e-3, atol=1e-4,
        err_msg="Analytical gradient doesn't match numerical gradient"
    )


def test_spearman_loss_gradient_correctness():
    """Test that analytical Spearman loss gradient matches numerical gradient."""
    predictions = np.array([0.1, 0.9, 0.3, 0.7, 0.5])
    targets = np.array([0.0, 1.0, 0.2, 0.8, 0.4])
    regularization_strength = 1.0
    
    # Compute analytical gradient
    pred_ranks = rank_relax.soft_rank(predictions.tolist(), regularization_strength)
    target_ranks = rank_relax.soft_rank(targets.tolist(), regularization_strength)
    grad_analytical = rank_relax.spearman_loss_gradient(
        predictions.tolist(), targets.tolist(),
        pred_ranks, target_ranks, regularization_strength
    )
    grad_analytical = np.array(grad_analytical)
    
    # Compute numerical gradient
    grad_numerical = numerical_gradient_spearman_loss(
        predictions, targets, regularization_strength
    )
    
    # Compare (allow small numerical errors)
    np.testing.assert_allclose(
        grad_analytical, grad_numerical, rtol=1e-3, atol=1e-4,
        err_msg="Analytical Spearman loss gradient doesn't match numerical gradient"
    )


def test_gradient_stability_across_regularization():
    """Test that gradients remain stable across different regularization strengths."""
    values = np.array([5.0, 1.0, 2.0, 4.0, 3.0])
    
    for reg_strength in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        ranks = rank_relax.soft_rank(values.tolist(), reg_strength)
        grad = rank_relax.soft_rank_gradient(
            values.tolist(), ranks, reg_strength
        )
        
        # Check for NaN or Inf
        grad_array = np.array(grad)
        assert not np.any(np.isnan(grad_array)), f"NaN in gradient at reg={reg_strength}"
        assert not np.any(np.isinf(grad_array)), f"Inf in gradient at reg={reg_strength}"
        
        # Check for reasonable magnitude (not exploding)
        max_grad = np.abs(grad_array).max()
        assert max_grad < 1e6, f"Gradient explosion at reg={reg_strength}: max={max_grad}"


def test_gradient_edge_cases():
    """Test gradient computation for edge cases."""
    # Equal values
    values_equal = [1.0, 1.0, 1.0]
    ranks_equal = rank_relax.soft_rank(values_equal, 1.0)
    grad_equal = rank_relax.soft_rank_gradient(values_equal, ranks_equal, 1.0)
    assert len(grad_equal) == 3
    assert len(grad_equal[0]) == 3
    
    # Single element
    values_single = [42.0]
    ranks_single = rank_relax.soft_rank(values_single, 1.0)
    grad_single = rank_relax.soft_rank_gradient(values_single, ranks_single, 1.0)
    assert len(grad_single) == 1
    assert len(grad_single[0]) == 1

