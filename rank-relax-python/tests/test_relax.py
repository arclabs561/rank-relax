"""Tests for rank-relax Python bindings."""

import pytest
import rank_relax


def test_soft_rank():
    """Test soft ranking."""
    values = [5.0, 1.0, 2.0, 4.0, 3.0]
    ranks = rank_relax.soft_rank(values, regularization_strength=1.0)
    
    assert len(ranks) == len(values)
    assert all(isinstance(r, float) for r in ranks)
    # Highest value should have highest rank
    assert ranks[0] > ranks[1]  # 5.0 > 1.0


def test_soft_sort():
    """Test soft sorting."""
    values = [5.0, 1.0, 2.0, 4.0, 3.0]
    sorted_values = rank_relax.soft_sort(values, regularization_strength=1.0)
    
    assert len(sorted_values) == len(values)
    assert all(isinstance(v, float) for v in sorted_values)
    # Should be sorted (monotonic)
    assert sorted_values == sorted(sorted_values)


def test_spearman_loss():
    """Test Spearman correlation loss."""
    predictions = [0.1, 0.9, 0.3, 0.7, 0.5]
    targets = [0.0, 1.0, 0.2, 0.8, 0.4]
    loss = rank_relax.spearman_loss(predictions, targets, regularization_strength=1.0)
    
    assert isinstance(loss, float)
    assert 0.0 <= loss <= 2.0  # Spearman correlation is [-1, 1], so loss is [0, 2]


def test_spearman_loss_perfect_correlation():
    """Test Spearman loss with perfect correlation."""
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    loss = rank_relax.spearman_loss(values, values, regularization_strength=1.0)
    
    # Perfect correlation should give loss close to 0
    assert loss < 0.1


def test_empty_input():
    """Test with empty input."""
    ranks = rank_relax.soft_rank([], regularization_strength=1.0)
    assert ranks == []
    
    sorted_values = rank_relax.soft_sort([], regularization_strength=1.0)
    assert sorted_values == []

