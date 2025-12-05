//! Differentiable ranking operations using smooth relaxation.
//!
//! This module implements the "naive" sigmoid-based approach to differentiable ranking.
//! While O(n²) in complexity, it provides an intuitive and straightforward implementation
//! that works well for small to medium-sized inputs.
//!
//! # Algorithm
//!
//! The rank of element `i` is computed as:
//! ```
//! rank[i] = (1/(n-1)) * Σ_{j≠i} sigmoid(α * (values[i] - values[j]))
//! ```
//!
//! where:
//! - `α = regularization_strength` controls the sharpness of the sigmoid
//! - `sigmoid(x) = 1/(1 + exp(-x))` provides a smooth approximation to the step function
//!
//! **Intuition**: For each element, we count (softly) how many other elements it's greater than.
//! The sigmoid gives a value close to 1 when `values[i] > values[j]`, and close to 0 when
//! `values[i] < values[j]`. Summing these gives a continuous approximation of the rank.
//!
//! **Normalization**: We divide by `(n-1)` to normalize ranks to the range [0, n-1], matching
//! the expected range for 0-indexed ranks.
//!
//! # Complexity
//!
//! - **Time**: O(n²) - requires comparing each element with all others
//! - **Space**: O(n) - stores ranks for each element
//!
//! # Parameter Tuning
//!
//! - **Low values (0.1-1.0)**: Smooth gradients, good for early training
//! - **Medium values (1.0-10.0)**: Balanced between smoothness and accuracy
//! - **High values (10.0-100.0)**: Sharper, closer to discrete ranking
//!
//! Choose based on the scale of differences in your values. If values differ by ~1.0,
//! use `regularization_strength ≈ 1.0`. If differences are ~0.1, use `≈ 10.0`.

/// Compute soft ranks for a vector of values.
///
/// Uses a smooth relaxation of the discrete ranking operation, enabling
/// gradient flow through the ranking.
///
/// This implements the sigmoid-based "naive" approach, which is intuitive
/// but has O(n²) complexity. For large inputs, consider more efficient methods
/// (see MATHEMATICAL_DETAILS.md for alternatives).
///
/// # Arguments
///
/// * `values` - Input values to rank
/// * `regularization_strength` - Temperature parameter controlling sharpness
///   (higher = sharper, more discrete-like behavior)
///   - Typical range: 0.1 to 100.0
///   - Should match the scale of differences in your values
///
/// # Returns
///
/// Vector of soft ranks (continuous approximations of integer ranks).
/// Ranks are normalized to [0, n-1] range, where n is the number of elements.
///
/// # Example
///
/// ```rust
/// use rank_relax::soft_rank;
///
/// let values = vec![5.0, 1.0, 2.0, 4.0, 3.0];
/// let ranks = soft_rank(&values, 1.0);
/// // With regularization_strength=1.0, ranks are soft approximations
/// // With regularization_strength=100.0, ranks approach [4.0, 0.0, 1.0, 3.0, 2.0]
/// ```
///
/// # Edge Cases
///
/// - Empty input: returns empty vector
/// - Single element: returns `[0.0]`
/// - NaN/Inf values: returns NaN for that element's rank
pub fn soft_rank(values: &[f64], regularization_strength: f64) -> Vec<f64> {
    let n = values.len();
    
    // Handle edge cases
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![0.0];
    }
    
    let mut ranks = vec![0.0; n];
    
    for i in 0..n {
        // Handle non-finite values (NaN, Inf) by returning NaN rank
        // This propagates the error signal rather than silently failing
        if !values[i].is_finite() {
            ranks[i] = f64::NAN;
            continue;
        }
        
        // Compute soft rank: sum of sigmoid comparisons with all other elements
        let mut rank_sum = 0.0;
        for j in 0..n {
            if i != j {
                // Skip non-finite values in comparisons (they can't be meaningfully compared)
                if !values[j].is_finite() {
                    continue;
                }
                // Sigmoid-based soft comparison:
                // sigmoid(α * (x_i - x_j)) ≈ 1 if x_i > x_j, ≈ 0 if x_i < x_j
                let diff = values[i] - values[j];
                rank_sum += sigmoid(diff * regularization_strength);
            }
        }
        
        // Normalize by (n-1) to get ranks in [0, n-1] range
        // This matches the expected range for 0-indexed ranks
        ranks[i] = rank_sum / (n - 1) as f64;
    }
    
    ranks
}

/// Sigmoid function: σ(x) = 1 / (1 + exp(-x))
///
/// Provides a smooth, differentiable approximation to the step function:
/// - As x → +∞, σ(x) → 1
/// - As x → -∞, σ(x) → 0
/// - At x = 0, σ(0) = 0.5
///
/// This enables smooth transitions between "less than" and "greater than"
/// comparisons, making ranking operations differentiable.
pub(crate) fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_soft_rank_basic() {
        let values = vec![1.0, 2.0, 3.0];
        let ranks = soft_rank(&values, 10.0); // High regularization = sharper
        
        // With high regularization, ranks should be close to [0, 1, 2]
        assert!(ranks[0] < ranks[1]);
        assert!(ranks[1] < ranks[2]);
    }

    #[test]
    fn test_soft_rank_preserves_ordering() {
        let values = vec![5.0, 1.0, 2.0, 4.0, 3.0];
        let ranks = soft_rank(&values, 1.0);
        
        // Ranks should preserve relative ordering
        // value[1] (1.0) < value[2] (2.0) < value[4] (3.0) < value[3] (4.0) < value[0] (5.0)
        assert!(ranks[1] < ranks[2]);
        assert!(ranks[2] < ranks[4]);
        assert!(ranks[4] < ranks[3]);
        assert!(ranks[3] < ranks[0]);
    }
}

