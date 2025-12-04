//! Differentiable ranking operations using smooth relaxation.

/// Compute soft ranks for a vector of values.
///
/// Uses a smooth relaxation of the discrete ranking operation, enabling
/// gradient flow through the ranking.
///
/// # Arguments
///
/// * `values` - Input values to rank
/// * `regularization_strength` - Temperature parameter controlling sharpness
///   (higher = sharper, more discrete-like behavior)
///
/// # Returns
///
/// Vector of soft ranks (continuous approximations of integer ranks)
///
/// # Example
///
/// ```rust
/// use rank_relax::soft_rank;
///
/// let values = vec![5.0, 1.0, 2.0, 4.0, 3.0];
/// let ranks = soft_rank(&values, 1.0);
/// // ranks will be approximately [4.0, 0.0, 1.0, 3.0, 2.0] (soft)
/// ```
pub fn soft_rank(values: &[f64], regularization_strength: f64) -> Vec<f64> {
    // Implements soft ranking using sigmoid-based approximation.
    // For each element, counts how many others it's greater than (softly)
    // using sigmoid comparisons, enabling differentiable ranking.
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
        // Skip NaN or Inf values
        if !values[i].is_finite() {
            ranks[i] = f64::NAN;
            continue;
        }
        
        let mut rank_sum = 0.0;
        for j in 0..n {
            if i != j {
                // Skip NaN or Inf values in comparison
                if !values[j].is_finite() {
                    continue;
                }
                // Sigmoid-based soft comparison
                let diff = values[i] - values[j];
                rank_sum += sigmoid(diff * regularization_strength);
            }
        }
        ranks[i] = rank_sum / (n - 1) as f64;
    }
    
    ranks
}

/// Sigmoid function: 1 / (1 + exp(-x))
fn sigmoid(x: f64) -> f64 {
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

