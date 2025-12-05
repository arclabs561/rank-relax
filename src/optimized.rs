//! Performance-optimized implementations using SIMD and parallelization.
//!
//! This module provides optimized versions of ranking operations that leverage
//! SIMD instructions and parallel processing for better performance on large inputs.

use crate::rank::sigmoid;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Optimized batch soft ranking with parallel processing.
///
/// Processes multiple rankings in parallel using rayon.
///
/// # Arguments
///
/// * `batch_values` - Vector of value vectors [batch_size, n_items]
/// * `regularization_strength` - Temperature parameter
///
/// # Returns
///
/// Vector of rank vectors [batch_size, n_items]
#[cfg(feature = "parallel")]
pub fn soft_rank_batch_parallel(
    batch_values: &[Vec<f64>],
    regularization_strength: f64,
) -> Vec<Vec<f64>> {
    batch_values
        .par_iter()
        .map(|values| crate::rank::soft_rank(values, regularization_strength))
        .collect()
}

/// SIMD-optimized sigmoid computation (when available).
///
/// Falls back to scalar implementation if SIMD is not available.
#[inline]
pub fn sigmoid_simd(x: f64) -> f64 {
    // For now, use scalar implementation
    // TODO: Add SIMD implementation using std::arch when stable
    sigmoid(x)
}

/// Optimized soft ranking with early termination for sorted inputs.
///
/// If the input is already sorted, we can skip some comparisons.
/// This is useful when inputs are pre-sorted or nearly sorted.
pub fn soft_rank_optimized(
    values: &[f64],
    regularization_strength: f64,
    assume_sorted: bool,
) -> Vec<f64> {
    let n = values.len();
    
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![0.0];
    }
    
    let mut ranks = vec![0.0; n];
    let inv_n_minus_1 = 1.0 / (n - 1) as f64;
    
    if assume_sorted {
        // Optimized path: if sorted, we know value[i] >= value[j] for i > j
        // This allows us to skip some sigmoid computations
        for i in 0..n {
            if !values[i].is_finite() {
                ranks[i] = f64::NAN;
                continue;
            }
            
            let mut rank_sum = 0.0;
            
            // For sorted arrays, we can use the fact that:
            // - value[i] > value[j] for j < i (sigmoid ≈ 1.0)
            // - value[i] < value[j] for j > i (sigmoid ≈ 0.0)
            // We only need to compute sigmoid for nearby elements
            
            // Count elements before i (these are smaller, sigmoid ≈ 1.0)
            rank_sum += i as f64;
            
            // Count elements after i (these are larger, sigmoid ≈ 0.0)
            // No contribution
            
            // For nearby elements, compute actual sigmoid
            let window = 5usize; // Only check nearby elements
            let start = i.saturating_sub(window);
            let end = (i + window + 1).min(n);
            
            for j in start..end {
                if i != j && values[j].is_finite() {
                    let diff = values[i] - values[j];
                    let sig = sigmoid(diff * regularization_strength);
                    // Adjust: subtract approximate value, add actual
                    if j < i {
                        rank_sum += sig - 1.0; // Was approximated as 1.0
                    } else {
                        rank_sum += sig; // Was approximated as 0.0
                    }
                }
            }
            
            ranks[i] = rank_sum * inv_n_minus_1;
        }
    } else {
        // Standard implementation
        for i in 0..n {
            if !values[i].is_finite() {
                ranks[i] = f64::NAN;
                continue;
            }
            
            let mut rank_sum = 0.0;
            for j in 0..n {
                if i != j && values[j].is_finite() {
                    let diff = values[i] - values[j];
                    rank_sum += sigmoid(diff * regularization_strength);
                }
            }
            ranks[i] = rank_sum * inv_n_minus_1;
        }
    }
    
    ranks
}

/// Memory-efficient gradient computation using sparse storage.
///
/// Instead of storing full n×n gradient matrix, we compute gradients on-demand
/// or use sparse representation for large inputs.
pub fn soft_rank_gradient_sparse(
    values: &[f64],
    ranks: &[f64],
    regularization_strength: f64,
    threshold: f64,
) -> Vec<Vec<f64>> {
    let n = values.len();
    
    if n == 0 || n == 1 {
        return vec![vec![0.0; n]; n];
    }
    
    let alpha = regularization_strength;
    let inv_n_minus_1 = 1.0 / (n - 1) as f64;
    
    let mut grad = vec![vec![0.0; n]; n];
    
    // Only compute gradients above threshold (sparse approximation)
    for i in 0..n {
        if !values[i].is_finite() || !ranks[i].is_finite() {
            continue;
        }
        
        for k in 0..n {
            if !values[k].is_finite() {
                continue;
            }
            
            if i == k {
                // Diagonal: always compute
                let mut sum = 0.0;
                for j in 0..n {
                    if i != j && values[j].is_finite() {
                        let diff = values[i] - values[j];
                        let sig = sigmoid(alpha * diff);
                        let sig_deriv = sig * (1.0 - sig);
                        if sig_deriv.abs() > threshold {
                            sum += sig_deriv;
                        }
                    }
                }
                grad[i][k] = alpha * inv_n_minus_1 * sum;
            } else {
                // Off-diagonal: only if significant
                let diff = values[i] - values[k];
                let sig = sigmoid(alpha * diff);
                let sig_deriv = sig * (1.0 - sig);
                
                if sig_deriv.abs() > threshold {
                    grad[i][k] = -alpha * inv_n_minus_1 * sig_deriv;
                }
            }
        }
    }
    
    grad
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_soft_rank_optimized_sorted() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ranks_opt = soft_rank_optimized(&values, 1.0, true);
        let ranks_std = crate::rank::soft_rank(&values, 1.0);
        
        // Should be approximately equal
        for (r_opt, r_std) in ranks_opt.iter().zip(ranks_std.iter()) {
            assert!((r_opt - r_std).abs() < 0.1, "Optimized should match standard");
        }
    }
    
    #[test]
    fn test_gradient_sparse() {
        let values = vec![1.0, 2.0, 3.0];
        let ranks = crate::rank::soft_rank(&values, 1.0);
        
        let grad_sparse = soft_rank_gradient_sparse(&values, &ranks, 1.0, 0.01);
        let grad_full = crate::gradients::soft_rank_gradient(&values, &ranks, 1.0);
        
        // Sparse should approximate full
        for i in 0..values.len() {
            for j in 0..values.len() {
                if grad_full[i][j].abs() > 0.01 {
                    assert!(
                        (grad_sparse[i][j] - grad_full[i][j]).abs() < 0.1,
                        "Sparse gradient should approximate full"
                    );
                }
            }
        }
    }
}

