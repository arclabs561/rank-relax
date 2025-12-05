//! Batch processing utilities for efficient multi-query ranking.
//!
//! This module provides functions to process multiple rankings simultaneously,
//! enabling efficient batch training and evaluation.
//!
//! # Why Batch Processing?
//!
//! In training scenarios, you often need to rank multiple queries or compute
//! losses for multiple prediction-target pairs. Batch processing:
//! - **Amortizes overhead**: Single function call for multiple operations
//! - **Enables parallelization**: Can process batches in parallel (future optimization)
//! - **Simplifies training loops**: Process entire batches at once
//!
//! # Performance
//!
//! Current implementation processes batches sequentially. For large batches,
//! consider parallelization (e.g., using `rayon` or similar).
//!
//! **Complexity**: O(batch_size × n²) where n is the number of items per query.
//!
//! # Example
//!
//! ```rust
//! use rank_relax::soft_rank_batch;
//!
//! // Batch of 3 queries, each with different numbers of items
//! let batch = vec![
//!     vec![5.0, 1.0, 2.0, 4.0, 3.0],  // Query 1: 5 items
//!     vec![3.0, 1.0, 2.0],             // Query 2: 3 items
//!     vec![10.0, 5.0, 8.0, 7.0],      // Query 3: 4 items
//! ];
//!
//! let ranks = soft_rank_batch(&batch, 1.0);
//! // Returns: [ranks_for_query1, ranks_for_query2, ranks_for_query3]
//! ```

use crate::rank::soft_rank;
use crate::spearman::spearman_loss;

/// Compute soft ranks for a batch of value vectors.
///
/// Processes multiple queries/rankings in a single call, useful for batch training.
///
/// **Note**: Each query can have a different number of items. The function handles
/// variable-length inputs gracefully.
///
/// # Arguments
///
/// * `batch_values` - Vector of value vectors, shape [batch_size, n_items]
///   - Each inner vector can have different length
///   - Empty vectors are handled (return empty ranks)
/// * `regularization_strength` - Temperature parameter (same for all queries)
///
/// # Returns
///
/// Vector of rank vectors, shape [batch_size, n_items]
/// - `result[i]` contains ranks for `batch_values[i]`
/// - Length of each rank vector matches corresponding input vector
///
/// # Example
///
/// ```rust
/// use rank_relax::soft_rank_batch;
///
/// let batch = vec![
///     vec![5.0, 1.0, 2.0, 4.0, 3.0],
///     vec![3.0, 1.0, 2.0],
/// ];
/// let ranks = soft_rank_batch(&batch, 1.0);
/// assert_eq!(ranks.len(), 2);
/// assert_eq!(ranks[0].len(), 5);
/// assert_eq!(ranks[1].len(), 3);
/// ```
pub fn soft_rank_batch(
    batch_values: &[Vec<f64>],
    regularization_strength: f64,
) -> Vec<Vec<f64>> {
    batch_values
        .iter()
        .map(|values| soft_rank(values, regularization_strength))
        .collect()
}

/// Compute Spearman loss for a batch of prediction-target pairs.
///
/// Processes multiple prediction-target pairs in a single call, computing
/// the Spearman correlation loss for each pair independently.
///
/// **Use case**: Training loops where you have a batch of queries, each with
/// predictions and ground truth targets.
///
/// # Arguments
///
/// * `batch_predictions` - Vector of prediction vectors [batch_size, n_items]
///   - Each inner vector contains model predictions for one query
///   - Can have different lengths per query
/// * `batch_targets` - Vector of target vectors [batch_size, n_items]
///   - Must have same length as `batch_predictions` (same batch_size)
///   - `batch_targets[i]` should have same length as `batch_predictions[i]`
/// * `regularization_strength` - Temperature parameter (same for all pairs)
///
/// # Returns
///
/// Vector of loss values [batch_size]
/// - `result[i]` = Spearman loss for `batch_predictions[i]` vs `batch_targets[i]`
/// - Loss range: [0, 2] (lower is better)
///
/// # Example
///
/// ```rust
/// use rank_relax::spearman_loss_batch;
///
/// let predictions = vec![
///     vec![0.1, 0.9, 0.3],  // Query 1 predictions
///     vec![1.0, 2.0, 3.0],  // Query 2 predictions
/// ];
/// let targets = vec![
///     vec![0.0, 1.0, 0.2],  // Query 1 targets
///     vec![1.0, 2.0, 3.0],  // Query 2 targets (perfect correlation)
/// ];
///
/// let losses = spearman_loss_batch(&predictions, &targets, 1.0);
/// assert_eq!(losses.len(), 2);
/// assert!(losses[1] < losses[0]); // Perfect correlation has lower loss
/// ```
///
/// # Edge Cases
///
/// - Mismatched lengths between predictions and targets: Returns 1.0 (max loss)
/// - Empty vectors: Returns 0.0 (no correlation defined)
/// - Single element: Returns 0.0 (no correlation possible)
pub fn spearman_loss_batch(
    batch_predictions: &[Vec<f64>],
    batch_targets: &[Vec<f64>],
    regularization_strength: f64,
) -> Vec<f64> {
    batch_predictions
        .iter()
        .zip(batch_targets.iter())
        .map(|(pred, targ)| spearman_loss(pred, targ, regularization_strength))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_soft_rank_batch() {
        let batch = vec![
            vec![5.0, 1.0, 2.0, 4.0, 3.0],
            vec![3.0, 1.0, 2.0],
            vec![10.0, 5.0, 8.0, 7.0],
        ];
        
        let ranks = soft_rank_batch(&batch, 1.0);
        
        assert_eq!(ranks.len(), 3);
        assert_eq!(ranks[0].len(), 5);
        assert_eq!(ranks[1].len(), 3);
        assert_eq!(ranks[2].len(), 4);
    }
    
    #[test]
    fn test_spearman_loss_batch() {
        let predictions = vec![
            vec![0.1, 0.9, 0.3],
            vec![1.0, 2.0, 3.0],
        ];
        let targets = vec![
            vec![0.0, 1.0, 0.2],
            vec![1.0, 2.0, 3.0],
        ];
        
        let losses = spearman_loss_batch(&predictions, &targets, 1.0);
        
        assert_eq!(losses.len(), 2);
        assert!(losses[0] >= 0.0 && losses[0] <= 2.0);
        assert!(losses[1] < losses[0]); // Perfect correlation should have lower loss
    }
}

