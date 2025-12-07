//! Spearman correlation utilities for differentiable ranking.
//!
//! Spearman correlation measures the monotonic relationship between two variables
//! by computing Pearson correlation on their ranks. This module provides a
//! differentiable version using soft ranks, enabling gradient-based optimization.

/// Compute Spearman correlation loss between predictions and targets.
///
/// **Spearman correlation** measures how well the rankings of predictions match
/// the rankings of targets, regardless of their absolute values. It's computed as:
///
/// ```text
/// Spearman = Pearson_correlation(rank(predictions), rank(targets))
/// ```
///
/// This function uses **soft ranks** (differentiable) instead of hard ranks,
/// enabling gradients to flow through the ranking operation during training.
///
/// **Loss formulation**: `loss = 1 - Spearman_correlation`
/// - Lower loss = better ranking agreement
/// - Loss range: [0, 2] (Spearman range: [-1, 1])
/// - Perfect correlation (ρ=1) → loss=0
/// - Perfect anti-correlation (ρ=-1) → loss=2
///
/// # Arguments
///
/// * `predictions` - Model predictions (any scale, ranking matters)
/// * `targets` - Ground truth values (any scale, ranking matters)
/// * `regularization_strength` - Temperature parameter for soft ranking
///   - Same tuning guidance as `soft_rank`
///   - Should match the scale of differences in your values
///
/// # Returns
///
/// Loss value in [0, 2] range (lower is better)
///
/// # Example
///
/// ```rust
/// use rank_relax::spearman_loss;
///
/// // Perfect correlation: predictions and targets have same ranking
/// let predictions = vec![0.1, 0.9, 0.3, 0.7, 0.5];
/// let targets = vec![0.0, 1.0, 0.2, 0.8, 0.4];
/// let loss = spearman_loss(&predictions, &targets, 10.0);
/// // Loss should be close to 0.0 (high regularization = accurate ranks)
/// ```
///
/// # Algorithm
///
/// 1. Compute soft ranks for predictions and targets
/// 2. Compute Pearson correlation between soft ranks
/// 3. Return `1 - correlation` as loss
///
/// # Edge Cases
///
/// - Mismatched lengths: returns 1.0 (maximum loss)
/// - < 2 elements: returns 0.0 (no correlation defined)
/// - Zero variance: returns 1.0 (maximum loss)
pub fn spearman_loss(predictions: &[f64], targets: &[f64], regularization_strength: f64) -> f64 {
    use crate::rank::soft_rank;
    
    // Edge case: mismatched lengths can't be compared
    if predictions.len() != targets.len() {
        return 1.0; // Maximum loss for mismatched lengths
    }
    
    // Edge case: need at least 2 elements for correlation
    if predictions.len() < 2 {
        return 0.0; // No correlation possible with < 2 elements
    }
    
    // Step 1: Compute soft ranks (differentiable!)
    // This is where gradients can flow through the ranking operation
    let pred_ranks = soft_rank(predictions, regularization_strength);
    let target_ranks = soft_rank(targets, regularization_strength);
    
    // Step 2: Compute Pearson correlation of ranks
    // Spearman correlation = Pearson correlation of ranks
    let pred_mean = pred_ranks.iter().sum::<f64>() / pred_ranks.len() as f64;
    let target_mean = target_ranks.iter().sum::<f64>() / target_ranks.len() as f64;
    
    // Compute covariance and variances
    let mut numerator = 0.0;  // Covariance: Σ(pred_diff * target_diff)
    let mut pred_var = 0.0;   // Variance of prediction ranks
    let mut target_var = 0.0; // Variance of target ranks
    
    for i in 0..pred_ranks.len() {
        let pred_diff = pred_ranks[i] - pred_mean;
        let target_diff = target_ranks[i] - target_mean;
        numerator += pred_diff * target_diff;
        pred_var += pred_diff * pred_diff;
        target_var += target_diff * target_diff;
    }
    
    // Pearson correlation = covariance / (std_pred * std_target)
    let denominator = (pred_var * target_var).sqrt();
    if denominator < 1e-8 {
        // Edge case: no variance means no correlation can be computed
        return 1.0; // Maximum loss
    }
    
    let spearman = numerator / denominator;
    
    // Step 3: Convert to loss (lower is better)
    // Loss = 1 - Spearman, so:
    // - Spearman = 1 (perfect) → loss = 0
    // - Spearman = 0 (no correlation) → loss = 1
    // - Spearman = -1 (anti-correlation) → loss = 2
    1.0 - spearman
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spearman_loss_perfect_correlation() {
        // Perfect correlation should give low loss
        let predictions = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let targets = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let loss = spearman_loss(&predictions, &targets, 10.0);
        
        assert!(loss < 0.1); // Should be very low for perfect correlation
    }

    #[test]
    fn test_spearman_loss_anti_correlation() {
        // Anti-correlation should give high loss
        let predictions = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let targets = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let loss = spearman_loss(&predictions, &targets, 10.0);
        
        assert!(loss > 1.5); // Should be high for anti-correlation
    }
}

