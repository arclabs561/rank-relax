//! Spearman correlation utilities for differentiable ranking.

/// Compute Spearman correlation loss between predictions and targets.
///
/// Uses soft ranking to compute differentiable Spearman correlation.
/// Loss = 1 - Spearman correlation (so lower is better, Spearman higher is better).
///
/// # Arguments
///
/// * `predictions` - Model predictions
/// * `targets` - Ground truth values
/// * `regularization_strength` - Temperature parameter for soft ranking
///
/// # Returns
///
/// Loss value (1 - Spearman correlation)
///
/// # Example
///
/// ```rust
/// use rank_relax::spearman_loss;
///
/// let predictions = vec![0.1, 0.9, 0.3, 0.7, 0.5];
/// let targets = vec![0.0, 1.0, 0.2, 0.8, 0.4];
/// let loss = spearman_loss(&predictions, &targets, 1.0);
/// ```
pub fn spearman_loss(predictions: &[f64], targets: &[f64], regularization_strength: f64) -> f64 {
    use crate::rank::soft_rank;
    
    if predictions.len() != targets.len() {
        return 1.0; // Maximum loss for mismatched lengths
    }
    
    if predictions.len() < 2 {
        return 0.0; // No correlation possible with < 2 elements
    }
    
    // Compute soft ranks
    let pred_ranks = soft_rank(predictions, regularization_strength);
    let target_ranks = soft_rank(targets, regularization_strength);
    
    // Compute Pearson correlation of ranks (this is Spearman correlation)
    let pred_mean = pred_ranks.iter().sum::<f64>() / pred_ranks.len() as f64;
    let target_mean = target_ranks.iter().sum::<f64>() / target_ranks.len() as f64;
    
    let mut numerator = 0.0;
    let mut pred_var = 0.0;
    let mut target_var = 0.0;
    
    for i in 0..pred_ranks.len() {
        let pred_diff = pred_ranks[i] - pred_mean;
        let target_diff = target_ranks[i] - target_mean;
        numerator += pred_diff * target_diff;
        pred_var += pred_diff * pred_diff;
        target_var += target_diff * target_diff;
    }
    
    let denominator = (pred_var * target_var).sqrt();
    if denominator < 1e-8 {
        return 1.0; // No variance, maximum loss
    }
    
    let spearman = numerator / denominator;
    
    // Loss = 1 - Spearman (so lower is better)
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

