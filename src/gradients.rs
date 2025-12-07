//! Analytical gradient computation for differentiable ranking operations.
//!
//! This module provides efficient analytical gradient formulas for soft ranking,
//! enabling fast backward passes without numerical differentiation.
//!
//! # Why Analytical Gradients?
//!
//! While automatic differentiation (autograd) can compute gradients automatically,
//! analytical formulas are:
//! - **Faster**: No need to trace through computation graph
//! - **More efficient**: Direct computation of gradient formulas
//! - **Explicit**: Clear understanding of how gradients flow
//!
//! # Gradient Formulas
//!
//! For sigmoid-based soft ranking, the gradients have closed-form expressions:
//! - **Soft rank gradient**: Uses sigmoid derivative `σ'(x) = σ(x)(1-σ(x))`
//! - **Spearman loss gradient**: Chain rule through correlation and ranking
//!
//! See function documentation for detailed formulas.

use crate::rank::sigmoid;

/// Compute the gradient of soft_rank with respect to input values.
///
/// For sigmoid-based soft ranking:
/// ```text
/// rank[i] = (1/(n-1)) * sum_{j != i} sigmoid(alpha * (values[i] - values[j]))
/// ```
///
/// The gradient with respect to `values[k]` is:
/// ```text
/// d(rank[i])/d(values[k]) = {
///   if i == k: (alpha/(n-1)) * sum_{j != i} sigmoid'(alpha * (values[i] - values[j]))
///   if i != k: -(alpha/(n-1)) * sigmoid'(alpha * (values[i] - values[k]))
/// }
/// ```
///
/// where `sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))`
///
/// # Arguments
///
/// * `values` - Input values [n]
/// * `ranks` - Precomputed soft ranks [n] (from forward pass)
/// * `regularization_strength` - Temperature parameter α
///
/// # Returns
///
/// Gradient matrix [n, n] where `grad[i][j] = d(rank[i])/d(values[j])`
pub fn soft_rank_gradient(
    values: &[f64],
    ranks: &[f64],
    regularization_strength: f64,
) -> Vec<Vec<f64>> {
    let n = values.len();
    
    if n == 0 || n == 1 {
        return vec![vec![0.0; n]; n];
    }
    
    let alpha = regularization_strength;
    let inv_n_minus_1 = 1.0 / (n - 1) as f64;
    
    let mut grad = vec![vec![0.0; n]; n];
    
    for i in 0..n {
        if !values[i].is_finite() || !ranks[i].is_finite() {
            continue; // Skip NaN/Inf
        }
        
        for k in 0..n {
            if !values[k].is_finite() {
                continue;
            }
            
            if i == k {
                // Diagonal: gradient w.r.t. same element
                let mut sum = 0.0;
                for j in 0..n {
                    if i != j && values[j].is_finite() {
                        let diff = values[i] - values[j];
                        let sig = sigmoid(alpha * diff);
                        let sig_deriv = sig * (1.0 - sig); // sigmoid'(x) = σ(x)(1-σ(x))
                        sum += sig_deriv;
                    }
                }
                grad[i][k] = alpha * inv_n_minus_1 * sum;
            } else {
                // Off-diagonal: gradient w.r.t. different element
                let diff = values[i] - values[k];
                let sig = sigmoid(alpha * diff);
                let sig_deriv = sig * (1.0 - sig);
                grad[i][k] = -alpha * inv_n_minus_1 * sig_deriv;
            }
        }
    }
    
    grad
}

/// Compute gradient of Spearman loss with respect to predictions.
///
/// Spearman loss = 1 - Pearson_correlation(rank(pred), rank(target))
///
/// The gradient requires:
/// 1. Gradient of soft_rank w.r.t. predictions
/// 2. Gradient of Pearson correlation w.r.t. ranks
/// 3. Chain rule: dL/dpred = (dL/drank_pred) * (drank_pred/dpred)
///
/// # Arguments
///
/// * `predictions` - Model predictions [n]
/// * `targets` - Ground truth [n]
/// * `pred_ranks` - Precomputed soft ranks of predictions [n]
/// * `target_ranks` - Precomputed soft ranks of targets [n]
/// * `regularization_strength` - Temperature parameter
///
/// # Returns
///
/// Gradient vector [n] where `grad[i] = d(loss)/d(predictions[i])`
pub fn spearman_loss_gradient(
    predictions: &[f64],
    _targets: &[f64],
    pred_ranks: &[f64],
    target_ranks: &[f64],
    regularization_strength: f64,
) -> Vec<f64> {
    let n = predictions.len();
    
    if n < 2 {
        return vec![0.0; n];
    }
    
    // Step 1: Compute gradient of Pearson correlation w.r.t. pred_ranks
    let pred_mean = pred_ranks.iter().sum::<f64>() / n as f64;
    let target_mean = target_ranks.iter().sum::<f64>() / n as f64;
    
    // Compute variances and covariance
    let mut pred_var = 0.0;
    let mut target_var = 0.0;
    let mut covariance = 0.0;
    
    for i in 0..n {
        let pred_diff = pred_ranks[i] - pred_mean;
        let target_diff = target_ranks[i] - target_mean;
        pred_var += pred_diff * pred_diff;
        target_var += target_diff * target_diff;
        covariance += pred_diff * target_diff;
    }
    
    let denominator = (pred_var * target_var).sqrt();
    if denominator < 1e-8 {
        return vec![0.0; n]; // No variance, no gradient
    }
    
    let correlation = covariance / denominator;
    
    // Gradient of loss = 1 - correlation w.r.t. pred_ranks
    // ∂(1 - corr)/∂rank_pred[i] = -∂corr/∂rank_pred[i]
    // 
    // For Pearson correlation:
    // corr = cov / (std_pred * std_target)
    // ∂corr/∂rank_pred[i] = (1/(n*std_pred*std_target)) * (
    //   target_diff[i] - corr * (pred_diff[i] / std_pred^2) * pred_var
    // )
    
    let pred_std = pred_var.sqrt();
    let target_std = target_var.sqrt();
    let inv_denom = 1.0 / denominator;
    
    // Gradient of Pearson correlation w.r.t. pred_ranks[i]
    // corr = cov / (std_pred * std_target)
    // ∂corr/∂rank_pred[i] = (1/(std_pred * std_target)) * (
    //   target_diff[i] - corr * (pred_diff[i] * std_target / std_pred)
    // )
    // Simplified: = (target_diff[i] - corr * pred_diff[i] * (std_target/std_pred)) / (std_pred * std_target)
    // Since std_target/std_pred = target_std/pred_std, and denominator = pred_std * target_std
    // = (target_diff[i] - corr * pred_diff[i] * (target_std/pred_std)) / (pred_std * target_std)
    // = (target_diff[i] - corr * pred_diff[i] * target_std / pred_std) * inv_denom
    
    let mut corr_grad_wrt_ranks = vec![0.0; n];
    for i in 0..n {
        let pred_diff = pred_ranks[i] - pred_mean;
        let target_diff = target_ranks[i] - target_mean;
        
        // Correct gradient formula for Pearson correlation
        let term1 = target_diff * inv_denom;
        let term2 = correlation * pred_diff * target_std * inv_denom / pred_std;
        corr_grad_wrt_ranks[i] = term1 - term2;
    }
    
    // Step 2: Gradient of loss w.r.t. pred_ranks (negative of correlation gradient)
    let loss_grad_wrt_ranks: Vec<f64> = corr_grad_wrt_ranks.iter().map(|&g| -g).collect();
    
    // Step 3: Gradient of soft_rank w.r.t. predictions
    let rank_grad = soft_rank_gradient(predictions, pred_ranks, regularization_strength);
    
    // Step 4: Chain rule: ∂loss/∂pred = (∂loss/∂rank) * (∂rank/∂pred)
    let mut grad = vec![0.0; n];
    for i in 0..n {
        for j in 0..n {
            grad[j] += loss_grad_wrt_ranks[i] * rank_grad[i][j];
        }
    }
    
    grad
}

/// Sigmoid derivative: σ'(x) = σ(x) * (1 - σ(x))
pub fn sigmoid_derivative(x: f64) -> f64 {
    let sig = sigmoid(x);
    sig * (1.0 - sig)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rank::soft_rank;
    
    #[test]
    fn test_soft_rank_gradient_basic() {
        let values = vec![1.0, 2.0, 3.0];
        let ranks = soft_rank(&values, 1.0);
        let grad = soft_rank_gradient(&values, &ranks, 1.0);
        
        // Gradient matrix should be n x n
        assert_eq!(grad.len(), 3);
        assert_eq!(grad[0].len(), 3);
        
        // Diagonal elements should be positive (increasing value increases rank)
        assert!(grad[0][0] > 0.0);
        assert!(grad[1][1] > 0.0);
        assert!(grad[2][2] > 0.0);
    }
    
    #[test]
    fn test_spearman_loss_gradient_basic() {
        let predictions = vec![0.1, 0.9, 0.3, 0.7, 0.5];
        let targets = vec![0.0, 1.0, 0.2, 0.8, 0.4];
        
        let pred_ranks = soft_rank(&predictions, 1.0);
        let target_ranks = soft_rank(&targets, 1.0);
        
        let grad = spearman_loss_gradient(
            &predictions,
            &targets,
            &pred_ranks,
            &target_ranks,
            1.0,
        );
        
        assert_eq!(grad.len(), predictions.len());
        // Gradients should be finite
        assert!(grad.iter().all(|&g| g.is_finite()));
    }
}

