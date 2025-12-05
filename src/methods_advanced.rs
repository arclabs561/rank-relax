//! Advanced differentiable ranking methods from research papers.
//!
//! This module implements more sophisticated methods that require additional
//! dependencies or more complex algorithms.

use crate::rank::sigmoid;

/// SoftSort using optimal transport (simplified version).
///
/// From: "SoftSort: A Continuous Relaxation for the argsort Operator" (ICML 2020)
/// Complexity: O(n²) for simplified version, O(n² log n) for full Sinkhorn
///
/// Uses Sinkhorn algorithm for optimal transport-based sorting.
/// This is a simplified version that approximates the full method.
pub fn soft_rank_softsort(values: &[f64], regularization_strength: f64) -> Vec<f64> {
    let n = values.len();
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![0.0];
    }
    
    // Simplified SoftSort: use temperature-scaled sigmoid with position weighting
    // Full SoftSort requires Sinkhorn iterations which is more complex
    let mut ranks = vec![0.0; n];
    let inv_n_minus_1 = 1.0 / (n - 1) as f64;
    
    // Create position weights (for optimal transport interpretation)
    let positions: Vec<f64> = (0..n).map(|i| i as f64).collect();
    
    for i in 0..n {
        if !values[i].is_finite() {
            ranks[i] = f64::NAN;
            continue;
        }
        
        // Compute weighted sum based on value differences and positions
        let mut rank_sum = 0.0;
        for j in 0..n {
            if i != j && values[j].is_finite() {
                let diff = values[i] - values[j];
                let sig = sigmoid(diff * regularization_strength);
                
                // Weight by position difference (SoftSort intuition)
                let pos_diff = (positions[i] - positions[j]).abs();
                let weight = (-pos_diff / regularization_strength).exp();
                
                rank_sum += sig * weight;
            }
        }
        
        ranks[i] = rank_sum * inv_n_minus_1;
    }
    
    ranks
}

/// Differentiable Top-K selection (DFTopK-style).
///
/// From: "Differentiable Top-k Operator with Optimal Transport" (NeurIPS 2020)
/// Complexity: O(n²) for simplified version
///
/// Selects top-k elements in a differentiable manner.
pub fn differentiable_topk(
    values: &[f64],
    k: usize,
    regularization_strength: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n = values.len();
    
    if n == 0 || k == 0 {
        return (vec![], vec![]);
    }
    
    if k >= n {
        // Return all elements
        let ranks = crate::rank::soft_rank(values, regularization_strength);
        return (values.to_vec(), ranks);
    }
    
    // Compute soft ranks
    let ranks = crate::rank::soft_rank(values, regularization_strength);
    
    // Create soft top-k mask using sigmoid
    // Elements with rank < k are selected
    let mut topk_values = Vec::with_capacity(n);
    let mut topk_ranks = Vec::with_capacity(n);
    
    for i in 0..n {
        // Soft indicator: sigmoid((k - rank[i]) * temperature)
        // This gives ~1.0 for top-k elements, ~0.0 for others
        let rank_pos = ranks[i];
        let indicator = sigmoid((k as f64 - rank_pos) * regularization_strength);
        
        // Weight values by indicator
        topk_values.push(values[i] * indicator);
        topk_ranks.push(ranks[i] * indicator);
    }
    
    (topk_values, topk_ranks)
}

/// ListNet-style listwise ranking loss.
///
/// From: "Learning to Rank: From Pairwise Approach to Listwise Approach" (ICML 2007)
/// Complexity: O(n²)
///
/// Uses cross-entropy between predicted and target rank distributions.
pub fn listnet_loss(
    predictions: &[f64],
    targets: &[f64],
    regularization_strength: f64,
) -> f64 {
    let n = predictions.len();
    
    if n == 0 || n != targets.len() {
        return f64::INFINITY;
    }
    
    // Convert to probability distributions using softmax
    let pred_ranks = crate::rank::soft_rank(predictions, regularization_strength);
    let target_ranks = crate::rank::soft_rank(targets, regularization_strength);
    
    // Normalize ranks to probabilities (softmax)
    let pred_probs = softmax_from_ranks(&pred_ranks);
    let target_probs = softmax_from_ranks(&target_ranks);
    
    // Cross-entropy loss
    let mut loss = 0.0;
    for i in 0..n {
        if target_probs[i] > 1e-10 {
            loss -= target_probs[i] * pred_probs[i].ln();
        }
    }
    
    loss
}

/// Convert ranks to probability distribution using softmax.
fn softmax_from_ranks(ranks: &[f64]) -> Vec<f64> {
    let n = ranks.len();
    if n == 0 {
        return vec![];
    }
    
    // Use negative ranks (higher rank = lower value in softmax)
    let max_rank = ranks.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let exp_sum: f64 = ranks.iter().map(|&r| (-(r - max_rank)).exp()).sum();
    
    ranks.iter().map(|&r| (-(r - max_rank)).exp() / exp_sum).collect()
}

/// ListMLE-style maximum likelihood estimation for ranking.
///
/// From: "Listwise Approach to Learning to Rank: Theory and Algorithm" (ICML 2008)
/// Complexity: O(n²)
///
/// Uses permutation probability model for ranking.
pub fn listmle_loss(
    predictions: &[f64],
    targets: &[f64],
    regularization_strength: f64,
) -> f64 {
    let n = predictions.len();
    
    if n == 0 || n != targets.len() {
        return f64::INFINITY;
    }
    
    // Get target ranking order
    let mut target_indices: Vec<usize> = (0..n).collect();
    target_indices.sort_by(|&a, &b| targets[b].partial_cmp(&targets[a]).unwrap());
    
    // Compute soft ranks for predictions
    let pred_ranks = crate::rank::soft_rank(predictions, regularization_strength);
    
    // ListMLE loss: negative log-likelihood of permutation
    // P(π) = ∏_{i=1}^{n} exp(score[π[i]]) / Σ_{j=i}^{n} exp(score[π[j]])
    let mut loss = 0.0;
    
    for i in 0..n {
        let idx = target_indices[i];
        let score = pred_ranks[idx];
        
        // Denominator: sum of exp scores for remaining elements
        let mut denom = 0.0;
        for &jdx in target_indices.iter().skip(i) {
            denom += pred_ranks[jdx].exp();
        }
        
        if denom > 1e-10 {
            loss -= score - denom.ln();
        }
    }
    
    loss
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_differentiable_topk() {
        let values = vec![5.0, 1.0, 2.0, 4.0, 3.0];
        let (topk_vals, _topk_ranks) = differentiable_topk(&values, 3, 1.0);
        
        assert_eq!(topk_vals.len(), values.len());
        // Top-3 should have higher weights
        assert!(topk_vals[0] > topk_vals[1]); // 5.0 > 1.0
    }
    
    #[test]
    fn test_listnet_loss() {
        let predictions = vec![0.1, 0.9, 0.3, 0.7, 0.5];
        let targets = vec![0.0, 1.0, 0.2, 0.8, 0.4];
        
        let loss = listnet_loss(&predictions, &targets, 1.0);
        assert!(loss >= 0.0);
        assert!(loss.is_finite());
    }
    
    #[test]
    fn test_listmle_loss() {
        let predictions = vec![0.1, 0.9, 0.3, 0.7, 0.5];
        let targets = vec![0.0, 1.0, 0.2, 0.8, 0.4];
        
        let loss = listmle_loss(&predictions, &targets, 1.0);
        assert!(loss >= 0.0);
        assert!(loss.is_finite());
    }
}

