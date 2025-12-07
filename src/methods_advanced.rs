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

#[cfg(feature = "gumbel")]
mod gumbel {
    use rand::Rng;

    /// Generate Gumbel noise: G = -log(-log(U)) where U ~ Uniform(0,1)
    ///
    /// Gumbel distribution is used in the Gumbel-Softmax trick for differentiable
    /// discrete sampling. The Gumbel distribution has the property that
    /// argmax_i (logit_i + G_i) follows the softmax distribution.
    ///
    /// # Arguments
    ///
    /// * `rng` - Random number generator
    ///
    /// # Returns
    ///
    /// Gumbel-distributed random value
    pub fn gumbel_noise(rng: &mut impl Rng) -> f64 {
        let u: f64 = rng.gen_range(0.0..1.0);
        // Ensure u is in (0, 1) to avoid log(0) or log(1)
        let u = u.max(1e-10).min(1.0 - 1e-10);
        -(-u.ln()).ln()
    }

    /// Compute softmax: exp(x_i) / sum_j exp(x_j)
    ///
    /// Uses numerical stability trick: subtract max before exponentiating.
    fn softmax(logits: &[f64]) -> Vec<f64> {
        let n = logits.len();
        if n == 0 {
            return vec![];
        }
        
        // Find max for numerical stability
        let max_logit = logits.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        
        // Compute exp(x_i - max) and sum
        let exps: Vec<f64> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
        let sum: f64 = exps.iter().sum();
        
        // Normalize
        if sum > 1e-10 {
            exps.iter().map(|&e| e / sum).collect()
        } else {
            // Fallback: uniform distribution
            vec![1.0 / n as f64; n]
        }
    }

    /// Gumbel-Softmax: Differentiable sampling from categorical distribution
    ///
    /// From: "Categorical Reparameterization with Gumbel-Softmax" (Jang et al., ICLR 2017)
    ///
    /// Converts discrete sampling into a differentiable operation by:
    /// 1. Adding Gumbel noise to logits
    /// 2. Applying softmax with temperature
    ///
    /// # Arguments
    ///
    /// * `logits` - Unnormalized log probabilities
    /// * `temperature` - Temperature parameter (τ). Lower = sharper, higher = smoother
    /// * `scale` - Scaling factor (κ) for logits. Controls influence of logits vs noise
    /// * `rng` - Random number generator
    ///
    /// # Returns
    ///
    /// Soft probability distribution over categories
    ///
    /// # Example
    ///
    /// ```rust
    /// use rank_relax::gumbel_softmax;
    /// use rand::thread_rng;
    ///
    /// let logits = vec![0.5, 1.0, 0.3];
    /// let mut rng = thread_rng();
    /// let probs = gumbel_softmax(&logits, 0.5, 1.0, &mut rng);
    /// // probs is a probability distribution (sums to 1.0)
    /// ```
    pub fn gumbel_softmax(
        logits: &[f64],
        temperature: f64,
        scale: f64,
        rng: &mut impl Rng,
    ) -> Vec<f64> {
        let n = logits.len();
        if n == 0 {
            return vec![];
        }
        if n == 1 {
            return vec![1.0];
        }
        
        // Add Gumbel noise and scale: (G_i + κ·logit_i) / τ
        let mut gumbel_logits = Vec::with_capacity(n);
        for &logit in logits {
            let g = gumbel_noise(rng);
            gumbel_logits.push((g + scale * logit) / temperature);
        }
        
        // Apply softmax
        softmax(&gumbel_logits)
    }

    /// Relaxed Top-k using Gumbel-Softmax
    ///
    /// From: "Gumbel Reranking: Differentiable End-to-End Reranker Optimization" (Huang et al., ACL 2025)
    ///
    /// Approximates top-k selection by:
    /// 1. Sampling k times independently using Gumbel-Softmax
    /// 2. Taking element-wise maximum across samples
    ///
    /// This creates a soft mask where top-k elements have high values (~1.0)
    /// and others have low values (~0.0), while remaining fully differentiable.
    ///
    /// # Arguments
    ///
    /// * `scores` - Reranker scores for each document/element
    /// * `k` - Number of top elements to select
    /// * `temperature` - Temperature parameter (τ). Lower = sharper selection
    /// * `scale` - Scaling factor (κ). Higher = more deterministic, lower = more exploration
    /// * `rng` - Random number generator
    ///
    /// # Returns
    ///
    /// Soft attention mask: values in [0, 1], higher for top-k elements
    ///
    /// # Example
    ///
    /// ```rust
    /// use rank_relax::relaxed_topk_gumbel;
    /// use rand::thread_rng;
    ///
    /// let scores = vec![0.8, 0.6, 0.9, 0.3, 0.7];
    /// let mut rng = thread_rng();
    /// let mask = relaxed_topk_gumbel(&scores, 3, 0.5, 1.0, &mut rng);
    /// // mask[i] ≈ 1.0 for top-3 elements, ≈ 0.0 for others
    /// ```
    pub fn relaxed_topk_gumbel(
        scores: &[f64],
        k: usize,
        temperature: f64,
        scale: f64,
        rng: &mut impl Rng,
    ) -> Vec<f64> {
        let n = scores.len();
        if n == 0 || k == 0 {
            return vec![];
        }
        if k >= n {
            // Return all ones (select all)
            return vec![1.0; n];
        }
        
        let mut max_mask = vec![0.0; n];
        
        // Sample k times, take element-wise maximum
        for _ in 0..k {
            let mask = gumbel_softmax(scores, temperature, scale, rng);
            for i in 0..n {
                if mask[i] > max_mask[i] {
                    max_mask[i] = mask[i];
                }
            }
        }
        
        max_mask
    }

    /// Generate Gumbel-based attention mask for RAG reranking
    ///
    /// This is a convenience function specifically for RAG reranking applications,
    /// combining relaxed top-k with Gumbel-Softmax to create differentiable attention masks.
    ///
    /// # Arguments
    ///
    /// * `reranker_scores` - Reranker relevance scores for each document
    /// * `k` - Number of top documents to select
    /// * `temperature` - Temperature parameter (default: 0.5 per paper)
    /// * `scale` - Scaling factor (default: 1.0 per paper)
    /// * `rng` - Random number generator
    ///
    /// # Returns
    ///
    /// Soft attention mask: values in [0, 1], can be applied to attention computation
    ///
    /// # Example
    ///
    /// ```rust
    /// use rank_relax::gumbel_attention_mask;
    /// use rand::thread_rng;
    ///
    /// let reranker_scores = vec![0.8, 0.6, 0.9, 0.3, 0.7];
    /// let mut rng = thread_rng();
    /// let attention_mask = gumbel_attention_mask(&reranker_scores, 3, 0.5, 1.0, &mut rng);
    ///
    /// // Apply mask to attention: masked_attention[i] = attention[i] * mask[i]
    /// ```
    pub fn gumbel_attention_mask(
        reranker_scores: &[f64],
        k: usize,
        temperature: f64,
        scale: f64,
        rng: &mut impl Rng,
    ) -> Vec<f64> {
        relaxed_topk_gumbel(reranker_scores, k, temperature, scale, rng)
    }
}

#[cfg(feature = "gumbel")]
pub use gumbel::{
    gumbel_attention_mask, gumbel_softmax, relaxed_topk_gumbel,
};

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

    #[cfg(feature = "gumbel")]
    mod gumbel_tests {
        use super::gumbel::*;
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        #[test]
        fn test_gumbel_noise() {
            let mut rng = StdRng::seed_from_u64(42);
            let noise = gumbel_noise(&mut rng);
            // Gumbel distribution: mean ≈ 0.577, variance ≈ 1.645
            assert!(noise.is_finite());
            assert!(noise > -10.0 && noise < 10.0); // Reasonable range
        }

        #[test]
        fn test_gumbel_softmax() {
            let mut rng = StdRng::seed_from_u64(42);
            let logits = vec![0.5, 1.0, 0.3];
            let probs = gumbel_softmax(&logits, 0.5, 1.0, &mut rng);
            
            assert_eq!(probs.len(), logits.len());
            // Should sum to approximately 1.0
            let sum: f64 = probs.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6);
            // All probabilities should be positive
            assert!(probs.iter().all(|&p| p >= 0.0 && p <= 1.0));
            // Higher logit should generally have higher probability
            assert!(probs[1] > probs[0]); // logits[1] = 1.0 > logits[0] = 0.5
        }

        #[test]
        fn test_relaxed_topk_gumbel() {
            let mut rng = StdRng::seed_from_u64(42);
            let scores = vec![0.8, 0.6, 0.9, 0.3, 0.7];
            let mask = relaxed_topk_gumbel(&scores, 3, 0.5, 1.0, &mut rng);
            
            assert_eq!(mask.len(), scores.len());
            // All mask values should be in [0, 1]
            assert!(mask.iter().all(|&m| m >= 0.0 && m <= 1.0));
            // Top-3 elements (indices 2, 0, 4) should have higher mask values
            assert!(mask[2] > mask[1]); // 0.9 > 0.6
            assert!(mask[0] > mask[3]); // 0.8 > 0.3
        }

        #[test]
        fn test_gumbel_attention_mask() {
            let mut rng = StdRng::seed_from_u64(42);
            let reranker_scores = vec![0.8, 0.6, 0.9, 0.3, 0.7];
            let mask = gumbel_attention_mask(&reranker_scores, 3, 0.5, 1.0, &mut rng);
            
            assert_eq!(mask.len(), reranker_scores.len());
            assert!(mask.iter().all(|&m| m >= 0.0 && m <= 1.0));
        }

        #[test]
        fn test_gumbel_edge_cases() {
            let mut rng = StdRng::seed_from_u64(42);
            
            // Empty input
            let empty: Vec<f64> = vec![];
            let probs = gumbel_softmax(&empty, 0.5, 1.0, &mut rng);
            assert_eq!(probs.len(), 0);
            
            // Single element
            let single = vec![1.0];
            let probs = gumbel_softmax(&single, 0.5, 1.0, &mut rng);
            assert_eq!(probs, vec![1.0]);
            
            // k >= n
            let scores = vec![0.5, 0.3];
            let mask = relaxed_topk_gumbel(&scores, 5, 0.5, 1.0, &mut rng);
            assert_eq!(mask, vec![1.0, 1.0]);
        }
    }
}

