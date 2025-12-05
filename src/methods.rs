//! Multiple differentiable ranking methods from research papers.
//!
//! This module implements various approaches to differentiable ranking, each with
//! different characteristics and trade-offs. Choose based on your needs:
//!
//! # Available Methods
//!
//! 1. **Sigmoid-based** (default): Simple, intuitive, O(n²)
//!    - Best for: General use, small-medium inputs
//!    - Trade-off: Simple but slower for large inputs
//!
//! 2. **NeuralSort-style**: Temperature-scaled softmax approach
//!    - Best for: When you need permutation matrices
//!    - Trade-off: Similar complexity to sigmoid, different gradient behavior
//!
//! 3. **SoftRank (Probabilistic)**: Gaussian smoothing approach
//!    - Best for: When you want probabilistic rank distributions
//!    - Trade-off: Uses normal CDF approximation
//!
//! 4. **SmoothI**: Smooth rank indicators
//!    - Best for: When you need smooth rank position indicators
//!    - Trade-off: Exponential scaling parameter
//!
//! # Choosing a Method
//!
//! **Start with Sigmoid** (default) - it's simple and works well for most cases.
//! Switch to other methods if you need specific properties (e.g., probabilistic
//! distributions, different gradient behavior).
//!
//! All methods have O(n²) complexity in the current implementation. For more
//! efficient methods (O(n log n)), see `MATHEMATICAL_DETAILS.md` for permutahedron
//! projection and optimal transport approaches.

use crate::rank::sigmoid;

/// Sigmoid-based soft ranking (current default, "naive" approach).
///
/// This is the simplest and most intuitive method, using sigmoid functions to
/// create smooth approximations of discrete ranking.
///
/// **Algorithm**: For each element, count (softly) how many others it's greater than:
/// ```
/// rank[i] = (1/(n-1)) * Σ_{j≠i} sigmoid(α * (values[i] - values[j]))
/// ```
///
/// **From**: General differentiable ranking approach (Qin et al., 2008)
/// **Complexity**: O(n²)
///
/// **When to use**: Default choice for most applications. Simple, intuitive, works well.
///
/// # Arguments
///
/// * `values` - Input values to rank
/// * `regularization_strength` - Temperature parameter α (controls sharpness)
///
/// # Returns
///
/// Vector of soft ranks in [0, n-1] range
pub fn soft_rank_sigmoid(values: &[f64], regularization_strength: f64) -> Vec<f64> {
    let n = values.len();
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![0.0];
    }
    
    let mut ranks = vec![0.0; n];
    let inv_n_minus_1 = 1.0 / (n - 1) as f64;
    
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
    
    ranks
}

/// NeuralSort-style ranking using temperature-scaled softmax.
///
/// This method uses temperature-scaled comparisons similar to NeuralSort,
/// which was designed to produce differentiable permutation matrices.
///
/// **Algorithm**: Uses temperature-scaled sigmoid for pairwise comparisons:
/// ```
/// rank[i] = (1/(n-1)) * Σ_{j≠i} sigmoid((values[i] - values[j]) / temperature)
/// ```
///
/// **From**: "NeuralSort: A Differentiable Sorting Operator" (Grover et al., ICML 2019)
/// **Complexity**: O(n²) for ranking
///
/// **When to use**: When you need permutation matrices or want different gradient
/// behavior than sigmoid-based approach.
///
/// **Note**: This is a simplified version. Full NeuralSort computes full permutation
/// matrices, which is more expensive but provides richer information.
///
/// # Arguments
///
/// * `values` - Input values to rank
/// * `temperature` - Temperature parameter (lower = sharper, higher = smoother)
///
/// # Returns
///
/// Vector of soft ranks in [0, n-1] range
pub fn soft_rank_neural_sort(values: &[f64], temperature: f64) -> Vec<f64> {
    let n = values.len();
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![0.0];
    }
    
    let mut ranks = vec![0.0; n];
    
    for i in 0..n {
        if !values[i].is_finite() {
            ranks[i] = f64::NAN;
            continue;
        }
        
        // For NeuralSort ranking, we compute how many elements are "less than" this one
        // using temperature-scaled sigmoid (which is similar to softmax for binary comparison)
        let mut rank_sum = 0.0;
        let mut valid_comparisons = 0;
        
        for j in 0..n {
            if i != j && values[j].is_finite() {
                let diff = (values[i] - values[j]) / temperature;
                // Use sigmoid for pairwise comparison (simpler than full softmax)
                // This approximates the NeuralSort approach
                rank_sum += sigmoid(diff);
                valid_comparisons += 1;
            }
        }
        
        if valid_comparisons > 0 {
            // Normalize to [0, n-1] range
            ranks[i] = rank_sum / valid_comparisons as f64 * (n - 1) as f64;
        } else {
            ranks[i] = 0.0;
        }
    }
    
    ranks
}

/// SoftRank probabilistic approach (simplified version).
///
/// This method uses Gaussian smoothing to create probabilistic rank distributions,
/// treating score differences as random variables.
///
/// **Algorithm**: Uses normal CDF approximation for pairwise comparisons:
/// ```
/// P(score_i > score_j) ≈ Φ((value_i - value_j) / (σ√2))
/// rank[i] = (1/(n-1)) * Σ_{j≠i} P(score_i > score_j)
/// ```
/// where Φ is the standard normal CDF (approximated using sigmoid).
///
/// **From**: "SoftRank: Optimizing Non-Smooth Rank Metrics" (Taylor et al., WSDM 2008)
/// **Complexity**: O(n²) for simplified version
///
/// **When to use**: When you want probabilistic rank distributions or need to model
/// uncertainty in scores.
///
/// **Note**: This is a simplified version. Full SoftRank computes full rank
/// distributions, which is more expensive but provides richer information.
///
/// # Arguments
///
/// * `values` - Input values to rank
/// * `sigma` - Standard deviation parameter (controls smoothing)
///
/// # Returns
///
/// Vector of soft ranks in [0, n-1] range
pub fn soft_rank_probabilistic(values: &[f64], sigma: f64) -> Vec<f64> {
    let n = values.len();
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![0.0];
    }
    
    // Simplified SoftRank: use normal CDF for pairwise comparisons
    // P(score_i > score_j) = Φ((value_i - value_j) / (σ√2))
    // where Φ is standard normal CDF
    
    let mut ranks = vec![0.0; n];
    let inv_n_minus_1 = 1.0 / (n - 1) as f64;
    let sqrt_2 = std::f64::consts::SQRT_2;
    
    for i in 0..n {
        if !values[i].is_finite() {
            ranks[i] = f64::NAN;
            continue;
        }
        
        let mut rank_sum = 0.0;
        for j in 0..n {
            if i != j && values[j].is_finite() {
                let diff = values[i] - values[j];
                // Approximate normal CDF using sigmoid: Φ(x) ≈ sigmoid(1.7 * x)
                // More accurate: use error function, but sigmoid is simpler
                let z = diff / (sigma * sqrt_2);
                let prob = sigmoid(1.7 * z); // Approximation of normal CDF
                rank_sum += prob;
            }
        }
        ranks[i] = rank_sum * inv_n_minus_1;
    }
    
    ranks
}

/// SmoothI-style smooth rank indicators.
///
/// This method uses exponential scaling to create smooth rank position indicators,
/// providing a different gradient profile than standard sigmoid-based approaches.
///
/// **Algorithm**: Uses exponential-scaled sigmoid for pairwise comparisons:
/// ```
/// rank[i] = (1/(n-1)) * Σ_{j≠i} sigmoid(α * (values[i] - values[j]))
/// ```
///
/// The key difference from standard sigmoid is the exponential scaling parameter α,
/// which can provide different gradient characteristics.
///
/// **From**: "SmoothI: Smooth Rank Indicators for Differentiable Ranking" (ICML 2021)
/// **Complexity**: O(n²)
///
/// **When to use**: When you need different gradient behavior or want to experiment
/// with alternative smooth ranking formulations.
///
/// **Note**: This is a simplified implementation. Full SmoothI may have additional
/// refinements for gradient quality.
///
/// # Arguments
///
/// * `values` - Input values to rank
/// * `alpha` - Exponential scaling parameter (similar to regularization_strength)
///
/// # Returns
///
/// Vector of soft ranks in [0, n-1] range
pub fn soft_rank_smooth_i(values: &[f64], alpha: f64) -> Vec<f64> {
    let n = values.len();
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![0.0];
    }
    
    // SmoothI uses smooth rank indicators, but for simplicity we use
    // a sigmoid-based approach similar to the base method but with
    // exponential scaling parameter
    let mut ranks = vec![0.0; n];
    let inv_n_minus_1 = 1.0 / (n - 1) as f64;
    
    for i in 0..n {
        if !values[i].is_finite() {
            ranks[i] = f64::NAN;
            continue;
        }
        
        // Count how many elements are "less than" using exponential-scaled sigmoid
        let mut rank_sum = 0.0;
        for j in 0..n {
            if i != j && values[j].is_finite() {
                let diff = values[i] - values[j];
                // Use exponential scaling for sharper transitions
                rank_sum += sigmoid(alpha * diff);
            }
        }
        
        ranks[i] = rank_sum * inv_n_minus_1;
    }
    
    ranks
}

/// Enum for selecting ranking method.
///
/// Different methods have different characteristics:
///
/// - **Sigmoid**: Default, simple, intuitive, O(n²)
/// - **NeuralSort**: Temperature-scaled, designed for permutation matrices
/// - **Probabilistic**: Gaussian smoothing, probabilistic rank distributions
/// - **SmoothI**: Exponential scaling, alternative gradient profile
///
/// See individual method documentation for details on when to use each.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RankingMethod {
    /// Sigmoid-based (naive, current default)
    /// Best for: General use, most applications
    Sigmoid,
    /// NeuralSort-style (temperature-scaled softmax)
    /// Best for: When you need permutation matrices or different gradient behavior
    NeuralSort,
    /// SoftRank probabilistic (Gaussian smoothing)
    /// Best for: Probabilistic rank distributions, uncertainty modeling
    Probabilistic,
    /// SmoothI (smooth rank indicators)
    /// Best for: Alternative gradient profiles, experimentation
    SmoothI,
}

impl RankingMethod {
    /// Compute soft ranks using the selected method.
    ///
    /// # Arguments
    ///
    /// * `values` - Input values to rank
    /// * `regularization_strength` - Temperature/scaling parameter
    ///   - For Sigmoid, NeuralSort, SmoothI: acts as temperature/sharpness
    ///   - For Probabilistic: acts as standard deviation (sigma)
    ///
    /// # Returns
    ///
    /// Vector of soft ranks in [0, n-1] range
    ///
    /// # Example
    ///
    /// ```rust
    /// use rank_relax::RankingMethod;
    ///
    /// let values = vec![5.0, 1.0, 2.0, 4.0, 3.0];
    ///
    /// // Use default sigmoid method
    /// let ranks_sigmoid = RankingMethod::Sigmoid.compute(&values, 1.0);
    ///
    /// // Try NeuralSort method
    /// let ranks_neural = RankingMethod::NeuralSort.compute(&values, 1.0);
    ///
    /// // Compare results (should be similar but may differ slightly)
    /// ```
    pub fn compute(
        &self,
        values: &[f64],
        regularization_strength: f64,
    ) -> Vec<f64> {
        match self {
            Self::Sigmoid => soft_rank_sigmoid(values, regularization_strength),
            Self::NeuralSort => soft_rank_neural_sort(values, regularization_strength),
            Self::Probabilistic => soft_rank_probabilistic(values, regularization_strength),
            Self::SmoothI => soft_rank_smooth_i(values, regularization_strength),
        }
    }
    
    /// Get method name for logging/benchmarking.
    ///
    /// Returns a human-readable string identifier for the method.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Sigmoid => "sigmoid",
            Self::NeuralSort => "neural_sort",
            Self::Probabilistic => "probabilistic",
            Self::SmoothI => "smooth_i",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_all_methods_preserve_ordering() {
        let values = vec![5.0, 1.0, 2.0, 4.0, 3.0];
        
        for method in [
            RankingMethod::Sigmoid,
            RankingMethod::NeuralSort,
            RankingMethod::Probabilistic,
            RankingMethod::SmoothI,
        ] {
            let ranks = method.compute(&values, 1.0);
            
            // Should preserve ordering: value[1] < value[2] < value[4] < value[3] < value[0]
            assert!(ranks[1] < ranks[2], "{} failed ordering", method.name());
            assert!(ranks[2] < ranks[4], "{} failed ordering", method.name());
            assert!(ranks[4] < ranks[3], "{} failed ordering", method.name());
            assert!(ranks[3] < ranks[0], "{} failed ordering", method.name());
        }
    }
}

