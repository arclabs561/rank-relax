//! Differentiable sorting and ranking operations for Rust ML frameworks.
//!
//! This crate provides smooth relaxations of discrete sorting and ranking operations,
//! enabling gradient-based optimization of objectives that depend on ordering.
//!
//! # Why Differentiable Ranking?
//!
//! Traditional ranking operations are **discrete** and **non-differentiable**:
//! - As you change a value, its rank jumps by integer steps (0, 1, 2, ...)
//! - These "jumps" have zero gradient almost everywhere
//! - This prevents optimizing ranking-based metrics (Spearman, NDCG) during training
//!
//! **Solution**: Replace discrete operations with smooth, differentiable approximations
//! that preserve ranking semantics while enabling gradient flow.
//!
//! # Overview
//!
//! The core concept is to replace non-differentiable discrete operations (sorting, ranking)
//! with continuous, differentiable approximations that:
//!
//! 1. **Preserve ranking semantics**: Maintain relative ordering information
//! 2. **Enable gradient flow**: Gradients can flow through ranking operations
//! 3. **Converge to discrete**: As regularization increases, behavior approaches discrete ranking
//!
//! # Example
//!
//! ```rust
//! use rank_relax::{soft_rank, spearman_loss};
//!
//! // Soft ranking (differentiable)
//! let values = vec![5.0, 1.0, 2.0, 4.0, 3.0];
//! let ranks = soft_rank(&values, 1.0); // regularization_strength = 1.0
//! // With high regularization, ranks approach [4.0, 0.0, 1.0, 3.0, 2.0]
//!
//! // Spearman correlation loss (for training)
//! let predictions = vec![0.1, 0.9, 0.3, 0.7, 0.5];
//! let targets = vec![0.0, 1.0, 0.2, 0.8, 0.4];
//! let loss = spearman_loss(&predictions, &targets, 1.0);
//! // Loss = 1 - Spearman correlation (lower is better)
//! ```
//!
//! # Algorithm: Sigmoid-Based Soft Ranking
//!
//! The current implementation uses the "naive" sigmoid-based approach:
//!
//! ```text
//! rank[i] = (1/(n-1)) * sum_{j != i} sigmoid(alpha * (values[i] - values[j]))
//! ```
//!
//! where `α = regularization_strength` controls sharpness.
//!
//! **Intuition**: For each element, count (softly) how many others it's greater than.
//! The sigmoid provides a smooth transition between "less than" and "greater than".
//!
//! **Trade-offs**:
//! - ✅ Simple and intuitive
//! - ✅ Works well for small-medium inputs
//! - ❌ O(n²) complexity (not suitable for very large inputs)
//!
//! For more efficient methods (permutahedron projection, optimal transport), see
//! `MATHEMATICAL_DETAILS.md`.
//!
//! # Parameter Tuning
//!
//! The `regularization_strength` parameter controls the sharpness of the relaxation:
//!
//! - **Low (0.1-1.0)**: Smooth gradients, good for early training
//! - **Medium (1.0-10.0)**: Balanced between smoothness and accuracy
//! - **High (10.0-100.0)**: Sharper, closer to discrete ranking
//!
//! **Rule of thumb**: Choose based on the scale of differences in your values.
//! If values differ by ~1.0, use `regularization_strength ~= 1.0`.
//! If differences are ~0.1, use `~= 10.0`.
//!
//! # Mathematical Background
//!
//! These methods implement **smooth relaxations** of discrete combinatorial operations.
//! See `MATHEMATICAL_DETAILS.md` for comprehensive mathematical formulations, derivations,
//! and theoretical foundations.

/// Differentiable ranking operations
pub mod rank;

/// Differentiable sorting operations
pub mod sort;

/// Spearman correlation utilities
pub mod spearman;

/// Analytical gradient computation
pub mod gradients;

/// Multiple ranking methods from research papers
pub mod methods;

/// Batch processing utilities
pub mod batch;

/// Performance-optimized implementations
pub mod optimized;

/// Advanced ranking methods from research papers
pub mod methods_advanced;

// Re-export advanced methods
pub use methods_advanced::{
    soft_rank_softsort, differentiable_topk, listnet_loss, listmle_loss
};

// Re-export Gumbel methods (requires "gumbel" feature)
#[cfg(feature = "gumbel")]
pub use methods_advanced::{
    gumbel_attention_mask, gumbel_softmax, relaxed_topk_gumbel,
};

// Re-export main functions
pub use rank::soft_rank;
pub use sort::soft_sort;
pub use spearman::spearman_loss;
pub use gradients::{soft_rank_gradient, spearman_loss_gradient, sigmoid_derivative};
pub use methods::{RankingMethod, soft_rank_sigmoid, soft_rank_neural_sort, soft_rank_probabilistic, soft_rank_smooth_i};
pub use batch::{soft_rank_batch, spearman_loss_batch};
#[cfg(feature = "parallel")]
pub use optimized::soft_rank_batch_parallel;
pub use optimized::{soft_rank_optimized, soft_rank_gradient_sparse};

#[cfg(test)]
mod proptests;
