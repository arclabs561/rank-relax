//! Differentiable sorting and ranking operations for Rust.
//!
//! This crate provides smooth relaxations of discrete sorting and ranking operations,
//! enabling gradient-based optimization of objectives that depend on ordering.
//!
//! # Overview
//!
//! The core concept is to replace non-differentiable discrete operations (sorting, ranking)
//! with continuous, differentiable approximations that:
//!
//! 1. Preserve ranking semantics
//! 2. Enable gradient flow through the ranking operation
//! 3. Converge to discrete behavior as regularization strength increases
//!
//! # Example
//!
//! ```rust
//! use rank_relax::soft_rank;
//!
//! let values = vec![5.0, 1.0, 2.0, 4.0, 3.0];
//! let ranks = soft_rank(&values, 1.0); // regularization_strength = 1.0
//! ```
//!
//! # Mathematical Background
//!
//! These methods implement **smooth relaxations** of discrete combinatorial operations.
//! See the documentation for detailed explanations of the mathematical framework.

/// Differentiable ranking operations
pub mod rank;

/// Differentiable sorting operations
pub mod sort;

/// Spearman correlation utilities
pub mod spearman;

// Re-export main functions
pub use rank::soft_rank;
pub use sort::soft_sort;
pub use spearman::spearman_loss;

#[cfg(test)]
mod proptests;
