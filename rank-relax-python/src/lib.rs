//! Python bindings for rank-relax using PyO3.
//!
//! Provides a Python API that mirrors the Rust API, enabling seamless
//! integration with Python ML training workflows.
//!
//! # Usage
//!
//! ```python
//! import rank_relax
//!
//! # Soft ranking
//! values = [5.0, 1.0, 2.0, 4.0, 3.0]
//! ranks = rank_relax.soft_rank(values, regularization_strength=1.0)
//!
//! # Soft sorting
//! sorted_values = rank_relax.soft_sort(values, regularization_strength=1.0)
//!
//! # Spearman correlation loss
//! predictions = [0.1, 0.9, 0.3, 0.7, 0.5]
//! targets = [0.0, 1.0, 0.2, 0.8, 0.4]
//! loss = rank_relax.spearman_loss(predictions, targets, regularization_strength=1.0)
//! ```

// TODO: Remove allow(deprecated) when upgrading to pyo3 0.25+ which uses IntoPyObject
#![allow(deprecated)]

use ::rank_relax::{soft_rank, soft_sort, spearman_loss};
use pyo3::prelude::*;
use pyo3::types::PyList;

/// Python module for rank-relax.
#[pymodule]
#[pyo3(name = "rank_relax")]
fn rank_relax_module(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(soft_rank_py, m)?)?;
    m.add_function(wrap_pyfunction!(soft_sort_py, m)?)?;
    m.add_function(wrap_pyfunction!(spearman_loss_py, m)?)?;
    Ok(())
}

/// Compute soft ranks for a vector of values.
///
/// Uses a smooth relaxation of the discrete ranking operation, enabling
/// gradient flow through the ranking.
///
/// # Arguments
///
/// * `values` - Input values to rank (list of floats)
/// * `regularization_strength` - Temperature parameter controlling sharpness
///   (higher = sharper, more discrete-like behavior)
///
/// # Returns
///
/// List of soft ranks (continuous approximations of integer ranks)
#[pyfunction]
fn soft_rank_py(values: &Bound<'_, PyList>, regularization_strength: f64) -> PyResult<Vec<f64>> {
    let rust_values: Vec<f64> = values
        .iter()
        .map(|v| v.extract::<f64>())
        .collect::<Result<Vec<_>, _>>()?;
    
    Ok(soft_rank(&rust_values, regularization_strength))
}

/// Compute soft sorted values for a vector.
///
/// Uses a smooth relaxation of the discrete sorting operation, enabling
/// gradient flow through the sorting.
///
/// # Arguments
///
/// * `values` - Input values to sort (list of floats)
/// * `regularization_strength` - Temperature parameter controlling sharpness
///
/// # Returns
///
/// List of soft sorted values (continuous approximations)
#[pyfunction]
fn soft_sort_py(values: &Bound<'_, PyList>, regularization_strength: f64) -> PyResult<Vec<f64>> {
    let rust_values: Vec<f64> = values
        .iter()
        .map(|v| v.extract::<f64>())
        .collect::<Result<Vec<_>, _>>()?;
    
    Ok(soft_sort(&rust_values, regularization_strength))
}

/// Compute Spearman correlation loss between predictions and targets.
///
/// Uses soft ranking to compute differentiable Spearman correlation.
/// Loss = 1 - Spearman correlation (so lower is better, Spearman higher is better).
///
/// # Arguments
///
/// * `predictions` - Model predictions (list of floats)
/// * `targets` - Ground truth values (list of floats)
/// * `regularization_strength` - Temperature parameter for soft ranking
///
/// # Returns
///
/// Loss value (1 - Spearman correlation)
#[pyfunction]
fn spearman_loss_py(
    predictions: &Bound<'_, PyList>,
    targets: &Bound<'_, PyList>,
    regularization_strength: f64,
) -> PyResult<f64> {
    let rust_predictions: Vec<f64> = predictions
        .iter()
        .map(|v| v.extract::<f64>())
        .collect::<Result<Vec<_>, _>>()?;
    
    let rust_targets: Vec<f64> = targets
        .iter()
        .map(|v| v.extract::<f64>())
        .collect::<Result<Vec<_>, _>>()?;
    
    Ok(spearman_loss(&rust_predictions, &rust_targets, regularization_strength))
}

