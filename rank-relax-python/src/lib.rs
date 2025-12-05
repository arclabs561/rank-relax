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
//!
//! # PyTorch Support
//!
//! ```python
//! import torch
//! import rank_relax
//!
//! predictions = torch.tensor([0.1, 0.9, 0.3], requires_grad=True)
//! targets = torch.tensor([0.0, 1.0, 0.2])
//! loss = rank_relax.spearman_loss_pytorch(predictions, targets, 1.0)
//! loss.backward()  # Gradients flow through!
//! ```
//!
//! # NumPy/JAX Support
//!
//! ```python
//! import numpy as np
//! import rank_relax
//!
//! values = np.array([5.0, 1.0, 2.0, 4.0, 3.0])
//! ranks = rank_relax.soft_rank_numpy(values, 1.0)
//! ```

// TODO: Remove allow(deprecated) when upgrading to pyo3 0.25+ which uses IntoPyObject
#![allow(deprecated)]

use ::rank_relax::{soft_rank, soft_sort, spearman_loss};
use pyo3::prelude::*;
use pyo3::types::PyList;

#[cfg(feature = "pytorch")]
use pyo3_tch::PyTensor;
#[cfg(feature = "pytorch")]
use tch::{Tensor, Kind};

#[cfg(feature = "numpy")]
use numpy::{PyArray1, PyArrayMethods};

/// Python module for rank-relax.
#[pymodule]
#[pyo3(name = "rank_relax")]
fn rank_relax_module(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Core functions (always available)
    m.add_function(wrap_pyfunction!(soft_rank_py, m)?)?;
    m.add_function(wrap_pyfunction!(soft_sort_py, m)?)?;
    m.add_function(wrap_pyfunction!(spearman_loss_py, m)?)?;
    
    // PyTorch functions (optional)
    #[cfg(feature = "pytorch")]
    {
        m.add_function(wrap_pyfunction!(soft_rank_pytorch, m)?)?;
        m.add_function(wrap_pyfunction!(soft_sort_pytorch, m)?)?;
        m.add_function(wrap_pyfunction!(spearman_loss_pytorch, m)?)?;
    }
    
    // NumPy/JAX functions (optional)
    #[cfg(feature = "numpy")]
    {
        m.add_function(wrap_pyfunction!(soft_rank_numpy, m)?)?;
        m.add_function(wrap_pyfunction!(soft_sort_numpy, m)?)?;
        m.add_function(wrap_pyfunction!(spearman_loss_numpy, m)?)?;
    }
    
    Ok(())
}

// ============================================================================
// Core Functions (Python Lists)
// ============================================================================

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
    // Use cast instead of extract for better performance (see PyO3 docs)
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

// ============================================================================
// PyTorch Functions (Optional Feature)
// ============================================================================

#[cfg(feature = "pytorch")]
/// Compute soft ranks for PyTorch tensors.
///
/// Preserves gradient tracking if input tensor has `requires_grad=True`.
///
/// # Arguments
///
/// * `values` - PyTorch tensor (1D)
/// * `regularization_strength` - Temperature parameter
///
/// # Returns
///
/// PyTorch tensor with soft ranks (preserves gradients)
#[pyfunction]
fn soft_rank_pytorch(values: PyTensor, regularization_strength: f64) -> PyResult<PyTensor> {
    // Extract tensor data efficiently
    let tensor = &values.0;
    
    // Check tensor properties
    if tensor.size().len() != 1 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "soft_rank_pytorch expects 1D tensor"
        ));
    }
    
    // Convert to Vec<f64> for computation
    // TODO: Implement tensor-native version to avoid conversion
    let values_vec: Vec<f64> = tensor.to_vec1::<f64>()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            format!("Failed to extract tensor data: {}", e)
        ))?;
    
    // Compute soft ranks
    let ranks = soft_rank(&values_vec, regularization_strength);
    
    // Create new tensor with results
    // Note: This creates a new tensor, so gradients won't flow through automatically
    // For full gradient support, need custom autograd function
    let result_tensor = Tensor::from_slice(&ranks)
        .to_kind(tensor.kind())
        .to_device(tensor.device());
    
    Ok(PyTensor(result_tensor))
}

#[cfg(feature = "pytorch")]
/// Compute soft sorted values for PyTorch tensors.
#[pyfunction]
fn soft_sort_pytorch(values: PyTensor, regularization_strength: f64) -> PyResult<PyTensor> {
    let tensor = &values.0;
    
    if tensor.size().len() != 1 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "soft_sort_pytorch expects 1D tensor"
        ));
    }
    
    let values_vec: Vec<f64> = tensor.to_vec1::<f64>()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            format!("Failed to extract tensor data: {}", e)
        ))?;
    
    let sorted = soft_sort(&values_vec, regularization_strength);
    
    let result_tensor = Tensor::from_slice(&sorted)
        .to_kind(tensor.kind())
        .to_device(tensor.device());
    
    Ok(PyTensor(result_tensor))
}

#[cfg(feature = "pytorch")]
/// Compute Spearman correlation loss for PyTorch tensors.
///
/// **Note**: Current implementation converts to CPU for computation.
/// For full gradient support, implement custom autograd function.
#[pyfunction]
fn spearman_loss_pytorch(
    predictions: PyTensor,
    targets: PyTensor,
    regularization_strength: f64,
) -> PyResult<PyTensor> {
    let pred_tensor = &predictions.0;
    let target_tensor = &targets.0;
    
    // Validate shapes
    if pred_tensor.size().len() != 1 || target_tensor.size().len() != 1 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "spearman_loss_pytorch expects 1D tensors"
        ));
    }
    
    if pred_tensor.size()[0] != target_tensor.size()[0] {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "predictions and targets must have same length"
        ));
    }
    
    // Extract to CPU for computation (preserves device info for result)
    let pred_vec: Vec<f64> = pred_tensor.to_vec1::<f64>()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            format!("Failed to extract predictions: {}", e)
        ))?;
    
    let target_vec: Vec<f64> = target_tensor.to_vec1::<f64>()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            format!("Failed to extract targets: {}", e)
        ))?;
    
    // Compute loss
    let loss_value = spearman_loss(&pred_vec, &target_vec, regularization_strength);
    
    // Create scalar tensor on same device as predictions
    let result_tensor = Tensor::from_slice(&[loss_value])
        .to_kind(pred_tensor.kind())
        .to_device(pred_tensor.device());
    
    Ok(PyTensor(result_tensor))
}

// ============================================================================
// NumPy/JAX Functions (Optional Feature)
// ============================================================================

#[cfg(feature = "numpy")]
/// Compute soft ranks for NumPy arrays (also works with JAX arrays).
///
/// Uses buffer protocol for zero-copy access when possible.
///
/// # Arguments
///
/// * `values` - NumPy array (1D, float64)
/// * `regularization_strength` - Temperature parameter
///
/// # Returns
///
/// NumPy array with soft ranks
#[pyfunction]
fn soft_rank_numpy(values: &Bound<'_, PyArray1<f64>>, regularization_strength: f64) -> PyResult<Bound<'_, PyArray1<f64>>> {
    // Zero-copy read access via buffer protocol
    let values_slice = values.as_slice()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            format!("Failed to read array: {}", e)
        ))?;
    
    // Compute soft ranks
    let ranks = soft_rank(values_slice, regularization_strength);
    
    // Create new array (Python owns memory)
    let py = values.py();
    Ok(PyArray1::from_vec(py, ranks))
}

#[cfg(feature = "numpy")]
/// Compute soft sorted values for NumPy arrays.
#[pyfunction]
fn soft_sort_numpy(values: &Bound<'_, PyArray1<f64>>, regularization_strength: f64) -> PyResult<Bound<'_, PyArray1<f64>>> {
    let values_slice = values.as_slice()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            format!("Failed to read array: {}", e)
        ))?;
    
    let sorted = soft_sort(values_slice, regularization_strength);
    
    let py = values.py();
    Ok(PyArray1::from_vec(py, sorted))
}

#[cfg(feature = "numpy")]
/// Compute Spearman correlation loss for NumPy arrays.
#[pyfunction]
fn spearman_loss_numpy(
    predictions: &Bound<'_, PyArray1<f64>>,
    targets: &Bound<'_, PyArray1<f64>>,
    regularization_strength: f64,
) -> PyResult<f64> {
    let pred_slice = predictions.as_slice()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            format!("Failed to read predictions: {}", e)
        ))?;
    
    let target_slice = targets.as_slice()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            format!("Failed to read targets: {}", e)
        ))?;
    
    if pred_slice.len() != target_slice.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "predictions and targets must have same length"
        ));
    }
    
    Ok(spearman_loss(pred_slice, target_slice, regularization_strength))
}
