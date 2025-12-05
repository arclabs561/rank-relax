//! PyTorch autograd functions with analytical gradients.
//!
//! This module provides proper PyTorch autograd functions that preserve gradients
//! and use analytical gradient computation for efficiency.

use pyo3::prelude::*;
use pyo3_tch::PyTensor;
use tch::{Tensor, Kind};

use crate::rank_relax::{
    soft_rank, spearman_loss,
    soft_rank_gradient, spearman_loss_gradient,
    RankingMethod,
};

/// PyTorch autograd function for soft_rank with analytical gradients.
#[pyclass]
pub struct SoftRankFunction;

#[pymethods]
impl SoftRankFunction {
    #[staticmethod]
    fn forward(
        predictions: PyTensor,
        regularization_strength: f64,
        method: Option<String>,
    ) -> PyResult<(PyTensor, PyTensor, f64, String)> {
        // Extract tensor data
        let tensor = &predictions.0;
        
        if tensor.size().len() != 1 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "soft_rank_pytorch expects 1D tensor"
            ));
        }
        
        let values: Vec<f64> = tensor.to_vec1::<f64>()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Failed to extract tensor: {}", e)
            ))?;
        
        // Select method
        let method_enum = match method.as_deref() {
            Some("neural_sort") => RankingMethod::NeuralSort,
            Some("probabilistic") => RankingMethod::Probabilistic,
            Some("smooth_i") => RankingMethod::SmoothI,
            _ => RankingMethod::Sigmoid,
        };
        
        // Compute ranks
        let ranks = method_enum.compute(&values, regularization_strength);
        
        // Create output tensor
        let ranks_tensor = Tensor::from_slice(&ranks)
            .to_kind(tensor.kind())
            .to_device(tensor.device());
        
        // Save for backward: predictions, ranks, regularization_strength, method
        Ok((
            PyTensor(predictions.0.shallow_clone()),
            PyTensor(ranks_tensor),
            regularization_strength,
            method_enum.name().to_string(),
        ))
    }
    
    #[staticmethod]
    fn backward(
        saved_predictions: PyTensor,
        saved_ranks: PyTensor,
        regularization_strength: f64,
        _method: String,
        grad_output: PyTensor,
    ) -> PyResult<PyTensor> {
        // Extract saved values
        let pred_tensor = &saved_predictions.0;
        let ranks_tensor = &saved_ranks.0;
        let grad_tensor = &grad_output.0;
        
        let predictions: Vec<f64> = pred_tensor.to_vec1::<f64>()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Failed to extract predictions: {}", e)
            ))?;
        
        let ranks: Vec<f64> = ranks_tensor.to_vec1::<f64>()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Failed to extract ranks: {}", e)
            ))?;
        
        let grad_output_vec: Vec<f64> = grad_tensor.to_vec1::<f64>()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Failed to extract grad_output: {}", e)
            ))?;
        
        // Compute analytical gradient
        let rank_grad = soft_rank_gradient(&predictions, &ranks, regularization_strength);
        
        // Apply chain rule: grad_input = grad_output * grad_rank
        let n = predictions.len();
        let mut grad_input = vec![0.0; n];
        
        for i in 0..n {
            for j in 0..n {
                grad_input[j] += grad_output_vec[i] * rank_grad[i][j];
            }
        }
        
        // Create gradient tensor
        let grad_tensor = Tensor::from_slice(&grad_input)
            .to_kind(pred_tensor.kind())
            .to_device(pred_tensor.device());
        
        Ok(PyTensor(grad_tensor))
    }
}

/// PyTorch autograd function for spearman_loss with analytical gradients.
#[pyclass]
pub struct SpearmanLossFunction;

#[pymethods]
impl SpearmanLossFunction {
    #[staticmethod]
    fn forward(
        predictions: PyTensor,
        targets: PyTensor,
        regularization_strength: f64,
        method: Option<String>,
    ) -> PyResult<(PyTensor, PyTensor, PyTensor, f64, String)> {
        // Extract tensors
        let pred_tensor = &predictions.0;
        let target_tensor = &targets.0;
        
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
        
        let pred_vec: Vec<f64> = pred_tensor.to_vec1::<f64>()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Failed to extract predictions: {}", e)
            ))?;
        
        let target_vec: Vec<f64> = target_tensor.to_vec1::<f64>()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Failed to extract targets: {}", e)
            ))?;
        
        // Select method
        let method_enum = match method.as_deref() {
            Some("neural_sort") => RankingMethod::NeuralSort,
            Some("probabilistic") => RankingMethod::Probabilistic,
            Some("smooth_i") => RankingMethod::SmoothI,
            _ => RankingMethod::Sigmoid,
        };
        
        // Compute ranks
        let pred_ranks = method_enum.compute(&pred_vec, regularization_strength);
        let target_ranks = method_enum.compute(&target_vec, regularization_strength);
        
        // Compute loss
        let loss_value = spearman_loss(&pred_vec, &target_vec, regularization_strength);
        
        // Create output tensors
        let loss_tensor = Tensor::from_slice(&[loss_value])
            .to_kind(pred_tensor.kind())
            .to_device(pred_tensor.device());
        
        let pred_ranks_tensor = Tensor::from_slice(&pred_ranks)
            .to_kind(pred_tensor.kind())
            .to_device(pred_tensor.device());
        
        let target_ranks_tensor = Tensor::from_slice(&target_ranks)
            .to_kind(target_tensor.kind())
            .to_device(target_tensor.device());
        
        Ok((
            PyTensor(predictions.0.shallow_clone()),
            PyTensor(targets.0.shallow_clone()),
            PyTensor(loss_tensor),
            regularization_strength,
            method_enum.name().to_string(),
        ))
    }
    
    #[staticmethod]
    fn backward(
        saved_predictions: PyTensor,
        saved_targets: PyTensor,
        _saved_loss: PyTensor,
        regularization_strength: f64,
        _method: String,
        grad_output: PyTensor,
    ) -> PyResult<(PyTensor, PyTensor)> {
        // Extract saved values
        let pred_tensor = &saved_predictions.0;
        let target_tensor = &saved_targets.0;
        let grad_tensor = &grad_output.0;
        
        let predictions: Vec<f64> = pred_tensor.to_vec1::<f64>()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Failed to extract predictions: {}", e)
            ))?;
        
        let targets: Vec<f64> = target_tensor.to_vec1::<f64>()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Failed to extract targets: {}", e)
            ))?;
        
        let grad_output_val: f64 = grad_tensor.to_scalar::<f64>()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Failed to extract grad_output: {}", e)
            ))?;
        
        // Recompute ranks for gradient computation
        let pred_ranks = soft_rank(&predictions, regularization_strength);
        let target_ranks = soft_rank(&targets, regularization_strength);
        
        // Compute analytical gradient
        let grad = spearman_loss_gradient(
            &predictions,
            &targets,
            &pred_ranks,
            &target_ranks,
            regularization_strength,
        );
        
        // Scale by grad_output
        let grad_scaled: Vec<f64> = grad.iter().map(|&g| g * grad_output_val).collect();
        
        // Create gradient tensor
        let grad_tensor = Tensor::from_slice(&grad_scaled)
            .to_kind(pred_tensor.kind())
            .to_device(pred_tensor.device());
        
        // No gradient for targets
        let zero_tensor = Tensor::zeros(
            &[targets.len() as i64],
            (pred_tensor.kind(), pred_tensor.device())
        );
        
        Ok((PyTensor(grad_tensor), PyTensor(zero_tensor)))
    }
}

