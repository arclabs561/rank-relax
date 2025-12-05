# PyTorch/JAX Integration for rank-relax Python Bindings

## Current State

**rank-relax Python bindings currently do NOT directly support PyTorch or JAX tensors.**

### How It Works Now

The current implementation uses **PyO3** to:
1. Accept Python lists of floats (`list[float]`)
2. Convert to Rust `Vec<f64>`
3. Run Rust computation
4. Return Python lists

```python
# Current usage (breaks gradient flow!)
import torch
import rank_relax

predictions = torch.tensor([0.1, 0.9, 0.3, 0.7, 0.5], requires_grad=True)
targets = torch.tensor([0.0, 1.0, 0.2, 0.8, 0.4])

# Must convert to list (loses gradient tracking)
loss = rank_relax.spearman_loss(
    predictions.tolist(),  # ❌ Breaks autograd
    targets.tolist(),
    regularization_strength=1.0
)

# Convert back to tensor (no gradients!)
loss_tensor = torch.tensor(loss)  # ❌ No gradient connection
```

**Problem**: Converting tensors to lists breaks PyTorch/JAX autograd, so gradients cannot flow through the ranking operation.

## How to Add PyTorch/JAX Support

### Option 1: PyTorch via `pyo3-tch` (Recommended)

Use the `pyo3-tch` crate to accept PyTorch tensors directly:

```rust
// In rank-relax-python/Cargo.toml
[dependencies]
pyo3 = { workspace = true }
pyo3-tch = "0.3"  // Add this
tch = "0.15"      // Add this
```

```rust
// In rank-relax-python/src/lib.rs
use pyo3_tch::PyTensor;
use tch::Tensor;

#[pyfunction]
fn spearman_loss_pytorch(
    predictions: PyTensor,
    targets: PyTensor,
    regularization_strength: f64,
) -> PyResult<PyTensor> {
    // Extract Rust tensors
    let pred_tensor = predictions.0;
    let target_tensor = targets.0;
    
    // Convert to Vec<f64> for rank-relax (or implement tensor-native version)
    let pred_vec: Vec<f64> = pred_tensor.to_vec1::<f64>()?;
    let target_vec: Vec<f64> = target_tensor.to_vec1::<f64>()?;
    
    // Compute loss
    let loss = spearman_loss(&pred_vec, &target_vec, regularization_strength);
    
    // Return as tensor (preserves gradient tracking if input had requires_grad=True)
    Ok(PyTensor(Tensor::from_slice(&[loss])))
}
```

**Limitation**: This still breaks gradient flow because we convert to `Vec<f64>`. To preserve gradients, you need to implement a **tensor-native version** that operates directly on `tch::Tensor` objects.

### Option 2: Custom Autograd Function (Full Gradient Support)

Implement a PyTorch autograd function that preserves gradients:

```rust
use pyo3::prelude::*;
use pyo3_tch::PyTensor;
use tch::{Tensor, Kind};

#[pyfunction]
fn spearman_loss_autograd(
    predictions: PyTensor,
    targets: PyTensor,
    regularization_strength: f64,
) -> PyResult<PyTensor> {
    // This would need to be implemented as a custom autograd function
    // that computes both forward and backward passes
    
    // Forward pass: compute soft ranks and Spearman correlation
    // Backward pass: compute gradients through soft ranking
    
    // For now, this is a placeholder - full implementation requires
    // implementing the gradient computation for soft ranking
    todo!("Implement tensor-native spearman_loss with autograd")
}
```

Then register it as a custom autograd function in Python:

```python
import torch
from torch.autograd import Function

class SpearmanLossFunction(Function):
    @staticmethod
    def forward(ctx, predictions, targets, regularization_strength):
        # Call Rust implementation
        loss = rank_relax.spearman_loss_autograd(
            predictions, targets, regularization_strength
        )
        ctx.save_for_backward(predictions, targets)
        ctx.regularization_strength = regularization_strength
        return loss
    
    @staticmethod
    def backward(ctx, grad_output):
        predictions, targets = ctx.saved_tensors
        # Compute gradients through soft ranking
        # This requires implementing the gradient formula
        grad_pred = rank_relax.spearman_loss_backward(
            predictions, targets, grad_output, ctx.regularization_strength
        )
        return grad_pred, None, None
```

### Option 3: JAX via `jax.numpy` Array Protocol

JAX arrays can be converted via the buffer protocol:

```rust
use pyo3::prelude::*;
use numpy::PyArray1;

#[pyfunction]
fn spearman_loss_jax(
    predictions: &PyArray1<f64>,
    targets: &PyArray1<f64>,
    regularization_strength: f64,
) -> PyResult<f64> {
    // Extract as Rust slice
    let pred_vec = predictions.as_slice()?;
    let target_vec = targets.as_slice()?;
    
    // Compute loss
    Ok(spearman_loss(pred_vec, target_vec, regularization_strength))
}
```

**Note**: This still breaks JAX's `jax.grad` because we're converting to Rust types. For full JAX support, you'd need to implement a JAX primitive.

## Implementation Requirements

### For Full PyTorch Support

1. **Add `pyo3-tch` dependency**
2. **Implement tensor-native operations** (don't convert to `Vec<f64>`)
3. **Implement backward pass** for gradient computation
4. **Register as autograd function** in Python

### For Full JAX Support

1. **Implement JAX primitive** using `jax.extend`
2. **Define `jvp` (forward-mode AD)** and `transpose` (reverse-mode AD) rules
3. **Handle device placement** (CPU/GPU/TPU)

## Current Workaround

Until tensor support is added, users can work around the limitation:

```python
import torch
import rank_relax

def spearman_loss_pytorch_wrapper(predictions, targets, regularization_strength=1.0):
    """Wrapper that preserves gradients (but breaks at rank-relax boundary)."""
    # Detach for computation (loses gradients)
    loss_value = rank_relax.spearman_loss(
        predictions.detach().tolist(),
        targets.detach().tolist(),
        regularization_strength
    )
    
    # Re-attach as leaf tensor (no gradient connection to inputs)
    return torch.tensor(loss_value, requires_grad=True)

# This won't work for training - gradients don't flow through!
predictions = torch.tensor([0.1, 0.9, 0.3], requires_grad=True)
loss = spearman_loss_pytorch_wrapper(predictions, targets)
loss.backward()  # ❌ predictions.grad will be None
```

## Recommendation

**Priority**: High - Adding PyTorch tensor support would make rank-relax much more useful for Python ML workflows.

**Approach**: 
1. Start with `pyo3-tch` integration (Option 1)
2. Implement tensor-native operations (avoid `Vec<f64>` conversion)
3. Add custom autograd function (Option 2) for full gradient support
4. Consider JAX support later (lower priority, more complex)

**Estimated Effort**: 
- Basic tensor support: 1-2 days
- Full autograd support: 1-2 weeks (requires implementing gradient formulas)

## Related Work

- **torchsort**: Python/PyTorch library with differentiable sorting (good reference)
- **softsort.pytorch**: Optimal transport-based sorting with PyTorch support
- **diffsort**: Differentiable sorting networks for PyTorch

These libraries show how to properly integrate differentiable ranking with PyTorch's autograd system.

