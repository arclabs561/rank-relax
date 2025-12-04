# Candle/Burn Integration for rank-relax

## Purpose

`rank-relax` provides differentiable ranking operations for training in Rust ML frameworks (candle/burn), enabling loss functions like Spearman correlation that require gradients through ranking operations.

## Current Status

🚧 **Early development** - Basic structure exists, needs candle/burn tensor integration

## Integration Plan

### For Candle

```rust
use candle_core::{Tensor, Device};
use rank_relax::spearman_loss;

// During training loop
fn training_step(
    model: &Model,
    batch: &Batch,
) -> Result<f32> {
    let predictions: Tensor = model.forward(&batch.inputs)?;  // [batch_size]
    let targets: Tensor = batch.targets.clone();              // [batch_size]
    
    // Compute Spearman correlation loss (differentiable!)
    let loss = spearman_loss(&predictions, &targets, 1.0)?;
    
    // Backprop - gradients flow through ranking operation
    let grads = loss.backward()?;
    optimizer.step(&grads)?;
    
    Ok(loss.to_scalar::<f32>()?)
}
```

### For Burn

```rust
use burn::tensor::{Tensor, Backend};
use rank_relax::spearman_loss;

// During training loop
fn training_step<B: Backend>(
    model: &Model<B>,
    batch: &Batch<B>,
) -> Tensor<B, 1> {
    let predictions = model.forward(batch.inputs.clone());  // [batch_size]
    let targets = batch.targets.clone();                    // [batch_size]
    
    // Compute Spearman correlation loss (differentiable!)
    spearman_loss(&predictions, &targets, 1.0)
    // Gradients automatically flow through burn's autograd
}
```

## Implementation Requirements

1. **Tensor Integration**: Support candle/burn tensor types
2. **Autograd Compatibility**: Ensure operations are differentiable
3. **GPU Support**: Work with CUDA tensors where applicable
4. **Performance**: Optimize for training workloads

## Next Steps

1. Add candle feature flag and tensor support
2. Add burn feature flag and tensor support
3. Implement differentiable operations using framework's autograd
4. Test with actual training loops
5. Benchmark performance vs Python implementations

