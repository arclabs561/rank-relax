# Candle/Burn Integration for rank-relax

## Purpose

`rank-relax` provides differentiable ranking operations for training in Rust ML frameworks (candle/burn), enabling loss functions like Spearman correlation that require gradients through ranking operations.

**Why this matters**: Traditional ranking operations are non-differentiable, preventing direct optimization of ranking metrics (Spearman, NDCG) during training. `rank-relax` solves this by providing smooth, differentiable approximations.

## Current Status

🚧 **Early development** - Basic structure exists, needs candle/burn tensor integration

**Current implementation**: Works with `Vec<f64>` (CPU-only, no autograd).  
**Planned**: Tensor integration with automatic differentiation support.

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
   - Convert between `Vec<f64>` and tensor types
   - Handle different backends (CPU, CUDA, Metal)
   - Support batch operations (rank multiple vectors)

2. **Autograd Compatibility**: Ensure operations are differentiable
   - Operations must be part of the computation graph
   - Gradients must flow through ranking operations
   - Test gradient correctness

3. **GPU Support**: Work with CUDA tensors where applicable
   - Current O(n²) algorithm may need optimization for GPU
   - Consider implementing permutahedron projection (O(n log n)) for better GPU performance

4. **Performance**: Optimize for training workloads
   - Current implementation: O(n²) per forward/backward pass
   - For large inputs (n > 1000), consider more efficient methods
   - Benchmark against Python implementations (torchsort, fast-soft-sort)

## Technical Challenges

### Challenge 1: Tensor Type Abstraction

**Problem**: Candle and Burn have different tensor types and APIs.

**Solution**: Use trait-based abstraction or separate implementations per framework.

### Challenge 2: Gradient Computation

**Problem**: Need to ensure gradients flow correctly through sigmoid-based ranking.

**Solution**: 
- Use framework's autograd (automatic differentiation)
- Verify gradients numerically (finite differences)
- Test with simple examples first

### Challenge 3: Performance

**Problem**: O(n²) complexity may be slow for large inputs.

**Solution**:
- Start with current implementation (proven to work)
- Later: implement permutahedron projection (O(n log n))
- Profile and optimize hot paths

## Next Steps

1. Add candle feature flag and tensor support
2. Add burn feature flag and tensor support
3. Implement differentiable operations using framework's autograd
4. Test with actual training loops
5. Benchmark performance vs Python implementations

