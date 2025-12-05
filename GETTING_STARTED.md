# Getting Started with rank-relax

This guide walks you through using `rank-relax` for differentiable ranking operations in Rust ML frameworks.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Candle Integration](#candle-integration)
4. [Burn Integration](#burn-integration)
5. [Use Cases](#use-cases)

---

## Overview

`rank-relax` provides **differentiable approximations** of sorting and ranking operations, enabling gradient-based optimization of objectives that depend on ordering.

### Key Concepts

- **Differentiable**: Gradients flow through ranking operations
- **Smooth relaxation**: Continuous approximations of discrete operations
- **ML framework support**: Works with candle and burn tensors
- **Training use**: Designed for training, not inference

### When to Use

**Use rank-relax when**:
- Training models with ranking-based loss functions (Spearman, NDCG)
- Need gradients through ranking operations
- Optimizing ranking quality directly

**Don't use rank-relax when**:
- Inference (use `std::sort` or `rank-fast` for speed)
- Non-differentiable ranking is acceptable
- Performance is critical (relaxations are slower than discrete operations)

---

## Quick Start

### Installation

```bash
cargo add rank-relax
```

### Basic Example: Soft Ranking

```rust
use rank_relax::soft_rank;

let values = vec![5.0, 1.0, 2.0, 4.0, 3.0];
let ranks = soft_rank(&values, 1.0);  // regularization_strength = 1.0

// Ranks are continuous approximations of integer ranks
// With high regularization, ranks approach [4.0, 0.0, 1.0, 3.0, 2.0]
```

### Spearman Correlation Loss

```rust
use rank_relax::spearman_loss;

let predictions = vec![0.1, 0.9, 0.3, 0.7, 0.5];
let targets = vec![0.0, 1.0, 0.2, 0.8, 0.4];

// Loss = 1 - Spearman correlation (lower is better)
let loss = spearman_loss(&predictions, &targets, 1.0);
```

---

## Candle Integration

🚧 **Status**: Integration in progress

### Planned API

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

**Note**: Candle integration requires the `candle` feature flag:
```toml
[dependencies]
rank-relax = { version = "0.1", features = ["candle"] }
```

---

## Burn Integration

🚧 **Status**: Planned (not yet implemented)

### Planned API

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

**Note**: Burn integration requires the `burn` feature flag (when implemented):
```toml
[dependencies]
rank-relax = { version = "0.1", features = ["burn"] }
```

---

## Use Cases

### 1. Training Ranking Models

Optimize models to produce good rankings directly:

```rust
// Model outputs relevance scores
let predictions = model.forward(inputs);

// Ground truth rankings
let targets = batch.targets;

// Optimize Spearman correlation
let loss = spearman_loss(&predictions, &targets, 1.0);
loss.backward()?;
```

### 2. Learning-to-Rank

Train models that learn to rank items:

```rust
// Query-document pairs
let query_doc_scores = model.score_pairs(queries, documents);

// Optimize ranking quality
let loss = spearman_loss(&query_doc_scores, &ground_truth_ranks, 1.0);
```

### 3. Multi-Objective Training

Combine ranking loss with other objectives:

```rust
let ranking_loss = spearman_loss(&predictions, &targets, 1.0);
let classification_loss = cross_entropy(&predictions, &labels);
let total_loss = 0.7 * ranking_loss + 0.3 * classification_loss;
```

---

## Mathematical Background

`rank-relax` implements **smooth relaxations** of discrete operations:

1. **Soft ranking**: Continuous approximation of integer ranks using sigmoid-based comparisons
   - Formula: `rank[i] = (1/(n-1)) * Σ_{j≠i} sigmoid(α * (values[i] - values[j]))`
   - Each element's rank is computed by softly counting how many others it's greater than
   - Complexity: O(n²) - suitable for small-medium inputs

2. **Soft sorting**: Continuous approximation of sorted order
   - ⚠️ **Note**: Current implementation is a placeholder (uses hard sorting)
   - True differentiable soft sort coming in future versions

3. **Differentiable correlation**: Spearman correlation computed on soft ranks
   - Spearman = Pearson correlation of ranks
   - Loss = 1 - Spearman correlation (lower is better)
   - Gradients flow through the soft ranking operation

The `regularization_strength` parameter controls sharpness:
- **Low** (0.1-1.0): Smooth, more differentiable, good for early training
- **Medium** (1.0-10.0): Balanced between smoothness and accuracy
- **High** (10-100): Sharper, closer to discrete behavior, better for accurate ranking

**Rule of thumb**: `regularization_strength ≈ 1.0 / typical_difference_between_values`

See **[MATHEMATICAL_DETAILS.md](MATHEMATICAL_DETAILS.md)** for comprehensive theory and **[PARAMETER_TUNING.md](PARAMETER_TUNING.md)** for detailed tuning guidance.

---

## Status

🚧 **Early development** - Basic structure exists, candle/burn integration in progress

### Current Status

- ✅ Core operations (soft_rank, soft_sort, spearman_loss)
- ✅ Basic tests
- 🚧 Candle integration (planned)
- 🚧 Burn integration (planned)
- ❌ Python bindings (not yet planned)
- ❌ Performance benchmarks (not yet)

### Roadmap

1. **Phase 1**: Complete Candle integration
2. **Phase 2**: Complete Burn integration
3. **Phase 3**: Performance optimization
4. **Phase 4**: Python bindings (if needed)

---

## Next Steps

- **See [CANDLE_BURN_INTEGRATION.md](CANDLE_BURN_INTEGRATION.md)** for integration details
- **See [README.md](README.md)** for overview
- **See [src/](src/)** for implementation details

---

## Getting Help

- **GitHub Issues**: https://github.com/arclabs561/rank-relax/issues
- **Documentation**: https://docs.rs/rank-relax (when published)

