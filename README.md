# rank-relax

Differentiable sorting and ranking operations for Rust ML frameworks (candle/burn).

[![CI](https://github.com/arclabs561/rank-relax/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/rank-relax/actions)
[![Crates.io](https://img.shields.io/crates/v/rank-relax.svg)](https://crates.io/crates/rank-relax)
[![Docs](https://docs.rs/rank-relax/badge.svg)](https://docs.rs/rank-relax)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)

## Overview

`rank-relax` provides differentiable approximations of sorting and ranking operations, enabling gradient-based optimization of objectives that depend on ordering (e.g., Spearman correlation, NDCG, ranking quality) in Rust ML frameworks.

## Why Differentiable Ranking?

Traditional ranking operations (sorting, ranking) are **discrete** and **non-differentiable**, which prevents gradient-based optimization of ranking objectives.

**Problem**: You can't directly optimize Spearman correlation or NDCG during training because gradients can't flow through discrete ranking operations.

**Solution**: `rank-relax` provides **smooth relaxations** that:
- Preserve ranking semantics
- Enable gradient flow through ranking operations
- Converge to discrete behavior as regularization increases

## Purpose

**For training in Rust ML frameworks** (candle/burn):
- Spearman correlation loss during training
- Gradient flow through ranking operations
- Direct optimization of ranking metrics

**Not for inference** - Use `std::sort` or `rank-fast` for fast, non-differentiable ranking after inference.

## Quick Start

```bash
cargo add rank-relax
```

```rust
use rank_relax::{soft_rank, spearman_loss};

// Soft ranking (differentiable)
let values = vec![5.0, 1.0, 2.0, 4.0, 3.0];
let ranks = soft_rank(&values, 1.0);  // regularization_strength = 1.0

// Spearman correlation loss (for training)
let predictions = vec![0.1, 0.9, 0.3, 0.7, 0.5];
let targets = vec![0.0, 1.0, 0.2, 0.8, 0.4];
let loss = spearman_loss(&predictions, &targets, 1.0);
```

### Python

**Install from PyPI:**

```bash
pip install rank-relax
```

```python
import rank_relax

# Soft ranking
values = [5.0, 1.0, 2.0, 4.0, 3.0]
ranks = rank_relax.soft_rank(values, regularization_strength=1.0)

# Soft sorting
sorted_values = rank_relax.soft_sort(values, regularization_strength=1.0)

# Spearman correlation loss
predictions = [0.1, 0.9, 0.3, 0.7, 0.5]
targets = [0.0, 1.0, 0.2, 0.8, 0.4]
loss = rank_relax.spearman_loss(predictions, targets, regularization_strength=1.0)
```

**For development/contributing:**

```bash
cd rank-relax-python
uv venv
source .venv/bin/activate
uv tool install maturin
maturin develop --uv
```

## Features

- **Differentiable ranking**: Smooth relaxation of discrete ranking operations
- **Candle integration**: Works with candle tensors and autograd (🚧 in progress)
- **Burn integration**: Works with burn tensors and autograd (🚧 planned)
- **Zero dependencies** (core): Minimal dependencies for maximum compatibility
- **Spearman correlation loss**: Direct optimization of ranking quality

## Usage

### Basic Operations

```rust
use rank_relax::{soft_rank, soft_sort, spearman_loss};

// Soft ranking (continuous approximation of integer ranks)
let values = vec![5.0, 1.0, 2.0, 4.0, 3.0];
let ranks = soft_rank(&values, 1.0);
// With high regularization, ranks approach [4.0, 0.0, 1.0, 3.0, 2.0]

// Soft sorting (continuous approximation of sorted order)
let sorted = soft_sort(&values, 1.0);
// With high regularization, sorted approaches [1.0, 2.0, 3.0, 4.0, 5.0]

// Spearman correlation loss (for training)
let predictions = vec![0.1, 0.9, 0.3, 0.7, 0.5];
let targets = vec![0.0, 1.0, 0.2, 0.8, 0.4];
let loss = spearman_loss(&predictions, &targets, 1.0);
// Loss = 1 - Spearman correlation (lower is better)
```

### With Candle (🚧 Planned)

```rust
use candle_core::{Tensor, Device};
use rank_relax::spearman_loss;

// During training loop
let predictions: Tensor = model.forward(&inputs)?;  // [batch_size]
let targets: Tensor = batch.targets.clone();         // [batch_size]

// Compute Spearman correlation loss (differentiable!)
let loss = spearman_loss(&predictions, &targets, 1.0)?;

// Backprop - gradients flow through ranking operation
let grads = loss.backward()?;
optimizer.step(&grads)?;
```

**Note**: Candle integration requires the `candle` feature flag (when implemented):
```toml
[dependencies]
rank-relax = { version = "0.1", features = ["candle"] }
```

### With Burn (🚧 Planned)

```rust
use burn::tensor::{Tensor, Backend};
use rank_relax::spearman_loss;

// During training loop
let predictions = model.forward(batch.inputs.clone());  // [batch_size]
let targets = batch.targets.clone();                    // [batch_size]

// Compute Spearman correlation loss (differentiable!)
spearman_loss(&predictions, &targets, 1.0)
// Gradients automatically flow through burn's autograd
```

## Mathematical Background

This crate implements **smooth relaxations** of discrete sorting/ranking operations. The core concept is to replace non-differentiable discrete operations with continuous, differentiable approximations that:

1. Preserve ranking semantics
2. Enable gradient flow through the ranking operation
3. Converge to discrete behavior as regularization strength increases

**Soft Ranking**: Uses sigmoid-based comparisons to compute continuous rank approximations:
- Each element's rank = average of sigmoid comparisons with all other elements
- Higher `regularization_strength` → sharper sigmoid → closer to discrete ranks

**Spearman Correlation**: Computed on soft ranks using Pearson correlation formula, enabling differentiable optimization.

See [`CANDLE_BURN_INTEGRATION.md`](CANDLE_BURN_INTEGRATION.md) for detailed explanation of the mathematical framework.

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

## Getting Started

See **[GETTING_STARTED.md](GETTING_STARTED.md)** for a complete walkthrough with examples.

## API

### Core Functions

| Function | Description | Returns |
|----------|-------------|---------|
| `soft_rank(values, regularization_strength)` | Compute soft ranks (differentiable) | `Vec<f64>` |
| `soft_sort(values, regularization_strength)` | Compute soft sorted values | `Vec<f64>` |
| `spearman_loss(predictions, targets, regularization_strength)` | Spearman correlation loss for training | `f64` |

### Configuration

- `regularization_strength`: Temperature parameter (higher = sharper, more discrete-like)
  - Typical range: `0.1` to `10.0`
  - Lower values = smoother gradients
  - Higher values = closer to discrete ranking

## Examples

See the [examples directory](examples/) for complete usage examples (when available).

## Benchmarks

Performance benchmarks are available via `cargo bench`. See [benches/](benches/) for benchmark implementations.

## Documentation

- **[Getting Started Guide](GETTING_STARTED.md)** - Complete walkthrough
- **[Candle/Burn Integration](CANDLE_BURN_INTEGRATION.md)** - Framework integration details (when available)
- **[API Documentation](https://docs.rs/rank-relax)** - Full API reference (when published)

## See Also

- **[rank-fusion](https://crates.io/crates/rank-fusion)**: Combine ranked lists from multiple retrievers
- **[rank-refine](https://crates.io/crates/rank-refine)**: Score embeddings with MaxSim (ColBERT)
- **[rank-eval](https://crates.io/crates/rank-eval)**: IR evaluation metrics and TREC format parsing

## License

MIT OR Apache-2.0
