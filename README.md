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

**The Problem**: 
- As you change a value, its rank jumps by integer steps (0, 1, 2, ...)
- These "jumps" have zero gradient almost everywhere
- You can't directly optimize Spearman correlation or NDCG during training because gradients can't flow through discrete ranking operations

**The Solution**: `rank-relax` provides **smooth relaxations** that:
- Preserve ranking semantics (maintain relative ordering)
- Enable gradient flow through ranking operations
- Converge to discrete behavior as regularization increases

**Simple Analogy**: Think of discrete ranking as a staircase (non-differentiable at steps). Soft ranking is like a smooth ramp (differentiable everywhere) that approximates the staircase.

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
- Formula: `rank[i] = (1/(n-1)) * Σ_{j≠i} sigmoid(α * (values[i] - values[j]))`
- Higher `regularization_strength` (α) → sharper sigmoid → closer to discrete ranks
- **Complexity**: O(n²) - suitable for small-medium inputs (< 1000 elements)

**Spearman Correlation**: Computed on soft ranks using Pearson correlation formula, enabling differentiable optimization:
- Spearman = Pearson correlation of ranks
- Loss = 1 - Spearman correlation (lower is better)
- Gradients flow through the soft ranking operation

For comprehensive mathematical formulations, derivations, and theoretical foundations, see **[MATHEMATICAL_DETAILS.md](MATHEMATICAL_DETAILS.md)**. This document covers:
- Optimal transport formulation (Sinkhorn algorithm, entropic regularization)
- Permutahedron projection (isotonic regression, Fenchel-Young losses)
- Sorting networks (sigmoid relaxations, monotonicity guarantees)
- NeuralSort/SoftSort (softmax-based relaxations)
- LapSum method (Laplace distribution approach)
- Gradient computation and automatic differentiation

For insights on pedagogical approaches and areas where explanations could be improved, see **[PEDAGOGICAL_IMPROVEMENTS.md](PEDAGOGICAL_IMPROVEMENTS.md)**. This document analyzes educational materials (UCSD CSE 291, MIT courses) and identifies intuitive explanations, visualizations, and step-by-step derivations that enhance understanding.

See [`CANDLE_BURN_INTEGRATION.md`](CANDLE_BURN_INTEGRATION.md) for framework-specific integration details.

## Status

🚧 **Early development** - Basic structure exists, candle/burn integration in progress

### Current Status

- ✅ Core operations (soft_rank, soft_sort, spearman_loss)
- ✅ Basic tests
- ✅ Python bindings (available via PyPI)
- 🚧 Candle integration (planned)
- 🚧 Burn integration (planned)
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

- `regularization_strength`: Temperature parameter controlling sharpness
  - **Typical range**: `0.1` to `100.0`
  - **Rule of thumb**: `≈ 1.0 / typical_difference_between_values`
  - **Lower values (0.1-1.0)**: Smoother gradients, good for early training
  - **Medium values (1.0-10.0)**: Balanced between smoothness and accuracy
  - **Higher values (10.0-100.0)**: Sharper, closer to discrete ranking
  
**Example**: If your values differ by ~1.0, use `regularization_strength ≈ 1.0`.
If differences are ~0.1, use `≈ 10.0`.

See **[PARAMETER_TUNING.md](PARAMETER_TUNING.md)** for detailed guidance.

## Examples

See the [examples directory](examples/) for complete usage examples (when available).

## Benchmarks

Performance benchmarks are available via `cargo bench`. See [benches/](benches/) for benchmark implementations.

## Documentation

- **[Getting Started Guide](GETTING_STARTED.md)** - Complete walkthrough
- **[Parameter Tuning Guide](PARAMETER_TUNING.md)** - How to choose `regularization_strength`
- **[Mathematical Details](MATHEMATICAL_DETAILS.md)** - Comprehensive theory and derivations
- **[Pedagogical Improvements](PEDAGOGICAL_IMPROVEMENTS.md)** - Educational insights and explanations
- **[Candle/Burn Integration](CANDLE_BURN_INTEGRATION.md)** - Framework integration details (when available)
- **[API Documentation](https://docs.rs/rank-relax)** - Full API reference (when published)

## Related Work

Differentiable ranking/sorting implementations and research:

### Implementations

- **[difftopk](https://github.com/Felix-Petersen/difftopk)** (Python/PyTorch): Top-k classification learning with TopKCrossEntropyLoss. Uses sorting networks (bitonic, odd_even). ICML 2022.
- **[diffsort](https://github.com/Felix-Petersen/diffsort)** (Python/PyTorch): Differentiable sorting networks with relaxed comparators. Implements NeuralSort, SoftSort. ICML 2021.
- **[torchsort](https://github.com/teddykoker/torchsort)** (Python/PyTorch): Fast differentiable sorting/ranking with CUDA kernels. 846 stars.
- **[fast-soft-sort](https://research.google/pubs/fast-differentiable-sorting-and-ranking/)** (Google Research): O(n log n) sorting via permutahedron projections. ICML 2020.
- **[softsort.pytorch](https://github.com/moskomule/softsort.pytorch)**: Optimal transport-based sorting (Sinkhorn). Cuturi et al. (2019).

### Research Papers

**Foundational**: NeuralSort (Grover et al., ICML 2019), Optimal Transport Sorting (Cuturi et al., ICML 2019), Fast Differentiable Sorting (Blondel et al., ICML 2020), SoftSort (Prillo & Eisenschlos, 2020).

**Sorting Networks**: Differentiable Sorting Networks (Petersen et al., ICML 2021), Monotonic Sorting Networks (Petersen et al., 2022), Generalized Neural Sorting Networks (Kim et al., 2023).

**Ranking-Specific**: NeuralNDCG (Pobrotyn & Białobrzeski, 2021), SortNet (Rigutini et al., 2023), LapSum (Struski et al., 2025).

**Underlying Theory**: Permutahedron projections, optimal transport (Sinkhorn algorithm), isotonic regression, sorting networks (bitonic, odd-even merge).

See **[RELATED_WORK.md](RELATED_WORK.md)** for comprehensive survey of implementations, papers, and theory.

**Differences**: `rank-relax` targets Rust ML frameworks (candle/burn), provides Spearman correlation loss, and emphasizes ranking operations over full sorting. Unlike difftopk's top-k focus or diffsort's full permutations, `rank-relax` prioritizes ranking semantics (order preservation) for gradient-based optimization of ranking metrics.

## See Also

- **[rank-fusion](https://crates.io/crates/rank-fusion)**: Combine ranked lists from multiple retrievers
- **[rank-refine](https://crates.io/crates/rank-refine)**: Score embeddings with MaxSim (ColBERT)
- **[rank-eval](https://crates.io/crates/rank-eval)**: IR evaluation metrics and TREC format parsing

## License

MIT OR Apache-2.0
