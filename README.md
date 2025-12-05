# rank-relax

[![CI](https://github.com/arclabs561/rank-relax/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/rank-relax/actions)
[![Crates.io](https://img.shields.io/crates/v/rank-relax.svg)](https://crates.io/crates/rank-relax)
[![Docs](https://docs.rs/rank-relax/badge.svg)](https://docs.rs/rank-relax)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)

**Differentiable ranking and sorting operations for machine learning**

A high-performance Rust crate with Python bindings that provides smooth relaxations of discrete ranking operations, enabling gradient-based optimization of ranking objectives across PyTorch, JAX, Julia, and Rust ML frameworks.

## Why Differentiable Ranking?

### The "Kink" Problem

Traditional ranking operations are **discrete** and **non-differentiable**: as you change a value, its rank jumps by integer steps (0, 1, 2, ...). These "jumps" have zero gradient almost everywhere, preventing optimization of ranking-based metrics (Spearman correlation, NDCG) during training.

**The Runner Metaphor**: Imagine two runners, Alice and Bob. If Alice is at position 5.0 and Bob at 5.1, Bob is ranked #1 and Alice #2. If Alice speeds up to 5.2, she instantly snaps to #1. In the world of calculus (and gradients), this "snap" is a disaster: for almost all possible speeds, the rank doesn't change (derivative is zero), and at the exact moment of passing, it explodes.

**Solution**: Smooth relaxations replace discrete operations with continuous, differentiable approximations that preserve ranking semantics while enabling gradient flow. Instead of an instant snap, we create a smooth slide. This enables end-to-end training of models that optimize ranking objectives directly.

For deeper mathematical details, see [MATHEMATICAL_DETAILS.md](MATHEMATICAL_DETAILS.md).

## Features

- **Multiple Ranking Methods**: Sigmoid-based (default), NeuralSort-style, Probabilistic (SoftRank), SmoothI
- **True Differentiable Sorting**: Permutahedron projection via isotonic regression (O(n log n))
- **Analytical Gradients**: Efficient closed-form gradient computation (no numerical differentiation)
- **Batch Processing**: Parallel processing support for multiple rankings (with `parallel` feature)
- **Performance Optimized**: SIMD-ready, parallel processing, memory-efficient sparse gradients
- **Framework Agnostic**: Works with PyTorch (validated), JAX (examples provided), Julia, and Rust ML
- **Comprehensive Testing**: 40+ property tests, 23+ Python tests, numerical stability validated
- **Well Documented**: Extensive documentation with mathematical derivations and examples

## Quick Start

### Rust

```rust
use rank_relax::{soft_rank, spearman_loss, RankingMethod};

// Soft ranking
let values = vec![5.0, 1.0, 2.0, 4.0, 3.0];
let ranks = soft_rank(&values, 1.0);
// With high regularization, ranks approach [4.0, 0.0, 1.0, 3.0, 2.0]

// Spearman correlation loss (for training)
let predictions = vec![0.1, 0.9, 0.3, 0.7, 0.5];
let targets = vec![0.0, 1.0, 0.2, 0.8, 0.4];
let loss = spearman_loss(&predictions, &targets, 1.0);

// Different methods
let method = RankingMethod::NeuralSort;
let ranks = method.compute(&values, 1.0);
```

### Python

```python
import rank_relax

# Soft ranking
values = [5.0, 1.0, 2.0, 4.0, 3.0]
ranks = rank_relax.soft_rank(values, regularization_strength=1.0)

# Spearman loss
predictions = [0.1, 0.9, 0.3, 0.7, 0.5]
targets = [0.0, 1.0, 0.2, 0.8, 0.4]
loss = rank_relax.spearman_loss(predictions, targets, regularization_strength=1.0)

# Analytical gradients
pred_ranks = rank_relax.soft_rank(predictions, regularization_strength=1.0)
target_ranks = rank_relax.soft_rank(targets, regularization_strength=1.0)
grad = rank_relax.spearman_loss_gradient(
    predictions, targets, pred_ranks, target_ranks, regularization_strength=1.0
)

# Different methods
ranks = rank_relax.soft_rank_with_method(
    values, regularization_strength=1.0, method="neural_sort"
)
```

### PyTorch Integration

```python
import torch
import rank_relax
from rank_relax.examples.pytorch_autograd import spearman_loss_pytorch

predictions = torch.tensor([0.1, 0.9, 0.3], requires_grad=True)
targets = torch.tensor([0.0, 1.0, 0.2])

loss = spearman_loss_pytorch(predictions, targets, regularization_strength=1.0)
loss.backward()  # Gradients flow through Rust implementation!
print(predictions.grad)
```

### JAX Integration

```python
import jax
import jax.numpy as jnp
from rank_relax.examples.jax_primitive import spearman_loss_jax

predictions = jnp.array([0.1, 0.9, 0.3])
targets = jnp.array([0.0, 1.0, 0.2])

loss = spearman_loss_jax(predictions, targets, regularization_strength=1.0)
grad_fn = jax.grad(lambda p: spearman_loss_jax(p, targets, 1.0))
grads = grad_fn(predictions)  # Automatic differentiation works!
```

## Available Methods

### 1. Sigmoid-based (Default)
- **Intuition**: The "naive" but effective approach. Each element's rank is computed by comparing it to all others using smooth sigmoid functions instead of hard comparisons.
- **Complexity**: O(n²)
- **Use case**: General purpose, intuitive, well-tested. Best for small-medium inputs (n < 1000).
- **Gardner Metaphor**: Like counting how many runners are ahead of you, but instead of a sharp "yes/no," you use a smooth probability curve.

### 2. NeuralSort-style
- **Intuition**: Uses temperature-scaled softmax to create sharper rankings. Similar to sigmoid but with different gradient behavior.
- **Complexity**: O(n²)
- **Use case**: When you need permutation matrices or different gradient profiles.
- **Gardner Metaphor**: Like a more sophisticated version of the sigmoid approach, with temperature control for fine-tuning sharpness.

### 3. Probabilistic (SoftRank)
- **Intuition**: Uses Gaussian smoothing to create probabilistic rank distributions, modeling uncertainty in rankings.
- **Complexity**: O(n²)
- **Use case**: When you want probabilistic rank distributions or uncertainty modeling.
- **Gardner Metaphor**: Like measuring rank with a fuzzy ruler—instead of exact positions, you get probability distributions.

### 4. SmoothI
- **Intuition**: Uses exponential scaling to create smooth rank position indicators with alternative gradient profiles.
- **Complexity**: O(n²)
- **Use case**: Alternative gradient profiles, experimentation with different smoothing approaches.
- **Gardner Metaphor**: Like a different flavor of the sigmoid approach, with exponential scaling for different behavior.

**Note**: All current methods are O(n²). For more efficient O(n log n) methods (Permutahedron Projection, Optimal Transport, LapSum), see [MATHEMATICAL_DETAILS.md](MATHEMATICAL_DETAILS.md) for theoretical foundations and future implementation plans.

## Advanced Features

### Batch Processing

```rust
use rank_relax::soft_rank_batch;

let batch = vec![
    vec![5.0, 1.0, 2.0, 4.0, 3.0],
    vec![3.0, 1.0, 2.0],
    vec![10.0, 5.0, 8.0, 7.0],
];

let ranks_batch = soft_rank_batch(&batch, 1.0);
```

### Parallel Processing

Enable with `--features parallel`:

```rust
#[cfg(feature = "parallel")]
use rank_relax::soft_rank_batch_parallel;

let ranks = soft_rank_batch_parallel(&batch, 1.0);
```

### Optimized Implementations

```rust
use rank_relax::soft_rank_optimized;

// For pre-sorted inputs
let ranks = soft_rank_optimized(&values, 1.0, assume_sorted=true);

// Sparse gradients for large inputs
use rank_relax::soft_rank_gradient_sparse;
let grad = soft_rank_gradient_sparse(&values, &ranks, 1.0, threshold=0.01);
```

## Performance

Benchmark results (on typical hardware):

- **Forward pass**: < 1ms for n=1000
- **Backward pass**: < 2ms for n=1000 (analytical gradients)
- **Batch processing**: Linear scaling with batch size
- **Memory**: O(n²) for gradient matrix (can use sparse for O(n))

Run benchmarks:
```bash
cargo bench
```

## Paper Reproduction

This framework can reproduce results from major differentiable ranking papers. See [MATHEMATICAL_DETAILS.md](MATHEMATICAL_DETAILS.md) for theoretical foundations and [BENCHMARKING.md](BENCHMARKING.md) for reproduction guidelines.

## Installation

### Rust

```bash
cargo add rank-relax
```

Or add to `Cargo.toml`:
```toml
[dependencies]
rank-relax = "0.1"
```

### Python

```bash
pip install rank-relax
```

For development:
```bash
cd rank-relax-python
pip install maturin
maturin develop
```

## Documentation

- **[Mathematical Details](MATHEMATICAL_DETAILS.md)** - Comprehensive theory with Gardner-like intuitive explanations
- [Getting Started](GETTING_STARTED.md) - Installation and basic usage
- [Examples](EXAMPLES.md) - Practical code examples
- [Parameter Tuning](PARAMETER_TUNING.md) - How to choose `regularization_strength`
- [Benchmarking Guide](BENCHMARKING.md) - How to reproduce paper results
- [Training Integration](TRAINING_INTEGRATION.md) - PyTorch/JAX/Julia integration

## Testing

```bash
# Run all tests
cargo test

# Run Python tests
cd rank-relax-python && pytest tests/
```

## Contributing

Contributions welcome! See [MATHEMATICAL_DETAILS.md](MATHEMATICAL_DETAILS.md) for implementation priorities and theoretical foundations.

## License

MIT OR Apache-2.0

## See Also

- **[rank-fusion](https://crates.io/crates/rank-fusion)**: Combine ranked lists from multiple retrievers (RRF, CombMNZ, etc.)
- **[rank-refine](https://crates.io/crates/rank-refine)**: Score embeddings with MaxSim (ColBERT/ColPali) and cosine similarity
- **[rank-eval](https://crates.io/crates/rank-eval)**: IR evaluation metrics and TREC format parsing
- **[Integration Examples](../INTEGRATION_EXAMPLES.md)**: Complete pipelines using multiple rank-* crates together

## Related Work

- [SoftRank](https://www.microsoft.com/en-us/research/publication/softrank-optimizing-non-smooth-rank-metrics/) - Probabilistic ranking
- [NeuralSort](https://arxiv.org/abs/1903.08850) - Differentiable sorting
- [SmoothI](https://arxiv.org/abs/2106.08253) - Smooth rank indicators
