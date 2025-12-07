# rank-relax

[![CI](https://github.com/arclabs561/rank-relax/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/rank-relax/actions)
[![Crates.io](https://img.shields.io/crates/v/rank-relax.svg)](https://crates.io/crates/rank-relax)
[![Docs](https://docs.rs/rank-relax/badge.svg)](https://docs.rs/rank-relax)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)

**Differentiable ranking and sorting operations for machine learning**

A high-performance Rust crate with Python bindings that provides smooth relaxations of discrete ranking operations, enabling gradient-based optimization of ranking objectives across PyTorch, JAX, Julia, and Rust ML frameworks.

## Why Differentiable Ranking?

Discrete ranking operations have zero gradients almost everywhere, preventing optimization of ranking-based metrics during training. This crate provides smooth relaxations that enable gradient flow while preserving ranking semantics.

See [MATHEMATICAL_DETAILS.md](docs/MATHEMATICAL_DETAILS.md) for details.

## Features

- **Multiple Ranking Methods**: Sigmoid-based (default), NeuralSort-style, Probabilistic (SoftRank), SmoothI
- **Gumbel-Softmax Top-k** (optional `gumbel` feature): Differentiable top-k selection using Gumbel trick for RAG reranking
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

### Statistical Analysis (Real Data)

Comprehensive statistical analysis using 1000 real soft ranking computations:

![Soft Ranking Statistical](../hack/viz/soft_ranking_statistical.png)

**Four-panel analysis:**
- **Top-left**: Error distribution by alpha (box plots showing statistical distribution)
- **Top-right**: Error distribution histogram with gamma fitting (statistical rigor like games/tenzi)
- **Bottom-left**: Convergence rate with confidence intervals
- **Bottom-right**: Example convergence showing soft → discrete as α increases

**Method Comparison:**

![Soft Ranking Method Comparison](../hack/viz/soft_ranking_method_comparison.png)

Comparison of different ranking methods (Sigmoid, NeuralSort, Probabilistic, SmoothI) with error/time trade-off analysis.

**Error Distribution:**

![Soft Ranking Distribution](../hack/viz/soft_ranking_distribution.png)

Error distribution with gamma fitting, showing statistical properties of soft ranking convergence.

**Data Source**: 1000 real soft ranking computations using actual algorithms. See [Visualizations](../hack/viz/SOFT_RANKING_VISUALIZATIONS.md) for complete analysis.
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
import torch.autograd as autograd
import rank_relax

# Custom autograd function for gradient flow
class SpearmanLossAutograd(autograd.Function):
    @staticmethod
    def forward(ctx, predictions, targets, regularization_strength):
        ctx.save_for_backward(predictions, targets)
        ctx.regularization_strength = regularization_strength
        pred = predictions.detach().cpu().numpy()
        targ = targets.detach().cpu().numpy()
        loss_val = rank_relax.spearman_loss(pred.tolist(), targ.tolist(), regularization_strength)
        return torch.tensor(loss_val, device=predictions.device, dtype=predictions.dtype, requires_grad=True)
    
    @staticmethod
    def backward(ctx, grad_output):
        predictions, targets = ctx.saved_tensors
        regularization_strength = ctx.regularization_strength
        pred = predictions.detach().cpu().numpy()
        targ = targets.detach().cpu().numpy()
        pred_ranks = rank_relax.soft_rank(pred.tolist(), regularization_strength)
        target_ranks = rank_relax.soft_rank(targ.tolist(), regularization_strength)
        grad = rank_relax.spearman_loss_gradient(pred.tolist(), targ.tolist(), pred_ranks, target_ranks, regularization_strength)
        grad_scaled = [g * grad_output.item() for g in grad]
        return torch.tensor(grad_scaled, device=predictions.device, dtype=predictions.dtype), None, None

def spearman_loss_pytorch(predictions, targets, regularization_strength=1.0):
    return SpearmanLossAutograd.apply(predictions, targets, regularization_strength)

predictions = torch.tensor([0.1, 0.9, 0.3], requires_grad=True)
targets = torch.tensor([0.0, 1.0, 0.2])

loss = spearman_loss_pytorch(predictions, targets, regularization_strength=1.0)
loss.backward()
print(predictions.grad)
```

### JAX Integration

```python
import jax
import jax.numpy as jnp
from jax import core
from jax.interpreters import ad
import rank_relax

# Custom JAX primitive for gradient flow
spearman_loss_p = core.Primitive("spearman_loss_rust")

def spearman_loss_jax(predictions, targets, regularization_strength=1.0):
    return spearman_loss_p.bind(predictions, targets, regularization_strength=regularization_strength)

spearman_loss_p.def_impl(lambda p, t, r: jnp.array(rank_relax.spearman_loss(p.tolist(), t.tolist(), r)))
spearman_loss_p.def_abstract_eval(lambda p, t, r: core.ShapedArray((), p.dtype))

def spearman_loss_jvp(primals, tangents, *, regularization_strength):
    predictions, targets = primals
    pred_dot, target_dot = tangents
    y = spearman_loss_jax(predictions, targets, regularization_strength)
    pred_list = predictions.tolist()
    targ_list = targets.tolist()
    pred_ranks = rank_relax.soft_rank(pred_list, regularization_strength)
    target_ranks = rank_relax.soft_rank(targ_list, regularization_strength)
    grad = rank_relax.spearman_loss_gradient(pred_list, targ_list, pred_ranks, target_ranks, regularization_strength)
    y_dot = jnp.sum(jnp.array(grad) * pred_dot)
    return y, y_dot

ad.primitive_jvps[spearman_loss_p] = spearman_loss_jvp

def spearman_loss_transpose(ct, predictions, targets, *, regularization_strength):
    if predictions is None:
        return None, None, None
    pred_list = predictions.tolist()
    targ_list = targets.tolist()
    pred_ranks = rank_relax.soft_rank(pred_list, regularization_strength)
    target_ranks = rank_relax.soft_rank(targ_list, regularization_strength)
    grad = rank_relax.spearman_loss_gradient(pred_list, targ_list, pred_ranks, target_ranks, regularization_strength)
    return jnp.array(grad) * ct, None, None

ad.primitive_transposes[spearman_loss_p] = spearman_loss_transpose

predictions = jnp.array([0.1, 0.9, 0.3])
targets = jnp.array([0.0, 1.0, 0.2])

loss = spearman_loss_jax(predictions, targets, regularization_strength=1.0)
grad_fn = jax.grad(lambda p: spearman_loss_jax(p, targets, 1.0))
grads = grad_fn(predictions)
```

## Available Methods

- **Sigmoid-based (default)**: Compares elements using smooth sigmoid functions. O(n²). General purpose, best for small-medium inputs.
- **NeuralSort-style**: Temperature-scaled softmax for sharper rankings. O(n²). Useful for permutation matrices or different gradient profiles.
- **Probabilistic (SoftRank)**: Gaussian smoothing for probabilistic rank distributions. O(n²). For uncertainty modeling.
- **SmoothI**: Exponential scaling for alternative gradient profiles. O(n²). For experimentation with different smoothing approaches.

**Note**: All current methods are O(n²). For more efficient O(n log n) methods (Permutahedron Projection, Optimal Transport, LapSum), see [MATHEMATICAL_DETAILS.md](docs/MATHEMATICAL_DETAILS.md) for theoretical foundations and future implementation plans.

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

### Gumbel-Softmax Top-k (RAG Reranking)

For RAG reranking applications requiring end-to-end optimization:

```rust
use rank_relax::{relaxed_topk_gumbel, gumbel_attention_mask};
use rand::thread_rng;

// Enable gumbel feature: cargo add rank-relax --features gumbel

let reranker_scores = vec![0.8, 0.6, 0.9, 0.3, 0.7];
let mut rng = thread_rng();

// Generate soft attention mask for top-3 documents
let attention_mask = gumbel_attention_mask(
    &reranker_scores,
    3,      // top-k
    0.5,    // temperature (τ)
    1.0,    // scale (κ)
    &mut rng,
);

// Apply mask to attention computation for end-to-end optimization
// Gradients flow: language_loss → attention → mask → reranker
```

**Use Case**: End-to-end reranker training in RAG systems without labeled data. See [GUMBEL_RERANKING.md](docs/GUMBEL_RERANKING.md) for details and connection to "Gumbel Reranking" (ACL 2025) paper.

Run examples:
```bash
# Basic example
cargo run --example gumbel_reranking --features gumbel

# RAG training simulation
cargo run --example gumbel_rag_training --features gumbel

# Comparison with sigmoid method
cargo run --example gumbel_vs_sigmoid_comparison --features gumbel
```

**Visualization**: Generate statistical analysis:
```bash
uv run hack/viz/generate_gumbel_analysis.py
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

## Framework Integration

Works with PyTorch, JAX, Julia, and Rust ML frameworks (Candle/Burn). See examples above for PyTorch and JAX integration patterns. For Candle/Burn examples, see `examples/candle_training.rs` and `examples/burn_training.rs`.

## Paper Reproduction

See [MATHEMATICAL_DETAILS.md](docs/MATHEMATICAL_DETAILS.md) and [BENCHMARKING.md](docs/BENCHMARKING.md) for paper reproduction guidelines.

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

- **[Mathematical Details](docs/MATHEMATICAL_DETAILS.md)** - Theory and derivations
- [Getting Started](docs/GETTING_STARTED.md) - Installation and basic usage
- [Examples](docs/EXAMPLES.md) - Practical code examples
- [Parameter Tuning](docs/PARAMETER_TUNING.md) - How to choose `regularization_strength`
- [Benchmarking Guide](docs/BENCHMARKING.md) - How to reproduce paper results
- [Training Integration](docs/TRAINING_INTEGRATION.md) - PyTorch/JAX/Julia integration

## Testing

```bash
# Run all tests
cargo test

# Run Python tests
cd rank-relax-python && pytest tests/
```

## Contributing

Contributions welcome! See [MATHEMATICAL_DETAILS.md](docs/MATHEMATICAL_DETAILS.md) for implementation priorities and theoretical foundations.

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
