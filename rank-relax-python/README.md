# rank-relax-python

Python bindings for [`rank-relax`](../README.md) — differentiable ranking operations for ML training.

[![PyPI](https://img.shields.io/pypi/v/rank-relax.svg)](https://pypi.org/project/rank-relax/)

## Installation

**Install from PyPI:**

```bash
pip install rank-relax
```

**For development/contributing:**

```bash
cd rank-relax-python
uv venv
source .venv/bin/activate
uv tool install maturin
maturin develop --uv
```

## Usage

```python
import rank_relax

# Soft ranking
values = [5.0, 1.0, 2.0, 4.0, 3.0]
ranks = rank_relax.soft_rank(values, regularization_strength=1.0)
# Returns soft ranks: approximately [4.0, 0.0, 1.0, 3.0, 2.0]

# Soft sorting
sorted_values = rank_relax.soft_sort(values, regularization_strength=1.0)
# Returns soft sorted values: approximately [1.0, 2.0, 3.0, 4.0, 5.0]

# Spearman correlation loss
predictions = [0.1, 0.9, 0.3, 0.7, 0.5]
targets = [0.0, 1.0, 0.2, 0.8, 0.4]
loss = rank_relax.spearman_loss(predictions, targets, regularization_strength=1.0)
# Returns loss = 1 - Spearman correlation
```

## API

| Function | Description | Returns |
|----------|-------------|---------|
| `soft_rank(values, regularization_strength)` | Compute soft ranks (differentiable) | `List[float]` |
| `soft_sort(values, regularization_strength)` | Compute soft sorted values | `List[float]` |
| `spearman_loss(predictions, targets, regularization_strength)` | Spearman correlation loss for training | `float` |

## See Also

- **[rank-relax Rust crate](../README.md)**: Core library documentation
- **[rank-fusion](https://crates.io/crates/rank-fusion)**: Combine ranked lists from multiple retrievers
- **[rank-refine](https://crates.io/crates/rank-refine)**: Score embeddings with MaxSim (ColBERT)

## License

MIT OR Apache-2.0

