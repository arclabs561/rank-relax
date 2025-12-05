# Documentation Index

Complete guide to all documentation in `rank-relax`.

## Quick Start

**New to rank-relax?** Start here:
1. **[README.md](README.md)** - Overview and quick start
2. **[GETTING_STARTED.md](GETTING_STARTED.md)** - Installation and basic usage
3. **[EXAMPLES.md](EXAMPLES.md)** - Practical code examples

## Core Documentation

### User Guides

- **[README.md](README.md)** - Main documentation, overview, features, API reference
- **[GETTING_STARTED.md](GETTING_STARTED.md)** - Step-by-step walkthrough for beginners
- **[EXAMPLES.md](EXAMPLES.md)** - Comprehensive examples for common use cases
- **[PARAMETER_TUNING.md](PARAMETER_TUNING.md)** - How to choose `regularization_strength`

### Theory and Mathematics

- **[MATHEMATICAL_DETAILS.md](MATHEMATICAL_DETAILS.md)** - Comprehensive mathematical formulations, derivations, and theoretical foundations
  - Problem formulation
  - Naive approach (sigmoid-based)
  - Optimal transport (Sinkhorn algorithm)
  - Permutahedron projection
  - Sorting networks
  - Complexity analysis
  - Worked examples

- **[PEDAGOGICAL_IMPROVEMENTS.md](PEDAGOGICAL_IMPROVEMENTS.md)** - Analysis of educational materials and pedagogical insights
  - What educational resources do well
  - Areas for improvement
  - Key pedagogical principles

- **[RELATED_WORK.md](RELATED_WORK.md)** - Survey of implementations, research papers, and underlying theory

### Integration Guides

- **[CANDLE_BURN_INTEGRATION.md](CANDLE_BURN_INTEGRATION.md)** - Framework integration details (planned)
- **[rank-relax-python/PYTORCH_JAX_INTEGRATION.md](rank-relax-python/PYTORCH_JAX_INTEGRATION.md)** - Python bindings integration

### Development

- **[PUBLISHING.md](PUBLISHING.md)** - Publishing and release process

## Documentation by Topic

### Understanding the Problem

- **Why differentiable ranking?**: [README.md#why-differentiable-ranking](README.md#why-differentiable-ranking)
- **The non-differentiability problem**: [MATHEMATICAL_DETAILS.md#problem-formulation](MATHEMATICAL_DETAILS.md#problem-formulation)
- **Intuitive explanation**: [README.md#why-differentiable-ranking](README.md#why-differentiable-ranking) (staircase vs. smooth ramp analogy)

### Algorithm Details

- **Current implementation (sigmoid-based)**: [MATHEMATICAL_DETAILS.md#the-naive-approach](MATHEMATICAL_DETAILS.md#the-naive-approach)
- **How it works**: [src/rank.rs](src/rank.rs) (code comments)
- **Complexity**: [MATHEMATICAL_DETAILS.md#complexity-analysis](MATHEMATICAL_DETAILS.md#complexity-analysis)
- **Alternative methods**: [MATHEMATICAL_DETAILS.md](MATHEMATICAL_DETAILS.md) (optimal transport, permutahedron, etc.)

### Parameter Tuning

- **Quick guide**: [README.md#configuration](README.md#configuration)
- **Detailed guide**: [PARAMETER_TUNING.md](PARAMETER_TUNING.md)
- **Rule of thumb**: `regularization_strength ≈ 1.0 / typical_difference_between_values`
- **Examples**: [EXAMPLES.md#parameter-tuning](EXAMPLES.md#parameter-tuning)

### Usage Examples

- **Basic ranking**: [EXAMPLES.md#basic-ranking](EXAMPLES.md#basic-ranking)
- **Training loops**: [EXAMPLES.md#training-with-spearman-loss](EXAMPLES.md#training-with-spearman-loss)
- **Edge cases**: [EXAMPLES.md#edge-cases](EXAMPLES.md#edge-cases)
- **Real-world scenarios**: [EXAMPLES.md#real-world-scenarios](EXAMPLES.md#real-world-scenarios)

### Mathematical Theory

- **Formulations**: [MATHEMATICAL_DETAILS.md](MATHEMATICAL_DETAILS.md)
- **Derivations**: [MATHEMATICAL_DETAILS.md](MATHEMATICAL_DETAILS.md) (Sinkhorn, permutahedron, etc.)
- **Gradient computation**: [MATHEMATICAL_DETAILS.md#gradient-computation](MATHEMATICAL_DETAILS.md#gradient-computation)
- **Worked examples**: [MATHEMATICAL_DETAILS.md#worked-examples](MATHEMATICAL_DETAILS.md#worked-examples)

### Related Work

- **Implementations**: [RELATED_WORK.md](RELATED_WORK.md) and [README.md#related-work](README.md#related-work)
- **Research papers**: [RELATED_WORK.md](RELATED_WORK.md)
- **Educational resources**: [PEDAGOGICAL_IMPROVEMENTS.md](PEDAGOGICAL_IMPROVEMENTS.md)

## Reading Paths

### For Users (Practitioners)

1. [README.md](README.md) - Get overview
2. [GETTING_STARTED.md](GETTING_STARTED.md) - Install and try it
3. [EXAMPLES.md](EXAMPLES.md) - See practical examples
4. [PARAMETER_TUNING.md](PARAMETER_TUNING.md) - Tune parameters

### For Researchers

1. [README.md](README.md) - Overview
2. [MATHEMATICAL_DETAILS.md](MATHEMATICAL_DETAILS.md) - Deep dive into theory
3. [RELATED_WORK.md](RELATED_WORK.md) - Survey of field
4. [PEDAGOGICAL_IMPROVEMENTS.md](PEDAGOGICAL_IMPROVEMENTS.md) - Educational insights

### For Contributors

1. [README.md](README.md) - Understand the project
2. [src/](src/) - Read the code (well-documented)
3. [CANDLE_BURN_INTEGRATION.md](CANDLE_BURN_INTEGRATION.md) - Integration plans
4. [PUBLISHING.md](PUBLISHING.md) - Release process

## Key Concepts Explained

### Differentiable Ranking

- **What it is**: Smooth approximation of discrete ranking
- **Why needed**: Enables gradient-based optimization of ranking metrics
- **How it works**: Uses sigmoid functions to create smooth transitions
- **See**: [README.md#why-differentiable-ranking](README.md#why-differentiable-ranking)

### Soft Ranking

- **Formula**: `rank[i] = (1/(n-1)) * Σ_{j≠i} sigmoid(α * (values[i] - values[j]))`
- **Intuition**: Count (softly) how many others each element is greater than
- **See**: [MATHEMATICAL_DETAILS.md#the-naive-approach](MATHEMATICAL_DETAILS.md#the-naive-approach)

### Spearman Correlation Loss

- **What it is**: Differentiable version of Spearman correlation
- **Formula**: `loss = 1 - Pearson_correlation(soft_rank(predictions), soft_rank(targets))`
- **Use case**: Training models to optimize ranking quality
- **See**: [src/spearman.rs](src/spearman.rs) and [EXAMPLES.md#training-with-spearman-loss](EXAMPLES.md#training-with-spearman-loss)

### Regularization Strength

- **What it controls**: Sharpness of the relaxation
- **Rule of thumb**: `≈ 1.0 / typical_difference_between_values`
- **Effects**: Low = smooth gradients, High = accurate ranking
- **See**: [PARAMETER_TUNING.md](PARAMETER_TUNING.md)

## Code Documentation

- **Library overview**: [src/lib.rs](src/lib.rs)
- **Ranking operations**: [src/rank.rs](src/rank.rs)
- **Sorting operations**: [src/sort.rs](src/sort.rs)
- **Spearman loss**: [src/spearman.rs](src/spearman.rs)
- **Property tests**: [src/proptests.rs](src/proptests.rs)

## External Resources

- **API Documentation**: https://docs.rs/rank-relax (when published)
- **GitHub Repository**: https://github.com/arclabs561/rank-relax
- **Related crates**: [rank-fusion](https://crates.io/crates/rank-fusion), [rank-refine](https://crates.io/crates/rank-refine), [rank-eval](https://crates.io/crates/rank-eval)

