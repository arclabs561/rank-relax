# Benchmarking Framework for Paper Reproduction

This document outlines the benchmarking framework to reproduce results from differentiable ranking papers.

## Papers to Reproduce

### 1. SoftRank (WSDM 2008)
**Paper**: "SoftRank: Optimizing Non-Smooth Rank Metrics"
**Key Results**: NDCG optimization on LETOR datasets
**Implementation**: Probabilistic ranking with Gaussian smoothing

### 2. NeuralSort (ICML 2019)
**Paper**: "NeuralSort: A Differentiable Sorting Operator"
**Key Results**: Sorting/ranking benchmarks, temperature scaling
**Implementation**: Temperature-scaled softmax

### 3. SmoothI (ICML 2021)
**Paper**: "SmoothI: Smooth Rank Indicators for Differentiable Ranking"
**Key Results**: Learning-to-rank datasets, approximation guarantees
**Implementation**: Smooth rank indicators with exponential functions

### 4. Differentiable Top-K
**Paper**: Various (DFTopK, SOFT, etc.)
**Key Results**: Top-k selection benchmarks
**Implementation**: Optimal transport or relaxed optimization

## Benchmark Structure

### Datasets
- **LETOR 3.0/4.0**: Standard learning-to-rank datasets
- **MS MARCO**: Large-scale ranking dataset
- **Yahoo Learning-to-Rank**: Industry benchmark
- **Synthetic**: Controlled experiments from papers

### Metrics
- **NDCG@K**: Normalized Discounted Cumulative Gain
- **MAP**: Mean Average Precision
- **MRR**: Mean Reciprocal Rank
- **Spearman Correlation**: Rank correlation
- **Training Time**: Wall-clock time per epoch
- **Gradient Quality**: Analytical vs numerical gradient comparison

### Methods to Compare
1. Sigmoid-based (current default)
2. NeuralSort-style
3. Probabilistic (SoftRank)
4. SmoothI
5. Baseline: Hard ranking (non-differentiable)

## Implementation Plan

### Phase 1: Core Benchmarking
- [ ] Create benchmark harness
- [ ] Load standard datasets
- [ ] Implement evaluation metrics
- [ ] Compare all methods on same datasets

### Phase 2: Paper-Specific Reproductions
- [ ] SoftRank: LETOR NDCG results
- [ ] NeuralSort: Sorting benchmarks
- [ ] SmoothI: Learning-to-rank datasets
- [ ] Document discrepancies and improvements

### Phase 3: Cross-Framework Comparison
- [ ] PyTorch implementation benchmarks
- [ ] JAX implementation benchmarks
- [ ] Julia implementation benchmarks
- [ ] Rust ML implementation benchmarks
- [ ] Performance comparison across frameworks

## Success Criteria

- ✅ Reproduce NDCG results within 2% of paper values
- ✅ Match or exceed paper-reported training times
- ✅ Analytical gradients match numerical gradients (within 1e-5)
- ✅ All methods work across all frameworks
- ✅ Performance is competitive with paper implementations

