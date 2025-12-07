# Related Work: Differentiable Sorting and Ranking

Comprehensive survey of implementations, research papers, and underlying theory for differentiable sorting and ranking operations.

## Recent Applications: Gumbel Reranking (ACL 2025)

**"Gumbel Reranking: Differentiable End-to-End Reranker Optimization"** (Huang et al., 2025) applies differentiable ranking techniques to RAG systems:

- **Technique**: Gumbel-Softmax + Relaxed Top-k for differentiable document selection
- **Application**: End-to-end reranker training in RAG without labeled data
- **Key Innovation**: Views reranking as learning a differentiable attention mask
- **Results**: 10.4% improvement on HotpotQA for multi-hop reasoning

**Connection to rank-relax**: This paper demonstrates a practical application of the techniques rank-relax provides. The paper uses Gumbel-Softmax and relaxed top-k sampling—techniques that could be added to rank-relax as advanced methods.

See [GUMBEL_RERANKING.md](GUMBEL_RERANKING.md) for detailed analysis and implementation notes.

## Implementations

### Python/PyTorch Libraries

#### difftopk
- **Repository**: [Felix-Petersen/difftopk](https://github.com/Felix-Petersen/difftopk)
- **Language**: Python/PyTorch
- **Focus**: Differentiable top-k classification learning with TopKCrossEntropyLoss
- **Key Features**:
  - Differentiable sorting networks (bitonic, odd_even, splitter_selection)
  - TopKCrossEntropyLoss as drop-in replacement for CrossEntropyLoss
  - Supports multiple k values simultaneously
  - Distribution options: cauchy, logistic, logistic_phi
- **Paper**: "Differentiable Top-k Classification Learning" (ICML 2022)
- **Use Case**: Classification tasks where top-k accuracy matters

#### diffsort
- **Repository**: [Felix-Petersen/diffsort](https://github.com/Felix-Petersen/diffsort)
- **Language**: Python/PyTorch
- **Focus**: Differentiable sorting networks using relaxed comparators
- **Key Features**:
  - Implements NeuralSort, SoftSort, and custom sorting networks
  - Full permutation matrices (n×n)
  - Monotonic differentiable sorting networks
  - Multiple sorting network architectures (bitonic, odd_even)
- **Paper**: "Differentiable Sorting Networks for Scalable Sorting and Ranking Supervision" (ICML 2021)
- **Use Case**: Full sorting operations with gradient flow

#### torchsort
- **Repository**: [teddykoker/torchsort](https://github.com/teddykoker/torchsort)
- **Language**: Python/PyTorch
- **Focus**: Fast, differentiable sorting and ranking in PyTorch
- **Key Features**:
  - CUDA kernels for performance
  - Soft ranking and sorting operations
  - Optimized for large-scale operations
- **Stars**: 846
- **Use Case**: High-performance differentiable sorting/ranking

#### softsort.pytorch
- **Repository**: [moskomule/softsort.pytorch](https://github.com/moskomule/softsort.pytorch)
- **Language**: Python/PyTorch
- **Focus**: Differentiable sorting using optimal transport (Sinkhorn)
- **Key Features**:
  - Implements Cuturi et al. (2019) optimal transport approach
  - Sinkhorn CDF and quantile operators
- **Paper**: "Differentiable Ranks and Sorting using Optimal Transport" (Cuturi et al., 2019)
- **Status**: Archived
- **Use Case**: Optimal transport-based differentiable sorting

#### Optimal Transport Implementations
- **BenJ-cell/Differentiable-Ranks-and-Sorting-using-Optimal-Transport**: NumPy/Jupyter implementation
- **VldKnd/Rankings-And-Sorting-OT**: Simple NumPy implementation of Cuturi et al. (2019)

### Google Research

#### fast-soft-sort
- **Research**: [Fast Differentiable Sorting and Ranking](https://research.google/pubs/fast-differentiable-sorting-and-ranking/)
- **Authors**: Blondel, Teboul, Berthet, Djolonga (2020)
- **Focus**: O(n log n) differentiable sorting via permutahedron projections
- **Key Innovation**: Isotonic optimization for computational efficiency
- **Complexity**: O(n log n) vs O(n²) for other methods
- **Use Case**: Large-scale differentiable sorting operations

## Research Papers

### Foundational Works

#### NeuralSort (2019)
- **Title**: "Stochastic Optimization of Sorting Networks via Continuous Relaxations"
- **Authors**: Grover, Wang, Zweig, Ermon
- **Venue**: ICML 2019
- **Key Contribution**: First general-purpose continuous relaxation of sorting operator
- **Method**: Relaxed swap operations in sorting networks
- **URL**: https://arxiv.org/abs/1903.08850

#### Optimal Transport Approach (2019)
- **Title**: "Differentiable Ranks and Sorting using Optimal Transport"
- **Authors**: Cuturi, Teboul, Vert
- **Venue**: ICML 2019
- **Key Contribution**: Formulates sorting as optimal assignment problem
- **Method**: Sinkhorn algorithm with entropic regularization
- **Complexity**: O(n²) per Sinkhorn iteration
- **URL**: https://arxiv.org/abs/1905.11885

#### Fast Differentiable Sorting (2020)
- **Title**: "Fast Differentiable Sorting and Ranking"
- **Authors**: Blondel, Teboul, Berthet, Djolonga
- **Venue**: ICML 2020
- **Key Contribution**: O(n log n) complexity via permutahedron projections
- **Method**: Isotonic regression and Fenchel-Young losses
- **URL**: https://arxiv.org/abs/2002.08871

#### SoftSort (2020)
- **Title**: "SoftSort: A Continuous Relaxation for the argsort Operator"
- **Authors**: Prillo, Eisenschlos
- **Key Contribution**: Continuous relaxation of argsort operator
- **Method**: Softmax-based relaxation
- **URL**: https://arxiv.org/abs/2006.16038

### Sorting Networks Approach

#### Differentiable Sorting Networks (2021)
- **Title**: "Differentiable Sorting Networks for Scalable Sorting and Ranking Supervision"
- **Authors**: Petersen, Borgelt, Kuehne, Deussen
- **Venue**: ICML 2021
- **Key Contribution**: Relaxed sorting networks with monotonicity guarantees
- **Method**: Relaxed comparators in classic sorting networks (bitonic, odd_even)
- **URL**: https://arxiv.org/abs/2105.04019

#### Monotonic Differentiable Sorting Networks (2022)
- **Title**: "Monotonic Differentiable Sorting Networks"
- **Authors**: Petersen, Borgelt, Kuehne, Deussen
- **Key Contribution**: Ensures monotonicity to prevent vanishing gradients
- **Method**: Monotonic relaxation of conditional swap operations
- **URL**: https://arxiv.org/abs/2203.09630

#### Generalized Neural Sorting Networks (2023)
- **Title**: "Generalized Neural Sorting Networks with Error-Free Differentiable Swap Functions"
- **Authors**: Kim, Yoon, Cho
- **Key Contribution**: Error-free differentiable swap functions
- **Method**: Improved swap function design
- **URL**: https://arxiv.org/abs/2310.07174

### Ranking-Specific Methods

#### NeuralNDCG (2021)
- **Title**: "NeuralNDCG: Direct Optimisation of a Ranking Metric via Differentiable Relaxation of Sorting"
- **Authors**: Pobrotyn, Białobrzeski
- **Key Contribution**: Direct optimization of NDCG metric
- **Method**: Differentiable relaxation of sorting for NDCG computation
- **URL**: https://arxiv.org/abs/2102.07831

#### Differentiable Top-k Classification (2022)
- **Title**: "Differentiable Top-k Classification Learning"
- **Authors**: Petersen, Kuehne, Borgelt, Deussen
- **Venue**: ICML 2022
- **Key Contribution**: TopKCrossEntropyLoss for multi-k optimization
- **Method**: Differentiable top-k operations via sorting networks
- **URL**: https://arxiv.org/abs/2202.xxxxx (see difftopk repository)

#### SortNet (2023)
- **Title**: "SortNet: Learning To Rank By a Neural-Based Sorting Algorithm"
- **Authors**: Rigutini, Papini, Maggini, Scarselli
- **Key Contribution**: Neural network-based sorting for learning-to-rank
- **Method**: End-to-end trainable sorting network
- **URL**: https://arxiv.org/abs/2311.01864

### Recent Advances

#### LapSum (2025)
- **Title**: "LapSum – One Method to Differentiate Them All: Ranking, Sorting and Top-k Selection"
- **Authors**: Struski, Bednarczyk, Podolak, Tabor
- **Key Contribution**: Unified method for ranking, sorting, and top-k via Laplace distribution
- **Method**: Closed-form inverse of LapSum function
- **URL**: https://arxiv.org/abs/2503.06242

## Underlying Theory

### Core Problem

Traditional sorting and ranking operations are **discrete** and **non-differentiable**:
- Sorting: Returns permutation matrix (integer-valued, piecewise constant)
- Ranking: Returns integer ranks (non-smooth)
- Top-k: Returns discrete selection (non-differentiable)

This prevents gradient-based optimization of objectives that depend on ordering (e.g., NDCG, Spearman correlation, ranking quality).

### Solution Approaches

#### 1. Softmax-Based Relaxation
- **Principle**: Replace discrete operations with softmax-weighted averages
- **Example**: NeuralSort uses softmax to approximate permutation matrices
- **Trade-off**: Smooth but may not preserve exact ordering semantics
- **Complexity**: O(n²) for n elements

#### 2. Optimal Transport
- **Principle**: Formulate sorting as optimal assignment problem
- **Method**: Sinkhorn algorithm with entropic regularization
- **Advantage**: Theoretically grounded, preserves ranking semantics
- **Complexity**: O(n²) per Sinkhorn iteration (typically 10-50 iterations)
- **Key Paper**: Cuturi et al. (2019)

#### 3. Permutahedron Projections
- **Principle**: Project onto permutahedron (convex hull of all permutations)
- **Method**: Isotonic regression and Fenchel-Young losses
- **Advantage**: O(n log n) complexity
- **Key Paper**: Blondel et al. (2020)

#### 4. Sorting Networks
- **Principle**: Relax comparators in classic sorting networks
- **Method**: Replace discrete swaps with smooth, differentiable operations
- **Advantage**: Preserves network structure, enables monotonicity guarantees
- **Complexity**: O(n log² n) for bitonic networks
- **Key Papers**: Petersen et al. (2021, 2022)

#### 5. Laplace Distribution (LapSum)
- **Principle**: Use sum of Laplace distributions for closed-form inverse
- **Method**: Efficient closed-form formula for ranking/sorting/top-k
- **Advantage**: Unified approach, error-free swap functions
- **Key Paper**: Struski et al. (2025)

### Mathematical Foundations

#### Permutahedron
- **Definition**: Convex hull of all permutation vectors
- **Dimension**: (n-1) for n elements
- **Use**: Projection onto permutahedron gives differentiable sorting
- **Reference**: Blondel et al. (2020)

#### Optimal Transport
- **Definition**: Minimum cost to transport one distribution to another
- **Regularization**: Entropic regularization enables Sinkhorn algorithm
- **Use**: Sorting as optimal assignment from input to sorted target
- **Reference**: Cuturi et al. (2019)

#### Isotonic Regression
- **Definition**: Regression under monotonicity constraints
- **Use**: Ensures sorted order in differentiable approximations
- **Complexity**: O(n log n) via pool adjacent violators algorithm
- **Reference**: Blondel et al. (2020)

#### Sorting Networks
- **Definition**: Fixed networks of comparators that sort any input
- **Types**: Bitonic, odd-even merge, etc.
- **Relaxation**: Replace discrete comparators with sigmoid/softmax
- **Reference**: Petersen et al. (2021)

### Complexity Comparison

| Method | Complexity | Preserves Ordering | Monotonicity | Notes |
|--------|-----------|-------------------|--------------|-------|
| NeuralSort | O(n²) | Approximate | No | Softmax-based |
| Optimal Transport | O(n²) per iter | Exact (in limit) | Yes | Sinkhorn iterations |
| Fast Soft Sort | O(n log n) | Exact (in limit) | Yes | Permutahedron projection |
| Sorting Networks | O(n log² n) | Exact (in limit) | Yes | Bitonic networks |
| LapSum | O(n log n) | Exact (in limit) | Yes | Closed-form inverse |

### Trade-offs

1. **Accuracy vs. Efficiency**: More accurate methods (OT, sorting networks) are slower
2. **Monotonicity**: Critical for preventing vanishing gradients
3. **Scalability**: O(n log n) methods (fast-soft-sort, LapSum) scale better
4. **Implementation Complexity**: Sorting networks require careful comparator design

## Applications

### Learning-to-Rank
- **Problem**: Optimize ranking metrics (NDCG, MAP) directly
- **Solution**: Differentiable ranking enables gradient-based optimization
- **Papers**: NeuralNDCG (2021), SortNet (2023)

### Top-k Classification
- **Problem**: Optimize top-k accuracy, not just top-1
- **Solution**: Differentiable top-k operations
- **Papers**: difftopk (2022)

### Ranking Supervision
- **Problem**: Only ordering constraints known, not absolute values
- **Solution**: Differentiable sorting networks
- **Papers**: Petersen et al. (2021, 2022)

### Object Detection
- **Problem**: Rank positives above negatives, sort by localization quality
- **Solution**: Rank & Sort Loss
- **Paper**: "Rank & Sort Loss for Object Detection" (2021)

## Comparison with rank-relax

**rank-relax** focuses on:
- **Rust ML frameworks** (candle/burn) - unique among implementations
- **Ranking semantics** over full sorting - prioritizes order preservation
- **Spearman correlation loss** - specific ranking metric optimization
- **Minimal dependencies** - zero dependencies in core

**Differences from other implementations**:
- Most libraries target Python/PyTorch
- Many focus on full sorting (permutation matrices)
- Few provide Spearman correlation loss specifically
- None target Rust ML frameworks

**Similarities**:
- All use smooth relaxations of discrete operations
- All enable gradient flow through ranking/sorting
- All converge to discrete behavior with high regularization

## Mathematical Details

For comprehensive mathematical formulations, derivations, and theoretical foundations, see **[MATHEMATICAL_DETAILS.md](MATHEMATICAL_DETAILS.md)**. This document provides:

- **Optimal Transport**: Full Sinkhorn algorithm derivation, entropic regularization, transport plan computation
- **Permutahedron Projection**: Convex hull formulation, isotonic regression (PAVA algorithm), Fenchel-Young losses
- **Sorting Networks**: Comparator relaxation, sigmoid-based swaps, monotonicity guarantees
- **NeuralSort/SoftSort**: Softmax relaxations, permutation matrix approximations
- **LapSum**: Laplace distribution formulation, closed-form inverse
- **Gradient Computation**: Explicit gradient formulas, automatic differentiation support

## References

### Key Papers (Chronological)

1. Grover et al. (2019) - NeuralSort: ICML 2019
2. Cuturi et al. (2019) - Optimal Transport Sorting: ICML 2019
3. Blondel et al. (2020) - Fast Differentiable Sorting: ICML 2020
4. Prillo & Eisenschlos (2020) - SoftSort
5. Petersen et al. (2021) - Differentiable Sorting Networks: ICML 2021
6. Pobrotyn & Białobrzeski (2021) - NeuralNDCG
7. Petersen et al. (2022) - Monotonic Sorting Networks
8. Petersen et al. (2022) - Differentiable Top-k: ICML 2022
9. Kim et al. (2023) - Generalized Neural Sorting Networks
10. Rigutini et al. (2023) - SortNet
11. Struski et al. (2025) - LapSum

### Implementation Repositories

1. [difftopk](https://github.com/Felix-Petersen/difftopk) - Top-k classification
2. [diffsort](https://github.com/Felix-Petersen/diffsort) - Sorting networks
3. [torchsort](https://github.com/teddykoker/torchsort) - Fast PyTorch sorting
4. [softsort.pytorch](https://github.com/moskomule/softsort.pytorch) - Optimal transport
5. [fast-soft-sort](https://research.google/pubs/fast-differentiable-sorting-and-ranking/) - Google Research

### Theoretical Foundations

- **Optimal Transport**: Cuturi (2013) - Sinkhorn algorithm
- **Permutahedron**: Classical convex geometry
- **Isotonic Regression**: Pool adjacent violators algorithm
- **Sorting Networks**: Knuth, "The Art of Computer Programming" Vol. 3

