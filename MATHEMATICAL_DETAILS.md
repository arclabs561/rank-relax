# Mathematical Details: Differentiable Sorting and Ranking

Comprehensive mathematical formulations, derivations, and theoretical foundations for differentiable sorting and ranking operations.

## Table of Contents

1. [Problem Formulation](#problem-formulation)
2. [Core Methods](#core-methods)
3. [Gradient Computation](#gradient-computation)
4. [Unifying Framework](#unifying-framework)
5. [Numerical Stability](#numerical-stability)
6. [Complexity & Practical Considerations](#complexity--practical-considerations)
7. [Worked Examples](#worked-examples)

---

## Problem Formulation

### The Non-Differentiability Problem

Traditional sorting maps input $x \in \mathbb{R}^n$ to sorted values and ranks via a permutation $\sigma$. The permutation is **piecewise constant** with respect to $x$:

$$
\frac{\partial \sigma}{\partial x} = 0 \quad \text{almost everywhere}
$$

**Intuition**: As a value increases, its rank jumps by integer steps (0→1→2), not smoothly. These "kinks" have zero gradient almost everywhere, preventing gradient-based optimization.

**Goal**: Find smooth relaxations $\tilde{\text{sort}}(x)$ and $\tilde{\text{rank}}(x)$ that are:
1. **Differentiable** everywhere
2. **Convergent**: $\lim_{\tau \to 0} \tilde{\text{sort}}(x, \tau) = \text{sort}(x)$
3. **Order-preserving**: Maintain relative ordering semantics

### The Naive Approach: Sigmoid Relaxation

The rank of element $i$ can be written as:

$$
\text{rank}(x)_i = \sum_{j \neq i} [x_j < x_i]
$$

Replace the discrete indicator with a sigmoid:

$$
\tilde{\text{rank}}(x)_i = \frac{1}{n-1} \sum_{j \neq i} \sigma(\alpha(x_i - x_j))
$$

where $\sigma(t) = 1/(1 + e^{-t})$ and $\alpha$ controls sharpness. As $\alpha \to \infty$, this recovers discrete ranking.

**Limitations**: $O(n^2)$ complexity, requires parameter tuning, gradients vanish when elements are well-separated.

---

## Core Methods

### 1. Permutahedron Projection

**Metaphor**: The permutahedron is a crystal whose corners are all possible permutations. Instead of snapping to the nearest corner (discrete), we project onto the crystal's interior (smooth).

**Formulation**: The permutahedron $\mathcal{P}_n$ is the convex hull of all permutation vectors. Differentiable sorting projects onto it:

$$
\tilde{\text{sort}}(x) = \arg\min_{y \in \mathcal{P}_n} \|y - x\|^2
$$

This is equivalent to **isotonic regression**, solved by the **Pool Adjacent Violators Algorithm (PAVA)**:
1. Start with $y = x$
2. While $y_i > y_{i+1}$ exists, pool (average) violating adjacent pairs
3. Continue until monotonic

**Properties**: $O(n \log n)$ complexity, exact gradients via implicit differentiation, block-sparse Jacobian.

### 2. Optimal Transport (Sinkhorn)

**Metaphor**: Moving dirt (input values) to holes (sorted positions) in fog (entropy). The fog prevents perfect assignments, creating smooth transport plans.

**Formulation**: Reformulate sorting as optimal assignment with entropic regularization:

$$
\text{OT}_\epsilon(\mu, \nu) = \min_{P \in \Pi(\mu, \nu)} \left( \langle P, C \rangle_F - \epsilon H(P) \right)
$$

Solved via **Sinkhorn iterations** (alternating row/column normalization):

$$
u^{(t+1)}_i = \frac{\mu_i}{\sum_j K_{ij} v^{(t)}_j}, \quad v^{(t+1)}_j = \frac{\nu_j}{\sum_i K_{ij} u^{(t+1)}_i}
$$

where $K_{ij} = \exp(-C_{ij}/\epsilon)$ is the kernel matrix.

**Properties**: $O(n^2 \cdot k)$ complexity (k ≈ 10-50 iterations), dense well-conditioned gradients, requires log-domain for numerical stability.

### 3. Sorting Networks

**Metaphor**: A railroad switchyard with leaky valves. Data flows through comparators (switches) that mix values based on weight, creating smooth transitions.

**Formulation**: Replace discrete comparators with sigmoid-based soft comparators:

$$
\tilde{\text{comp}}(x_i, x_j; \tau) = (\text{softmin}(x_i, x_j), \text{softmax}(x_i, x_j))
$$

**Properties**: $O(n \log^2 n)$ complexity, parallel-friendly, preserves ordering when inputs are well-separated.

### 4. LapSum (2025)

**Metaphor**: Numbers are hills (Laplace distributions). Sum them into a mountain range, then find the water level (quantile) where volume equals the desired rank.

**Formulation**: Use mixture of Laplace CDFs:

$$
\tilde{F}(t; x, \tau) = \frac{1}{n} \sum_{i=1}^n F(t; x_i, \tau)
$$

Soft rank: $\tilde{\text{rank}}(x)_i = n \cdot \tilde{F}(x_i; x, \tau)$  
Soft sort: $\tilde{\text{sort}}(x)_j = \tilde{F}^{-1}((j-1)/(n-1); x, \tau)$

**Properties**: $O(n \log n)$ complexity, closed-form inverse (no iterations), smooth gradients, fastest for large inputs.

---

## Gradient Computation

### The Jacobian Problem

We need gradients $\nabla_x L$ but the Jacobian $J = \partial \hat{x}/\partial x$ is often dense and expensive. **Key insight**: We only need JVPs/VJPs, not the full matrix.

### Gradient Flow (Sigmoid Method)

**Forward**: $\text{rank}_i = \frac{1}{n-1} \sum_{j \neq i} \sigma(\alpha(x_i - x_j))$  
**Backward**: $\frac{\partial \text{rank}_i}{\partial x_i} = \frac{\alpha}{n-1} \sum_{j \neq i} \sigma'(\alpha(x_i - x_j))$

**Intuition**: Gradients are largest when values are close (uncertain ranking), smallest when well-separated (settled ranking).

### Implicit Differentiation (Sinkhorn)

Instead of unrolling iterations (memory: $O(T \cdot N^2)$), use the **Implicit Function Theorem** at the fixed point. Solve a single linear system (memory: $O(N^2)$, independent of $T$).

**Schur Complement**: Reduce $(mn + m + n) \times (mn + m + n)$ system to $(m+n-1) \times (m+n-1)$ via block matrix inversion.

### Isotonic Regression Gradients

PAVA creates **block-sparse** Jacobian: elements in the same pool share gradients equally. At block boundaries, gradients are discontinuous (requires smoothing for neural network training).

---

## Unifying Framework

### Fenchel-Young Losses

All methods are projections onto convex sets with different regularizations:

| Method | Set $\mathcal{C}$ | Regularization $\Omega$ | Result |
| :-- | :-- | :-- | :-- |
| Hard Sort | Permutahedron | 0 | Discrete permutation |
| Isotonic | Permutahedron | L2 | Euclidean projection |
| Sinkhorn | Birkhoff Polytope | Entropy | Sinkhorn iteration |

**Insight**: Every soft sort answers: "What point in the crystal maximizes inner product minus regularization?"

### The 1D Case: Sorting = Optimal Transport

In 1D, optimal transport map is: $T^* = F_\beta^{-1} \circ F_\alpha$ — this is **sorting**. The Wasserstein distance is the $L^p$ distance between quantile functions.

**Connection to LapSum**: Replace hard empirical CDF with smooth Laplace mixture for closed-form inverse.

### Birkhoff-von Neumann

The extreme points of doubly-stochastic matrices are permutation matrices. Entropic regularization pushes solutions into the interior, enabling differentiability.

**Gradient trick**: $\nabla_C L_C(a, b) = P^*$ (gradient equals optimal coupling via envelope theorem).

---

## Numerical Stability

### Exponential Underflow

Methods use $\exp(-C/\epsilon)$, $\text{softmax}(x)$, $\sigma(x)$ which can overflow/underflow.

**Solutions**:
1. **Log-domain Sinkhorn**: Store potentials $f, g$ where $\log P_{ij} = (f_i + g_j - C_{ij})/\epsilon$
2. **LogSumExp**: $\text{LSE}(x) = \max(x) + \log \sum_i \exp(x_i - \max(x))$
3. **Sigmoid clipping**: Clamp inputs to $[-500, 500]$
4. **Softmax stabilization**: Subtract maximum before exponentiating

### Edge Cases

- **Empty**: Return empty vector
- **Single element**: Return `[0.0]`
- **NaN/Inf**: Propagate NaN, handle Inf separately
- **Identical values**: Ranks ≈ $(n-1)/2$ (equal)
- **Very large inputs**: Use log-domain, consider normalization

---

## Complexity & Practical Considerations

### Time Complexity

| Method | Forward | Backward | Space | Best For |
| :-- | :-- | :-- | :-- | :-- |
| **Sigmoid** (current) | $O(n^2)$ | $O(n^2)$ | $O(n)$ | Small-medium inputs |
| **Permutahedron** | $O(n \log n)$ | $O(n \log n)$ | $O(n)$ | Large inputs, exact |
| **Sinkhorn** | $O(n^2 \cdot k)$ | $O(n^2)$ | $O(n^2)$ | Learning global structure |
| **Sorting Networks** | $O(n \log^2 n)$ | $O(n \log^2 n)$ | $O(n)$ | Parallel processing |
| **LapSum** | $O(n \log n)$ | $O(n)$ | $O(n)$ | Speed-critical |

### Hierarchy of Sophistication

1. **Sigmoid**: Simple, $O(n^2)$, vanishing gradients
2. **Isotonic**: Fast, $O(n \log n)$, block-sparse gradients
3. **Sinkhorn (naive)**: Dense gradients, memory-intensive
4. **Sinkhorn (implicit)**: Memory-efficient, production-grade
5. **LapSum**: Fastest, closed-form, emerging standard

**Current implementation**: Level 1 (Sigmoid). For production: aim for Levels 4-5.

### Practical Guidelines

- **n < 100**: All methods fast, choose by accuracy needs
- **100 < n < 1000**: Permutahedron or sorting networks
- **n > 1000**: Permutahedron or LapSum ($O(n \log n)$)

---

## Worked Examples

### Example 1: Soft Ranking with Different α

**Input**: $x = [5.0, 1.0, 2.0, 4.0, 3.0]$

- **α = 0.1**: Ranks ≈ [2.4, 2.1, 2.2, 2.3, 2.2] (smooth, poor discrimination)
- **α = 1.0**: Ranks ≈ [4.0, 0.1, 1.2, 3.1, 2.0] (balanced)
- **α = 10.0**: Ranks ≈ [4.0, 0.0, 1.0, 3.0, 2.0] (sharp, close to discrete)

### Example 2: Permutahedron Projection (n=3)

**Input**: $x = [3.0, 1.0, 2.0]$

**PAVA steps**:
1. Start: $y = [3.0, 1.0, 2.0]$
2. $y_0 > y_2$ violates monotonicity
3. Pool: $y_0 = y_2 = (3.0 + 2.0)/2 = 2.5$
4. Result: $y = [2.5, 1.0, 2.5]$ (monotonic)

**Gradients**: $\partial y_0/\partial x_0 = \partial y_0/\partial x_2 = 1/2$ (shared), $\partial y_1/\partial x_1 = 1$ (isolated)

### Example 3: Sinkhorn (n=2)

**Input**: $x = [5.0, 1.0]$, **Target**: $y = [1.0, 5.0]$

**Cost**: $C = \begin{bmatrix} 4 & 0 \\ 0 & 4 \end{bmatrix}$, **Kernel**: $K = \exp(-C) \approx \begin{bmatrix} 0.018 & 1.0 \\ 1.0 & 0.018 \end{bmatrix}$

After convergence: $P^* \approx \begin{bmatrix} 0.018 & 0.982 \\ 0.982 & 0.018 \end{bmatrix}$

Interpretation: 98.2% of $x_1=5.0$ goes to $y_2=5.0$ (correct assignment).

---

## References

1. Cuturi, Teboul, Vert (2019). "Differentiable Ranks and Sorting using Optimal Transport". ICML 2019.
2. Blondel, Teboul, Berthet, Djolonga (2020). "Fast Differentiable Sorting and Ranking". ICML 2020.
3. Grover, Wang, Zweig, Ermon (2019). "Stochastic Optimization of Sorting Networks via Continuous Relaxations". ICML 2019.
4. Prillo, Eisenschlos (2020). "SoftSort: A Continuous Relaxation for the argsort Operator".
5. Petersen, Borgelt, Kuehne, Deussen (2021). "Differentiable Sorting Networks for Scalable Sorting and Ranking Supervision". ICML 2021.
6. Struski, Bednarczyk, Podolak, Tabor (2025). "LapSum – One Method to Differentiate Them All: Ranking, Sorting and Top-k Selection".
