# Mathematical Details: Differentiable Sorting and Ranking

Comprehensive mathematical formulations, derivations, and theoretical foundations for differentiable sorting and ranking operations.

## Table of Contents

1. [Problem Formulation](#problem-formulation)
2. [Optimal Transport Approach](#optimal-transport-approach)
3. [Permutahedron Projection](#permutahedron-projection)
4. [Sorting Networks](#sorting-networks)
5. [NeuralSort and SoftSort](#neuralsort-and-softsort)
6. [Isotonic Regression](#isotonic-regression)
7. [Fenchel-Young Losses](#fenchel-young-losses)
8. [LapSum Method](#lapsum-method)
9. [Gradient Computation](#gradient-computation)
10. [Complexity Analysis](#complexity-analysis)
11. [Worked Examples](#worked-examples)

---

## Problem Formulation

### The Non-Differentiability Problem

Traditional sorting is a **discrete operation** that maps an input vector \(x \in \mathbb{R}^n\) to:
- **Sorted values**: \(\text{sort}(x) = (x_{\sigma(1)}, x_{\sigma(2)}, \ldots, x_{\sigma(n)})\) where \(\sigma\) is the sorting permutation
- **Ranks**: \(\text{rank}(x) = (\sigma^{-1}(1), \sigma^{-1}(2), \ldots, \sigma^{-1}(n))\)

The sorting permutation \(\sigma\) is **piecewise constant** with respect to \(x\), meaning:
\[
\frac{\partial \sigma}{\partial x} = 0 \quad \text{almost everywhere}
\]

**Intuitive explanation**: As we increase the value of an element, its rank decreases abruptly (by integer steps), not smoothly. The rank function has "jumps" or "kinks" where it's not differentiable.

This prevents gradient-based optimization of objectives that depend on sorting or ranking.

### Goal

Find **smooth relaxations** \(\tilde{\text{sort}}(x)\) and \(\tilde{\text{rank}}(x)\) such that:
1. **Differentiability**: Gradients exist everywhere
2. **Convergence**: \(\lim_{\tau \to 0} \tilde{\text{sort}}(x, \tau) = \text{sort}(x)\) (where \(\tau\) is temperature)
3. **Ordering Preservation**: Maintains relative order semantics

---

## The "Naive" Approach: Intuitive but Problematic

### Indicator Function Formulation

The rank of element \(i\) can be written as a sum of indicator functions:
\[
\text{rank}(x)_i = \sum_{j \neq i} [x_j < x_i]
\]
where \([x_j < x_i]\) is 1 if \(x_j < x_i\), 0 otherwise.

**Intuition**: For each element, count how many other elements are smaller than it. This gives its rank (0-indexed).

### Sigmoid Relaxation

The natural relaxation is to replace the discrete indicator with a sigmoid:
\[
\tilde{\text{rank}}(x)_i = \sum_{j \neq i} \sigma(\alpha(x_i - x_j))
\]
where \(\sigma(t) = \frac{1}{1 + e^{-t}}\) is the sigmoid function and \(\alpha > 0\) controls sharpness.

**Why this works**: As \(\alpha \to \infty\), \(\sigma(\alpha(x_i - x_j)) \to [x_i > x_j]\), recovering the discrete indicator.

### Problems with the Naive Approach

1. **Complexity**: Requires \(O(n^2)\) comparisons for \(n\) elements
2. **Parameter tuning**: The \(\alpha\) parameter must be tuned based on the scale of differences between elements
3. **Gradient quality**: Gradients can be very small when elements are well-separated

**Example**: For \(x = [-0.3, 0.8, -5.0, 3.0, 1.0]\), computing soft rank requires 20 sigmoid evaluations (5×4 comparisons).

**Worked Example** (n=3, α=1.0):
Let \(x = [2.0, 1.0, 3.0]\).

For element 0 (value 2.0):
- Compare with element 1: \(\sigma(1.0 \cdot (2.0 - 1.0)) = \sigma(1.0) \approx 0.73\)
- Compare with element 2: \(\sigma(1.0 \cdot (2.0 - 3.0)) = \sigma(-1.0) \approx 0.27\)
- Rank: \((0.73 + 0.27) / 2 = 0.5\)

For element 1 (value 1.0):
- Compare with element 0: \(\sigma(1.0 \cdot (1.0 - 2.0)) = \sigma(-1.0) \approx 0.27\)
- Compare with element 2: \(\sigma(1.0 \cdot (1.0 - 3.0)) = \sigma(-2.0) \approx 0.12\)
- Rank: \((0.27 + 0.12) / 2 = 0.195\)

For element 2 (value 3.0):
- Compare with element 0: \(\sigma(1.0 \cdot (3.0 - 2.0)) = \sigma(1.0) \approx 0.73\)
- Compare with element 1: \(\sigma(1.0 \cdot (3.0 - 1.0)) = \sigma(2.0) \approx 0.88\)
- Rank: \((0.73 + 0.88) / 2 = 0.805\)

Result: ranks ≈ [0.5, 0.195, 0.805], which preserves ordering (1.0 < 2.0 < 3.0).

With higher α (e.g., 10.0), ranks approach [1.0, 0.0, 2.0] (discrete ranks).

This motivates more efficient and robust methods discussed below.

---

## Optimal Transport Approach

### Formulation (Cuturi et al., 2019)

Sorting is reformulated as an **optimal assignment problem** between input values and a sorted target sequence.

#### Setup

- **Input**: \(x = (x_1, \ldots, x_n) \in \mathbb{R}^n\)
- **Target**: \(y = (y_1, \ldots, y_m)\) where \(y_1 < y_2 < \cdots < y_m\) (sorted sequence)
- **Input measure**: \(\mu = \sum_{i=1}^n \mu_i \delta_{x_i}\) (discrete measure on input values)
- **Target measure**: \(\nu = \sum_{j=1}^m \nu_j \delta_{y_j}\) (discrete measure on target values)

#### Unregularized Optimal Transport

The optimal transport problem is:
\[
\text{OT}(\mu, \nu) = \min_{P \in \Pi(\mu, \nu)} \langle P, C \rangle_F = \min_{P} \sum_{i,j} P_{ij} C_{ij}
\]

where:
- \(P \in \mathbb{R}^{n \times m}\) is a **transport plan** (matrix)
- \(C_{ij} = |x_i - y_j|\) or \(C_{ij} = (x_i - y_j)^2\) is the **cost matrix**
- \(\Pi(\mu, \nu) = \{P \geq 0 : P\mathbf{1}_m = \mu, P^T\mathbf{1}_n = \nu\}\) are the **marginal constraints**

The solution \(P^*\) is typically **sparse** (only \(n\) non-zero entries), making it non-differentiable.

#### Entropic Regularization

To enable differentiability, add **entropic regularization**:
\[
\text{OT}_\epsilon(\mu, \nu) = \min_{P \in \Pi(\mu, \nu)} \left( \langle P, C \rangle_F - \epsilon H(P) \right)
\]

where:
- \(H(P) = -\sum_{i,j} P_{ij} \log P_{ij}\) is the **entropy** of the transport plan
- \(\epsilon > 0\) is the **regularization parameter** (temperature)

The regularized solution \(P_\epsilon^*\) is **dense** and **differentiable** everywhere.

#### Sinkhorn Algorithm

The regularized problem is solved via **Sinkhorn iterations**, an efficient coordinate ascent method.

**Why exponential form?** The entropy term \(-\epsilon H(P)\) in the objective leads to solutions of the form \(P_{ij} \propto \exp(-C_{ij}/\epsilon)\), motivating the kernel matrix.

**Kernel matrix**:
\[
K = \exp(-C / \epsilon) \in \mathbb{R}^{n \times m}
\]

where \(C_{ij} = |x_i - y_j|\) or \((x_i - y_j)^2\) is the cost matrix.

**Iterative updates** (alternating projections):
\[
u^{(t+1)}_i = \frac{\mu_i}{\sum_j K_{ij} v^{(t)}_j}, \quad v^{(t+1)}_j = \frac{\nu_j}{\sum_i K_{ij} u^{(t+1)}_i}
\]

**Intuition**: 
- Update \(u\) to satisfy row constraints (sum of row \(i\) equals \(\mu_i\))
- Update \(v\) to satisfy column constraints (sum of column \(j\) equals \(\nu_j\))
- Alternating between these projections converges to the optimal solution

**Optimal transport plan**:
\[
P_\epsilon^* = \text{diag}(u^*) K \text{diag}(v^*)
\]

where \(u^*\) and \(v^*\) are the fixed points of the iterations.

**Convergence**: Typically requires 10-50 iterations, with convergence rate depending on \(\epsilon\) and the scale of costs.

#### Sinkhorn Ranking and Sorting Operators

**S-Rank operator** (differentiable ranking):
\[
\text{S-rank}(x)_i = \sum_{j=1}^m (j-1) \cdot P_{\epsilon,ij}^*
\]

**S-Sort operator** (differentiable sorting):
\[
\text{S-sort}(x)_j = \sum_{i=1}^n x_i \cdot P_{\epsilon,ij}^*
\]

**S-CDF operator** (cumulative distribution):
\[
\text{S-CDF}(x; t) = \sum_{j: y_j \leq t} \sum_{i=1}^n P_{\epsilon,ij}^*
\]

#### Complexity

- **Per iteration**: \(O(nm)\) for matrix-vector products
- **Convergence**: Typically 10-50 iterations
- **Total**: \(O(nm \cdot \text{iterations})\)

---

## Permutahedron Projection

### Formulation (Blondel et al., 2020)

The **permutahedron** \(\mathcal{P}_n\) is the convex hull of all permutation vectors:
\[
\mathcal{P}_n = \text{conv}\{(\sigma(1), \sigma(2), \ldots, \sigma(n)) : \sigma \in S_n\}
\]

where \(S_n\) is the symmetric group of all permutations.

#### Visual Intuition

**Low-dimensional examples**:
- **n=2**: Permutahedron is a line segment from \((0,1)\) to \((1,0)\) in 2D space
- **n=3**: Permutahedron is a regular hexagon in 3D space (2D polytope)
- **n=4**: Permutahedron is a truncated octahedron with 24 vertices (3D polytope)

**Key insight**: The permutahedron captures all possible ranking vectors. Each vertex corresponds to a unique permutation/ranking.

#### Key Properties

- **Dimension**: \((n-1)\)-dimensional polytope in \(\mathbb{R}^n\)
- **Vertices**: All \(n!\) permutation vectors
- **Facets**: Defined by inequalities \(\sum_{i \in I} x_i \geq \binom{|I|+1}{2}\) for all \(I \subseteq [n]\)

#### Ranking as Optimization

**Observation**: Ranking can be written as finding the vertex of the permutahedron that maximizes the dot product with the input:
\[
\text{rank}(x) = \arg\max_{v \in \text{vertices}(\mathcal{P}_n)} (x \cdot v)
\]

**Intuition**: For \(n=2\), if \(x = (a, b)\):
- If \(b > a\), then \((0,1) \cdot (a,b) = b > a = (1,0) \cdot (a,b)\), so output is \((0,1)\)
- If \(a > b\), then \((1,0) \cdot (a,b) = a > b = (0,1) \cdot (a,b)\), so output is \((1,0)\)

**Extension**: The optimization domain can be extended from vertices to the entire permutahedron without changing the optimum (by linearity of dot product).

#### Projection Formulation

Differentiable sorting is achieved by **projecting onto the permutahedron**:
\[
\tilde{\text{sort}}(x) = \arg\min_{y \in \mathcal{P}_n} \|y - x\|^2
\]

**Intuition**: Find the point on the permutahedron closest to the input vector \(x\). This point is a convex combination of permutation vectors, providing a smooth approximation.

**Regularization trick**: Instead of projecting directly (which is still non-differentiable), add quadratic regularization:
\[
\tilde{\text{rank}}(x) = \arg\max_{v \in \mathcal{P}_n} (x \cdot v) + \epsilon \|v\|^2
\]

This makes the permutahedron "curved" and soft, enabling differentiability.

**Equivalence**: This is equivalent to:
\[
\tilde{\text{rank}}(x) = \arg\min_{v \in \mathcal{P}_n} \frac{1}{2}\left\|\frac{x}{\epsilon} - v\right\|^2
\]

i.e., projecting \(x/\epsilon\) onto the permutahedron.

This projection is:
- **Differentiable**: Via implicit differentiation
- **Exact**: In the limit as \(\epsilon \to 0\)
- **Efficient**: \(O(n \log n)\) via isotonic regression

#### Isotonic Regression Connection

The projection onto the permutahedron is equivalent to **isotonic regression**:
\[
\tilde{\text{sort}}(x) = \arg\min_{y: y_1 \leq y_2 \leq \cdots \leq y_n} \|y - x\|^2
\]

This is solved by the **pool adjacent violators algorithm (PAVA)**:
1. Start with \(y = x\)
2. While there exists \(i\) such that \(y_i > y_{i+1}\):
   - Pool \(y_i\) and \(y_{i+1}\) (replace with their average)
   - Continue until monotonic

**Complexity**: \(O(n \log n)\) worst case, \(O(n)\) average case.

#### Fenchel-Young Losses

The projection can be expressed via **Fenchel-Young losses**:
\[
L(x, y) = \max_{z \in \mathcal{P}_n} \langle z, x \rangle - \Omega(z) - \langle y, x \rangle
\]

where \(\Omega(z)\) is a regularization function. This enables efficient computation and differentiation.

---

## Sorting Networks

### Formulation (Petersen et al., 2021)

**Sorting networks** are fixed networks of comparators that sort any input. Examples:
- **Bitonic network**: \(O(n \log^2 n)\) comparators
- **Odd-even merge**: \(O(n \log^2 n)\) comparators
- **Bubble sort network**: \(O(n^2)\) comparators

#### Comparator Function

A **comparator** at position \((i, j)\) with \(i < j\) outputs:
\[
\text{comp}(x_i, x_j) = (\min(x_i, x_j), \max(x_i, x_j))
\]

#### Differentiable Relaxation

Replace the discrete comparator with a **smooth relaxation** using sigmoid functions.

**Softmin and softmax operators**:
\[
\text{softmin}(a_i, a_j) = \alpha_{ij} \cdot a_i + (1 - \alpha_{ij}) \cdot a_j
\]
\[
\text{softmax}(a_i, a_j) = (1 - \alpha_{ij}) \cdot a_i + \alpha_{ij} \cdot a_j
\]

where:
\[
\alpha_{ij} = \frac{e^{(a_j - a_i)/\tau}}{1 + e^{(a_j - a_i)/\tau}} = \sigma((a_j - a_i)/\tau)
\]

**Sigmoid-based comparator**:
\[
\tilde{\text{comp}}(x_i, x_j; \tau) = (\text{softmin}(x_i, x_j), \text{softmax}(x_i, x_j))
\]

**Intuition**: 
- When \(x_i \gg x_j\), \(\alpha_{ij} \approx 0\), so softmin ≈ \(x_j\) and softmax ≈ \(x_i\) (correct ordering)
- When \(x_i \approx x_j\), \(\alpha_{ij} \approx 0.5\), so outputs are averages (smooth transition)
- As \(\tau \to 0\), sigmoid becomes step function, recovering discrete comparator

**Properties**:
- \(\lim_{\tau \to 0} \tilde{\text{comp}}(x_i, x_j; \tau) = \text{comp}(x_i, x_j)\)
- Differentiable everywhere
- Preserves ordering when inputs are well-separated

#### Monotonic Differentiable Sorting Networks

To ensure **monotonicity** (gradients have correct sign), use a **monotonic sigmoid family**:
\[
\sigma_{\tau,\lambda}(t) = \frac{1}{1 + \exp(-(t + \lambda)/\tau)}
\]

where \(\lambda\) is a bias parameter that ensures:
\[
\frac{\partial \tilde{\text{comp}}_1}{\partial x_i} \geq 0, \quad \frac{\partial \tilde{\text{comp}}_2}{\partial x_j} \geq 0
\]

This prevents **vanishing gradients** and ensures stable training.

#### Complexity

- **Comparators**: \(O(n \log^2 n)\) for bitonic networks
- **Per comparator**: \(O(1)\) operations
- **Total**: \(O(n \log^2 n)\)

---

## NeuralSort and SoftSort

### NeuralSort (Grover et al., 2019)

**NeuralSort** uses a **softmax-based relaxation** of the sorting permutation matrix.

#### Permutation Matrix

The sorting permutation can be represented as a matrix \(P \in \{0,1\}^{n \times n}\) where:
\[
P_{ij} = \begin{cases}
1 & \text{if } x_i \text{ is the } j\text{-th smallest element} \\
0 & \text{otherwise}
\end{cases}
\]

#### NeuralSort Relaxation

Replace the discrete permutation with a **softmax relaxation**:
\[
\tilde{P}_{ij} = \frac{\exp(-|x_i - \text{sort}(x)_j| / \tau)}{\sum_{k=1}^n \exp(-|x_i - \text{sort}(x)_k| / \tau)}
\]

**Differentiable sorting**:
\[
\tilde{\text{sort}}(x)_j = \sum_{i=1}^n \tilde{P}_{ij} x_i
\]

**Differentiable ranking**:
\[
\tilde{\text{rank}}(x)_i = \sum_{j=1}^n j \cdot \tilde{P}_{ij}
\]

#### Complexity

- **Computation**: \(O(n^2)\) for all pairwise distances
- **Softmax**: \(O(n^2)\) per output position
- **Total**: \(O(n^2)\)

### SoftSort (Prillo & Eisenschlos, 2020)

**SoftSort** uses a **softmax relaxation of argsort** (the inverse permutation).

#### Argsort Operator

The argsort operator returns the indices that would sort the array:
\[
\text{argsort}(x) = (\sigma^{-1}(1), \sigma^{-1}(2), \ldots, \sigma^{-1}(n))
\]

#### SoftSort Relaxation

\[
\tilde{\text{argsort}}(x)_j = \sum_{i=1}^n i \cdot \frac{\exp(-|x_i - \text{sort}(x)_j| / \tau)}{\sum_{k=1}^n \exp(-|x_k - \text{sort}(x)_j| / \tau)}
\]

**Differentiable sorting**:
\[
\tilde{\text{sort}}(x) = x[\tilde{\text{argsort}}(x)]
\]

where \(x[\cdot]\) denotes indexing.

---

## Isotonic Regression

### Formulation

**Isotonic regression** finds the best monotonic approximation:
\[
\min_{y: y_1 \leq y_2 \leq \cdots \leq y_n} \sum_{i=1}^n w_i (y_i - x_i)^2
\]

where \(w_i\) are weights.

### Pool Adjacent Violators Algorithm (PAVA)

**Algorithm**:
1. Initialize \(y = x\)
2. While there exists \(i\) such that \(y_i > y_{i+1}\):
   - Find all consecutive violators: indices \(i, i+1, \ldots, j\) where \(y_i > y_{i+1} > \cdots > y_j\)
   - Replace with weighted average: \(y_k = \frac{\sum_{\ell=i}^j w_\ell x_\ell}{\sum_{\ell=i}^j w_\ell}\) for \(k \in [i, j]\)
3. Return \(y\)

**Complexity**: \(O(n)\) average case, \(O(n \log n)\) worst case.

### Differentiability

The PAVA solution is **piecewise linear** in \(x\), making it differentiable almost everywhere. The gradient is:
\[
\frac{\partial y_i}{\partial x_j} = \begin{cases}
\frac{w_j}{\sum_{k \in \text{pool}(i)} w_k} & \text{if } j \in \text{pool}(i) \\
0 & \text{otherwise}
\end{cases}
\]

where \(\text{pool}(i)\) is the set of indices pooled together with \(i\).

---

## Fenchel-Young Losses

### Formulation (Blondel et al., 2020)

**Fenchel-Young losses** provide a unified framework for differentiable sorting and ranking.

#### Fenchel Conjugate

Given a function \(f: \mathcal{Y} \to \mathbb{R}\), its **Fenchel conjugate** is:
\[
f^*(y) = \sup_{z \in \mathcal{Y}} \langle y, z \rangle - f(z)
\]

#### Fenchel-Young Loss

For sorting, define:
\[
L(x, y) = f^*(x) + f(y) - \langle x, y \rangle
\]

where:
- \(f(y) = \Omega(y)\) is a regularization function (e.g., entropy)
- \(y \in \mathcal{P}_n\) is the target (sorted vector)
- \(x\) is the input

#### Properties

1. **Non-negativity**: \(L(x, y) \geq 0\) with equality when \(y = \arg\max_{z \in \mathcal{P}_n} \langle x, z \rangle - \Omega(z)\)
2. **Differentiability**: Gradient is \(\nabla_x L(x, y) = \arg\max_{z \in \mathcal{P}_n} \langle x, z \rangle - \Omega(z) - y\)
3. **Convexity**: Convex in \(x\) for fixed \(y\)

#### Connection to Sorting

When \(\Omega\) is the **entropy** and \(\mathcal{Y} = \mathcal{P}_n\), the Fenchel-Young loss recovers:
- **Differentiable sorting**: Via the argmax
- **Differentiable ranking**: Via the gradient

---

## LapSum Method

### Formulation (Struski et al., 2025)

**LapSum** uses the sum of **Laplace distributions** for a closed-form differentiable ranking/sorting operator.

#### Laplace Distribution

The Laplace distribution has PDF:
\[
f(x; \mu, b) = \frac{1}{2b} \exp\left(-\frac{|x - \mu|}{b}\right)
\]

#### LapSum Function

For input \(x = (x_1, \ldots, x_n)\), define:
\[
\text{LapSum}(x; t) = \sum_{i=1}^n \text{Laplace}(t; x_i, \tau)
\]

where \(\tau\) is the temperature parameter.

#### Closed-Form Inverse

The key innovation is a **closed-form formula for the inverse**:
\[
\text{LapSum}^{-1}(x; p) = \text{quantile of LapSum at probability } p
\]

This enables efficient computation of:
- **Soft ranking**: Via quantile computation
- **Soft sorting**: Via inverse mapping
- **Top-k selection**: Via quantile thresholds

#### Complexity

- **Computation**: \(O(n \log n)\) for sorting quantiles
- **Differentiation**: \(O(n)\) via closed-form gradients
- **Total**: \(O(n \log n)\)

---

## Gradient Computation

### Automatic Differentiation

All differentiable sorting methods support **automatic differentiation** via:
- **PyTorch**: `torch.autograd`
- **TensorFlow**: `tf.GradientTape`
- **JAX**: `jax.grad`
- **Rust ML frameworks**: candle, burn (when integrated)

### Gradient Flow Through Soft Ranking (Current Implementation)

For the sigmoid-based approach used in this crate, gradients flow as follows:

**Forward pass**:
\[
\text{rank}_i = \frac{1}{n-1} \sum_{j \neq i} \sigma(\alpha(x_i - x_j))
\]

**Backward pass** (gradient w.r.t. \(x_i\)):
\[
\frac{\partial \text{rank}_i}{\partial x_i} = \frac{\alpha}{n-1} \sum_{j \neq i} \sigma'(\alpha(x_i - x_j))
\]

where \(\sigma'(t) = \sigma(t)(1 - \sigma(t))\) is the derivative of sigmoid.

**Intuition**: The gradient is proportional to:
- The regularization strength α (higher = larger gradients)
- The sum of sigmoid derivatives (largest when \(x_i\) is close to other values)

**Gradient quality**: Gradients are largest when values are close together (where ranking is ambiguous), and smallest when values are well-separated (where ranking is clear). This matches intuition: the model needs more gradient signal when rankings are uncertain.

### Explicit Gradients

For some methods, **explicit gradient formulas** are available:

#### Optimal Transport Gradients

For Sinkhorn-based methods:
\[
\frac{\partial \text{S-sort}(x)_j}{\partial x_i} = P_{\epsilon,ij}^* + \sum_{k,\ell} \frac{\partial P_{\epsilon,k\ell}^*}{\partial x_i} x_k
\]

where the second term is computed via **implicit differentiation** of the Sinkhorn fixed point.

#### Sorting Network Gradients

For differentiable sorting networks:
\[
\frac{\partial \tilde{\text{sort}}(x)}{\partial x_i} = \prod_{\text{comparators } c \text{ involving } i} \frac{\partial \tilde{\text{comp}}_c}{\partial x_i}
\]

computed via **backpropagation** through the network.

#### Isotonic Regression Gradients

For PAVA-based methods:
\[
\frac{\partial y_i}{\partial x_j} = \begin{cases}
\frac{w_j}{\sum_{k \in \text{pool}(i)} w_k} & \text{if } j \in \text{pool}(i) \\
0 & \text{otherwise}
\end{cases}
\]

---

## Complexity Analysis

### Time Complexity Comparison

| Method | Forward | Backward | Space | Notes |
|--------|---------|----------|-------|-------|
| **Naive Sigmoid** (current impl) | O(n²) | O(n²) | O(n) | Simple, intuitive |
| **Permutahedron Projection** | O(n log n) | O(n log n) | O(n) | Efficient, exact |
| **Optimal Transport (Sinkhorn)** | O(n²·k) | O(n²) | O(n²) | k = iterations (~10-50) |
| **Sorting Networks** | O(n log² n) | O(n log² n) | O(n) | Parallel-friendly |
| **NeuralSort/SoftSort** | O(n²) | O(n²) | O(n²) | Permutation matrices |

where \(n\) is the number of elements to rank/sort.

### Space Complexity

- **Naive Sigmoid**: O(n) - stores ranks only
- **Permutahedron**: O(n) - efficient projection
- **Optimal Transport**: O(n²) - transport plan matrix
- **Sorting Networks**: O(n) - intermediate values
- **NeuralSort/SoftSort**: O(n²) - permutation matrix

### Practical Considerations

**For small inputs (n < 100)**: All methods are fast enough. Choose based on accuracy needs.

**For medium inputs (100 < n < 1000)**: Permutahedron projection or sorting networks are preferred.

**For large inputs (n > 1000)**: Permutahedron projection is most efficient (O(n log n)).

**Current implementation**: Uses naive sigmoid approach (O(n²)), suitable for small-medium inputs. For larger inputs, consider implementing permutahedron projection.

---

## Worked Examples

### Example 1: Soft Ranking with Different Regularization

**Input**: \(x = [5.0, 1.0, 2.0, 4.0, 3.0]\)

**With α = 0.1** (low regularization):
- Ranks are very smooth: approximately [2.4, 2.1, 2.2, 2.3, 2.2]
- Poor discrimination between elements
- Good for early training (smooth gradients)

**With α = 1.0** (medium regularization):
- Ranks: approximately [4.0, 0.1, 1.2, 3.1, 2.0]
- Moderate discrimination
- Balanced for general use

**With α = 10.0** (high regularization):
- Ranks: approximately [4.0, 0.0, 1.0, 3.0, 2.0]
- Close to discrete ranks [4, 0, 1, 3, 2]
- Good for accurate ranking

**With α = 100.0** (very high regularization):
- Ranks: [4.0, 0.0, 1.0, 3.0, 2.0] (essentially discrete)
- Maximum accuracy, but potential numerical issues

### Example 2: Spearman Correlation Loss

**Predictions**: \(p = [0.1, 0.9, 0.3, 0.7, 0.5]\)
**Targets**: \(t = [0.0, 1.0, 0.2, 0.8, 0.4]\)

**Step 1**: Compute soft ranks (α = 10.0)
- pred_ranks ≈ [0.0, 4.0, 1.0, 3.0, 2.0]
- target_ranks ≈ [0.0, 4.0, 1.0, 3.0, 2.0]

**Step 2**: Compute Pearson correlation
- Both have same ranks → perfect correlation
- Spearman = 1.0

**Step 3**: Compute loss
- loss = 1 - 1.0 = 0.0 (perfect)

**With different predictions** \(p = [0.9, 0.1, 0.7, 0.3, 0.5]\):
- pred_ranks ≈ [4.0, 0.0, 3.0, 1.0, 2.0]
- target_ranks ≈ [0.0, 4.0, 1.0, 3.0, 2.0]
- Ranks are reversed → anti-correlation
- Spearman ≈ -1.0
- loss = 1 - (-1.0) = 2.0 (maximum loss)

---

## Summary of Mathematical Properties

| Method | Formulation | Differentiability | Convergence | Complexity |
|--------|-------------|-------------------|-------------|------------|
| **Optimal Transport** | \(\min_P \langle P, C \rangle - \epsilon H(P)\) | Everywhere | Exact (as \(\epsilon \to 0\)) | \(O(n^2 \cdot \text{iter})\) |
| **Permutahedron** | \(\arg\min_{y \in \mathcal{P}_n} \|y - x\|^2\) | Almost everywhere | Exact | \(O(n \log n)\) |
| **Sorting Networks** | Relaxed comparators | Everywhere | Exact (as \(\tau \to 0\)) | \(O(n \log^2 n)\) |
| **NeuralSort** | Softmax over distances | Everywhere | Approximate | \(O(n^2)\) |
| **SoftSort** | Softmax argsort | Everywhere | Approximate | \(O(n^2)\) |
| **LapSum** | Laplace quantiles | Everywhere | Exact (as \(\tau \to 0\)) | \(O(n \log n)\) |

---

## References

1. Cuturi, Teboul, Vert (2019). "Differentiable Ranks and Sorting using Optimal Transport". ICML 2019.
2. Blondel, Teboul, Berthet, Djolonga (2020). "Fast Differentiable Sorting and Ranking". ICML 2020.
3. Grover, Wang, Zweig, Ermon (2019). "Stochastic Optimization of Sorting Networks via Continuous Relaxations". ICML 2019.
4. Prillo, Eisenschlos (2020). "SoftSort: A Continuous Relaxation for the argsort Operator".
5. Petersen, Borgelt, Kuehne, Deussen (2021). "Differentiable Sorting Networks for Scalable Sorting and Ranking Supervision". ICML 2021.
6. Petersen, Borgelt, Kuehne, Deussen (2022). "Monotonic Differentiable Sorting Networks".
7. Struski, Bednarczyk, Podolak, Tabor (2025). "LapSum – One Method to Differentiate Them All: Ranking, Sorting and Top-k Selection".

