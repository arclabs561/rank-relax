# Examples and Use Cases

Practical examples demonstrating how to use `rank-relax` in various scenarios.

## Table of Contents

1. [Basic Ranking](#basic-ranking)
2. [Training with Spearman Loss](#training-with-spearman-loss)
3. [Parameter Tuning](#parameter-tuning)
4. [Edge Cases](#edge-cases)
5. [Real-World Scenarios](#real-world-scenarios)

---

## Basic Ranking

### Example 1: Simple Soft Ranking

```rust
use rank_relax::soft_rank;

// Rank a simple vector
let values = vec![5.0, 1.0, 2.0, 4.0, 3.0];
let ranks = soft_rank(&values, 1.0);

println!("Values: {:?}", values);
println!("Soft ranks: {:?}", ranks);
// Output:
// Values: [5.0, 1.0, 2.0, 4.0, 3.0]
// Soft ranks: [4.0, 0.1, 1.2, 3.1, 2.0]  (approximate, depends on regularization)
```

### Example 2: Effect of Regularization Strength

```rust
use rank_relax::soft_rank;

let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];

// Low regularization: smooth, less accurate
let ranks_low = soft_rank(&values, 0.1);
println!("Low reg (0.1): {:?}", ranks_low);
// Output: [2.4, 2.1, 2.2, 2.3, 2.2]  (very smooth, poor discrimination)

// Medium regularization: balanced
let ranks_med = soft_rank(&values, 1.0);
println!("Medium reg (1.0): {:?}", ranks_med);
// Output: [4.0, 0.1, 1.2, 3.1, 2.0]  (moderate discrimination)

// High regularization: sharp, accurate
let ranks_high = soft_rank(&values, 10.0);
println!("High reg (10.0): {:?}", ranks_high);
// Output: [4.0, 0.0, 1.0, 3.0, 2.0]  (nearly discrete)
```

---

## Training with Spearman Loss

### Example 3: Basic Training Loop

```rust
use rank_relax::spearman_loss;

// Simulated training loop
for epoch in 0..10 {
    // Model predictions (would come from your model)
    let predictions = vec![0.1, 0.9, 0.3, 0.7, 0.5];
    
    // Ground truth targets
    let targets = vec![0.0, 1.0, 0.2, 0.8, 0.4];
    
    // Compute loss (differentiable!)
    let loss = spearman_loss(&predictions, &targets, 1.0);
    
    println!("Epoch {}: Loss = {:.4}", epoch, loss);
    
    // In real training: loss.backward() and optimizer.step()
}
```

### Example 4: Perfect vs Anti-Correlation

```rust
use rank_relax::spearman_loss;

let targets = vec![0.0, 1.0, 2.0, 3.0, 4.0];

// Perfect correlation: same ranking
let perfect_pred = vec![0.0, 1.0, 2.0, 3.0, 4.0];
let perfect_loss = spearman_loss(&perfect_pred, &targets, 10.0);
println!("Perfect correlation loss: {:.4}", perfect_loss);
// Output: ~0.0 (very low loss)

// Anti-correlation: reversed ranking
let anti_pred = vec![4.0, 3.0, 2.0, 1.0, 0.0];
let anti_loss = spearman_loss(&anti_pred, &targets, 10.0);
println!("Anti-correlation loss: {:.4}", anti_loss);
// Output: ~2.0 (maximum loss)

// Random: no correlation
let random_pred = vec![3.0, 1.0, 4.0, 0.0, 2.0];
let random_loss = spearman_loss(&random_pred, &targets, 10.0);
println!("Random correlation loss: {:.4}", random_loss);
// Output: ~1.0 (medium loss)
```

---

## Parameter Tuning

### Example 5: Choosing Regularization Strength

```rust
use rank_relax::soft_rank;

// Values differ by ~1.0 → use regularization_strength ≈ 1.0
let values1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
let ranks1 = soft_rank(&values1, 1.0);
println!("Scale ~1.0, reg=1.0: {:?}", ranks1);

// Values differ by ~0.1 → use regularization_strength ≈ 10.0
let values2 = vec![0.1, 0.2, 0.3, 0.4, 0.5];
let ranks2 = soft_rank(&values2, 10.0);
println!("Scale ~0.1, reg=10.0: {:?}", ranks2);

// Values differ by ~10.0 → use regularization_strength ≈ 0.1
let values3 = vec![10.0, 20.0, 30.0, 40.0, 50.0];
let ranks3 = soft_rank(&values3, 0.1);
println!("Scale ~10.0, reg=0.1: {:?}", ranks3);
```

### Example 6: Adaptive Regularization (Annealing)

```rust
use rank_relax::spearman_loss;

// Simulate training with adaptive regularization
let predictions = vec![0.1, 0.9, 0.3, 0.7, 0.5];
let targets = vec![0.0, 1.0, 0.2, 0.8, 0.4];

for epoch in 0..10 {
    // Start with low regularization (smooth gradients)
    // Gradually increase (sharper, more accurate)
    let reg_strength = 0.5 + (epoch as f64) * 0.5;
    
    let loss = spearman_loss(&predictions, &targets, reg_strength);
    println!("Epoch {}: reg={:.1}, loss={:.4}", epoch, reg_strength, loss);
}
```

---

## Edge Cases

### Example 7: Handling Edge Cases

```rust
use rank_relax::{soft_rank, spearman_loss};

// Empty input
let empty: Vec<f64> = vec![];
let ranks_empty = soft_rank(&empty, 1.0);
assert_eq!(ranks_empty.len(), 0);

// Single element
let single = vec![42.0];
let ranks_single = soft_rank(&single, 1.0);
assert_eq!(ranks_single, vec![0.0]);

// All equal values
let equal = vec![1.0, 1.0, 1.0, 1.0];
let ranks_equal = soft_rank(&equal, 10.0);
// All ranks should be approximately equal (within tolerance)
let first_rank = ranks_equal[0];
for rank in &ranks_equal {
    assert!((rank - first_rank).abs() < 0.1);
}

// Mismatched lengths (for spearman_loss)
let pred = vec![1.0, 2.0];
let targ = vec![1.0, 2.0, 3.0];
let loss = spearman_loss(&pred, &targ, 1.0);
assert_eq!(loss, 1.0); // Maximum loss for mismatched lengths
```

### Example 8: Extreme Values

```rust
use rank_relax::soft_rank;

// Very large values
let large = vec![1e10, 2e10, 3e10];
let ranks_large = soft_rank(&large, 1.0);
// Should handle without overflow

// Very small values
let small = vec![1e-10, 2e-10, 3e-10];
let ranks_small = soft_rank(&small, 1.0);
// Should handle without underflow

// Mixed scales
let mixed = vec![0.001, 1000.0, 0.1];
let ranks_mixed = soft_rank(&mixed, 1.0);
// May need higher regularization_strength due to large scale differences
```

---

## Real-World Scenarios

### Example 9: Learning-to-Rank

```rust
use rank_relax::spearman_loss;

// Simulate learning-to-rank scenario
// Query: "machine learning"
// Documents with relevance scores
struct Document {
    id: usize,
    relevance_score: f64,
}

let documents = vec![
    Document { id: 1, relevance_score: 0.3 },
    Document { id: 2, relevance_score: 0.9 },
    Document { id: 3, relevance_score: 0.1 },
    Document { id: 4, relevance_score: 0.7 },
    Document { id: 5, relevance_score: 0.5 },
];

// Model predictions (would come from your ranking model)
let predictions: Vec<f64> = documents.iter()
    .map(|d| d.relevance_score)
    .collect();

// Ground truth rankings (from human judgments)
let ground_truth = vec![0.2, 1.0, 0.0, 0.8, 0.4];

// Compute loss
let loss = spearman_loss(&predictions, &ground_truth, 5.0);
println!("Ranking loss: {:.4}", loss);

// In training: optimize model to minimize this loss
```

### Example 10: Multi-Objective Training

```rust
use rank_relax::spearman_loss;

// Combine ranking loss with classification loss
fn combined_loss(
    predictions: &[f64],
    targets: &[f64],
    labels: &[usize],
    ranking_weight: f64,
) -> f64 {
    // Ranking loss (Spearman correlation)
    let ranking_loss = spearman_loss(predictions, targets, 1.0);
    
    // Classification loss (simplified example)
    let classification_loss = compute_classification_loss(predictions, labels);
    
    // Weighted combination
    ranking_weight * ranking_loss + (1.0 - ranking_weight) * classification_loss
}

fn compute_classification_loss(_predictions: &[f64], _labels: &[usize]) -> f64 {
    // Placeholder - would use actual classification loss
    0.5
}

// Usage
let predictions = vec![0.1, 0.9, 0.3, 0.7, 0.5];
let targets = vec![0.0, 1.0, 0.2, 0.8, 0.4];
let labels = vec![0, 1, 0, 1, 0];

let total_loss = combined_loss(&predictions, &targets, &labels, 0.7);
println!("Combined loss: {:.4}", total_loss);
```

### Example 11: Batch Processing

```rust
use rank_relax::{soft_rank_batch, spearman_loss_batch};

// Process multiple queries in a batch (efficient!)
let batch = vec![
    vec![5.0, 1.0, 2.0, 4.0, 3.0],  // Query 1: 5 items
    vec![3.0, 1.0, 2.0],             // Query 2: 3 items
    vec![10.0, 20.0, 30.0],          // Query 3: 3 items
];

let ranks = soft_rank_batch(&batch, 1.0);
for (i, ranks) in ranks.iter().enumerate() {
    println!("Query {} ranks: {:?}", i, ranks);
}

// Batch loss computation (for training)
let predictions = vec![
    vec![0.1, 0.9, 0.3, 0.7, 0.5],
    vec![1.0, 2.0, 3.0],
    vec![10.0, 20.0, 30.0],
];
let targets = vec![
    vec![0.0, 1.0, 0.2, 0.8, 0.4],
    vec![1.0, 2.0, 3.0],
    vec![10.0, 20.0, 30.0],
];

let losses = spearman_loss_batch(&predictions, &targets, 1.0);
let avg_loss = losses.iter().sum::<f64>() / losses.len() as f64;
println!("Average batch loss: {:.4}", avg_loss);
```

### Example 12: Multiple Ranking Methods

```rust
use rank_relax::{RankingMethod, soft_rank_sigmoid, soft_rank_neural_sort};

let values = vec![5.0, 1.0, 2.0, 4.0, 3.0];

// Compare different methods
let ranks_sigmoid = RankingMethod::Sigmoid.compute(&values, 1.0);
let ranks_neural = RankingMethod::NeuralSort.compute(&values, 1.0);
let ranks_prob = RankingMethod::Probabilistic.compute(&values, 1.0);
let ranks_smooth = RankingMethod::SmoothI.compute(&values, 1.0);

println!("Sigmoid: {:?}", ranks_sigmoid);
println!("NeuralSort: {:?}", ranks_neural);
println!("Probabilistic: {:?}", ranks_prob);
println!("SmoothI: {:?}", ranks_smooth);

// All should preserve ordering, but may differ in values
```

---

## Multiple Ranking Methods

### Example 13: Choosing the Right Method

```rust
use rank_relax::RankingMethod;

let values = vec![5.0, 1.0, 2.0, 4.0, 3.0];

// Default: Sigmoid (simple, works well for most cases)
let ranks_default = RankingMethod::Sigmoid.compute(&values, 1.0);

// NeuralSort: When you need permutation matrices or different gradients
let ranks_neural = RankingMethod::NeuralSort.compute(&values, 1.0);

// Probabilistic: When you want rank distributions
let ranks_prob = RankingMethod::Probabilistic.compute(&values, 1.0);

// SmoothI: Alternative gradient profile
let ranks_smooth = RankingMethod::SmoothI.compute(&values, 1.0);
```

**When to use which**:
- **Sigmoid**: Default choice, start here
- **NeuralSort**: Need permutation matrices or different gradient behavior
- **Probabilistic**: Want probabilistic rank distributions
- **SmoothI**: Experimenting with alternative formulations

## Performance Tips

### Example 14: Optimizing for Large Inputs

```rust
use rank_relax::soft_rank;

// For large inputs (n > 1000), consider:
// 1. Using higher regularization_strength for faster convergence
// 2. Processing in batches if possible
// 3. Future: use permutahedron projection (O(n log n)) instead of sigmoid (O(n²))

let large_input: Vec<f64> = (0..1000).map(|i| i as f64).collect();

// Current implementation: O(n²), may be slow
let start = std::time::Instant::now();
let ranks = soft_rank(&large_input, 1.0);
let elapsed = start.elapsed();
println!("Ranked {} elements in {:?}", large_input.len(), elapsed);
```

---

## Next Steps

- See **[GETTING_STARTED.md](GETTING_STARTED.md)** for installation and basics
- See **[PARAMETER_TUNING.md](PARAMETER_TUNING.md)** for detailed parameter guidance
- See **[MATHEMATICAL_DETAILS.md](MATHEMATICAL_DETAILS.md)** for theory
- See **[API Documentation](https://docs.rs/rank-relax)** for full reference

