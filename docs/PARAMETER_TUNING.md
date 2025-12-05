# Parameter Tuning Guide

Practical guidance for choosing `regularization_strength` and understanding its effects.

## Understanding `regularization_strength`

The `regularization_strength` parameter (also called "temperature" or "sharpness") controls how close the soft ranking is to discrete ranking:

- **Low values**: Smooth, differentiable, but less accurate
- **High values**: Sharper, more accurate, but potentially less stable gradients

## Rule of Thumb

**Match the parameter to the scale of differences in your values:**

```
regularization_strength ≈ 1.0 / typical_difference_between_values
```

### Examples

**Example 1: Values differ by ~1.0**
```rust
let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
// Differences are ~1.0, so use regularization_strength ≈ 1.0
let ranks = soft_rank(&values, 1.0);
```

**Example 2: Values differ by ~0.1**
```rust
let values = vec![0.1, 0.2, 0.3, 0.4, 0.5];
// Differences are ~0.1, so use regularization_strength ≈ 10.0
let ranks = soft_rank(&values, 10.0);
```

**Example 3: Values differ by ~10.0**
```rust
let values = vec![10.0, 20.0, 30.0, 40.0, 50.0];
// Differences are ~10.0, so use regularization_strength ≈ 0.1
let ranks = soft_rank(&values, 0.1);
```

## Practical Guidelines

### For Training

**Start with medium values (1.0-10.0)** and adjust based on:

1. **Gradient quality**: If gradients are too small, reduce `regularization_strength`
2. **Ranking accuracy**: If ranks are too soft, increase `regularization_strength`
3. **Training stability**: If training is unstable, reduce `regularization_strength`

### Adaptive Strategy

Some practitioners use **annealing**: start with low values and gradually increase:

```rust
// Early training: smooth gradients
let reg_strength = 0.5;

// Mid training: balanced
let reg_strength = 1.0;

// Late training: sharper, more accurate
let reg_strength = 10.0;
```

### For Different Use Cases

**Spearman correlation loss:**
- Use medium-high values (5.0-20.0) for accurate rank computation
- The loss depends on ranking accuracy, so sharper is usually better

**Gradient-based optimization:**
- Use medium values (1.0-5.0) for stable gradients
- Too high can cause vanishing gradients
- Too low can cause inaccurate rankings

## Visual Guide

```
regularization_strength = 0.1
  ────────────────────────────────────────
  Smooth, very differentiable
  Ranks: [2.1, 2.3, 2.5, 2.7, 2.9]  (soft)
  
regularization_strength = 1.0
  ────────────────────────────────────────
  Balanced
  Ranks: [0.1, 1.2, 2.0, 3.1, 4.0]  (moderate)
  
regularization_strength = 10.0
  ────────────────────────────────────────
  Sharp, close to discrete
  Ranks: [0.0, 1.0, 2.0, 3.0, 4.0]  (nearly discrete)
```

## Common Mistakes

### ❌ Too Low

```rust
// BAD: Too low for the scale of values
let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
let ranks = soft_rank(&values, 0.01);  // Too low!
// Result: All ranks are very similar (~2.0), poor discrimination
```

### ❌ Too High

```rust
// BAD: Too high can cause numerical issues
let values = vec![0.1, 0.2, 0.3, 0.4, 0.5];
let ranks = soft_rank(&values, 1000.0);  // Too high!
// Result: Potential overflow, unstable gradients
```

### ✅ Just Right

```rust
// GOOD: Matched to scale
let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
let ranks = soft_rank(&values, 1.0);  // Good!
// Result: Clear ranking with smooth gradients
```

## Testing Your Choice

**Quick test**: Compare soft ranks to expected discrete ranks:

```rust
let values = vec![5.0, 1.0, 2.0, 4.0, 3.0];
let ranks = soft_rank(&values, regularization_strength);

// Expected discrete ranks: [4, 0, 1, 3, 2]
// Check if soft ranks are close:
assert!((ranks[0] - 4.0).abs() < 0.5);  // Should be close to 4.0
assert!((ranks[1] - 0.0).abs() < 0.5);  // Should be close to 0.0
// ... etc
```

If soft ranks are too far from discrete ranks, increase `regularization_strength`.
If they're close but gradients are unstable, decrease it slightly.

## Summary

| Value Range | Use Case | Characteristics |
|------------|----------|----------------|
| 0.1 - 1.0 | Early training, smooth gradients | Very smooth, less accurate |
| 1.0 - 10.0 | General use, balanced | Good balance |
| 10.0 - 100.0 | Accurate ranking, late training | Sharp, accurate, may be less stable |

**Remember**: There's no single "best" value. Choose based on your data scale and training needs.

