# Gumbel Reranking: Connection to rank-relax

**Paper**: "Gumbel Reranking: Differentiable End-to-End Reranker Optimization" (ACL 2025)
**Authors**: Siyuan Huang, Zhiyuan Ma, Jintao Du, et al.

## Summary

This paper proposes an end-to-end training framework for rerankers in RAG systems using **Gumbel-Softmax** and **Relaxed Top-k** techniques. It reformulates reranking as learning a differentiable attention mask, enabling direct optimization of language modeling loss.

## Key Techniques (Highly Relevant to rank-relax)

### 1. Gumbel-Softmax Trick

The paper uses the Gumbel trick to convert discrete document selection into a differentiable process:

```python
# Gumbel noise
G_i = -log(-log(u_i)), u_i ~ Uniform(0,1)

# Soft attention mask
M̂_i = softmax((G_i + κ·w_i) / τ)
```

Where:
- `w_i` = reranker score for document i
- `τ` = temperature parameter (controls sharpness)
- `κ` = scaling factor (controls reranker influence)

### 2. Relaxed Top-k Sampling

To approximate top-k selection, they sample k times independently and take element-wise maximum:

```python
# Sample k times
for j in range(k):
    M̂_j = softmax((G + κ·w) / τ)
    
# Relaxed top-k mask
M̂ = max(M̂_1, M̂_2, ..., M̂_k)
```

This creates a soft mask where top-k documents have high values (~1.0) and others have low values (~0.0).

### 3. Differentiable Masked Attention (DMA)

The soft mask is applied to attention computation:

```
DMA(Q_m, K_i,t) = (M̂_i · exp(Q_m K_i,t^T / √d_k)) / Σ
```

This enables end-to-end optimization: gradients flow from language loss → attention → mask → reranker.

## Connection to rank-relax

### What rank-relax Already Has

1. **`differentiable_topk()`** - Basic differentiable top-k selection
   - Current implementation: sigmoid-based soft mask
   - Location: `src/methods_advanced.rs`
   - **Gap**: Not Gumbel-based, doesn't use relaxed top-k sampling

2. **Soft ranking methods** - Multiple approaches (Sigmoid, NeuralSort, Probabilistic, SmoothI)
   - **Gap**: No Gumbel-Softmax variant

3. **Differentiable operations** - Core infrastructure for gradient flow
   - ✅ Already supports this

### What rank-relax Should Add

1. **Gumbel-Softmax Top-k Relaxation**
   ```rust
   pub fn gumbel_topk_relaxed(
       scores: &[f64],
       k: usize,
       temperature: f64,
       scale: f64,
   ) -> Vec<f64>
   ```

2. **Relaxed Top-k with Multiple Samples**
   ```rust
   pub fn relaxed_topk_gumbel(
       scores: &[f64],
       k: usize,
       n_samples: usize,  // Sample k times
       temperature: f64,
       scale: f64,
   ) -> Vec<f64>
   ```

3. **Attention Mask Generation**
   ```rust
   pub fn gumbel_attention_mask(
       reranker_scores: &[f64],
       k: usize,
       temperature: f64,
       scale: f64,
   ) -> Vec<f64>  // Soft mask values in [0, 1]
   ```

## Implementation Notes

### Gumbel Noise Generation

```rust
use rand::Rng;

fn gumbel_noise(rng: &mut impl Rng) -> f64 {
    let u: f64 = rng.gen_range(0.0..1.0);
    -(-u.ln()).ln()
}
```

### Gumbel-Softmax

```rust
fn gumbel_softmax(
    logits: &[f64],
    temperature: f64,
    scale: f64,
    rng: &mut impl Rng,
) -> Vec<f64> {
    let n = logits.len();
    let mut gumbel_logits = Vec::with_capacity(n);
    
    for &logit in logits {
        let g = gumbel_noise(rng);
        gumbel_logits.push((g + scale * logit) / temperature);
    }
    
    // Softmax
    softmax(&gumbel_logits)
}
```

### Relaxed Top-k

```rust
fn relaxed_topk_gumbel(
    scores: &[f64],
    k: usize,
    temperature: f64,
    scale: f64,
    rng: &mut impl Rng,
) -> Vec<f64> {
    let n = scores.len();
    let mut max_mask = vec![0.0; n];
    
    // Sample k times, take element-wise max
    for _ in 0..k {
        let mask = gumbel_softmax(scores, temperature, scale, rng);
        for i in 0..n {
            max_mask[i] = max_mask[i].max(mask[i]);
        }
    }
    
    max_mask
}
```

## Use Case: RAG Reranking

This technique is specifically designed for RAG systems where:

1. **No labeled data**: Only query-answer pairs, no document relevance labels
2. **End-to-end optimization**: Want to optimize reranker directly on language loss
3. **Multi-hop reasoning**: Need to capture document interdependencies
4. **Differentiable selection**: Need gradients to flow through document selection

### Example Workflow

```rust
use rank_relax::gumbel_attention_mask;

// 1. Reranker scores documents
let reranker_scores = vec![0.8, 0.6, 0.9, 0.3, 0.7];

// 2. Generate soft attention mask (top-3)
let attention_mask = gumbel_attention_mask(
    &reranker_scores,
    k: 3,
    temperature: 0.5,
    scale: 1.0,
);

// 3. Apply mask to attention computation
// (in LLM forward pass)
let masked_attention = apply_mask(attention, &attention_mask);

// 4. Compute language loss
let loss = language_model_loss(masked_attention, answer);

// 5. Backpropagate through mask → reranker
// Gradients flow through differentiable mask
```

## Comparison with Existing Methods

| Method | Differentiable | Captures Dependencies | End-to-End |
|--------|--------------|----------------------|------------|
| **Hard Top-k** | ❌ | ✅ | ❌ |
| **Sigmoid Top-k** (current) | ✅ | ❌ | ✅ |
| **Gumbel Top-k** (paper) | ✅ | ✅ | ✅ |

**Key Advantage**: Gumbel approach captures document interdependencies through relaxed top-k sampling, making it suitable for multi-hop QA.

## Experimental Results (from paper)

- **HotpotQA**: 10.4% improvement in Recall@5 for indirectly relevant documents
- **Multi-hop tasks**: Better than perplexity-based distillation methods
- **End-to-end**: Directly optimizes language loss (not LLM-supervised loss)

## Integration Path

1. **Add Gumbel utilities** to `src/methods_advanced.rs`
2. **Add relaxed top-k** variant using Gumbel-Softmax
3. **Document in README** as advanced method for RAG/reranking
4. **Add example** showing RAG reranking use case
5. **Benchmark** against existing `differentiable_topk()`

## References

- **Paper**: Huang et al. (2025). "Gumbel Reranking: Differentiable End-to-End Reranker Optimization". ACL 2025.
- **Gumbel-Softmax**: Jang et al. (2017). "Categorical Reparameterization with Gumbel-Softmax". ICLR 2017.
- **Relaxed Top-k**: Chen et al. (2018). "Learning to Explain: An Information-Theoretic Perspective on Model Interpretation". ICML 2018.

## Status

- ✅ **Identified**: Paper is highly relevant to rank-relax
- ✅ **Implemented**: Gumbel-Softmax top-k relaxation
- ✅ **Documented**: Added to RELATED_WORK.md and README
- ✅ **Example**: RAG reranking use case (`examples/gumbel_reranking.rs`)

## Implementation

The Gumbel-Softmax methods are available in rank-relax with the `gumbel` feature:

```bash
cargo add rank-relax --features gumbel
```

**Available Functions**:
- `gumbel_softmax()` - Gumbel-Softmax sampling
- `relaxed_topk_gumbel()` - Relaxed top-k with Gumbel
- `gumbel_attention_mask()` - Convenience function for RAG reranking

**Tests**: All functions are tested and passing (23 tests total). See `cargo test --features gumbel`.

**Test Coverage**:
- ✅ Unit tests (5 tests)
- ✅ Integration tests (8 tests) 
- ✅ Property tests (7 tests)
- ✅ Comparison tests (3 tests)

See [GUMBEL_TESTING.md](GUMBEL_TESTING.md) for comprehensive testing documentation.

