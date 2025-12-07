# Gumbel-Softmax Usage Examples

Practical examples demonstrating Gumbel-Softmax for RAG reranking.

## Quick Start

```bash
# Enable gumbel feature
cargo add rank-relax --features gumbel

# Run examples
cargo run --example gumbel_reranking --features gumbel
cargo run --example gumbel_rag_training --features gumbel
cargo run --example gumbel_vs_sigmoid_comparison --features gumbel
```

## Example 1: Basic Reranking

```rust
use rank_relax::gumbel_attention_mask;
use rand::thread_rng;

let reranker_scores = vec![0.9, 0.8, 0.7, 0.6, 0.5];
let mut rng = thread_rng();

let mask = gumbel_attention_mask(&reranker_scores, 3, 0.5, 1.0, &mut rng);
// mask[i] ≈ 1.0 for top-3, ≈ 0.0 for others
```

## Example 2: RAG Training Loop

See `examples/gumbel_rag_training.rs` for a complete RAG training simulation showing:
- Document retrieval
- Reranker scoring
- Gumbel mask generation
- Attention masking
- Language loss computation
- Gradient-based reranker updates

## Example 3: Comparison with Sigmoid

See `examples/gumbel_vs_sigmoid_comparison.rs` for side-by-side comparison:
- Deterministic vs stochastic behavior
- Exploration properties
- When to use each method

## Visualization

Generate statistical analysis:
```bash
uv run hack/viz/generate_gumbel_analysis.py
```

Produces:
- `gumbel_analysis.png` - Comprehensive statistical analysis
- `gumbel_vs_sigmoid.png` - Comparison visualization

## Integration with ML Frameworks

### PyTorch (via Python bindings)

```python
import rank_relax
import torch
import numpy as np

# Reranker scores (from your reranker model)
scores = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5], requires_grad=True)

# Generate mask (would need Python bindings for Gumbel)
# For now, use Rust implementation and convert
mask_np = rank_relax.relaxed_topk_gumbel(
    scores.detach().numpy().tolist(),
    k=3,
    temperature=0.5,
    scale=1.0,
    seed=42
)
mask = torch.tensor(mask_np, requires_grad=True)

# Apply to attention
attention = compute_attention(query, documents)  # Your LLM attention
masked_attention = attention * mask.unsqueeze(0)

# Compute loss
loss = language_model_loss(masked_attention, answer)
loss.backward()  # Gradients flow to mask → reranker
```

### Candle (Rust)

```rust
use candle_core::{Tensor, Device};
use rank_relax::gumbel_attention_mask;

// Reranker scores as Candle tensor
let scores_tensor: Tensor = /* your reranker output */;
let scores_vec: Vec<f64> = scores_tensor.to_vec1::<f64>()?;

// Generate mask
let mut rng = thread_rng();
let mask = gumbel_attention_mask(&scores_vec, k, 0.5, 1.0, &mut rng);

// Convert back to tensor
let mask_tensor = Tensor::from_vec(mask, (scores_vec.len(),), &Device::Cpu)?;

// Apply to attention
let masked_attention = attention.broadcast_mul(&mask_tensor)?;
```

## Parameter Tuning

### Temperature (τ)

- **Low (0.1-0.3)**: Sharp selection, less exploration
- **Medium (0.5)**: Balanced (paper default)
- **High (1.0-2.0)**: Smooth selection, more exploration

### Scale (κ)

- **Low (0.1-0.5)**: More random, Gumbel noise dominates
- **Medium (1.0)**: Balanced (paper default)
- **High (2.0-3.0)**: More deterministic, reranker scores dominate

### k (top-k)

- Choose based on your use case
- Typical: 3-10 for RAG reranking
- Larger k = more documents, more computation

## Best Practices

1. **Start with defaults**: `temperature=0.5`, `scale=1.0`
2. **Use fixed seeds during training** for reproducibility
3. **Monitor selection frequency** to ensure exploration
4. **Compare with sigmoid** to verify Gumbel provides benefit
5. **Tune temperature** based on convergence behavior

## Troubleshooting

**Problem**: Mask values too uniform
- **Solution**: Lower temperature or increase scale

**Problem**: Too much randomness
- **Solution**: Increase scale or lower temperature

**Problem**: Not selecting relevant documents
- **Solution**: Check reranker scores, may need more training

## See Also

- `docs/GUMBEL_RERANKING.md` - Full implementation details
- `docs/GUMBEL_TESTING.md` - Testing documentation
- `rank-rank/docs/RESEARCH_CONNECTIONS.md` - Research paper connection

