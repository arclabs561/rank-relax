#set page(margin: (x: 2.5cm, y: 2cm))
#set text(font: "New Computer Modern", size: 11pt)
#set heading(numbering: "1.")
#set par(justify: true, leading: 0.65em)

#show heading: set text(weight: "bold")

= rank-relax: Differentiable Ranking Documentation

#align(center)[
  #text(size: 14pt, weight: "bold")[Differentiable Sorting and Ranking Operations]
  
  #text(size: 10pt)[Using Smooth Relaxation for Rust ML Frameworks]
  
  #v(0.5cm)
  #text(size: 9pt, style: "italic")[Version 0.1.0]
]

== Introduction

`rank-relax` provides differentiable sorting and ranking operations using smooth relaxation techniques, enabling gradient flow through discrete ranking operations. This is essential for training ranking models end-to-end in machine learning frameworks like Candle and Burn.

== Features

- *Multiple Ranking Methods*: Sigmoid-based (default), NeuralSort-style, Probabilistic (SoftRank), SmoothI
- *Gumbel-Softmax Top-k*: Differentiable top-k selection using Gumbel trick for RAG reranking
- *True Differentiable Sorting*: Permutahedron projection via isotonic regression (O(n log n))
- *Analytical Gradients*: Efficient closed-form gradient computation (no numerical differentiation)
- *Batch Processing*: Parallel processing support for multiple rankings
- *Framework Agnostic*: Works with PyTorch, JAX, Julia, and Rust ML frameworks

== Quick Start

=== Rust

```rust
use rank_relax::{soft_rank, spearman_loss, RankingMethod};

// Soft ranking
let values = vec![5.0, 1.0, 2.0, 4.0, 3.0];
let ranks = soft_rank(&values, 1.0);

// Spearman correlation loss (for training)
let predictions = vec![0.1, 0.9, 0.3, 0.7, 0.5];
let targets = vec![0.0, 1.0, 0.2, 0.8, 0.4];
let loss = spearman_loss(&predictions, &targets, 1.0);
```

=== Gumbel-Softmax for RAG Reranking

```rust
use rank_relax::gumbel_attention_mask;
use rand::thread_rng;

let reranker_scores = vec![0.8, 0.6, 0.9, 0.3, 0.7];
let mut rng = thread_rng();

let attention_mask = gumbel_attention_mask(
    &reranker_scores,
    3,      // top-k
    0.5,    // temperature (τ)
    1.0,    // scale (κ)
    &mut rng,
);
```

== Gumbel Reranking

The Gumbel-Softmax top-k relaxation enables end-to-end training of rerankers in RAG systems without labeled relevance data. This implementation is based on "Gumbel Reranking: Differentiable End-to-End Reranker Optimization" (ACL 2025).

=== Key Techniques

#v(0.3em)
- Gumbel-Softmax Trick: Converts discrete document selection into differentiable operation
- Relaxed Top-k: Samples k times independently, takes element-wise maximum
- Differentiable Masked Attention: Enables gradient flow from language loss to reranker

=== Use Cases

- End-to-end reranker training in RAG systems
- Multi-hop question answering
- Document retrieval with interdependencies
- Training without labeled relevance data

=== Implementation Details

The Gumbel-Softmax method uses the Gumbel trick to convert discrete sampling into a differentiable process:

$ G_i = -log(-log(U_i)), quad U_i ~ "Uniform"(0,1) $

$ hat(M)_i = "softmax"((G_i + kappa * w_i) / tau) $

For relaxed top-k, we sample k times and take the element-wise maximum:

$ hat(M) = max(hat(M)_1, hat(M)_2, ..., hat(M)_k) $

This creates a soft mask where top-k documents have high values (~1.0) and others have low values (~0.0), while remaining fully differentiable.

=== Experimental Results

From the paper:
- HotpotQA: 10.4% improvement in Recall at 5 for indirectly relevant documents
- Multi-hop tasks: Better than perplexity-based distillation methods
- End-to-end: Directly optimizes language loss (not LLM-supervised loss)

== Algorithm Details

=== Soft Ranking

The rank of element `i` is computed as:

$ "rank"[i] = (1/(n-1)) * sum_("j" != "i") "sigmoid"("alpha" * ("values"["i"] - "values"["j"])) $

where alpha = regularization_strength controls the sharpness of the sigmoid.

=== Gumbel-Softmax

For Gumbel-Softmax, we add Gumbel noise to logits:

$ G_i = -log(-log(U_i)), quad U_i ~ "Uniform"(0,1) $

$ hat(M)_i = "softmax"((G_i + kappa * w_i) / tau) $

where:
- w_i = reranker score for document i
- tau = temperature parameter: controls sharpness
- kappa = scaling factor: controls reranker influence

== Performance

Benchmark results (on typical hardware):

- *Forward pass*: < 1ms for n=1000
- *Backward pass*: < 2ms for n=1000 (analytical gradients)
- *Batch processing*: Linear scaling with batch size
- *Memory*: O(n²) for gradient matrix (can use sparse for O(n))

== Installation

```bash
# Basic installation
cargo add rank-relax

# With Gumbel feature
cargo add rank-relax --features gumbel
```

== Examples

See the `examples/` directory for:

- `gumbel_reranking.rs`: Basic Gumbel-Softmax usage
- `gumbel_rag_training.rs`: Complete RAG training simulation
- `gumbel_vs_sigmoid_comparison.rs`: Method comparison
- `complete_workflow.rs`: Full ranking workflow
- `candle_training.rs`: Candle framework integration

== Testing

All functions are thoroughly tested:

- Unit tests (5 tests)
- Integration tests (8 tests)
- Property tests (7 tests)
- Comparison tests (3 tests)

Run tests: cargo test --features gumbel

== Documentation

- `docs/GUMBEL_RERANKING.md`: Gumbel implementation details
- `docs/GUMBEL_TESTING.md`: Testing documentation
- `docs/RELATED_WORK.md`: Research paper connections
- `USAGE_EXAMPLES.md`: Practical usage examples

== References

=== Primary Sources

- Huang, S., Ma, Z., Du, J., et al. (2025). "Gumbel Reranking: Differentiable End-to-End Reranker Optimization". In *Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics* (ACL 2025).

- Jang, E., Gu, S., & Poole, B. (2017). "Categorical Reparameterization with Gumbel-Softmax". In *5th International Conference on Learning Representations* (ICLR 2017).

- Grover, A., Wang, E., Zweig, A., & Ermon, S. (2019). "Stochastic Optimization of Sorting Networks via Continuous Relaxations". In *7th International Conference on Learning Representations* (ICLR 2019).

=== Related Work

- Blondel, M., Teboul, O., Berthet, Q., & Djolonga, J. (2020). "Fast Differentiable Sorting and Ranking". In *Proceedings of the 37th International Conference on Machine Learning* (ICML 2020).

- Cuturi, M., Teboul, O., & Vert, J. P. (2019). "Differentiable Ranks and Sorting using Optimal Transport". In *Proceedings of the 36th International Conference on Machine Learning* (ICML 2019).

- Petersen, F., Borgelt, C., Kuehne, H., & Deussen, O. (2021). "Differentiable Sorting Networks for Scalable Sorting and Ranking Supervision". In *Proceedings of the 38th International Conference on Machine Learning* (ICML 2021).

For a comprehensive survey, see `docs/RELATED_WORK.md`.

== License

MIT OR Apache-2.0

