//! Example: Gumbel Reranking for RAG Systems
//!
//! This example demonstrates how to use Gumbel-Softmax top-k relaxation
//! for end-to-end reranker training in RAG systems, as described in:
//! "Gumbel Reranking: Differentiable End-to-End Reranker Optimization" (ACL 2025)
//!
//! Run with: cargo run --example gumbel_reranking --features gumbel

#[cfg(not(feature = "gumbel"))]
fn main() {
    eprintln!("This example requires the 'gumbel' feature.");
    eprintln!("Run with: cargo run --example gumbel_reranking --features gumbel");
}

#[cfg(feature = "gumbel")]
fn main() {
    use rank_relax::{gumbel_attention_mask, relaxed_topk_gumbel};
    use rand::thread_rng;

    println!("Gumbel Reranking Example: RAG End-to-End Training");
    println!("{}", "=".repeat(60));

    // Simulate reranker scores for candidate documents
    let reranker_scores = vec![
        0.8,  // doc1: highly relevant
        0.6,  // doc2: moderately relevant
        0.9,  // doc3: very relevant
        0.3,  // doc4: low relevance
        0.7,  // doc5: moderately relevant
        0.2,  // doc6: very low relevance
    ];

    println!("\nReranker Scores:");
    for (i, &score) in reranker_scores.iter().enumerate() {
        println!("  doc{}: {:.2}", i + 1, score);
    }

    let mut rng = thread_rng();
    let k = 3; // Select top-3 documents
    let temperature = 0.5; // Paper default
    let scale = 1.0; // Paper default

    // Generate soft attention mask using Gumbel-Softmax
    let attention_mask = gumbel_attention_mask(
        &reranker_scores,
        k,
        temperature,
        scale,
        &mut rng,
    );

    println!("\nSoft Attention Mask (Top-{}):", k);
    for (i, &mask) in attention_mask.iter().enumerate() {
        let selected = if mask > 0.5 { "✓" } else { " " };
        println!("  doc{}: {:.3} {}", i + 1, mask, selected);
    }

    // Show how mask would be applied to attention
    println!("\nMasked Attention (conceptual):");
    println!("  Before: attention[i] = exp(Q·K_i^T / √d_k)");
    println!("  After:  attention[i] = mask[i] · exp(Q·K_i^T / √d_k)");
    println!("\n  This enables end-to-end optimization:");
    println!("    language_loss → attention → mask → reranker");

    // Demonstrate multiple samples (relaxed top-k)
    println!("\nRelaxed Top-k (sampling {} times, taking max):", k);
    let relaxed_mask = relaxed_topk_gumbel(
        &reranker_scores,
        k,
        temperature,
        scale,
        &mut rng,
    );

    for (i, &mask) in relaxed_mask.iter().enumerate() {
        println!("  doc{}: {:.3}", i + 1, mask);
    }

    println!("\nKey Properties:");
    println!("  ✓ Differentiable: gradients flow through mask");
    println!("  ✓ Captures dependencies: relaxed top-k considers document interactions");
    println!("  ✓ End-to-end: directly optimizes language loss");
    println!("  ✓ No labeled data: only needs query-answer pairs");

    println!("\nSee docs/GUMBEL_RERANKING.md for full details.");
}

