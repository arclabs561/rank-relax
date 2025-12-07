//! Comparison: Gumbel vs Sigmoid Top-k Selection
//!
//! Demonstrates the differences between Gumbel-Softmax and sigmoid-based
//! top-k selection methods, showing when each is more appropriate.
//!
//! Run with: cargo run --example gumbel_vs_sigmoid_comparison --features gumbel

#[cfg(not(feature = "gumbel"))]
fn main() {
    eprintln!("This example requires the 'gumbel' feature.");
    eprintln!("Run with: cargo run --example gumbel_vs_sigmoid_comparison --features gumbel");
}

#[cfg(feature = "gumbel")]
fn main() {
    use rank_relax::{relaxed_topk_gumbel, differentiable_topk};
    use rand::{SeedableRng, Rng};
    use rand::rngs::StdRng;

    println!("Gumbel vs Sigmoid Top-k Comparison");
    println!("{}", "=".repeat(60));
    println!();

    let scores = vec![0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0];
    let k = 3;

    println!("Input Scores: {:?}", scores);
    println!("Selecting top-{}", k);
    println!();

    // Sigmoid-based (deterministic)
    let (sigmoid_vals, sigmoid_ranks) = differentiable_topk(&scores, k, 1.0);
    let sigmoid_mask: Vec<f64> = sigmoid_vals.iter()
        .zip(scores.iter())
        .map(|(&val, &score)| if score.abs() > 1e-10 { val / score } else { 0.0 })
        .collect();

    println!("Sigmoid-based Top-k (deterministic):");
    for (i, &mask) in sigmoid_mask.iter().enumerate() {
        let selected = if mask > 0.5 { "✓" } else { " " };
        println!("  [{}] score={:.1}, mask={:.3} {}", i, scores[i], mask, selected);
    }
    println!();

    // Gumbel-based (stochastic)
    println!("Gumbel-Softmax Top-k (stochastic, 3 runs):");
    for run in 0..3 {
        let mut rng = StdRng::seed_from_u64(100 + run);
        let gumbel_mask = relaxed_topk_gumbel(&scores, k, 0.5, 1.0, &mut rng);
        
        println!("  Run {}:", run + 1);
        for (i, &mask) in gumbel_mask.iter().enumerate() {
            let selected = if mask > 0.5 { "✓" } else { " " };
            println!("    [{}] score={:.1}, mask={:.3} {}", i, scores[i], mask, selected);
        }
        println!();
    }

    // Compare properties
    println!("Comparison:");
    println!("{}", "-".repeat(60));
    println!("Property              | Sigmoid    | Gumbel");
    println!("{}", "-".repeat(60));
    println!("Differentiable       | ✅ Yes     | ✅ Yes");
    println!("Deterministic        | ✅ Yes     | ❌ No (stochastic)");
    println!("Exploration          | ❌ Limited | ✅ Good");
    println!("Document Dependencies| ❌ No      | ✅ Yes (relaxed top-k)");
    println!("End-to-end Training  | ✅ Yes     | ✅ Yes");
    println!("Multi-hop Reasoning  | ⚠️  Limited| ✅ Better");
    println!();

    // Show exploration advantage
    println!("Exploration Analysis (10 runs with different seeds):");
    let mut gumbel_results = Vec::new();
    for seed in 0..10 {
        let mut rng = StdRng::seed_from_u64(seed);
        let mask = relaxed_topk_gumbel(&scores, k, 0.5, 1.0, &mut rng);
        gumbel_results.push(mask);
    }

    // Calculate variance (measure of exploration)
    let variance: Vec<f64> = (0..scores.len())
        .map(|i| {
            let values: Vec<f64> = gumbel_results.iter().map(|m| m[i]).collect();
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            values.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64
        })
        .collect();

    println!("  Variance per document:");
    for (i, &var) in variance.iter().enumerate() {
        println!("    [{}] variance={:.4}", i, var);
    }
    println!("  → Higher variance = more exploration");
    println!();

    println!("When to Use Each:");
    println!("  Sigmoid: Fast, deterministic, when exploration not needed");
    println!("  Gumbel:  Better for multi-hop reasoning, when exploration helps");
    println!("           (e.g., RAG reranking with document interdependencies)");
}

