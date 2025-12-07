//! Comparison tests: Gumbel vs Sigmoid top-k
//!
//! Validates that Gumbel method provides different behavior than sigmoid-based approach.

#[cfg(feature = "gumbel")]
mod tests {
    use rank_relax::{relaxed_topk_gumbel, differentiable_topk};
    use rand::{SeedableRng, Rng};
    use rand::rngs::StdRng;

    #[test]
    fn test_gumbel_vs_sigmoid_difference() {
        // Gumbel and sigmoid should produce different results
        // (they use different mechanisms)
        let scores = vec![0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2];
        let k = 3;

        // Sigmoid-based (existing method)
        let (sigmoid_vals, _) = differentiable_topk(&scores, k, 1.0);
        let sigmoid_mask: Vec<f64> = sigmoid_vals.iter()
            .zip(scores.iter())
            .map(|(&val, &score)| if score > 0.0 { val / score } else { 0.0 })
            .collect();

        // Gumbel-based (new method)
        let mut rng = StdRng::seed_from_u64(42);
        let gumbel_mask = relaxed_topk_gumbel(&scores, k, 0.5, 1.0, &mut rng);

        // They should be different (different algorithms)
        let mut differences = 0;
        for i in 0..scores.len() {
            if (sigmoid_mask[i] - gumbel_mask[i]).abs() > 0.01 {
                differences += 1;
            }
        }

        // Should have some differences (they're different methods)
        assert!(differences > 0, "Gumbel and sigmoid should produce different results");
    }

    #[test]
    fn test_gumbel_exploration_advantage() {
        // Gumbel should provide better exploration during training
        // (stochastic sampling vs deterministic sigmoid)
        let scores = vec![0.9, 0.8, 0.7, 0.6, 0.5];
        let k = 2;

        // Run Gumbel multiple times with different seeds
        let mut gumbel_results = Vec::new();
        for seed in 0..20 {
            let mut rng = StdRng::seed_from_u64(seed);
            let mask = relaxed_topk_gumbel(&scores, k, 0.5, 1.0, &mut rng);
            gumbel_results.push(mask);
        }

        // Check variance (Gumbel should have exploration)
        let variance: f64 = (0..scores.len())
            .map(|i| {
                let values: Vec<f64> = gumbel_results.iter().map(|m| m[i]).collect();
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                values.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64
            })
            .sum::<f64>() / scores.len() as f64;

        // Should have some variance (exploration)
        assert!(variance > 0.01, "Gumbel should provide exploration (variance > 0)");
    }

    #[test]
    fn test_both_methods_differentiable() {
        // Both methods should be differentiable (no panics, finite values)
        let scores = vec![0.9, 0.8, 0.7, 0.6, 0.5];
        let k = 2;

        // Sigmoid
        let (sigmoid_vals, _) = differentiable_topk(&scores, k, 1.0);
        assert!(sigmoid_vals.iter().all(|&v| v.is_finite()));

        // Gumbel
        let mut rng = StdRng::seed_from_u64(42);
        let gumbel_mask = relaxed_topk_gumbel(&scores, k, 0.5, 1.0, &mut rng);
        assert!(gumbel_mask.iter().all(|&v| v.is_finite()));
    }
}

