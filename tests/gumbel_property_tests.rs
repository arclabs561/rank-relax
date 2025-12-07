//! Property-based tests for Gumbel-Softmax methods
//!
//! Tests mathematical properties and invariants.

#[cfg(feature = "gumbel")]
mod tests {
    use rank_relax::{gumbel_softmax, relaxed_topk_gumbel, gumbel_attention_mask};
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn test_gumbel_softmax_sum_to_one() {
        // Property: Gumbel-Softmax output should sum to 1.0
        for seed in 0..20 {
            let mut rng = StdRng::seed_from_u64(seed);
            let logits: Vec<f64> = (0..10).map(|i| i as f64 * 0.1).collect();
            
            let probs = gumbel_softmax(&logits, 0.5, 1.0, &mut rng);
            let sum: f64 = probs.iter().sum();
            
            assert!((sum - 1.0).abs() < 1e-6, 
                "Probabilities should sum to 1.0, got {}", sum);
        }
    }

    #[test]
    fn test_gumbel_softmax_non_negative() {
        // Property: All probabilities should be non-negative
        for seed in 0..20 {
            let mut rng = StdRng::seed_from_u64(seed);
            let logits: Vec<f64> = (0..15).map(|i| i as f64 * 0.1).collect();
            
            let probs = gumbel_softmax(&logits, 0.5, 1.0, &mut rng);
            
            assert!(probs.iter().all(|&p| p >= 0.0 && p <= 1.0),
                "All probabilities should be in [0, 1]");
        }
    }

    #[test]
    fn test_relaxed_topk_bounds() {
        // Property: Mask values should be in [0, 1]
        for seed in 0..20 {
            let mut rng = StdRng::seed_from_u64(seed);
            let scores: Vec<f64> = (0..20).map(|i| i as f64 * 0.05).collect();
            
            let mask = relaxed_topk_gumbel(&scores, 5, 0.5, 1.0, &mut rng);
            
            assert!(mask.iter().all(|&m| m >= 0.0 && m <= 1.0),
                "All mask values should be in [0, 1]");
        }
    }

    #[test]
    fn test_relaxed_topk_ordering() {
        // Property: Higher scores should generally have higher mask values
        let mut rng = StdRng::seed_from_u64(100);
        let scores = vec![0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0];
        
        // Run multiple times and check ordering is preserved on average
        let mut mask_sums = vec![0.0; scores.len()];
        let n_runs = 50;
        
        for _ in 0..n_runs {
            let mask = relaxed_topk_gumbel(&scores, 3, 0.5, 1.0, &mut rng);
            for i in 0..scores.len() {
                mask_sums[i] += mask[i];
            }
        }
        
        // Average mask values should preserve score ordering
        // (higher scores should have higher average mask values)
        for i in 0..(scores.len() - 1) {
            if scores[i] > scores[i + 1] {
                let avg_i = mask_sums[i] / n_runs as f64;
                let avg_next = mask_sums[i + 1] / n_runs as f64;
                // Allow some variance but generally should hold
                assert!(avg_i >= avg_next - 0.1,
                    "Score ordering should be preserved: score[{}]={} > score[{}]={}, but mask[{}]={} < mask[{}]={}",
                    i, scores[i], i+1, scores[i+1], i, avg_i, i+1, avg_next);
            }
        }
    }

    #[test]
    fn test_temperature_monotonicity() {
        // Property: Lower temperature should produce sharper distributions
        let scores = vec![0.9, 0.8, 0.7, 0.6, 0.5];
        let k = 2;
        
        let mut rng_low = StdRng::seed_from_u64(200);
        let mut rng_high = StdRng::seed_from_u64(200);
        
        let mask_low = relaxed_topk_gumbel(&scores, k, 0.1, 1.0, &mut rng_low);
        let mask_high = relaxed_topk_gumbel(&scores, k, 1.0, 1.0, &mut rng_high);
        
        // Low temperature should have more extreme values
        // (higher max, lower min, or more variance)
        let max_low: f64 = mask_low.iter().fold(0.0f64, |a, &b| a.max(b));
        let max_high: f64 = mask_high.iter().fold(0.0f64, |a, &b| a.max(b));
        
        // Low temperature should have higher maximum (sharper selection)
        assert!(max_low >= max_high - 0.1,
            "Low temperature should produce sharper selection");
    }

    #[test]
    fn test_scale_effect_on_determinism() {
        // Property: Higher scale should make selection more deterministic
        let scores = vec![0.9, 0.8, 0.7, 0.6, 0.5];
        let k = 2;
        
        // Run with low scale (more random)
        let mut low_scale_masks = Vec::new();
        for seed in 0..10 {
            let mut rng = StdRng::seed_from_u64(seed);
            let mask = relaxed_topk_gumbel(&scores, k, 0.5, 0.1, &mut rng);
            low_scale_masks.push(mask);
        }
        
        // Run with high scale (more deterministic)
        let mut high_scale_masks = Vec::new();
        for seed in 0..10 {
            let mut rng = StdRng::seed_from_u64(seed);
            let mask = relaxed_topk_gumbel(&scores, k, 0.5, 3.0, &mut rng);
            high_scale_masks.push(mask);
        }
        
        // High scale should have lower variance across runs
        // (more consistent selection of top elements)
        let low_scale_variance: f64 = (0..scores.len())
            .map(|i| {
                let values: Vec<f64> = low_scale_masks.iter().map(|m| m[i]).collect();
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                values.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64
            })
            .sum::<f64>() / scores.len() as f64;
            
        let high_scale_variance: f64 = (0..scores.len())
            .map(|i| {
                let values: Vec<f64> = high_scale_masks.iter().map(|m| m[i]).collect();
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                values.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64
            })
            .sum::<f64>() / scores.len() as f64;
        
        // High scale should have lower variance (more deterministic)
        assert!(high_scale_variance < low_scale_variance,
            "High scale should be more deterministic: high_var={}, low_var={}",
            high_scale_variance, low_scale_variance);
    }

    #[test]
    fn test_k_selection_effect() {
        // Property: Larger k should result in more non-zero mask values
        let scores: Vec<f64> = (0..20).map(|i| i as f64 * 0.05).collect();
        let mut rng = StdRng::seed_from_u64(300);
        
        let mask_k3 = relaxed_topk_gumbel(&scores, 3, 0.5, 1.0, &mut rng);
        let mask_k10 = relaxed_topk_gumbel(&scores, 10, 0.5, 1.0, &mut rng);
        
        // Count non-negligible mask values (> 0.1)
        let count_k3 = mask_k3.iter().filter(|&&m| m > 0.1).count();
        let count_k10 = mask_k10.iter().filter(|&&m| m > 0.1).count();
        
        // Larger k should select more elements
        assert!(count_k10 >= count_k3,
            "Larger k should select more elements: k=3 selected {}, k=10 selected {}",
            count_k3, count_k10);
    }
}

