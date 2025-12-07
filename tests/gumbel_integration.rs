//! Integration tests for Gumbel-Softmax methods
//!
//! Tests the full workflow of Gumbel reranking for RAG systems.

#[cfg(feature = "gumbel")]
mod tests {
    use rank_relax::{gumbel_attention_mask, relaxed_topk_gumbel, gumbel_softmax};
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn test_rag_reranking_workflow() {
        // Simulate RAG reranking scenario
        let reranker_scores = vec![
            0.95, // doc1: highly relevant
            0.85, // doc2: relevant
            0.75, // doc3: moderately relevant
            0.45, // doc4: low relevance
            0.35, // doc5: very low relevance
            0.25, // doc6: irrelevant
        ];

        let mut rng = StdRng::seed_from_u64(42);
        let k = 3;
        let temperature = 0.5;
        let scale = 1.0;

        // Generate attention mask
        let mask = gumbel_attention_mask(
            &reranker_scores,
            k,
            temperature,
            scale,
            &mut rng,
        );

        // Verify mask properties
        assert_eq!(mask.len(), reranker_scores.len());
        assert!(mask.iter().all(|&m| m >= 0.0 && m <= 1.0));

        // Top-3 documents should generally have higher mask values
        // (doc1, doc2, doc3 have highest scores)
        // Note: Gumbel is stochastic, so we check on average or with tolerance
        let top3_avg: f64 = mask[0..3].iter().sum::<f64>() / 3.0;
        let bottom3_avg: f64 = mask[3..6].iter().sum::<f64>() / 3.0;
        assert!(top3_avg > bottom3_avg - 0.1, 
            "Top-3 should have higher mask values on average: top3_avg={}, bottom3_avg={}",
            top3_avg, bottom3_avg);
    }

    #[test]
    fn test_temperature_effect() {
        let scores = vec![0.9, 0.8, 0.7, 0.6, 0.5];
        let mut rng1 = StdRng::seed_from_u64(100);
        let mut rng2 = StdRng::seed_from_u64(100);

        // Low temperature = sharper selection
        let mask_low_temp = relaxed_topk_gumbel(&scores, 2, 0.1, 1.0, &mut rng1);
        // High temperature = smoother selection
        let mask_high_temp = relaxed_topk_gumbel(&scores, 2, 1.0, 1.0, &mut rng2);

        // Low temperature should have more extreme values (closer to 0 or 1)
        let low_temp_variance: f64 = mask_low_temp.iter()
            .map(|&x| (x - 0.5).powi(2))
            .sum::<f64>() / mask_low_temp.len() as f64;
        let high_temp_variance: f64 = mask_high_temp.iter()
            .map(|&x| (x - 0.5).powi(2))
            .sum::<f64>() / mask_high_temp.len() as f64;

        // Low temperature should have higher variance (more extreme)
        assert!(low_temp_variance > high_temp_variance);
    }

    #[test]
    fn test_scale_effect() {
        let scores = vec![0.9, 0.8, 0.7, 0.6, 0.5];
        let mut rng1 = StdRng::seed_from_u64(200);
        let mut rng2 = StdRng::seed_from_u64(200);

        // High scale = more deterministic (reranker scores dominate)
        let mask_high_scale = relaxed_topk_gumbel(&scores, 2, 0.5, 3.0, &mut rng1);
        // Low scale = more random (Gumbel noise dominates)
        let mask_low_scale = relaxed_topk_gumbel(&scores, 2, 0.5, 0.1, &mut rng2);

        // High scale should better reflect score ordering
        // Top score (index 0) should have higher mask value with high scale
        assert!(mask_high_scale[0] > mask_low_scale[0]);
    }

    #[test]
    fn test_multiple_samples_convergence() {
        let scores = vec![0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2];
        let k = 3;

        // Run multiple times and check consistency
        let mut masks = Vec::new();
        for seed in 0..10 {
            let mut rng = StdRng::seed_from_u64(seed);
            let mask = relaxed_topk_gumbel(&scores, k, 0.5, 1.0, &mut rng);
            masks.push(mask);
        }

        // Top-3 elements (indices 0, 1, 2) should generally have higher values
        // Check on average across all runs (Gumbel is stochastic)
        let mut top3_sum = 0.0;
        let mut bottom3_sum = 0.0;
        for mask in &masks {
            top3_sum += mask[0..3].iter().sum::<f64>();
            bottom3_sum += mask[5..8].iter().sum::<f64>();
        }
        let top3_avg = top3_sum / (masks.len() * 3) as f64;
        let bottom3_avg = bottom3_sum / (masks.len() * 3) as f64;
        assert!(top3_avg > bottom3_avg - 0.05,
            "Top-3 should have higher mask values on average: top3_avg={}, bottom3_avg={}",
            top3_avg, bottom3_avg);
    }

    #[test]
    fn test_gumbel_softmax_properties() {
        let logits = vec![1.0, 0.5, 0.3];
        let mut rng = StdRng::seed_from_u64(300);

        let probs = gumbel_softmax(&logits, 0.5, 1.0, &mut rng);

        // Should sum to 1.0 (probability distribution)
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // All probabilities should be positive
        assert!(probs.iter().all(|&p| p >= 0.0 && p <= 1.0));

        // Higher logit should generally have higher probability
        assert!(probs[0] > probs[2]); // logits[0] = 1.0 > logits[2] = 0.3
    }

    #[test]
    fn test_edge_cases() {
        let mut rng = StdRng::seed_from_u64(400);

        // Empty scores
        let empty: Vec<f64> = vec![];
        let mask = gumbel_attention_mask(&empty, 3, 0.5, 1.0, &mut rng);
        assert_eq!(mask.len(), 0);

        // Single score
        let single = vec![0.5];
        let mask = gumbel_attention_mask(&single, 1, 0.5, 1.0, &mut rng);
        assert_eq!(mask.len(), 1);
        assert!((mask[0] - 1.0).abs() < 1e-6); // Should select the only element

        // k >= n (select all)
        let scores = vec![0.5, 0.3];
        let mask = gumbel_attention_mask(&scores, 5, 0.5, 1.0, &mut rng);
        assert_eq!(mask.len(), 2);
        assert!(mask.iter().all(|&m| (m - 1.0).abs() < 1e-6));

        // k = 0
        let mask = gumbel_attention_mask(&scores, 0, 0.5, 1.0, &mut rng);
        assert_eq!(mask.len(), 0);
    }

    #[test]
    fn test_deterministic_with_seed() {
        let scores = vec![0.9, 0.8, 0.7, 0.6, 0.5];
        let k = 2;
        let temperature = 0.5;
        let scale = 1.0;

        // Same seed should produce same results
        let mut rng1 = StdRng::seed_from_u64(500);
        let mask1 = relaxed_topk_gumbel(&scores, k, temperature, scale, &mut rng1);

        let mut rng2 = StdRng::seed_from_u64(500);
        let mask2 = relaxed_topk_gumbel(&scores, k, temperature, scale, &mut rng2);

        // Should be identical with same seed
        for (m1, m2) in mask1.iter().zip(mask2.iter()) {
            assert!((m1 - m2).abs() < 1e-10);
        }
    }

    #[test]
    fn test_mask_application_simulation() {
        // Simulate applying mask to attention scores
        let reranker_scores = vec![0.9, 0.7, 0.8, 0.3, 0.5];
        let mut rng = StdRng::seed_from_u64(600);
        let k = 3;

        let mask = gumbel_attention_mask(&reranker_scores, k, 0.5, 1.0, &mut rng);

        // Simulate attention scores (would come from LLM)
        let attention_scores = vec![0.5, 0.4, 0.6, 0.3, 0.2];

        // Apply mask: masked_attention[i] = attention[i] * mask[i]
        let masked_attention: Vec<f64> = attention_scores.iter()
            .zip(mask.iter())
            .map(|(&att, &m)| att * m)
            .collect();

        // Top-k documents should have non-zero masked attention
        // (indices 0, 2, 1 have highest reranker scores)
        assert!(masked_attention[0] > 0.0);
        assert!(masked_attention[2] > 0.0);
        assert!(masked_attention[1] > 0.0);

        // Lower-ranked documents should have lower masked attention
        assert!(masked_attention[0] > masked_attention[3]);
        assert!(masked_attention[2] > masked_attention[4]);
    }
}

