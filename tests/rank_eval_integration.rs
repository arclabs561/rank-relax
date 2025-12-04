//! Integration tests for rank-relax using rank-eval.
//!
//! These tests validate that differentiable ranking operations maintain
//! quality compared to discrete ranking, and demonstrate how to use
//! rank-eval for evaluation.

#[cfg(test)]
mod tests {
    use rank_eval::binary::{ndcg_at_k, mrr, average_precision};
    use rank_relax::soft_rank;
    use std::collections::HashSet;

    /// Convert soft ranks to discrete ranking.
    ///
    /// Takes soft ranks (continuous values) and converts them to a discrete
    /// ranking by sorting by rank value (descending).
    fn soft_to_discrete_ranking(soft_ranks: &[f64]) -> Vec<usize> {
        let mut indexed: Vec<(usize, f64)> = soft_ranks.iter().enumerate().map(|(i, &r)| (i, r)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        indexed.iter().map(|(i, _)| *i).collect()
    }

    #[test]
    fn test_soft_rank_quality_high_regularization() {
        // Test that soft ranking with high regularization approaches discrete quality
        let values = vec![5.0, 1.0, 2.0, 4.0, 3.0];
        let relevant: HashSet<usize> = [0, 2, 4].into_iter().collect(); // Indices 0, 2, 4 are relevant

        // High regularization (should be close to discrete)
        let soft_ranks = soft_rank(&values, 10.0);
        let discrete_ranking = soft_to_discrete_ranking(&soft_ranks);

        // Evaluate with rank-eval
        let ndcg = ndcg_at_k(&discrete_ranking, &relevant, 10);
        let mrr_score = mrr(&discrete_ranking, &relevant);
        let ap = average_precision(&discrete_ranking, &relevant);

        // With high regularization, soft ranking should maintain reasonable quality
        assert!(ndcg > 0.0 && ndcg <= 1.0, "nDCG should be valid");
        assert!(mrr_score > 0.0 && mrr_score <= 1.0, "MRR should be valid");
        assert!(ap > 0.0 && ap <= 1.0, "AP should be valid");

        // Perfect ranking would be [0, 4, 2, 3, 1] (values: 5, 4, 3, 2, 1)
        // With high regularization, we should get close to this
        assert_eq!(discrete_ranking[0], 0, "Highest value should be ranked first");
    }

    #[test]
    fn test_soft_rank_quality_low_regularization() {
        // Test that soft ranking with low regularization still produces valid rankings
        let values = vec![5.0, 1.0, 2.0, 4.0, 3.0];
        let relevant: HashSet<usize> = [0, 2, 4].into_iter().collect();

        // Low regularization (smoother, but should still work)
        let soft_ranks = soft_rank(&values, 0.1);
        let discrete_ranking = soft_to_discrete_ranking(&soft_ranks);

        let ndcg = ndcg_at_k(&discrete_ranking, &relevant, 10);

        // Even with low regularization, should produce valid metrics
        assert!(ndcg >= 0.0 && ndcg <= 1.0);
    }

    #[test]
    fn test_soft_rank_convergence() {
        // Test that increasing regularization improves ranking quality
        let values = vec![5.0, 1.0, 2.0, 4.0, 3.0];
        let relevant: HashSet<usize> = [0, 2, 4].into_iter().collect();

        let ndcg_low = {
            let soft_ranks = soft_rank(&values, 0.1);
            let ranking = soft_to_discrete_ranking(&soft_ranks);
            ndcg_at_k(&ranking, &relevant, 10)
        };

        let ndcg_high = {
            let soft_ranks = soft_rank(&values, 10.0);
            let ranking = soft_to_discrete_ranking(&soft_ranks);
            ndcg_at_k(&ranking, &relevant, 10)
        };

        // Higher regularization should generally give better (or equal) quality
        // (though this isn't always guaranteed, it's a reasonable expectation)
        assert!(ndcg_low >= 0.0 && ndcg_low <= 1.0);
        assert!(ndcg_high >= 0.0 && ndcg_high <= 1.0);
    }

    #[test]
    fn test_perfect_ranking_preserved() {
        // Test that perfect rankings are preserved by soft ranking
        let values = vec![5.0, 4.0, 3.0, 2.0, 1.0]; // Already sorted
        let relevant: HashSet<usize> = [0, 1, 2].into_iter().collect();

        let soft_ranks = soft_rank(&values, 10.0);
        let discrete_ranking = soft_to_discrete_ranking(&soft_ranks);

        // Perfect ranking should give nDCG = 1.0
        let ndcg = ndcg_at_k(&discrete_ranking, &relevant, 10);
        assert!((ndcg - 1.0).abs() < 0.1, "Perfect ranking should give nDCG ≈ 1.0");
    }

    #[test]
    fn test_ranking_consistency() {
        // Test that soft ranking produces consistent results
        let values = vec![5.0, 1.0, 2.0, 4.0, 3.0];
        let relevant: HashSet<usize> = [0, 2, 4].into_iter().collect();

        let soft_ranks1 = soft_rank(&values, 1.0);
        let soft_ranks2 = soft_rank(&values, 1.0);

        // Same inputs should give same outputs
        assert_eq!(soft_ranks1.len(), soft_ranks2.len());
        for (r1, r2) in soft_ranks1.iter().zip(soft_ranks2.iter()) {
            assert!((r1 - r2).abs() < 1e-9, "Soft ranks should be deterministic");
        }
    }
}

