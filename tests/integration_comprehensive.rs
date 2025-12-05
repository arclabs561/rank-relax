//! Comprehensive integration tests for all ranking methods and features.

use rank_relax::*;

#[test]
fn test_all_methods_consistency() {
    let values = vec![5.0, 1.0, 2.0, 4.0, 3.0];
    
    let methods = [
        RankingMethod::Sigmoid,
        RankingMethod::NeuralSort,
        RankingMethod::Probabilistic,
        RankingMethod::SmoothI,
    ];
    
    let mut results = Vec::new();
    
    for method in methods {
        let ranks = method.compute(&values, 1.0);
        results.push((method.name(), ranks));
    }
    
    // All methods should preserve ordering
    for (name, ranks) in &results {
        assert!(ranks[1] < ranks[2], "{}: value[1] < value[2]", name);
        assert!(ranks[2] < ranks[4], "{}: value[2] < value[4]", name);
        assert!(ranks[4] < ranks[3], "{}: value[4] < value[3]", name);
        assert!(ranks[3] < ranks[0], "{}: value[3] < value[0]", name);
    }
    
    // All methods should produce ranks in [0, n-1] range
    for (name, ranks) in &results {
        for &rank in ranks {
            assert!(rank >= 0.0 && rank <= (values.len() - 1) as f64,
                "{}: rank {} out of bounds", name, rank);
        }
    }
}

#[test]
fn test_gradient_accuracy() {
    let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let ranks = soft_rank(&values, 1.0);
    
    // Compute analytical gradient
    let grad_analytical = soft_rank_gradient(&values, &ranks, 1.0);
    
    // Verify gradient properties
    // Diagonal elements should be positive (increasing value increases rank)
    for i in 0..values.len() {
        assert!(grad_analytical[i][i] > 0.0, "Diagonal gradient should be positive");
    }
    
    // Gradient matrix should be square
    assert_eq!(grad_analytical.len(), values.len());
    for row in &grad_analytical {
        assert_eq!(row.len(), values.len());
    }
}

#[test]
fn test_spearman_loss_gradient_flow() {
    let predictions = vec![0.1, 0.9, 0.3, 0.7, 0.5];
    let targets = vec![0.0, 1.0, 0.2, 0.8, 0.4];
    
    let pred_ranks = soft_rank(&predictions, 1.0);
    let target_ranks = soft_rank(&targets, 1.0);
    
    // Compute loss
    let loss = spearman_loss(&predictions, &targets, 1.0);
    assert!(loss >= 0.0 && loss <= 2.0, "Loss should be in [0, 2]");
    
    // Compute gradient
    let grad = spearman_loss_gradient(
        &predictions,
        &targets,
        &pred_ranks,
        &target_ranks,
        1.0,
    );
    
    assert_eq!(grad.len(), predictions.len());
    assert!(grad.iter().all(|&g| g.is_finite()), "All gradients should be finite");
}

#[test]
fn test_batch_processing() {
    let batch = vec![
        vec![5.0, 1.0, 2.0, 4.0, 3.0],
        vec![3.0, 1.0, 2.0],
        vec![10.0, 5.0, 8.0, 7.0],
    ];
    
    let ranks_batch = soft_rank_batch(&batch, 1.0);
    
    assert_eq!(ranks_batch.len(), batch.len());
    for (values, ranks) in batch.iter().zip(ranks_batch.iter()) {
        assert_eq!(ranks.len(), values.len());
        
        // Verify ordering is preserved
        let mut sorted_values: Vec<(usize, f64)> = values.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        sorted_values.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        
        for i in 1..sorted_values.len() {
            let prev_idx = sorted_values[i-1].0;
            let curr_idx = sorted_values[i].0;
            assert!(ranks[prev_idx] < ranks[curr_idx], "Ordering should be preserved");
        }
    }
}

#[test]
fn test_optimized_vs_standard() {
    let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
    
    let ranks_std = soft_rank(&values, 1.0);
    let ranks_opt = soft_rank_optimized(&values, 1.0, true);
    
    // Should be approximately equal for sorted input
    for (r_std, r_opt) in ranks_std.iter().zip(ranks_opt.iter()) {
        assert!((r_std - r_opt).abs() < 1.0, "Optimized should match standard");
    }
}

#[test]
fn test_advanced_methods() {
    use rank_relax::methods_advanced::{soft_rank_softsort, differentiable_topk, listnet_loss, listmle_loss};
    
    let values = vec![5.0, 1.0, 2.0, 4.0, 3.0];
    
    // Test SoftSort
    let ranks_softsort = soft_rank_softsort(&values, 1.0);
    assert_eq!(ranks_softsort.len(), values.len());
    
    // Test Differentiable Top-K
    let (topk_vals, topk_ranks) = differentiable_topk(&values, 3, 1.0);
    assert_eq!(topk_vals.len(), values.len());
    assert_eq!(topk_ranks.len(), values.len());
    
    // Test ListNet
    let predictions = vec![0.1, 0.9, 0.3, 0.7, 0.5];
    let targets = vec![0.0, 1.0, 0.2, 0.8, 0.4];
    let loss_listnet = listnet_loss(&predictions, &targets, 1.0);
    assert!(loss_listnet >= 0.0 && loss_listnet.is_finite());
    
    // Test ListMLE
    let loss_listmle = listmle_loss(&predictions, &targets, 1.0);
    assert!(loss_listmle >= 0.0 && loss_listmle.is_finite());
}

#[test]
fn test_edge_cases() {
    // Empty input
    let empty: Vec<f64> = vec![];
    let ranks = soft_rank(&empty, 1.0);
    assert!(ranks.is_empty());
    
    // Single element
    let single = vec![1.0];
    let ranks = soft_rank(&single, 1.0);
    assert_eq!(ranks, vec![0.0]);
    
    // All equal
    let equal = vec![1.0, 1.0, 1.0];
    let ranks = soft_rank(&equal, 1.0);
    for &rank in &ranks {
        assert!(rank >= 0.0 && rank <= 2.0);
    }
    
    // NaN handling
    let with_nan = vec![1.0, f64::NAN, 3.0];
    let ranks = soft_rank(&with_nan, 1.0);
    assert!(ranks[1].is_nan(), "NaN input should produce NaN rank");
}

#[test]
fn test_regularization_strength_effects() {
    let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    
    let ranks_low = soft_rank(&values, 0.1);
    let ranks_high = soft_rank(&values, 10.0);
    
    // Higher regularization should produce sharper (more discrete-like) ranks
    // Check that high regularization produces ranks closer to integer values
    let mut low_variance = 0.0;
    let mut high_variance = 0.0;
    
    for i in 0..values.len() {
        let expected_rank = i as f64;
        low_variance += (ranks_low[i] - expected_rank).powi(2);
        high_variance += (ranks_high[i] - expected_rank).powi(2);
    }
    
    // High regularization should have lower variance (closer to discrete)
    assert!(high_variance < low_variance, "High regularization should be sharper");
}

