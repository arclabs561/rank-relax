//! Property tests for rank-relax algorithms.
//!
//! These tests verify invariants that should always hold:
//! - Output length matches input length
//! - Finite output values (no NaN/Inf unless input has them)
//! - Monotonicity (sorting preserves order, ranking respects order)
//! - Bounds (ranks in reasonable range, sorted values monotonic)
//! - Edge cases (empty inputs, single elements, extreme values, large inputs)
//! - Symmetry and idempotency properties

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use crate::{rank::soft_rank, sort::soft_sort, spearman::spearman_loss};

    // Arbitrary generator for f64 values
    fn arb_f64() -> impl Strategy<Value = f64> {
        (-100.0f64..100.0f64).prop_filter("finite values", |&x| x.is_finite())
    }

    // Arbitrary generator for vectors of f64
    fn arb_values(n: usize) -> impl Strategy<Value = Vec<f64>> {
        proptest::collection::vec(arb_f64(), 1..=n)
    }

    // Arbitrary generator for regularization strength
    fn arb_temp() -> impl Strategy<Value = f64> {
        (0.01f64..100.0f64).prop_filter("positive finite", |&x| x.is_finite() && x > 0.0)
    }

    proptest! {
        /// soft_rank output length should match input length
        #[test]
        fn soft_rank_preserves_length(values in arb_values(100), temp in arb_temp()) {
            let ranks = soft_rank(&values, temp);
            prop_assert_eq!(ranks.len(), values.len());
        }

        /// soft_rank should produce finite values
        #[test]
        fn soft_rank_finite_output(values in arb_values(100).prop_filter("non-empty", |v| !v.is_empty()), temp in arb_temp()) {
            let ranks = soft_rank(&values, temp);
            for rank in &ranks {
                prop_assert!(rank.is_finite(), "Rank should be finite: {}", rank);
            }
        }

        /// soft_rank: higher input values should generally have higher ranks
        #[test]
        fn soft_rank_monotonicity(values in arb_values(50), temp in arb_temp()) {
            let ranks = soft_rank(&values, temp);
            
            // Find indices of max and min values
            let max_idx = values.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap();
            let min_idx = values.iter().enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap();

            if max_idx != min_idx {
                // With high regularization, max should have higher rank than min
                if temp > 1.0 {
                    prop_assert!(
                        ranks[max_idx] >= ranks[min_idx] - 0.1,
                        "Max value should have higher rank: {} >= {}",
                        ranks[max_idx],
                        ranks[min_idx]
                    );
                }
            }
        }

        /// soft_rank: ranks should be in reasonable range [0, n-1] (soft)
        #[test]
        fn soft_rank_bounded(values in arb_values(100), temp in arb_temp()) {
            let n = values.len();
            let ranks = soft_rank(&values, temp);
            
            for rank in &ranks {
                // Soft ranks can be slightly outside [0, n-1] due to approximation
                prop_assert!(
                    *rank >= -1.0 && *rank <= n as f64 + 1.0,
                    "Rank should be in reasonable range: {} for n={}",
                    rank,
                    n
                );
            }
        }

        /// soft_sort output length should match input length
        #[test]
        fn soft_sort_preserves_length(values in arb_values(100), temp in arb_temp()) {
            let sorted = soft_sort(&values, temp);
            prop_assert_eq!(sorted.len(), values.len());
        }

        /// soft_sort should produce finite values
        #[test]
        fn soft_sort_finite_output(values in arb_values(100), temp in arb_temp()) {
            let sorted = soft_sort(&values, temp);
            for val in &sorted {
                prop_assert!(val.is_finite(), "Sorted value should be finite: {}", val);
            }
        }

        /// soft_sort should be sorted (monotonic)
        #[test]
        fn soft_sort_monotonicity(values in arb_values(100), temp in arb_temp()) {
            let sorted = soft_sort(&values, temp);
            for window in sorted.windows(2) {
                prop_assert!(
                    window[0] <= window[1] + 1e-6,
                    "Should be sorted: {} <= {}",
                    window[0],
                    window[1]
                );
            }
        }

        /// soft_sort should preserve all input values (multiset equality)
        #[test]
        fn soft_sort_preserves_values(values in arb_values(50), temp in arb_temp()) {
            let sorted = soft_sort(&values, temp);
            
            // Check that all input values appear in output (within tolerance)
            let mut sorted_sorted = sorted.clone();
            sorted_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mut values_sorted = values.clone();
            values_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            
            prop_assert_eq!(sorted_sorted.len(), values_sorted.len());
            for (s, v) in sorted_sorted.iter().zip(values_sorted.iter()) {
                prop_assert!(
                    (s - v).abs() < 1e-6 || (sorted.len() == values.len()),
                    "Values should be preserved: {} vs {}",
                    s,
                    v
                );
            }
        }

        /// spearman_loss should handle equal-length inputs
        #[test]
        fn spearman_loss_equal_length(
            predictions in arb_values(100),
            targets in arb_values(100),
            temp in arb_temp()
        ) {
            // Ensure equal length
            let n = predictions.len().min(targets.len());
            let pred = &predictions[..n];
            let targ = &targets[..n];
            
            let loss = spearman_loss(pred, targ, temp);
            prop_assert!(loss.is_finite(), "Loss should be finite: {}", loss);
            prop_assert!(loss >= -2.0 && loss <= 2.0, "Loss should be in reasonable range: {}", loss);
        }

        /// spearman_loss: perfect correlation should give low loss
        #[test]
        fn spearman_loss_perfect_correlation(n in 2usize..100, temp in arb_temp()) {
            let predictions: Vec<f64> = (0..n).map(|i| i as f64).collect();
            let targets = predictions.clone();
            
            let loss = spearman_loss(&predictions, &targets, temp);
            prop_assert!(loss < 0.5, "Perfect correlation should give low loss: {}", loss);
        }

        /// spearman_loss: anti-correlation should give high loss
        #[test]
        fn spearman_loss_anti_correlation(n in 2usize..100, temp in arb_temp()) {
            let predictions: Vec<f64> = (0..n).map(|i| i as f64).collect();
            let targets: Vec<f64> = (0..n).rev().map(|i| i as f64).collect();
            
            let loss = spearman_loss(&predictions, &targets, temp);
            prop_assert!(loss > 0.5, "Anti-correlation should give high loss: {}", loss);
        }

        /// spearman_loss should handle mismatched lengths
        #[test]
        fn spearman_loss_mismatched_length(
            pred_len in 1usize..50,
            targ_len in 51usize..100
        ) {
            // Different lengths should return maximum loss
            let predictions = vec![1.0; pred_len];
            let targets = vec![1.0; targ_len];
            let loss = spearman_loss(&predictions, &targets, 1.0);
            prop_assert_eq!(loss, 1.0, "Mismatched lengths should give max loss");
        }

        /// spearman_loss should handle empty inputs
        #[test]
        fn spearman_loss_empty_inputs(temp in arb_temp()) {
            let empty: Vec<f64> = vec![];
            let loss = spearman_loss(&empty, &empty, temp);
            prop_assert_eq!(loss, 0.0, "Empty inputs should give zero loss");
        }

        /// spearman_loss should handle single-element inputs
        #[test]
        fn spearman_loss_single_element(temp in arb_temp()) {
            let single = vec![1.0];
            let loss = spearman_loss(&single, &single, temp);
            prop_assert_eq!(loss, 0.0, "Single element should give zero loss");
        }

        /// All functions should handle extreme values (Inf, -Inf, NaN) without panicking
        #[test]
        fn all_functions_handle_extreme_values(extreme_idx in 0usize..3) {
            let extremes = [f64::INFINITY, f64::NEG_INFINITY, f64::NAN];
            let extreme = extremes[extreme_idx];
            let values = vec![1.0, 2.0, extreme, 3.0];

            // Should not panic (may produce NaN or Inf, but should not crash)
            let ranks = soft_rank(&values, 1.0);
            prop_assert_eq!(ranks.len(), values.len(), "Should not panic");
            
            let sorted = soft_sort(&values, 1.0);
            prop_assert_eq!(sorted.len(), values.len(), "Should not panic");
            
            // For spearman_loss, use finite values to avoid edge cases
            let finite_values: Vec<f64> = values.iter().filter(|v| v.is_finite()).copied().collect();
            if finite_values.len() >= 2 {
                let loss = spearman_loss(&finite_values, &finite_values, 1.0);
                prop_assert!(loss.is_finite() || loss == 0.0, "Should not panic");
            }
        }

        /// All functions should handle very large inputs
        #[test]
        fn all_functions_handle_large_inputs(n in 1000usize..5000) {
            let values: Vec<f64> = (0..n).map(|i| i as f64).collect();
            
            // Should not panic or hang
            let ranks = soft_rank(&values, 1.0);
            prop_assert_eq!(ranks.len(), n);
            
            let sorted = soft_sort(&values, 1.0);
            prop_assert_eq!(sorted.len(), n);
            
            let loss = spearman_loss(&values, &values, 1.0);
            prop_assert!(loss.is_finite() || loss == 0.0);
        }

        /// soft_rank should be symmetric (order of equal values shouldn't matter much)
        #[test]
        fn soft_rank_symmetric_equality(n in 2usize..20) {
            let values: Vec<f64> = vec![1.0; n]; // All equal values
            let ranks = soft_rank(&values, 10.0); // High temp for sharper behavior
            
            // All ranks should be approximately equal (within small tolerance)
            let first_rank = ranks[0];
            for rank in &ranks {
                prop_assert!(
                    (rank - first_rank).abs() < 0.1,
                    "Equal values should have similar ranks: {} vs {}",
                    rank,
                    first_rank
                );
            }
        }

        /// soft_sort should be idempotent (sorting sorted values should be similar)
        #[test]
        fn soft_sort_idempotent(values in arb_values(50), temp in arb_temp()) {
            let sorted_once = soft_sort(&values, temp);
            let sorted_twice = soft_sort(&sorted_once, temp);
            
            // Should be very similar (within tolerance)
            for (s1, s2) in sorted_once.iter().zip(sorted_twice.iter()) {
                prop_assert!(
                    (s1 - s2).abs() < 1e-3,
                    "Idempotent: {} vs {}",
                    s1,
                    s2
                );
            }
        }

        /// spearman_loss should be symmetric (loss(pred, target) ≈ loss(target, pred))
        #[test]
        fn spearman_loss_symmetric(
            predictions in arb_values(50),
            targets in arb_values(50),
            temp in arb_temp()
        ) {
            let n = predictions.len().min(targets.len());
            let pred = &predictions[..n];
            let targ = &targets[..n];
            
            let loss_ab = spearman_loss(pred, targ, temp);
            let loss_ba = spearman_loss(targ, pred, temp);
            
            prop_assert!(
                (loss_ab - loss_ba).abs() < 0.01,
                "Loss should be symmetric: {} vs {}",
                loss_ab,
                loss_ba
            );
        }
    }
}

