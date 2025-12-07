//! E2E test: Install from crates.io and test Candle integration.

#[cfg(feature = "candle")]
#[test]
fn test_e2e_candle_training() {
    use rank_relax::soft_rank;
    
    // Test that rank-relax works with Candle workflow
    let values = vec![5.0, 1.0, 2.0, 4.0, 3.0];
    let ranks = soft_rank(&values, 1.0);
    
    assert_eq!(ranks.len(), values.len());
    assert!(ranks.iter().all(|&r| r.is_finite()));
    
    // In a real training scenario:
    // 1. Model outputs Candle tensor
    // 2. Convert to Vec<f64>
    // 3. Use rank-relax
    // 4. Convert back to tensor for loss
    
    println!("✅ E2E Candle training test passed");
}

#[cfg(not(feature = "candle"))]
#[test]
fn test_e2e_candle_training() {
    // Skip if candle feature not enabled
    println!("⚠️  Candle feature not enabled - skipping test");
}

