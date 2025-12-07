//! Example: Using rank-relax in Candle training loop.
//!
//! This demonstrates how to use rank-relax with Candle tensors for training.

use rank_relax::soft_rank;

#[cfg(feature = "candle")]
use candle_core::{Device, Tensor};

#[cfg(feature = "candle")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;
    
    // Example: Soft ranking with Candle tensors
    // Note: Current implementation works with Vec<f64>, so we convert
    let values = vec![5.0, 1.0, 2.0, 4.0, 3.0];
    let ranks = soft_rank(&values, 1.0);
    
    // Convert to Candle tensor
    let ranks_tensor = Tensor::new(&ranks[..], &device)?;
    
    println!("Values: {:?}", values);
    println!("Soft ranks: {:?}", ranks);
    println!("Ranks tensor shape: {:?}", ranks_tensor.shape());
    
    // In a real training loop, you would:
    // 1. Get predictions from your model (Candle tensor)
    // 2. Convert to Vec<f64> for rank-relax
    // 3. Compute soft ranks
    // 4. Convert back to tensor
    // 5. Use in loss computation
    
    println!("✅ Candle integration example complete");
    Ok(())
}

#[cfg(not(feature = "candle"))]
fn main() {
    println!("This example requires the 'candle' feature flag:");
    println!("  cargo run --example candle_training --features candle");
}

