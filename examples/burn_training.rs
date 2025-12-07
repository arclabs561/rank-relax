//! Example: Using rank-relax in Burn training loop.
//!
//! This demonstrates how to use rank-relax with Burn tensors for training.
//!
//! Note: Burn integration is planned but not yet fully implemented.
//! This example shows the intended usage pattern.

use rank_relax::soft_rank;

fn main() {
    // Example: Soft ranking (current implementation works with Vec<f64>)
    let values = vec![5.0, 1.0, 2.0, 4.0, 3.0];
    let ranks = soft_rank(&values, 1.0);
    
    println!("Values: {:?}", values);
    println!("Soft ranks: {:?}", ranks);
    
    // In a real Burn training loop, you would:
    // 1. Get predictions from your model (Burn tensor)
    // 2. Convert to Vec<f64> for rank-relax
    // 3. Compute soft ranks
    // 4. Convert back to tensor
    // 5. Use in loss computation
    // 6. Gradients automatically flow through Burn's autograd
    
    println!("✅ Burn integration example complete");
    println!("Note: Full Burn tensor integration is planned for future release");
}

