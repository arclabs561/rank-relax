//! Paper reproduction benchmarks.
//!
//! This module contains benchmarks designed to reproduce results from
//! differentiable ranking research papers.

use rank_relax::*;
use std::time::Instant;

/// Reproduce SoftRank paper results (simplified).
///
/// Paper: "SoftRank: Optimizing Non-Smooth Rank Metrics" (WSDM 2008)
/// Expected: NDCG optimization on ranking datasets
pub fn benchmark_softrank_ndcg() {
    println!("=== SoftRank NDCG Benchmark ===");
    
    // Simulate ranking scenario: 10 documents per query
    let n_docs = 10;
    let n_queries = 100;
    
    let mut total_ndcg = 0.0;
    let mut total_time = 0.0;
    
    for query in 0..n_queries {
        // Generate relevance scores
        let mut scores: Vec<f64> = (0..n_docs)
            .map(|i| (i as f64) * 0.1 + (query as f64) * 0.01)
            .collect();
        
        // Shuffle to make it non-trivial
        scores.reverse();
        
        // Compute soft ranks
        let start = Instant::now();
        let ranks = soft_rank(&scores, 1.0);
        let elapsed = start.elapsed().as_secs_f64();
        
        total_time += elapsed;
        
        // Compute approximate NDCG (simplified)
        // In real scenario, would use actual relevance labels
        let mut dcg = 0.0;
        for (i, &rank) in ranks.iter().enumerate() {
            let relevance = if rank > (n_docs / 2) as f64 { 1.0 } else { 0.0 };
            dcg += relevance / ((i + 2) as f64).ln_2();
        }
        
        total_ndcg += dcg;
    }
    
    println!("Average NDCG: {:.4}", total_ndcg / n_queries as f64);
    println!("Average time per query: {:.4}ms", total_time / n_queries as f64 * 1000.0);
    println!("Total time: {:.4}s", total_time);
}

/// Reproduce NeuralSort paper results (simplified).
///
/// Paper: "NeuralSort: A Differentiable Sorting Operator" (ICML 2019)
/// Expected: Fast sorting/ranking with temperature scaling
pub fn benchmark_neural_sort() {
    println!("=== NeuralSort Benchmark ===");
    
    let sizes = vec![10, 50, 100, 500, 1000];
    
    for size in sizes {
        let values: Vec<f64> = (0..size).map(|i| (i as f64) * 0.1).collect();
        
        // Test different temperatures
        let temperatures = vec![0.1, 1.0, 10.0];
        
        for temp in temperatures {
            let start = Instant::now();
            let method = RankingMethod::NeuralSort;
            let ranks = method.compute(&values, temp);
            let elapsed = start.elapsed().as_secs_f64();
            
            // Verify ordering is preserved
            let mut sorted = true;
            for i in 1..size {
                if ranks[i-1] >= ranks[i] {
                    sorted = false;
                    break;
                }
            }
            
            println!(
                "Size: {}, Temp: {:.1}, Time: {:.4}ms, Sorted: {}",
                size, temp, elapsed * 1000.0, sorted
            );
        }
    }
}

/// Compare all methods on same dataset.
pub fn benchmark_method_comparison() {
    println!("=== Method Comparison Benchmark ===");
    
    let sizes = vec![50, 100, 500];
    
    for size in sizes {
        let values: Vec<f64> = (0..size).map(|i| (i as f64) * 0.1).collect();
        
        println!("\nSize: {}", size);
        
        let methods = [
            ("Sigmoid", RankingMethod::Sigmoid),
            ("NeuralSort", RankingMethod::NeuralSort),
            ("Probabilistic", RankingMethod::Probabilistic),
            ("SmoothI", RankingMethod::SmoothI),
        ];
        
        for (name, method) in methods {
            let start = Instant::now();
            let ranks = method.compute(&values, 1.0);
            let elapsed = start.elapsed().as_secs_f64();
            
            // Check ordering preservation
            let mut correct_order = true;
            for i in 1..size {
                if ranks[i-1] >= ranks[i] {
                    correct_order = false;
                    break;
                }
            }
            
            println!(
                "  {}: {:.4}ms, Order preserved: {}",
                name, elapsed * 1000.0, correct_order
            );
        }
    }
}

/// Benchmark gradient computation performance.
pub fn benchmark_gradients() {
    println!("=== Gradient Computation Benchmark ===");
    
    let sizes = vec![10, 50, 100, 500];
    
    for size in sizes {
        let values: Vec<f64> = (0..size).map(|i| (i as f64) * 0.1).collect();
        let ranks = soft_rank(&values, 1.0);
        
        let start = Instant::now();
        let grad = soft_rank_gradient(&values, &ranks, 1.0);
        let elapsed = start.elapsed().as_secs_f64();
        
        // Verify gradient properties
        let mut max_grad = 0.0;
        for row in &grad {
            for &g in row {
                max_grad = max_grad.max(g.abs());
            }
        }
        
        println!(
            "Size: {}, Time: {:.4}ms, Max gradient: {:.4}",
            size, elapsed * 1000.0, max_grad
        );
    }
}

fn main() {
    benchmark_softrank_ndcg();
    println!();
    
    benchmark_neural_sort();
    println!();
    
    benchmark_method_comparison();
    println!();
    
    benchmark_gradients();
}

