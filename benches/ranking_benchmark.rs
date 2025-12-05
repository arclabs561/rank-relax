//! Benchmarking suite for ranking operations.
//!
//! Compares performance across different methods and input sizes.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use rank_relax::*;

fn bench_soft_rank_methods(c: &mut Criterion) {
    let mut group = c.benchmark_group("soft_rank_methods");
    
    let sizes = vec![10, 50, 100, 500, 1000];
    
    for size in sizes {
        let values: Vec<f64> = (0..size).map(|i| (i as f64) * 0.1).collect();
        
        group.bench_with_input(
            BenchmarkId::new("sigmoid", size),
            &values,
            |b, v| b.iter(|| soft_rank(black_box(v), black_box(1.0)))
        );
        
        group.bench_with_input(
            BenchmarkId::new("neural_sort", size),
            &values,
            |b, v| {
                let method = RankingMethod::NeuralSort;
                b.iter(|| method.compute(black_box(v), black_box(1.0)))
            }
        );
        
        group.bench_with_input(
            BenchmarkId::new("probabilistic", size),
            &values,
            |b, v| {
                let method = RankingMethod::Probabilistic;
                b.iter(|| method.compute(black_box(v), black_box(1.0)))
            }
        );
        
        group.bench_with_input(
            BenchmarkId::new("smooth_i", size),
            &values,
            |b, v| {
                let method = RankingMethod::SmoothI;
                b.iter(|| method.compute(black_box(v), black_box(1.0)))
            }
        );
    }
    
    group.finish();
}

fn bench_gradient_computation(c: &mut Criterion) {
    let mut group = c.benchmark_group("gradient_computation");
    
    let sizes = vec![10, 50, 100, 500];
    
    for size in sizes {
        let values: Vec<f64> = (0..size).map(|i| (i as f64) * 0.1).collect();
        let ranks = soft_rank(&values, 1.0);
        
        group.bench_with_input(
            BenchmarkId::new("analytical", size),
            &(&values, &ranks),
            |b, (v, r)| {
                b.iter(|| soft_rank_gradient(black_box(v), black_box(r), black_box(1.0)))
            }
        );
    }
    
    group.finish();
}

fn bench_batch_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_processing");
    
    let batch_sizes = vec![10, 50, 100];
    let item_sizes = vec![50, 100, 500];
    
    for batch_size in batch_sizes {
        for item_size in &item_sizes {
            let batch: Vec<Vec<f64>> = (0..batch_size)
                .map(|_| (0..*item_size).map(|i| (i as f64) * 0.1).collect())
                .collect();
            
            group.bench_with_input(
                BenchmarkId::new("sequential", format!("{}_{}", batch_size, item_size)),
                &batch,
                |b, batch| {
                    b.iter(|| {
                        for values in batch {
                            black_box(soft_rank(black_box(values), black_box(1.0)));
                        }
                    })
                }
            );
            
            #[cfg(feature = "parallel")]
            group.bench_with_input(
                BenchmarkId::new("parallel", format!("{}_{}", batch_size, item_size)),
                &batch,
                |b, batch| {
                    b.iter(|| black_box(soft_rank_batch_parallel(black_box(batch), black_box(1.0))))
                }
            );
        }
    }
    
    group.finish();
}

fn bench_spearman_loss(c: &mut Criterion) {
    let mut group = c.benchmark_group("spearman_loss");
    
    let sizes = vec![10, 50, 100, 500, 1000];
    
    for size in sizes {
        let predictions: Vec<f64> = (0..size).map(|i| (i as f64) * 0.1).collect();
        let targets: Vec<f64> = (0..size).map(|i| (i as f64) * 0.1 + 0.05).collect();
        
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &(&predictions, &targets),
            |b, (p, t)| b.iter(|| spearman_loss(black_box(p), black_box(t), black_box(1.0)))
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_soft_rank_methods,
    bench_gradient_computation,
    bench_batch_processing,
    bench_spearman_loss
);
criterion_main!(benches);

