use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use rank_relax::{soft_rank, soft_sort, spearman_loss};
use rank_relax::methods::{soft_rank_sigmoid, soft_rank_neural_sort, soft_rank_probabilistic, soft_rank_smooth_i};

fn bench_soft_rank(c: &mut Criterion) {
    let mut group = c.benchmark_group("soft_rank");
    
    for size in [10, 50, 100, 500, 1000].iter() {
        let values: Vec<f64> = (0..*size).map(|i| (i as f64).sin()).collect();
        
        group.bench_with_input(
            BenchmarkId::new("sigmoid", size),
            &values,
            |b, v| {
                b.iter(|| soft_rank(black_box(v), black_box(1.0)));
            },
        );
    }
    
    group.finish();
}

fn bench_soft_sort(c: &mut Criterion) {
    let mut group = c.benchmark_group("soft_sort");
    
    for size in [10, 50, 100, 500, 1000, 5000].iter() {
        let values: Vec<f64> = (0..*size).map(|i| (i as f64).sin()).collect();
        
        group.bench_with_input(
            BenchmarkId::new("permutahedron", size),
            &values,
            |b, v| {
                b.iter(|| soft_sort(black_box(v), black_box(1.0)));
            },
        );
    }
    
    group.finish();
}

fn bench_spearman_loss(c: &mut Criterion) {
    let mut group = c.benchmark_group("spearman_loss");
    
    for size in [10, 50, 100, 500, 1000].iter() {
        let predictions: Vec<f64> = (0..*size).map(|i| (i as f64).sin()).collect();
        let targets: Vec<f64> = (0..*size).map(|i| ((i as f64) * 1.1).sin()).collect();
        
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &(predictions, targets),
            |b, (p, t)| {
                b.iter(|| spearman_loss(black_box(p), black_box(t), black_box(1.0)));
            },
        );
    }
    
    group.finish();
}

fn bench_all_methods(c: &mut Criterion) {
    let mut group = c.benchmark_group("ranking_methods");
    
    let values: Vec<f64> = (0..100).map(|i| (i as f64).sin()).collect();
    
    group.bench_with_input(
        BenchmarkId::from_parameter("sigmoid"),
        &values,
        |b, v| {
            b.iter(|| soft_rank_sigmoid(black_box(v), black_box(1.0)));
        },
    );
    
    group.bench_with_input(
        BenchmarkId::from_parameter("neural_sort"),
        &values,
        |b, v| {
            b.iter(|| soft_rank_neural_sort(black_box(v), black_box(1.0)));
        },
    );
    
    group.bench_with_input(
        BenchmarkId::from_parameter("probabilistic"),
        &values,
        |b, v| {
            b.iter(|| soft_rank_probabilistic(black_box(v), black_box(1.0)));
        },
    );
    
    group.bench_with_input(
        BenchmarkId::from_parameter("smooth_i"),
        &values,
        |b, v| {
            b.iter(|| soft_rank_smooth_i(black_box(v), black_box(1.0)));
        },
    );
    
    group.finish();
}

criterion_group!(benches, bench_soft_rank, bench_soft_sort, bench_spearman_loss, bench_all_methods);
criterion_main!(benches);

