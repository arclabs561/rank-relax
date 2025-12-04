use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rank_relax::{soft_rank, soft_sort, spearman_loss};

fn random_values(n: usize, seed: u64) -> Vec<f64> {
    // Simple LCG for reproducible "random" values
    let mut x = seed;
    (0..n)
        .map(|_| {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
            (x as f64 / u64::MAX as f64) * 100.0 - 50.0
        })
        .collect()
}

fn bench_soft_rank(c: &mut Criterion) {
    let mut g = c.benchmark_group("soft_rank");

    for &n in &[10, 100, 1000] {
        let values = random_values(n, 1);

        for &temp in &[0.1, 1.0, 10.0] {
            g.bench_with_input(
                BenchmarkId::new(format!("n{}_temp{}", n, temp), n),
                &n,
                |bench, _| {
                    bench.iter(|| black_box(soft_rank(&values, temp)));
                },
            );
        }
    }

    g.finish();
}

fn bench_soft_sort(c: &mut Criterion) {
    let mut g = c.benchmark_group("soft_sort");

    for &n in &[10, 100, 1000] {
        let values = random_values(n, 2);

        for &temp in &[0.1, 1.0, 10.0] {
            g.bench_with_input(
                BenchmarkId::new(format!("n{}_temp{}", n, temp), n),
                &n,
                |bench, _| {
                    bench.iter(|| black_box(soft_sort(&values, temp)));
                },
            );
        }
    }

    g.finish();
}

fn bench_spearman_loss(c: &mut Criterion) {
    let mut g = c.benchmark_group("spearman_loss");

    for &n in &[10, 100, 1000] {
        let predictions = random_values(n, 3);
        let targets = random_values(n, 4);

        for &temp in &[0.1, 1.0, 10.0] {
            g.bench_with_input(
                BenchmarkId::new(format!("n{}_temp{}", n, temp), n),
                &n,
                |bench, _| {
                    bench.iter(|| black_box(spearman_loss(&predictions, &targets, temp)));
                },
            );
        }
    }

    g.finish();
}

criterion_group!(benches, bench_soft_rank, bench_soft_sort, bench_spearman_loss);
criterion_main!(benches);

