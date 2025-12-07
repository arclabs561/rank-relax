use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use rank_relax::{gumbel_attention_mask, relaxed_topk_gumbel, differentiable_topk};
use rand::{SeedableRng, Rng};
use rand::rngs::StdRng;

fn random_scores(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n).map(|_| rng.gen_range(0.0..1.0)).collect()
}

fn bench_gumbel_vs_sigmoid_topk(c: &mut Criterion) {
    let mut group = c.benchmark_group("topk_comparison");

    for &n in &[10, 50, 100, 200] {
        let scores = random_scores(n, 1);
        let k = n / 3;

        // Gumbel-Softmax top-k
        group.bench_with_input(
            BenchmarkId::new("gumbel_topk", n),
            &n,
            |b, _| {
                let mut rng = StdRng::seed_from_u64(42);
                b.iter(|| {
                    black_box(relaxed_topk_gumbel(
                        &scores,
                        k,
                        0.5,  // temperature
                        1.0,  // scale
                        &mut rng,
                    ));
                });
            },
        );

        // Sigmoid-based top-k (existing method)
        group.bench_with_input(
            BenchmarkId::new("sigmoid_topk", n),
            &n,
            |b, _| {
                b.iter(|| {
                    black_box(differentiable_topk(&scores, k, 1.0));
                });
            },
        );
    }

    group.finish();
}

fn bench_gumbel_attention_mask(c: &mut Criterion) {
    let mut group = c.benchmark_group("gumbel_attention_mask");

    for &n in &[10, 50, 100, 200, 500] {
        let scores = random_scores(n, 2);
        let k = n / 3;

        group.bench_with_input(
            BenchmarkId::from_parameter(n),
            &n,
            |b, _| {
                let mut rng = StdRng::seed_from_u64(42);
                b.iter(|| {
                    black_box(gumbel_attention_mask(
                        &scores,
                        k,
                        0.5,
                        1.0,
                        &mut rng,
                    ));
                });
            },
        );
    }

    group.finish();
}

fn bench_temperature_scaling(c: &mut Criterion) {
    let scores = random_scores(100, 3);
    let k = 30;
    let mut group = c.benchmark_group("temperature_scaling");

    for &temp in &[0.1, 0.5, 1.0, 2.0] {
        group.bench_with_input(
            BenchmarkId::new("temp", temp),
            &temp,
            |b, &t| {
                let mut rng = StdRng::seed_from_u64(42);
                b.iter(|| {
                    black_box(relaxed_topk_gumbel(&scores, k, t, 1.0, &mut rng));
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_gumbel_vs_sigmoid_topk,
    bench_gumbel_attention_mask,
    bench_temperature_scaling
);
criterion_main!(benches);

