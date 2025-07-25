use concurrency::{multiply, multiply_simd, multiply_threaded, Matrix};
use criterion::{criterion_group, criterion_main, Criterion};

// benchmark for multiply
fn multiply_benchmark(c: &mut Criterion) {
    let a = Matrix::new((0..2500).map(|x| x as f32).collect(), 50, 50);
    let b = Matrix::new((0..2500).map(|x| x as f32).collect(), 50, 50);
    c.bench_function("multiply", |bencher| bencher.iter(|| multiply(&a, &b)));
}

fn multiply_threaded_benchmark(c: &mut Criterion) {
    let a = Matrix::new((0..2500).map(|x| x as f32).collect(), 50, 50);
    let b = Matrix::new((0..2500).map(|x| x as f32).collect(), 50, 50);
    c.bench_function("multiply_threaded", |bencher| {
        bencher.iter(|| multiply_threaded(&a, &b))
    });
}

fn multiply_simd_benchmark(c: &mut Criterion) {
    let a = Matrix::new((0..2500).map(|x| x as f32).collect(), 50, 50);
    let b = Matrix::new((0..2500).map(|x| x as f32).collect(), 50, 50);
    c.bench_function("multiply_simd", |bencher| {
        bencher.iter(|| multiply_simd(&a, &b))
    });
}

criterion_group!(
    benches,
    multiply_benchmark,
    multiply_threaded_benchmark,
    multiply_simd_benchmark
);
criterion_main!(benches);
