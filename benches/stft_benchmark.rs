use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use emo_audio::stft::*;
use rand::distributions::Uniform;
use rand::prelude::*;

pub fn stft_benchmark(c: &mut Criterion) {
    // KB in bytes divided by 4 bytes for a float
    const KB: usize = 1024 / 4;
    // Audio files we process range from around 40KB to 125KB this is kinda
    // large so I'm going to work with roughly 8KB as the max for benchmarking

    let mut rng = thread_rng();
    let side = Uniform::new(-1.0, 1.0);
    let mut signal = vec![];
    for _ in 0..(8 * KB) {
        signal.push(rng.sample(side));
    }
    let stft = ShortTimeFourierTransform::default();
    let mut group = c.benchmark_group("stft");
    for size in [KB / 4, KB / 2, KB, 2 * KB, 4 * KB, 8 * KB].iter() {
        group.throughput(Throughput::Bytes(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| stft.run(&signal[0..size]));
        });
    }
    group.finish();
}

criterion_group!(benches, stft_benchmark);
criterion_main!(benches);
