use criterion::{Criterion, criterion_group, criterion_main};
use niodoo_real_integrated::torus::TorusPadMapper;

fn bench_torus_projection(c: &mut Criterion) {
    let mut mapper = TorusPadMapper::new(1337);
    let embedding: Vec<f32> = (0..896)
        .map(|i| ((i as f32 * 0.0137).sin() * 0.5) as f32)
        .collect();

    c.bench_function("torus_project", |b| {
        b.iter(|| mapper.project(&embedding).unwrap())
    });
}

criterion_group!(benches, bench_torus_projection);
criterion_main!(benches);
