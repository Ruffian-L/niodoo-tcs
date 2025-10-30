use anyhow::Result;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use niodoo_real_integrated::pipeline::proto::ConsciousnessState;
use serde_json;
use std::time::SystemTime;
use tracing::info;

fn create_sample_state() -> ConsciousnessState {
    ConsciousnessState {
        topology: Some(niodoo_real_integrated::pipeline::proto::TopologyState {
            entropy: 1.946,
            iit_phi: 0.36,
            knots: vec![15.0],
            betti_numbers: vec![7, 15, 0],
            spectral_gap: 1.46,
            persistent_entropy: 1.46,
        }),
        pad_ghost: Some(niodoo_real_integrated::pipeline::proto::PadGhostState {
            pad: vec![0.9947, 0.9950, 0.9948, 0.9937, 0.9609, 0.9906, 0.9948],
            mu: vec![0.0025, -0.0006, 0.0037, 0.0006, 0.0017, 0.0013, -0.0074],
            sigma: vec![0.1227, 0.0939, 0.0949, 0.0994, 0.0905, 0.0976, 0.0947],
            raw_stds: vec![0.0443, 0.0339, 0.0342, 0.0359, 0.0326, 0.0352, 0.0342],
        }),
        quadrant: "Discover".to_string(),
        threat: false,
        healing: true,
        rouge_score: 0.244,
        timestamp: SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs() as i64,
    }
}

fn bench_protobuf_encode(c: &mut Criterion) {
    let state = create_sample_state();
    c.bench_function("Protobuf encode 1000x", |b| {
        b.iter(|| {
            for _ in 0..1000 {
                let _bytes = black_box(state.encode_to_vec());
            }
        })
    });
}

fn bench_protobuf_decode(c: &mut Criterion) {
    let state = create_sample_state();
    let bytes = state.encode_to_vec();
    c.bench_function("Protobuf decode 1000x", |b| {
        b.iter(|| {
            for _ in 0..1000 {
                let _decoded = black_box(ConsciousnessState::decode(&bytes[..]).unwrap());
            }
        })
    });
}

fn bench_json_encode(c: &mut Criterion) {
    let state = create_sample_state();
    c.bench_function("JSON encode 1000x", |b| {
        b.iter(|| {
            for _ in 0..1000 {
                let _json = black_box(serde_json::to_vec(&state).unwrap());
            }
        })
    });
}

fn bench_json_decode(c: &mut Criterion) {
    let state = create_sample_state();
    let json = serde_json::to_vec(&state).unwrap();
    c.bench_function("JSON decode 1000x", |b| {
        b.iter(|| {
            for _ in 0..1000 {
                let _decoded: ConsciousnessState = black_box(serde_json::from_slice(&json).unwrap());
            }
        })
    });
}

fn bench_size(c: &mut Criterion) {
    let state = create_sample_state();
    let proto_bytes = state.encode_to_vec();
    let json = serde_json::to_vec(&state).unwrap();

    c.bench_function("Size comparison", |b| {
        b.iter(|| {
            black_box((&proto_bytes.len(), &json.len()))
        })
    });

    info!("Proto size: {} bytes, JSON size: {} bytes (Proto ~{}% smaller)", 
          proto_bytes.len(), json.len(), 
          (1.0 - proto_bytes.len() as f64 / json.len() as f64) * 100.0);
}

criterion_group!(
    benches,
    bench_protobuf_encode,
    bench_protobuf_decode,
    bench_json_encode,
    bench_json_decode,
    bench_size
);
criterion_main!(benches);
