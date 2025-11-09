# Niodoo-TCT Roadmap

## Phase 0 — Minimal Prototype (This Week)
- INT8 quantizer with per-channel scaling
- Persistent homology (H0/H1) on consumer GPU/CPU
- Segment-based sheaf approximation with static restriction matrices
- Random data + sentence embedding smoke tests
- Topology feature vectorisation layer (Betti curves, persistence summaries, sheaf energy)
- Hidden-state helper to convert transformer activations into nToken feature vectors

## Phase 1 — Real Data Hooks
- Integrate HuggingFace sentence encoders for seed corpora
- Cache layer backed by DuckDB for reuse
- Benchmark suite comparing against cosine similarity baselines

## Phase 2 — Acceleration & Compression
- Ripser++ GPU backend integration
- Optional FP8/INT4 quantization paths
- Sheaf restriction learning via lightweight regression heads

## Phase 3 — Distillation & Deployment
- Train high-capacity teacher model on A100 baseline
- Distill into <500M parameter student with quantized heads
- Package as deployable gRPC/REST microservice

## Phase 4 — Federation & Edge Tiering
- Device → edge → cloud orchestration of nToken fidelity levels
- Global cache synchronization + eviction policies
- Topological monitoring dashboards and alerts
