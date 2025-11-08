# Testing, Benchmarking, and Observability Plan for nTokens

## 1. Testing Strategy

### 1.1 Unit Tests

- **Compositional**: verify functor preserves composition using hand-crafted grammar examples.
- **Topology**: confirm persistence diagrams for synthetic complexes match known results.
- **Sheaf**: ensure restriction maps commute and Laplacian remains PSD.
- **Hyperbolic**: numerical tests for gyrovector operations (closure, associativity up to tolerance).
- **Constraints**: validate geodesic penalty gradients against finite differences.

### 1.2 Property-Based Tests

- Leverage `proptest` to randomize simple graphs and ensure Bottleneck Stability (`d_B ≤ ε`).
- Generate random sheaf assignments and confirm `d^2 = 0` (cochain complex property).

### 1.3 Integration Tests

- Create mocked lambeq outputs and ERAG responses to run full `Builder` pipeline.
- Test pipeline stage wiring via `cargo test --features ntokens` harness, asserting `PromptContext.current_ntoken` is populated and reused by downstream components.
- Validate memory updates store nToken metadata and retrieval pipelines can read back the same.

### 1.4 Regression & Acceptance

- Compare curator scores before/after nToken integration on reference prompts.
- Ensure failure handling path triggers fallback without panic.
- Maintain golden persistence diagrams for canonical inputs.

## 2. Performance & Benchmarking

- Implement `benches/ntokens_bench.rs` running criterion benchmarks for:
  - Complex construction
  - Multiparameter persistence
  - Sheaf diffusion
  - Hyperbolic updates
- Add GPU soak test script (`scripts/bench_ntokens.sh`) executing 10k/100k simplex scenarios, recording latency and VRAM usage.
- Capture benchmark outputs in `results/ntokens/` with metadata (git revision, hardware profile).

## 3. Observability Additions

### 3.1 Metrics

- Prometheus gauges/histograms:
  - `ntoken_synthesis_latency_seconds`
  - `ntoken_persistence_pairs_total`
  - `ntoken_constraint_violation_ratio`
  - `ntoken_hyperbolic_norm`
  - `ntoken_gpu_memory_bytes`
- Increment counters on failure paths: `ntoken_build_failures_total`, `ntoken_fallback_cpu_total`.

### 3.2 Logging

- Structured events (JSON) summarizing persistence entropy, cohomology rank, constraint scores per prompt.
- Error logs include nToken ID, prompt ID, CUDA stream, and kernel name.

### 3.3 Tracing

- Add tracing spans in `Pipeline::build_ntoken`: `ntoken.build`, `ntoken.persistence`, `ntoken.sheaf`, `ntoken.memory`.
- Propagate hyperbolic norm and constraint satisfaction as span fields for cross-component correlation.

### 3.4 Dashboards

- Update Grafana dashboards with new panels for latency distribution, VRAM utilization, constraint violations, and throughput.
- Create heat map correlating persistence entropy with curator failures.

## 4. Alerting

- Define alert rules:
  - `ntoken_build_failures_total` increasing by >5/min.
  - `ntoken_gpu_memory_bytes` > 0.9 * capacity for 5 min.
  - Constraint violation ratio > 0.2 over 15 min rolling window.
- Alerts route to existing topology channel with runbook reference (`docs/ntokens/OPERATIONS.md`, forthcoming).

## 5. Release Criteria

- Pass all unit/integration tests with `--features ntokens,ntokens-gpu` in CI.
- Meet performance targets (<1 s for 10k complexes, <10 s for 100k).
- Observability metrics present and dashboards updated.
- Run soak test for 1 hour with zero critical alerts.

This plan ensures the nToken rollout is verifiable, measurable, and observable throughout development and production deployment.



