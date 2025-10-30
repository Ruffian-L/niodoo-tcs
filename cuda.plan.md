# Niodoo CUDA + All Targets Modernization (Rust 2024)

## Current state excerpt

```48:56:/workspace/Niodoo-Final/Cargo.toml
[workspace.package]
edition = "2024"
rust-version = "1.87"
version = "0.1.0"
```

## Goals

- Rust 2024 edition, MSRV pinned, reproducible builds.
- CUDA acceleration path for ONNX/Qwen; CPU/WebGPU/ROCm gated for portability.
- Ship to Linux services, CLI/desktop, browser/WASM, and embedded/edge via feature sets.
- Strong CI: lint, audit, deny, nextest, miri, fuzz, benches; perf and telemetry wired.

## 1) Toolchain and editions

- rust-toolchain pinned at 1.87.0 with clippy, rustfmt, rust-src.
- In Cargo (workspace and all packages): `edition = "2024"`, `[workspace.package] rust-version = "1.87"`.
- Keep resolver = "2".

Example:

```toml
# /workspace/Niodoo-Final/rust-toolchain.toml
[toolchain]
channel = "1.87.0"
components = ["clippy", "rustfmt", "rust-src"]
```

```toml
# /workspace/Niodoo-Final/Cargo.toml (excerpt)
[workspace.package]
edition = "2024"
rust-version = "1.87"
```

## 2) Dependency hygiene and upgrades

- Use `cargo-outdated` to bump workspace deps to latest compatible; prioritize: tokio, axum, tonic, candle-*, ort, tokenizers, qdrant-client, tracing, tracing-subscriber, prometheus.
- Keep `ort` `load-dynamic` feature for runtime provider; CPU fallback by default.
- Consolidate `tokio` at workspace; gate where not needed (e.g., `tcs-core` for WASM).

## 3) Feature gating architecture

- Consistent features across relevant crates (`tcs-ml`, `tcs-core`, `niodoo_real_integrated`):
  - `cpu` (default)
  - `cuda` (NVIDIA via `ort/cuda`, `cudarc`)
  - `webgpu` (via `wgpu`)
  - `wasm` (wasm32 target)
  - `onnx`, `tokenizers` retained; `onnx-cuda` aliases `onnx,cuda`.

```toml
# tcs-ml/Cargo.toml (features)
[features]
default = ["cpu"]
cpu = []
cuda = ["dep:cudarc", "ort/cuda"]
webgpu = ["dep:wgpu"]
wasm = []
onnx = ["dep:ort", "dep:half"]
onnx-cuda = ["onnx", "cuda"]
```

```rust
// tcs-ml/src/lib.rs (ORT session sketch)
let session = ort::session::SessionBuilder::new(&environment)?
    .with_model_from_file(model_path)?;
// ORT selects available providers (CUDA/CPU) at runtime when linked.
```

## 4) WASM/browser target

- `tcs-core` builds on `wasm32-unknown-unknown`:
  - `tokio` behind `runtime` feature (off for wasm).
  - Avoid threads/fs in wasm builds.
- Minimal wasm entry crate `tcs-core-wasm` exposing `WasmEmbeddingBuffer` via `wasm-bindgen`.

## 5) Embedded/edge profile

- `edge` profile + `edge` feature; service deps are optional behind `svc`.
- Build tiny binary:
  `cargo build -p niodoo_real_integrated --profile edge --no-default-features --features beelink,edge`

## 6) CUDA acceleration path (NVIDIA)

- `tcs-ml`: ORT session with dynamic provider selection; CPU fallback by default.
- Validate `QWEN_MODEL_PATH` and tokenizer; warn with actionable hints.

## 7) Clean protobuf build

- Localize proto builds to crate `build.rs` (e.g., `niodoo_real_integrated/build.rs`); no root `build.rs`.

## 8) CI and quality gates

- Scripts/configs:
  - `scripts/ci.sh`: fmt, clippy, nextest, audit, deny, miri (subset), udeps.
  - `deny.toml`, `nextest.toml`.
- Optional `sccache` hooks for faster builds.

## 9) Tests: property + fuzz

- `proptest` invariants for `EquivariantLayer` and `EmbeddingBuffer`.
- `cargo-fuzz` for parser/protocol paths.

## 10) Telemetry and perf

- Optional `tracing-opentelemetry`, `opentelemetry-otlp`; export traces/metrics.
- `prometheus` metrics; per-stage timings in pipeline.
- `criterion` benches for CPU vs CUDA (feature-gated).

## 11) Semantic code search and governance

- Add `tcs-code-tools` with `code_indexer` and `code_query` using Qdrant.

## 12) Security & memory analysis

- `cargo-audit`, `cargo-deny` in CI; optional rCanary for leak detection.

## Deliverables

- Updated Cargo editions and features.
- ONNX runtime in `tcs-ml` with dynamic provider select + CPU fallback.
- WASM-targetable `tcs-core` and a minimal wasm wrapper crate.
- CI scripts/configs; tests (property/fuzz); benches; telemetry hooks.

### To-dos

- [x] Pin toolchain and bump all crates to Rust 2024 with MSRV 1.87
- [ ] Upgrade key workspace dependencies and align features across crates
- [x] Add cpu/cuda/webgpu/wasm features to tcs-ml and consumers
- [x] Implement ONNX Runtime provider selection with CPU fallback
- [x] Make tcs-core compile to wasm and add wasm wrapper crate
- [x] Create edge feature/profile to strip heavy deps and shrink binaries
- [x] Localize protobuf build scripts and remove top-level build.rs if redundant
- [x] Add ci.sh, nextest, audit, deny, miri, udeps, sccache cache
- [x] Add proptest invariants for EquivariantLayer and EmbeddingBuffer
- [x] Set up cargo-fuzz for parsers and protocol handling paths
- [x] Add OpenTelemetry option, stage timings, and criterion benches (CPU vs CUDA)
- [x] Implement code indexer/query tools using qdrant-client
- [x] Integrate cargo-audit/deny and optional rCanary memory leak checks

