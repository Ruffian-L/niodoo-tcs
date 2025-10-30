# Developer Quickstart (2025 Modernization)

## Toolchain
- Rust: 1.87.0 (pinned in `rust-toolchain.toml`)
- Editions: 2024 across workspace

Install:
```bash
rustup toolchain install 1.87.0
rustup default 1.87.0
```

## Build profiles and targets
- Standard dev: `cargo build --workspace`
- Release: `cargo build --workspace --release`
- Edge (tiny binaries): `cargo build -p niodoo_real_integrated --profile edge --no-default-features --features beelink,edge`
- WASM core: `tcs-core-wasm` exposes `WasmEmbeddingBuffer`

## Feature flags (high-level)
- tcs-ml: `cpu` (default), `onnx`, `tokenizers`, `cuda`, `webgpu`, `wasm`
- niodoo_real_integrated: `svc` (default), `edge`, `otel`, `qdrant`, `embedded-qdrant`
- tcs-core: `runtime` gates `tokio` (off for wasm)

## Env vars (common)
```bash
export RUST_LOG=info
# Qdrant (code search + ERAG)
export QDRANT_URL=http://127.0.0.1:6333
export QDRANT_COLLECTION=code_index
# Telemetry (optional)
export OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4317
# Models
export QWEN_MODEL_PATH=/models/qwen/onnx/model.onnx
export TOKENIZER_JSON=/models/tokenizer.json
export MODELS_DIR=/models
```

## Semantic code search (tcs-code-tools)
1) Start Qdrant:
```bash
docker run -p 6333:6333 -v $PWD/qdrant_storage:/qdrant/storage qdrant/qdrant:latest
```
2) Index repo:
```bash
cargo run -p tcs-code-tools --bin code_indexer -- .
```
3) Query:
```bash
cargo run -p tcs-code-tools --bin code_query -- "where is TopologyEngine implemented?" 10
```

## Observability (OpenTelemetry)
- Enable: `--features otel` on `niodoo_real_integrated` and set `OTEL_EXPORTER_OTLP_ENDPOINT`.
- Logs always available via `RUST_LOG` + `tracing-subscriber`.

## CUDA/ONNX runtime
- `tcs-ml` uses ORT dynamically; CUDA EP is selected by ORT when present in system libs.
- Set `QWEN_MODEL_PATH` to load model; tokenizer auto-detected near model path.

## CI and quality gates
- One-shot CI locally:
```bash
bash scripts/ci.sh
```
- Security:
```bash
bash scripts/security.sh
```
- Dependencies:
```bash
bash scripts/deps_update.sh
```

## Common commands
```bash
# Integrated pipeline
cargo run -p niodoo_real_integrated -- --prompt "hello"
# Million cycle stress
cargo run -p niodoo_real_integrated --bin million_cycle_test
# Bench (criterion)
cargo bench -p tcs-core
# Tests
cargo test --workspace --all-features
```

## Troubleshooting
- Qt build errors: the transitional `src/` crate is excluded from the workspace to avoid `qmake`; keep using `niodoo_real_integrated` and libraries.
- ORT provider errors: ensure system ORT libs present or keep CPU path; verify `QWEN_MODEL_PATH` and `TOKENIZER_JSON`.
- Qdrant connection: verify `QDRANT_URL`, collection name, and that the container is running.

