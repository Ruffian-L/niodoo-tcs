# Changelog

## Unreleased
- Workspace upgraded to Rust 2024 edition; MSRV pinned to 1.87.
- ONNX runtime: tcs-ml uses dynamic provider selection with safe CPU fallback.
- Optional OpenTelemetry in niodoo_real_integrated via `otel` feature + OTLP endpoint.
- WASM: added tcs-core-wasm with WasmEmbeddingBuffer bindings; tokio gated in tcs-core.
- Edge: `[profile.edge]` and niodoo_real_integrated `edge`/`svc` features (svc optional).
- Semantic code search: new crate tcs-code-tools (`code_indexer`, `code_query`) using Qdrant.
- CI/Quality: scripts/ci.sh, deny.toml, nextest.toml, security script; overflow-checks in release.
- Testing: added proptest invariants and a cargo-fuzz target.
- Deps: bumped reqwest to 0.12 across crates; aligned toolchain and workspace deps.
- Docs: added docs/DEVELOPER_QUICKSTART.md and docs/CODE_SEARCH.md; README modernization quickstart.
