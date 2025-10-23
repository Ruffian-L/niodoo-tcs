# Qwen2.5-Coder ONNX Integration Status — October 2025 Update

This document replaces the earlier pre-integration plan. All legacy TODOs that mentioned missing position IDs, absent KV cache support, or tokenizer stubs have been cleared. Use this file as the single source of truth for Qwen embedder status.

## ✅ Completed Scope
- **Stateful embedder**: `tcs-ml/src/qwen_embedder.rs` now drives the full 51-input contract (input IDs, attention mask, position IDs, 48× KV tensors) and streams incremental tokens while keeping the cache synchronized.
- **Configurable architecture**: `QwenConfig` validates layer/head dimensions, exposes presets, and is re-exported from `tcs_ml` for downstream consumers.
- **Tokenizer integration**: When built with `onnx-with-tokenizers`, the embedder loads `tokenizer.json`; otherwise it degrades gracefully to character encoding.
- **f16 + f32 outputs**: Mixed-precision logits and KV tensors are normalized to `Vec<f32>` via the shared `f16` helper.
- **Smoke-test coverage**: `cargo run -p tcs-ml --bin test_qwen_stateful --features onnx-with-tokenizers` verifies cache growth, cosine drift, and reset behavior end to end.
- **CI automation**: `.github/workflows/ci.yml` provisions libonig, downloads the ONNX artifacts, exports the runtime path, and executes the smoke test on every push/PR to `main`.
- **Sliding cache window**: `QwenConfig.cache_window` (default 2048) trims KV tensors after each step so long sessions don’t exceed the configured window.

## 🚧 Active / Planned Work
- **Orchestrator hookup**: Swap existing MotorBrain embedding calls to use `QwenEmbedder`, including lifecycle management for cache reset.
- **Cache hygiene**: Design eviction or windowing for long-running sessions once real workloads arrive.
- **Batch + concurrency story**: Evaluate expanding the embedder to support multi-request batching or per-thread cache instances.
- **Documentation sync**: Continue merging duplicated status notes into the newer `QWEN_STATEFUL_SUCCESS.md` and future READMEs.

## 📂 Key Artifacts
- `tcs-ml/Cargo.toml` — feature flags (`onnx`, `tokenizers`, `onnx-with-tokenizers`) and dependency declarations.
- `tcs-ml/src/qwen_embedder.rs` — primary implementation (state machine, cache merge, embedding extraction).
- `tcs-ml/src/bin/test_qwen_stateful.rs` — executable smoke test used locally and in CI.
- `.github/workflows/ci.yml` — GitHub Actions workflow ensuring model/runtime prerequisites are present.
- `QWEN_STATEFUL_SUCCESS.md` — narrative write-up of the completed integration.

## 🔁 Retired Notes
- Older references to “Missing Input: position_ids” or “need to add KV cache tensors” are obsolete and are preserved only in version control history.
- Tokenizer linking is no longer blocking; the fallback path intentionally remains for minimal builds.

## 📝 Maintenance Tips
- When updating model weights, adjust `QwenConfig` presets and regenerate the cached model download hash in the CI workflow.
- Keep the smoke test fast—limit prompts and logging so CI stays under five minutes.
- Treat this document as the living changelog for embedder workstreams; append new sections rather than reintroducing separate status files.