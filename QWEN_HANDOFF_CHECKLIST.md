# Qwen2.5-Coder Stateful Embedder Handoff Checklist

Use this list when resuming work on the Qwen embedder so the next AI (or teammate) can hit the ground running.

## Environment
- export `RUSTONIG_SYSTEM_LIBONIG=1`
- export `LD_LIBRARY_PATH=/path/to/onnxruntime/lib:$LD_LIBRARY_PATH` *(update for your install or CI runner)*
- export `QWEN_MODEL_PATH=/path/to/models/qwen2.5-coder-0.5b-instruct-onnx/onnx/model_quantized.onnx`
- optional: set `ORT_DYLIB_PATH=/path/to/onnxruntime/lib/libonnxruntime.so` if older examples expect it

## Build and Test Commands
- `cargo check -p tcs-ml --lib --features onnx`
- `cargo run -p tcs-ml --bin test_qwen_stateful --features onnx-with-tokenizers`
- `cargo fmt && cargo clippy --all-targets --all-features` (pre-PR hygiene)

## Critical Files
- `tcs-ml/src/qwen_embedder.rs` — stateful embedder implementation
- `tcs-ml/src/qwen_config.rs` — model presets and validation
- `tcs-ml/src/bin/test_qwen_stateful.rs` — smoke test used by CI
- `.github/workflows/ci.yml` — ensures model/runtime downloads for automation

## Current Focus Items
- integrate `QwenEmbedder` into the orchestrator pipeline (replace legacy MotorBrain path)
- monitor cache-window behaviour on real transcripts; adjust defaults if trimming proves too aggressive
- explore batching/concurrency patterns if multiple threads will share one embedder
- consolidate documentation into `finalREADME.md` once the orchestrator wiring lands

## Configuration Notes
- `QwenConfig.cache_window` (default 2048) defines the post-inference KV cache cap; adjust if longer contexts are required but stay ≤ `max_seq_len`

## Known Good Outputs
- first prompt cosine drift negative (~-0.2) after second prompt indicates cache reuse
- cache reset should drop context length to 0 and rebuild to ~9 tokens on the next call
- embeddings default to `QwenConfig.embed_dim` (currently 512) with non-zero counts matching length

## Observability and Logging
- `QWEN_STATEFUL_SUCCESS.md` captures the feature tour and test evidence
- smoke test logs include context length and cosine similarity; keep them concise for CI

## Open Questions
- confirm desired eviction policy (pure reset, window, or LRU) before implementing
- determine whether tokenizer model should be cached in a shared location or bundled per environment
- align naming conventions with downstream TCS components once integration is underway

Keep this checklist updated after each major edit so the next helper knows exactly where to resume.
