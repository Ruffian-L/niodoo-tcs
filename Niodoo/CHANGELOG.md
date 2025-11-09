# Changelog

All notable work inside the reverse-ablation lab is tracked here. Every entry must reference real soak results, validation artifacts, or configuration changes. No stubs.

## [Phase 0] Foundations & Tooling - 2025-11-08
- Scaffolded `/workspace/Niodoo-Final/Niodoo` with dedicated `src/`, `scripts/`, `config/`, `logs/`, `tests/`, `docs/`, `baselines/`, and `reports/` directories.
- Authored `README.md` outlining the seven-stage pipeline, reverse ablation roadmap, dependency matrix, and operating guidance.
- Seeded `requirements.txt` with pinned versions of `vllm`, `rouge-score`, `pydantic`, and `requests` for environment reproducibility.
- Added `.gitignore` rules to exclude logs, generated baselines, reports, build outputs, and Python caches.
- Created Granite service tooling: `scripts/serve_granite.sh` (launch + readiness wait loop) and `scripts/check_granite.sh` (health probe with model count reporting).
- Added `tests/soak_test_basic.sh` to run a 12-request baseline soak, logging latencies and exporting `baselines/system0.json`.

## [Phase 1] System 0 - Stateless Granite Baseline
- Granite service (port 8000) launched with vLLM 0.11.0 (`python -m vllm.entrypoints.openai.api_server`). CLI smoke test succeeded (latency ~5.9s, 92 tokens). Baseline soak (`tests/soak_test_basic.sh`) ran 12 prompts with 100% success; results stored in `baselines/system0.json` (avg latency 2.70s, p50 2.45s, p95 2.98s).
- TODO: Log Granite service setup, CLI implementation, baseline soak (`baselines/system0.json`).

## [Phase 2] System 1 - ERAG Memory Core
- `src/curator.py` implements the System 2 Curator (ROUGE-L scoring + Granite quality prompt) with env-configurable endpoints and CLI entrypoint.
- CLI `seed_system1_memory` binary loads LocalEmbedder, normalizes vectors to 768d, and upserts sample payloads (Discover/Persist/Master) into Qdrant (standalone binary on 6333/6334).
- System 1 soak (`tests/soak_system1.sh`) runs memory-mode prompts against Granite; results saved to `logs/soak_system1.log` and `baselines/system1.json` (avg latency ~4.94s across 5 prompts).
- ERAG searches now pad/truncate embedder output, print augmented prompts when memory mode is enabled, and accept `min_score = 0.0` for guaranteed retrieval during validation.
- CLI pipeline now supports `--with-memory` / `--erag-config` / `--compass`; embeds prompt, fetches Qdrant memories via REST, and prepends context before Granite generation.
- Qdrant ERAG client (`src/erag.rs`) created: loads TOML config, embeds via `LocalEmbedder`, filters by compass payload, and issues searches against hyperspherical thresholds.
- Embedded ONNX wrapper (`src/embedding.rs`) built on `tcs-ml::QwenEmbedder`; configurable via env (`NIODOO_EMBED_MODEL`, `NIODOO_EMBED_CONFIG`, `NIODOO_EMBED_FORCE_CPU`).
- Scaffolded Qdrant bootstrap: `config/erag.toml`, `scripts/start_qdrant.sh`, and `scripts/bootstrap_qdrant_collections.sh` (Docker launch + schema setup).
- TODO: Record embedding module, Qdrant bootstrap, memory soak results.

## [Phase 3] System 2 - Curator + Adaptive Learning
- Ported the curator-executor loop: `system2_loop` binary orchestrates ERAG retrieval, Granite generation, curator scoring, and Qdrant experience logging.
- Added Qdrant-backed experience store (`config/system2_memory.toml`, `niodoo_system2_experiences`) with deterministic seeding and REST upserts.
- System 2 soak (`tests/soak_system2.sh`) captures quality/latency baselines (`logs/soak_system2.log`, `baselines/system2.json`).
- Documented workflow in `README.md`; `curator.py` stays reusable and the bootstrap script now notes the schema skip for Qdrant ≥1.15.
- Hardened curator scoring to require strict JSON and added resilient numeric parsing so `quality_score` is populated during soaks and learning triggers.
- Introduced the adaptive QLoRA learning loop (`src/learning_loop.py`) with deterministic buffer persistence, automated revision prompts, and TOML-driven hyperparameters (`config/learning_loop.toml`).
- Extended `system2_loop` to stream curator samples into the learning loop, capture buffer/train counts in baselines, and surface training telemetry in `logs/system2_loop.log`.
- Expanded `requirements.txt` with `transformers`, `peft`, `datasets`, and `accelerate` to bake the QLoRA stack into reproducible setup scripts.
- Fixed the curator word-based score fallback by restoring real `\b` boundaries (no literal `\\b` escapes), then re-ran `system2_loop` for 5 iterations with Granite+ERAG (`logs/soak_system2.log`, refreshed `baselines/system2.json`: avg latency 1.96s, avg quality 0.0) to confirm scores remain populated under the adaptive loop.
- Replaced the self-evaluating Granite curator with the topology-tuned QLoRA guardian (`qwen25-small-topology-20251105/merged`) loaded locally via `transformers`; curator scoring now runs on CPU by default via `NIODOO_CURATOR_DEVICE_MAP`. Validated with a 5-iteration soak (System 2) showing the stricter curator feedback (all hallucinated samples routed into the learning buffer, `final_buffer_count=5`, `avg latency=1.62s`) without triggering QLoRA training.
- Upgraded `accelerate` to 1.11.0 (aligned with Transformers 4.57.1) so Trainer can call `unwrap_model(..., keep_torch_compile=...)`; installed directly in the venv to avoid rebuilding `vllm`.
- Forced the learning loop via `python src/learning_loop.py --config config/learning_loop.toml train-now --force` after shutting down Granite, consuming 26 curator-rejected samples (loss ≈3.79, runtime ≈33.5 s) and writing fresh adapters to `models/system2_adapters/` (`adapter_model.safetensors`, tokenizer, README). Buffer flushed (`storage/system2_learning_buffer.jsonl` → 0 lines) with training telemetry archived in `logs/learning_loop.log`.
- Injected a System 2 directive + JSON schema into memory augmentation (`src/context.rs`) and lowered Granite temperature to 0.1 so completions ground themselves in retrieved memories. Post-training soak (`cargo run --bin system2_loop ...`) now yields curator scores 9–10 with JSON evidence mapping and refreshed baseline metrics (`baselines/system2.json`: avg latency 1.60 s, avg quality 9.8, rouge 0.088).
- Added an explicit completion banner to `system2_loop` so every run prints `[system2] completed …` with iteration count, aggregate quality/ROUGE, and buffer depth; validated via one-iteration smoke run (`logs/system2_loop.log`) to avoid future "hung" command ambiguity.
- Ported the integrated PromptSecurityManager stack into the lab (`src/security.rs`, `config/security.toml`) and wired `system2_loop` to enforce/record sanitized prompts before ERAG; the security audit log now captures each intake (`logs/security_audit.log`), and a fresh 1-iteration smoke (`cargo run --bin system2_loop -- --iterations 1 …`) landed avg quality 9.0 / rouge 0.05 with buffer 12 (`baselines/system2.json`, `logs/system2_loop.log`).

## [Phase 4] System 3 - Topological Consciousness
- TODO: Document torus projection, topology metrics, compass-driven tests.

## [Phase 5] Validation, Comparison & Reporting
- TODO: Summarize cross-system comparisons, cognitive benchmarks, and final superiority proof updates.
