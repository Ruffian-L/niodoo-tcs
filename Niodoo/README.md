# Niodoo Reverse Ablation Lab

This workspace rebuilds the production `niodoo_real_integrated` cognitive pipeline from the ground up, starting with a stateless Granite-3B baseline and adding one capability at a time until the full topology-aware system is online. Every phase captures quantitative baselines, reuses the existing validation stack, and logs results in `CHANGELOG.md`.

## Seven-Stage Cognitive Pipeline
1. **Security Manager** – Validates prompts, enforces policies.
2. **Embedding** – Local ONNX `tcs-ml::QwenStatefulEmbedder` generates 768D vectors.
3. **Torus Projection** – Maps embeddings to the 7D PAD+Ghost Möbius manifold.
4. **TCS Analysis** – Computes Betti numbers, knot complexity, and `β_meta` (conditional).
5. **Consciousness Compass** – 2-bit entropy quadrant (Panic, Persist, Discover, Master).
6. **ERAG Memory Retrieval** – Hyperspherical Qdrant search conditioned on compass state.
7. **Dynamic Tokenization** – CRDT-managed vocabulary promotion before generation.

**Post-generation loop:** vLLM generation → Curator quality scoring → QLoRA learning loop → ERAG memory storage.

## Reverse Ablation Roadmap
| Phase | System | New Capability | Primary Artifacts |
|-------|--------|----------------|-------------------|
| 0 | Foundations | Workspace scaffold, docs, tooling | `README.md`, `docs/`, scripts, configs |
| 1 | System 0 | Stateless Granite baseline | `src/main.rs`, `scripts/serve_granite.sh`, `tests/soak_system0.sh` |
| 2 | System 1 | ERAG memory core | `src/embedding.rs`, `src/erag.rs`, `tests/soak_system1.sh` |
| 3 | System 2 | Curator + adaptive learning | `src/curator.py`, `src/learning_loop.py`, `tests/soak_system2.sh` |
| 4 | System 3 | Topological consciousness | `src/torus.rs`, `src/tcs_analysis.rs`, `src/compass.rs`, `tests/soak_system3.sh` |
| 5 | Validation | Comparative & cognitive proofs | `reports/reverse_ablation_report.md`, updated dashboards |

Progress is recorded phase-by-phase in `CHANGELOG.md`. Gate criteria match the refined plan stored in `/niodoo-reverse-ablation-setup.plan.md`.

## Dependency Matrix
| Component | Requirement | Notes |
|-----------|-------------|-------|
| Qdrant | Ports 6333/6334 | Required from System 1 onward. Use `scripts/start_qdrant.sh`. |
| vLLM (Granite) | Port 8000 | Required for all systems. `scripts/serve_granite.sh` runs `ibm-granite/granite-3b-code-instruct`. |
| ONNX Runtime | `onnxruntime-linux-x64-1.16.3/` | Must be on `LD_LIBRARY_PATH` for local embeddings. |
| Optional Ollama | Port 11434 | Alternate curator backend; default is vLLM. |
| Optional RL Server | Port 8080 | Advanced reinforcement scenarios (future). |
| Prometheus/Grafana | Prometheus scrape on services | Reuse existing dashboards for topology metrics and throughput SLIs. |

## System 0 CLI Usage
The baseline CLI calls the OpenAI-compatible Granite endpoint:

```
cargo run -- --prompt "Summarize the NIODOO pipeline."
```

- Override the endpoint: `--endpoint http://localhost:8000/v1/completions`
- Switch output to JSON for logging: `--output json`
- Provide prompts via stdin: `echo "Explain the Consciousness Compass" | cargo run --`

### Running the Granite Baseline Soak
After `scripts/serve_granite.sh` reports readiness, run:

```
bash tests/soak_test_basic.sh
```

Results stream to `logs/soak_system0.log` and `baselines/system0.json`.

### Enabling ERAG Memory (System 1)
```
cargo run --bin niodoo-cli -- --with-memory --compass Discover --prompt "Explain Möbius memory."
```

1. Launch Qdrant standalone (prior steps set up `/workspace/qdrant_config/config.yaml`).
2. Seed memories: `cargo run --bin seed_system1_memory`.
3. Run memory-mode queries as above.

- `--with-memory` turns on the ERAG pipeline (embeds prompt, searches Qdrant, prepends memories).
- `--erag-config` defaults to `config/erag.toml`; override to target other deployments.
- `--compass` filters retrieved payloads by consciousness quadrant (`Panic`, `Persist`, `Discover`, `Master`).

Run the System 1 soak once Qdrant + Granite are healthy:
```
source ../venv/bin/activate
./tests/soak_system1.sh
```
Outputs: `logs/soak_system1.log`, `baselines/system1.json`.

## System 1 Embeddings (WIP)
## System 2 Curator (WIP)
- Run the end-to-end loop (generation → curator → memory logging):
  ```
  cargo run --bin system2_loop -- --iterations 3
  ```
- Experiences land in `config/system2_memory.toml`'s collection (`niodoo_system2_experiences`).
- System 2 soak: `tests/soak_system2.sh` (writes `logs/soak_system2.log`, `baselines/system2.json`).

- Quality scorer lives in `src/curator.py`.
- Configure via env (`NIODOO_CURATOR_ENDPOINT`, `NIODOO_CURATOR_MODEL`, `NIODOO_CURATOR_TIMEOUT`).
- Run a quick evaluation:
  ```
  source ../venv/bin/activate
  python src/curator.py "<prompt>" "<granite response>"
  ```
  Output JSON includes `rouge_l`, parsed `quality_score`, and raw Granite feedback.

- ONNX model: defaults to `/workspace/models/Qwen-Embedding/onnx/model_fp16.onnx`
- Config: optional TOML via `NIODOO_EMBED_CONFIG` (falls back to Qwen2.5 Coder 0.5B preset)
- Runtime flags:
  - `NIODOO_EMBED_MODEL` – override model path
  - `NIODOO_EMBED_FORCE_CPU=true` – force CPU execution when CUDA is unavailable
  - `QWEN_CUDA_MEM_LIMIT_MB` – adjust GPU memory budget for the ONNX session

The `src/embedding.rs` helper wraps `tcs-ml::QwenEmbedder` with a mutexed stateful interface and batch helpers (`LocalEmbedder::embed_batch`).

## How to Use This Workspace
1. Follow Phase 0 tasks to set up environment and service scripts.
2. For each subsequent phase:
   - Start required services in documented order.
   - Implement the new capability under `src/`, `scripts/`, or `docs/`.
   - Run the associated soak test, capture metrics in `baselines/`, and update `CHANGELOG.md`.
   - Track progress using the pre-created TODO list (see `todo` panel).
3. Use the validation tooling under `docs/validation/` (metrics runner, cognitive benchmarks, Prometheus dashboards) to compare results.

## References
- **Forensics Report & Architecture Notes:** `/workspace/Niodoo-Final/SYSTEM_ARCHITECTURE.md`, `/workspace/Niodoo-Final/COMPONENT_INVENTORY.md`, `/workspace/Niodoo-Final/RUNPOD_ENDPOINT_RESEARCH_PROMPT.md`
- **Plan:** `/niodoo-reverse-ablation-setup.plan.md`
- **Existing Validation Stack:** `docs/validation/` (metrics runner, cognitive suites, baseline scripts)

All work in this lab must update `CHANGELOG.md` with real test evidence—no stubs, no fake data.
