# NIODOO-TCS Integration Battle Plan (2025-10-22)

This file distills the current system state, verified runs, and the next optimization targets following the 2025-10-22 rut gauntlet execution.

## Captured Baselines

| Run | Command | Cycles | Latency avg (ms) | ROUGE-L avg | Threat % | Healing % | Artifacts |
|-----|---------|--------|------------------|-------------|----------|-----------|-----------|
| Smoke | `cargo run -p niodoo_real_integrated --bin niodoo_real_integrated -- --prompt "soul-eater bug" --output=csv` | 1 | 1861.95 | 0.753 | 100% | 0% | `logs/2025-10-22-smoke*` |
| Rut Gauntlet | `cargo run -p niodoo_real_integrated --bin niodoo_real_integrated -- --output=csv` | 100 | 2003.61 | 0.652 | 55% | 9% | `logs/2025-10-22-rut-gauntlet*` |

## Gaps vs. Targets

- **Latency**: target < 500 ms; current 1.9–2.0 s. Requires caching + parallelism + continuous batching.
- **ROUGE-L**: target > 0.70; current 0.65 mean (0.48–0.82 range). Needs self-consistency pruning and temperature/topp tweaks.
- **Healing Rate**: target 100%; current 9%. Requires compass recalibration.

## Immediate Workstreams (Next Moves)

1. **Performance Pipeline**
   - Parallelize embed/compass/ERAG/vLLM stages using `tokio::join!` and `spawn_blocking` for MCTS.
   - Layered cache: L1 in-memory LRU → L2 Redis → L3 Qdrant.
   - Enable vLLM continuous batching (`--max-num-seqs 256`, `--enable-lora` for QLoRA updates).
   - Split workloads across Quadro (embedding/ERAG) and 5080-Q (generation).

2. **Echo Quality & ROUGE**
   - Claude/GPT echo pruning: limit to 512 tokens, strip constitutional fluff.
   - Add self-consistency ensemble (AirRAG style) and fallback to majority vote when variance > 0.1.
   - Adjust generation params: temperature 0.5, top_p 0.6, repetition penalty 1.2.

3. **Compass Calibration**
   - Tighten stagnation detection: require entropy variance < 0.1 and absolute entropy < 1.5 for yawn triggers.
   - Periodically recalibrate threat threshold based on last N cycles to converge to 40% threat / 100% healing.

4. **Observability**
   - Promote new metrics (latency histograms, threat/healing gauges, ROUGE percentiles) to Grafana.
   - Persist CSV→Parquet for longer-term trend analysis.

## Snapshot & Rollback

- Use `scripts/niodoo_snapshot.sh` to capture tarball snapshots under `backups/`.
- Recommended Git workflow:
  ```bash
  git checkout -b niodoo-stable-20251022
  git add logs/ scripts/niodoo_snapshot.sh docs/NIODOO_TCS_Battle_Plan.md
  git commit -m "snapshot: 2025-10-22 rut gauntlet baseline"
  ```
  Then branch off for experiments (`niodoo-cache-spike`, `niodoo-rouge-tune`, etc.).

## References

- MCTS-RAG (Yale EMNLP 2025) – validates compass ↔ retrieval design.
- AirRAG (arXiv 2501.10053) – provides self-consistency pattern for hybrid outputs.
- Non-Euclidean Affective Computing (iComputing 2025) – backs torus PAD / ghost manifold mapping.

Keep this file updated after each major run so we can diff baselines and avoid frying the NIODOO cortex.
