# 2025-10-22 Rut Gauntlet Run

**Command**
```
cargo run -p niodoo_real_integrated --bin niodoo_real_integrated -- --output=csv
```
(environment overrides: `http_proxy` / `https_proxy` unset)

**Input**
- Default rut gauntlet prompt set (100 prompts)

**Outputs**
- CSV: `logs/2025-10-22-rut-gauntlet.csv`
- Prometheus metrics: `logs/2025-10-22-rut-gauntlet.prom`
- JSON summary: `logs/2025-10-22-rut-gauntlet.json`
- Plots:
  - Entropy: `logs/2025-10-22-rut-gauntlet-entropy.png`
  - Latency: `logs/2025-10-22-rut-gauntlet-latency.png`
  - ROUGE: `logs/2025-10-22-rut-gauntlet-rouge.png`
- Full stdout/stderr:
  - `logs/2025-10-22-rut-gauntlet.stdout`
  - `logs/2025-10-22-rut-gauntlet.log`

**Aggregate Metrics**
- Cycles: 100
- Entropy: mean 1.862, σ 0.048, min 1.745, max 1.958
- Latency: mean 2003.61 ms, σ 119.32 ms, min 1730.09 ms, max 2345.45 ms
- ROUGE-L: mean 0.652, σ 0.072, min 0.488, max 0.821
- Threat rate: 55%
- Healing rate: 9%

**Targets vs. Reality**
- Entropy stability (σ < 0.3): ✅
- Threat coverage (>20%): ✅
- Healing coverage (~100%): ⚠️ low (9%)
- ROUGE-L (>0.7): ⚠️ 0.652 average
- Latency (<500 ms): ❌ 2003.61 ms average

**Notable Observations**
- vLLM latency fluctuates between 1.7–2.3 s with current single-request execution.
- Claude echo occasionally returns long “constitutional” monologues; pruning will likely boost ROUGE.
- Healing detections below desired level; compass thresholds need recalibration before live deployment.

**Next Actions (from battle plan)**
1. Implement 3-level cache + parallel stages to hit <500 ms goal.
2. Add self-consistency check + directive pruning to raise ROUGE above 0.7.
3. Tune compass stagnation/yawn thresholds to raise healing detections toward 100%.
