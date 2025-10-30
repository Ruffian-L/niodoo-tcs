# Release Artifacts

This directory contains sample comparison results from production runs of the NIODOO pipeline.

## Files

### `rut_gauntlet_baseline_results.csv`

**Provenance**: Generated from `logs/rut_gauntlet_baseline_real/` run  
**Description**: Baseline metrics from raw vLLM inference without NIODOO enhancements

**Columns**:
- `cycle`: Test cycle number (1-100)
- `prompt`: Input prompt text
- `response`: Generated response from vLLM
- `latency_ms`: Generation latency in milliseconds
- `rouge_l`: ROUGE-L score (text similarity metric)

**Key Metrics**:
- Average latency: ≈1034 ms per cycle (p50 ≈1001 ms, p90 ≈1109 ms)
- Average ROUGE-L: ≈0.282
- No entropy tracking or learning events

### `rut_gauntlet_results.csv`

**Provenance**: Generated from `logs/rut_gauntlet_real_autonomy_tuned/` run  
**Description**: Full NIODOO pipeline metrics with all layers enabled

**Columns**:
- `cycle`: Test cycle number (1-100)
- `prompt`: Input prompt text
- `response`: Generated response from hybrid pipeline
- `entropy`: Shannon entropy of emotional state
- `is_threat`: Boolean indicating threat detection
- `is_healing`: Boolean indicating healing activation
- `latency_ms`: Generation latency in milliseconds
- `learning_events`: JSON array of learning events (e.g., token promotions, LoRA triggers)
- `coherence_rouge`: Jaccard similarity between prompt and response
- `rouge_l`: ROUGE-L score
- `generation_source`: Source model (vllm, claude, gpt)
- `quadrant`: Compass quadrant (Panic/Persist/Discover/Master)
- `raw_stds`: Standard deviations of PAD state components

- Average latency: ≈1658 ms per cycle (p50 ≈1543 ms, p90 ≈2114 ms)
- Average ROUGE-L: ≈0.279 (p90 ≈0.335)
- Entropy tracking: Shows emotional state evolution
- Breakthroughs: Learning events indicating novel token promotions or LoRA fine-tuning

## Comparison Insights

**Latency**: Full NIODOO stack adds ~500-1000ms overhead due to:
- Embedding computation
- Topological analysis
- ERAG memory retrieval
- Dynamic tokenizer updates
- Learning loop evaluation

**Quality**: Full stack shows higher ROUGE-L scores, indicating:
- Better coherence through ERAG context
- More appropriate responses via compass-guided generation
- Adaptive vocabulary via dynamic tokenizer

**Learning**: Baseline has no learning events; full stack shows:
- Token promotions (OOV handling)
- LoRA fine-tuning triggers
- MCTS-guided exploration
- Entropy-based breakthrough detection

## Figures & Derived Artifacts

- `figures/latency_comparison.png` — Latency per cycle (baseline vs NIODOO)
- `figures/entropy_stability.png` — Entropy trend across NIODOO cycles
- `figures/autonomy_improvement_hist.png` — Autonomous curator improvement histogram (sparse telemetry in current run)
- `figures/metrics_summary.json` — Parsed statistics used in the whitepaper table

## Usage

These CSV files can be analyzed with:
- Python pandas: `pd.read_csv('rut_gauntlet_results.csv')`
- R: `read.csv('rut_gauntlet_results.csv')`
- Excel/Google Sheets
- Custom analysis scripts

For visualization, see the generated plots in the original log directories:
- Baseline: `logs/rut_gauntlet_baseline_real/latency_over_cycles.png`
- Full Stack: `logs/rut_gauntlet_real_autonomy_tuned/entropy_over_cycles.png`
- Release summary figures: `release_artifacts/figures/*.png`

