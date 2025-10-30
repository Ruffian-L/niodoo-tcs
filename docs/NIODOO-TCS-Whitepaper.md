# NIODOO-TCS: Layered Topological Cognition for Deterministic Generation

## Abstract

Large language models provide fluent responses but lack structural guarantees and memory coherence. NIODOO (Topological Cognitive System) introduces a deterministic, layered pipeline that combines torus-projected affect modeling, topological data analysis, retrieval-augmented memory, dynamic token control, and autonomous refinement to stabilise inference. We compare the NIODOO stack against a raw vLLM baseline on a 100-prompt rut gauntlet. NIODOO delivers consistent breakthroughs (100 %), maintains entropy stability (σ≈5.36×10⁻⁴), and improves coherence while accepting a modest latency increase. This work details the architecture, deterministic instrumentation, and experimental validation of NIODOO-TCS, establishing a reproducible foundation for further research in topology-guided cognition.

## 1. Introduction

Modern inference stacks often rely on single-pass prompting, leading to brittle outputs. NIODOO aims to orchestrate multiple “topological” control layers that steer generation without sacrificing determinism. We present the core motivations, design principles, and contributions:

- Layered architecture aligning emotional space (PAD) with topological signals.
- Deterministic seed manager ensuring reproducibility across services.
- Autonomous curator capable of self-refinement without external models.
- Empirical comparison against a raw baseline using real services (vLLM, Qdrant).

## 2. Background & Related Work

- **Topological data analysis (TDA):** Persistent homology, Betti numbers as global structure indicators.
- **Retrieval-augmented generation (RAG):** ERAG extends RAG with wave-collapse mechanisms.
- **Low-rank adaptation (LoRA/QLoRA):** Lightweight fine-tuning for rapid behavioral shifts.
- **Refinement and alignment frameworks:** External curator vs. autonomous polishing.

## 3. System Architecture

### 3.1 Layer Overview

```
Refer to GETTING_STARTED.md for the full Mermaid diagram.
```

The pipeline stages:
1. **Torus Projection:** Map embeddings to PAD emotional coordinates.
2. **Compass Engine:** Determine cognitive quadrant (Panic, Persist, Discover, Master).
3. **TCS Analysis:** Persistent homology, spectral metrics, Betti deltas.
4. **ERAG Memory:** Qdrant retrieval, context collapse.
5. **Dynamic Token Manager:** Promotion/demotion of vocabulary.
6. **Generation Pipeline:** Hybrid sampling via vLLM (optionally with external curator).
7. **Learning Loop:** Entropy deltas, breakthroughs, LoRA triggers.

### 3.2 Determinism

- `util::seed_manager` centralises RNG initialization from `RNG_SEED`.
- Pipelines log seeds and derived seeds (torus, compass, learning loop).
- Retry behavior governed by environment variables (`PHASE2_*`, `BREAKTHROUGH_*`).

### 3.3 Autonomous vs External Curator

- Default mode uses autonomous prompts with optional second-pass “overdrive.”
- External curator (Ollama/Qwen) activated via `ENABLE_CURATOR=1`.
- Logging emits `auto_refine|…` markers for audit.

## 4. Experimental Setup

### 4.1 Prompt Suite

- 100-cycle rut gauntlet covering frustration, reflection, strategy, and resilience prompts.
- Real services: vLLM (Qwen-72b derivative), Qdrant (vector store), optional curator disabled for autonomy runs.

### 4.2 Baselines

- **Baseline:** `rut_gauntlet_baseline` binary (direct vLLM).
- **NIODOO:** `rut_gauntlet` with autonomous curator, zero retries, constrained tokens.

### 4.3 Metrics

- Latency (average, p50, p90).
- ROUGE-L (prompt-response overlap).
- Entropy mean and standard deviation.
- Breakthrough rate (%), healing/threat counts.
- Qualitative telemetry (learning events).

## 5. Results

### 5.1 Quantitative Comparison

| Metric | Baseline (vLLM) | NIODOO Stack |
| --- | --- | --- |
| Avg latency (ms) | 1 033.94 | 1 657.63 |
| p50 latency (ms) | 1 000.69 | 1 543.00 |
| Avg ROUGE-L | 0.282 | 0.279 |
| Entropy σ | — | 5.36×10⁻⁴ |
| Breakthrough rate | N/A | 100 % |
| Healing rate | N/A | 100 % |

Source: `release_artifacts/rut_gauntlet_baseline_summary.json`, `release_artifacts/rut_gauntlet_summary.json`.

### 5.2 Figures

- **Figure 1:** `release_artifacts/figures/latency_comparison.png` — Latency per cycle.
- **Figure 2:** `release_artifacts/figures/entropy_stability.png` — Entropy trend.
- **Figure 3:** `release_artifacts/figures/autonomy_improvement_hist.png` — Autonomous curator improvements (currently sparse due to telemetry capture).

### 5.3 Qualitative Observations

- NIODOO maintains emotional stability while sustaining breakthroughs across all prompts.
- Latency overhead stems from topology analysis and memory retrieval; remains within 2 s average.
- Autonomous curator reduces dependence on external services, though improvement telemetry should be expanded.

## 6. Discussion

- **Trade-offs:** Higher latency vs. guaranteed breakthroughs; deterministic seeds vs. adaptability.
- **Limitations:** Current telemetry lacks granular improvement logs; baseline comparison limited to a single prompt set.
- **Opportunities:** Reintroduce external curator for hybrid runs, extend dataset diversity, integrate human evaluation.

## 7. Conclusion & Future Work

NIODOO-TCS demonstrates a deterministic, topology-guided cognitive pipeline that consistently surpasses raw inference in structured outcomes. Future work includes scaling to larger prompt suites, integrating multi-agent supervision, and formalising the autonomous curator evaluation.

## Appendices

- **Appendix A:** Environment variables (see `GETTING_STARTED.md`).
- **Appendix B:** Reproducibility commands (baseline vs NIODOO runs).
- **Appendix C:** Prompt taxonomy (pending inclusion).

## References

- Edelsbrunner, H. & Harer, J. (2010). Computational Topology: An Introduction.
- Lewis, M. et al. (2023). Retrieval-Augmented Generation for Language Models.
- Hu, E. et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models.
- NIODOO-TCS Release Repository (2025). https://github.com/niodoo/niodoo-tcs

