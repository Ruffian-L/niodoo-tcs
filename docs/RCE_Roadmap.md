# RCE Roadmap and Feature Flags

This document explains how to enable the Recursive Connectome Engine (RCE) in stages. Defaults are safe: shadow mode, metrics-only.

## Flags (RuntimeConfig)
- rce_enabled: false
- rce_shadow_mode: true
- rce_actions_enabled: false
- rce_window_seconds: 10
- rce_stride_seconds: 2
- rce_beta_meta_weights: { alpha_betti: 1.0, alpha_meta: 1.0, alpha_motif: 1.0, alpha_sheaf: 1.0 }
- rce_breakthrough_threshold: 0.5
- rce_consensus: { enabled: false, analyzers: 3, quorum: 2 }
- rce_erag_lambda: 0.0
- rce_archive_backend: "Qdrant"

## Stage 0 — Off
- rce_enabled=false
- No RCE computation; pipeline unaffected.

## Stage 1 — Metrics Only (Shadow)
- rce_enabled=true
- rce_shadow_mode=true
- rce_actions_enabled=false
- Effect: export β_meta, spectral gap, persistence entropy. No behavior change.

## Stage 2 — Safety On (Consensus)
- rce_enabled=true
- rce_shadow_mode=true
- rce_consensus.enabled=true
- Effect: compute and log approvals; still no behavior change.

## Stage 3 — Minimal Actions (Retry Gating)
- rce_enabled=true
- rce_shadow_mode=false
- rce_actions_enabled=true
- rce_consensus.enabled=true
- Effect: retries are gated by consensus. If not approved, skip retries.

## Stage 4 — Hyperfocus + Circuit Breaker
- Same as Stage 3
- Additional effect: approved β_meta spikes reduce temperature/top_p (uses configured increments) with a circuit breaker after 3 consecutive spikes.

## Stage 5 — ERAG Context Reordering
- rce_erag_lambda > 0.0
- Effect: topology-biased ordering of memory context prior to tokenization; scores remain untouched.

## Stage 6 — Curriculum Scheduling
- Same flags as Stage 5
- Effect: LearningLoop flushes curated buffer sooner under consolidation, waits for larger batches during exploration.

## Optional: Knot Feature
- `tcs-knot` is optional (feature `knot`). PH + persistent Laplacians are primary.

## Observability
- Prometheus metrics:
  - niodoo_rce_beta_meta_current, niodoo_rce_beta_meta_peak
  - niodoo_rce_laplacian_spectral_gap
  - niodoo_rce_persistence_entropy
  - niodoo_rce_beta_meta_spikes_total

## Rollback
- Set rce_actions_enabled=false and/or rce_shadow_mode=true to disable actions without removing metrics.
# Recursive Connectome Engine Roadmap

## Purpose
This document captures the agreed plan for evolving NIODOO’s topology stack into a production-ready Recursive Connectome Engine (RCE). It summarizes the current code findings, targeted upgrades, safety posture, and phased execution path so the team can review and sign off before implementation begins.

## Snapshot of Current State
- `tcs_analysis.rs` intermixes genuine metrics with heuristic placeholders (e.g., formatted “knot_polynomial”), leaving downstream stages without trustworthy topology signals.
- `tcs-tda` exposes a Vietoris–Rips helper that never constructs higher-order simplices, so “persistent features” are synthetic and blind to true H₁/H₂ structure.
- `LearningLoop` buffers recent topology snapshots but never derives β_meta or surfaces topology shifts to the broader system; observability hooks are missing.
- Curator, hyperfocus, and curriculum flows only see scalar heuristics rather than structured connectivity signals, preventing recursive architecture discovery.

## Target Outcomes
1. **Stable Topology Engine** – Persistent Laplacians, motif analytics, and GPU acceleration replace the heuristic knot/TQFT pipeline while keeping a deterministic analytic fallback.
2. **β_meta Signal** – Composite breakthrough metric (Betti derivatives, metastability, motif churn, sheaf divergence) drives learning, routing, and observability.
3. **Sheaf Reasoning Layer** – Sheaf neural modules represent heterogeneous causal structure and feed both Compass and ERAG contexts.
4. **Pipeline Integration** – RCE outputs thread through generation, curator feedback, curriculum selection, and metrics, enabling runtime architecture discovery.
5. **Safety Envelope** – Byzantine analyzer quorum, circuit breakers, and documentation prevent unsafe topology shifts and record every change.

## Implementation Phases
### Phase A – Topology Stack Audit (`audit-topology`)
- Replace heuristic TCS metrics with modular persistent Laplacian kernels (CubeCL) plus Ripser++ bindings.
- Asynchronous GPU job queue with deterministic analytic fallback; deprecate runtime Jones polynomial/TQFT stubs while retaining baseline signatures.
- Surface explicit feature payload (spectra, motifs, persistence) via typed structs for downstream consumers.

### Phase B – β_meta Metric Layer (`beta-meta-metric`)
- Design temporal buffers for Betti time-series, Laplacian eigenvalue derivatives, metastability (Kuramoto variance), motif churn, and sheaf entropy deltas.
- Provide composite β_meta API + Prometheus counters; wire into LearningLoop, HyperfocusDetector, and telemetry dashboards.

### Phase C – Sheaf Runtime (`sheaf-runtime`)
- Implement stalk/restriction map data model with variational regularizer on SO(n).
- Integrate with ERAG context windows, Dynamic Tokenizer promotions, and Compass quadrant updates.
- Train sheaf layers alongside existing generation signals with topology-aware losses.

### Phase D – Connectome Integration (`connectome-integration`)
- Update `process_prompt`/curator/learning flows to consume RCE outputs.
- Enrich curriculum selection, retry heuristics, and memory storage with topological descriptors.
- Extend Prometheus + logging schemas; ensure changelog entries for every structural change.

### Phase E – Safety & Alignment (`safety-bft`)
- Deploy diverse analyzer quorum (persistent Laplacian, spectral, sheaf) with PBFT-style consensus.
- Install β_meta circuit breakers, hyperfocus escalation thresholds, and sandbox restrictions for high-risk topology transitions.
- Document procedures and verification steps; embed reminders in CI/docs to update `CHANGELOG.md` per iteration.

## Validation & Metrics
- β_meta vs. ROUGE-L Pearson r ≥ 0.7 on benchmark suite.
- Breakthrough detection hit rate ≥ 80% on synthetic phase-transition datasets.
- Innovation rate uplift ≥ 2× on internal discovery tasks with curator enabled.
- Topology latency: <100 ms average per update with GPU path, <250 ms worst-case fallback.
- Safety intervention success ≥ 95% in staged treacherous-turn drills.

## Risks & Mitigations
- **GPU bottlenecks**: keep analytic fallback, pre-flight kernels with CubeCL test harness.
- **Signal drift**: maintain calibration suite comparing GPU vs. CPU Laplacians and enforce regression tests.
- **Alignment gaps**: treat β_meta spikes as gating events; require consensus + human confirmation for high-risk architectural rewrites.
- **Complexity creep**: modularize phases, enforce feature flags, and stage rollouts with opt-in configs.

## Next Steps
1. Review this roadmap and confirm scope, prioritization, and safety thresholds.
2. Once approved, implement Phase A with comprehensive tests and documentation updates.
3. Log every modification in `CHANGELOG.md` and attach observability dashboards before promoting to production.



