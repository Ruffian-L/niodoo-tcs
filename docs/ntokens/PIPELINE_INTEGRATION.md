# nToken Pipeline Integration Plan

This document enumerates the concrete changes required to embed Topological Connection Tokens (nTokens) within the NIODOO pipeline. It focuses on `niodoo_real_integrated` and its immediate dependencies.

## 1. Stage Placement Overview

```
Security → Embedding → ERAG Retrieval → **NToken Synthesis** → Torus/TCS → Compass → Tokenizer → Generation → Curator → Consonance → Failure → Learning → Memory
```

- Insert a dedicated stage (`PipelineStage::NTokenSynthesis`) between ERAG retrieval and Torus projection in `pipeline/stages.rs`.
- Ensure the stage executes regardless of topology mode so downstream components can consume nToken metadata even when TCS is disabled.

## 2. Core Pipeline Changes

### 2.1 `pipeline/core.rs`

1. **Builder Invocation**
   - Add a method `Pipeline::build_ntoken(&mut self, ctx: &mut PromptContext) -> Result<NToken>`.
   - Inputs: parsed prompt, ERAG results, runtime config, previous cycle nToken (if any), timestamp.
   - Outputs: populated `NToken` plus instrumentation data.

2. **Processing Flow**
   - Call `build_ntoken` immediately after ERAG retrieval.
   - Store the produced nToken in `ctx.current_ntoken` (new field on `PromptContext`).
   - Pass references to Torus, TCS, Compass, Curator, Learning, and Memory stages.

3. **Async Boundaries**
   - `build_ntoken` executes GPU-heavy work; wrap in `tokio::task::spawn_blocking` with CUDA stream guard to avoid holding async runtime threads.
   - Ensure new stage updates `self.timings.ntoken_synthesis`.

4. **Config Wiring**
   - Extend `RuntimeConfig` with `ntoken` section (filtration params, kernel tuning).
   - Validate config during `initialise` (e.g., check Multipers availability, GPU mode).

### 2.2 `pipeline/stages.rs`

1. Define `PipelineStage::NTokenSynthesis` enum variant.
2. Implement `StageExecutor` arm invoking `Pipeline::build_ntoken` and recording metrics.
3. Add Prometheus gauge/histogram for synthesis latency and GPU occupancy.

### 2.3 `pipeline/context.rs`

- Introduce `current_ntoken: Option<Arc<NToken>>` in `PromptContext`.
- Provide helper methods to clone references for downstream stages.

## 3. Downstream Consumers

### 3.1 Torus Projection / TCS

- Accept optional nToken reference; when provided, use `TopologySignature` to prime spectral calculations and skip redundant persistence runs.
- Cache results so TCS reuses nToken persistence data.

### 3.2 Compass Engine

- Consume `ValueProfile` and `TemporalFlux` to adjust quadrant decisions (e.g., high constraint tension triggers caution state).
- Log hyperbolic norm deltas for cascade tracker.

### 3.3 Token Manager

- Use nToken attention hints to bias segmentation around high persistence regions.

### 3.4 Generation Engine

- Provide nToken context to prompt templates (e.g., include topological cues in system message when enabled).

### 3.5 Curator

- Integrate `sheaf.cohomology` and `metrics` for failure detection, enabling topological anomaly escalation.
- Update learning signals to feed persistence entropy and constraint violation rates.

### 3.6 Learning Loop and Memory

- Expand `CuratedExperience` to store nToken metadata.
- Ensure QLoRA training datapoints retain persistence signatures for topology-aware finetuning.

## 4. Metrics & Telemetry

- New Prometheus metrics (`metrics_topology_full.prom` additions):
  - `ntoken_synthesis_latency_seconds`
  - `ntoken_persistence_pairs_total`
  - `ntoken_constraint_violation_ratio`
  - `ntoken_hyperbolic_norm`
- Tracing spans: `ntoken.build`, `ntoken.persistence`, `ntoken.sheaf_diffusion`.
- Structured logs summarizing key invariants per prompt.

## 5. Error Handling

- Graceful degradation path: if nToken synthesis fails, log error, emit metric, continue pipeline with fallback (skip TCS reuse, degrade to legacy behavior) while triggering alert.
- Provide `ntoken_mandatory` config flag for environments that must fail fast.

## 6. Testing Checklist

- Extend soak/load tests to ensure `NTokenSynthesis` stage runs without deadlocks.
- Add integration tests mocking nToken builder outputs, verifying stage wiring.
- Update golden metrics for `metrics_topology_full.prom` to include new gauges.

## 7. Rollout Considerations

- Default `ntoken_enabled = false` in production until benchmarks confirm latency budgets.
- Provide feature flag per environment (config + CLI) to simplify A/B testing.
- Document manual override path in `docs/ntokens/OPERATIONS.md` (future work).

This plan establishes the precise code touchpoints needed for nToken adoption. Subsequent documents detail module implementation and GPU execution specifics.



