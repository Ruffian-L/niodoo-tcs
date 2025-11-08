# Empirical Validation Plan for niodoo_real_integrated

## Executive Summary

This document defines the comprehensive validation framework for empirically validating the niodoo_real_integrated cognitive architecture. The framework moves beyond traditional performance metrics to include Quality Service Level Indicators (SLIs) that measure functional correctness and cognitive stability.

## Validation Objectives

1. **Foundational Integrity**: Verify dependency health and pipeline stability
2. **Cognitive Capabilities**: Validate long-term memory, complex reasoning, and meta-cognitive functions
3. **Component Contributions**: Quantify the impact of each architectural layer via ablation analysis
4. **Resilience**: Test system behavior under infrastructure failures and adversarial conditions
5. **Statistical Rigor**: Apply formal statistical methods for significance testing and effect size analysis

## I. Foundational Validation: Dependency and Pipeline Integrity

### 1.1 Service Dependency Health Check Matrix

All external dependencies are instrumented with Prometheus scraping:

- **vLLM**: `/metrics` endpoint (port 5001)
- **Qdrant**: `/metrics` endpoint (port 6333) + gRPC health checks
- **NVIDIA GPU**: nvidia-ml-py exporter (port 9400)
- **ONNX Runtime**: JSON profiling traces (not Prometheus)

See `docs/validation/PROMETHEUS_METRICS.md` for detailed metric names and health check queries.

### 1.2 Service Level Objectives (SLOs)

Comprehensive SLOs defined in Table 1 of the original validation plan. All SLOs are tracked via Prometheus alerts (`prometheus-alerts.yml`) and visualized in Grafana dashboards.

**Key Innovation**: Quality SLIs extend beyond latency/availability:
- **TCS Stability SLI**: Coefficient of variation of persistence_entropy (SLO: < 0.1)
- **RCE Governance SLI**: β_meta range compliance (SLO: in [0.8, 1.2])

## II. Cognitive and Behavioral Validation Framework

### 2.1 Long-Term Memory and Contextual Coherence

**Benchmark**: LoCoMo (Long-Context Conversational Memory)
- Context ingestion of full conversational histories
- Single-hop, multi-hop, temporal, and adversarial QA tasks
- Topological signature correlation analysis

**Status**: Integration pending (VAL-03-locomo)

### 2.2 Complex and Process-Aware Reasoning

**Benchmarks**:
- AQA-Bench: Interactive algorithmic reasoning (DFS tasks)
- DocPuzzle: Multi-step reasoning with checklist-guided process analysis
- CounterBench: Counterfactual reasoning validation

**Status**: Integration pending (VAL-03 tasks)

### 2.3 Meta-Cognitive and Self-Correction

**Benchmark**: CriticBench (Generation, Critique, Correction protocol)
- Tests CompassEngine and Curator feedback loop effectiveness

**Status**: Integration pending (VAL-03-criticbench)

## III. Ablation Analysis: Quantifying Component Contributions

### 3.1 Ablation Experiment Matrix

Defined experiments (ABL-001 through ABL-006) to isolate component contributions:

- **ABL-001**: RCE_ENABLED=0 (expected: CounterBench degradation, β_meta instability)
- **ABL-002**: nTokens layer bypassed (expected: DocPuzzle degradation)
- **ABL-003**: TCS_ENABLE_GPU=0 (expected: latency increase, no cognitive change)
- **ABL-004**: USE_GPU_FITNESS=0 (expected: latency increase, no cognitive change)
- **ABL-005**: Curator disabled (expected: CriticBench degradation)
- **ABL-006**: ERAG bypassed (expected: LoCoMo catastrophic degradation)

**Status**: Ablation runner pending (VAL-04-ablation-runner)

## IV. Resilience and Safety Validation

### 4.1 Chaos Engineering

Planned experiments:
- Dependency latency injection (NetworkChaos)
- Pod failure (PodChaos)
- CPU starvation (StressChaos)

**Steady State Definition**: System meets latency AND cognitive quality SLOs under baseline load.

**Status**: Chaos Mesh integration pending (VAL-06)

### 4.2 Adversarial Testing

Attack vectors:
- Prompt injection
- Jailbreaking
- Simulated data poisoning
- Affective alignment stress tests

**Status**: Red teaming pending (VAL-07)

## V. Statistical Rigor and Automated Governance

### 5.1 Statistical Analysis Protocol

**Bootstrap Analysis**: 95% confidence intervals for latency percentiles
**Effect Size**: Cohen's d for comparing distributions
**Hypothesis Testing**: Mann-Whitney U test for non-parametric comparisons

**Formal Rollback Criterion**: Regression requires investigation if:
- Statistically significant (p < 0.05) AND
- Medium or larger effect size (|Cohen's d| >= 0.5)

**Implementation**: `niodoo_real_integrated/src/validation/stats.rs`

### 5.2 CI/CD Integration

**Status**: GitHub Actions workflow pending (VAL-05-ci-workflow)

Planned workflow:
- Lightweight regression suite (1-minute latency barrage)
- 20 golden cognitive probes
- Topological stability check
- Blocks PR merge on regression detection

## VI. Implementation Status

### Completed (VAL-01, VAL-02)

- ✅ Prometheus scraping configuration
- ✅ Grafana dashboards (System Health, Cognitive Performance, Topological State)
- ✅ Quality SLI metrics infrastructure
- ✅ Prometheus alerting rules
- ✅ Metrics runner CLI tool
- ✅ Baseline storage and comparison scripts
- ✅ Statistical analysis library

### In Progress / Pending

- ⏳ Cognitive benchmark integration (VAL-03)
- ⏳ Ablation runner implementation (VAL-04)
- ⏳ CI workflow integration (VAL-05)
- ⏳ Documentation completion (VAL-06)

## Usage

See `RUNNING_TESTS.md` for practical execution instructions.

