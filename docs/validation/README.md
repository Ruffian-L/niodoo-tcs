# NIODOO Validation Framework Documentation

This directory contains documentation for the empirical validation framework implemented for the niodoo_real_integrated cognitive architecture.

## Overview

The validation framework provides comprehensive instrumentation, benchmarking, and statistical analysis capabilities to empirically validate the system's cognitive capabilities and performance characteristics.

## Documentation Files

- **VALIDATION_PLAN.md**: Complete validation plan and methodology
- **RUNNING_TESTS.md**: Practical runbooks for executing validation tests
- **PROMETHEUS_METRICS.md**: Service dependency metrics documentation

## Quick Start

### 1. Capture Baseline

```bash
./scripts/capture_baseline.sh
```

This runs a standardized test suite and saves golden metrics to `baselines/baseline-{timestamp}.json`.

### 2. Run Load Test

```bash
cargo run --bin metrics_runner -- \
    --scenario load_test \
    --concurrent-users 16 \
    --duration-secs 60 \
    --target-tokens 2048 \
    --output metrics_report.json
```

### 3. Compare with Baseline

```bash
./scripts/compare_baseline.sh metrics_report.json
```

## Validation Components

### Foundational Observability (VAL-01)

- **Prometheus Configuration**: `prometheus.yml` - Scrape configs for all dependencies
- **Grafana Dashboards**: `grafana-dashboards/` - System health, cognitive performance, topological state
- **Quality SLIs**: Extended metrics.rs with TCS stability and RCE governance tracking
- **Alerting Rules**: `prometheus-alerts.yml` - SLO breach alerts

### Metrics Runner (VAL-02)

- **CLI Tool**: `niodoo_real_integrated/src/bin/metrics_runner.rs`
- **Scenarios**: LoadTest, Baseline, Cognitive
- **Output**: Structured JSON reports with latency, throughput, quality SLIs, topological metrics

### Statistical Analysis (VAL-05)

- **Library**: `niodoo_real_integrated/src/validation/stats.rs`
- **Functions**: Bootstrap CI, Cohen's d, Mann-Whitney U test
- **Usage**: Integrated into comparison scripts and CI workflow

## SLOs (Service Level Objectives)

See Table 1 in VALIDATION_PLAN.md for complete SLO definitions. Key targets:

- **PromptSecurityManager**: p99 latency < 10ms, 99.99% availability
- **QwenStatefulEmbedder**: p99 latency < 50ms, >500 embeddings/sec/core
- **ERAG**: p99 latency < 200ms, 99.9% gRPC success rate
- **TCSAnalyzer**: p99 latency < 150ms, stability CV < 0.1
- **GenerationEngine**: TTFT p99 < 500ms @ 16 users, >3000 tokens/sec
- **RCE Layer**: β_meta in [0.8, 1.2] range

## Quality SLIs

Unlike traditional metrics, Quality SLIs measure functional correctness:

- **TCS Stability**: Coefficient of variation of persistence_entropy (lower is better)
- **RCE Governance**: β_meta range compliance (1.0 = compliant, 0.0 = non-compliant)

## Next Steps

- VAL-03: Cognitive benchmark integration (LoCoMo, AQA-Bench, DocPuzzle, CounterBench, CriticBench)
- VAL-04: Ablation testing framework
- VAL-05: CI integration with regression detection

