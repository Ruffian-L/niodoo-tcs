# Baseline Storage Directory

This directory contains captured baseline metrics for comparison with current runs.

## Files

- `baseline-YYYYMMDD_HHMMSS.json`: Timestamped baseline captures
- `baseline-latest.json`: Symlink to the most recent baseline

## Usage

```bash
# Capture a new baseline
./scripts/capture_baseline.sh

# Compare current run with baseline
./scripts/compare_baseline.sh metrics_report.json [baseline.json]
```

## Baseline Format

Baselines are JSON files containing:
- Latency metrics (p50, p95, p99, mean, min, max)
- Throughput metrics (requests/sec, tokens/sec, embeddings/sec)
- Quality SLIs (TCS stability CV, RCE β_meta compliance)
- Topological metrics (persistence entropy, spectral gap, Betti numbers)
- Cognitive metrics (if available)

## Statistical Analysis

The comparison script performs:
- Bootstrap confidence intervals for percentile metrics
- Cohen's d effect size calculation
- SLO compliance checking

