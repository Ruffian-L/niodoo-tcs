# Running Validation Tests - Practical Runbooks

This document provides step-by-step instructions for executing validation tests in the niodoo_real_integrated system.

## Prerequisites

1. **Services Running**:
   - vLLM server on port 5001 (with `--metrics` flag)
   - Qdrant server on port 6333
   - Prometheus server (optional, for real-time monitoring)
   - Grafana instance (optional, for dashboards)

2. **Environment Setup**:
   ```bash
   export VLLM_ENDPOINT=http://127.0.0.1:5001
   export QDRANT_URL=http://127.0.0.1:6333
   ```

## Baseline Capture

### Step 1: Capture Initial Baseline

```bash
./scripts/capture_baseline.sh
```

This will:
- Run a 60-second load test with 16 concurrent users
- Generate 2048-token responses
- Save metrics to `baselines/baseline-{timestamp}.json`
- Create `baselines/baseline-latest.json` symlink

**Expected Output**: Baseline JSON file with latency, throughput, quality SLIs, and topological metrics.

### Step 2: Verify Baseline Capture

```bash
cat baselines/baseline-latest.json | jq '.latency.p99_ms'
cat baselines/baseline-latest.json | jq '.quality_slis'
```

## Load Testing

### Basic Load Test

```bash
cargo run --bin metrics_runner -- \
    --scenario load_test \
    --concurrent-users 16 \
    --duration-secs 60 \
    --target-tokens 2048 \
    --output load_test_report.json
```

### High-Concurrency Load Test

```bash
cargo run --bin metrics_runner -- \
    --scenario load_test \
    --concurrent-users 32 \
    --duration-secs 120 \
    --target-tokens 4096 \
    --output high_load_report.json
```

### Mock Mode (No External Services)

```bash
MOCK_MODE=true cargo run --bin metrics_runner -- \
    --scenario load_test \
    --mock-mode \
    --output mock_test_report.json
```

## Comparison with Baseline

### Compare Current Run with Baseline

```bash
./scripts/compare_baseline.sh load_test_report.json
```

This will:
- Load baseline metrics
- Compute statistical differences (bootstrap CI, Cohen's d)
- Report regressions (statistically significant + medium+ effect size)
- Display SLO compliance status

### Manual Comparison

```bash
python3 <<EOF
import json

current = json.load(open('load_test_report.json'))
baseline = json.load(open('baselines/baseline-latest.json'))

p99_diff = current['latency']['p99_ms'] - baseline['latency']['p99_ms']
print(f"p99 latency change: {p99_diff:+.2f}ms")
EOF
```

## Monitoring During Tests

### View Prometheus Metrics

```bash
# Pipeline metrics
curl http://127.0.0.1:9090/metrics | grep niodoo_

# vLLM metrics (if available)
curl http://127.0.0.1:5001/metrics | grep vllm_

# Qdrant metrics
curl http://127.0.0.1:6333/metrics | grep qdrant_
```

### Check Quality SLIs

```bash
curl http://127.0.0.1:9090/metrics | grep quality_sli
```

Expected output:
```
niodoo_quality_sli_tcs_stability_cv 0.05
niodoo_quality_sli_rce_beta_meta_compliance 1.0
```

### Grafana Dashboards

1. Import dashboards from `grafana-dashboards/`:
   - System Health Dashboard
   - Cognitive Performance Dashboard
   - Topological State Dashboard

2. Configure Prometheus data source

3. View real-time metrics during test execution

## Troubleshooting

### Pipeline Initialization Fails

- Check service availability: `./test_services.sh`
- Verify config: `cargo run --bin metrics_runner -- --help`
- Check logs: `RUST_LOG=debug cargo run --bin metrics_runner -- ...`

### Metrics Not Appearing

- Verify `/metrics` endpoint: `curl http://127.0.0.1:9090/metrics`
- Check Prometheus scrape config: `prometheus.yml`
- Ensure `svc` feature is enabled: `cargo build --features svc`

### Baseline Comparison Fails

- Ensure baseline file exists: `ls baselines/baseline-latest.json`
- Check JSON format: `jq . baselines/baseline-latest.json`
- Verify Python dependencies: `python3 -c "import json, math"`

## Advanced Usage

### Custom Prompt File

```bash
cargo run --bin metrics_runner -- \
    --scenario load_test \
    --prompt-file custom_prompts.txt \
    --output custom_report.json
```

### Statistical Analysis Directly

```rust
use niodoo_real_integrated::validation::stats::*;

let baseline: Vec<f64> = vec![...];
let current: Vec<f64> = vec![...];

let (lower, upper) = bootstrap_percentile_ci(&current, 0.99, 10000, 0.95);
let effect_size = cohens_d(&baseline, &current);
let (u_stat, p_value) = mann_whitney_u(&baseline, &current);

if requires_regression_action(&baseline, &current, 0.05, 0.5) {
    println!("⚠️  Regression detected: requires investigation");
}
```

## Next Steps

After baseline capture and load testing:
1. Run cognitive benchmarks (when VAL-03 is implemented)
2. Execute ablation experiments (when VAL-04 is implemented)
3. Set up CI workflow (when VAL-05 is implemented)

