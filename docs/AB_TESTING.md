# A/B Testing Guide

## Overview

A/B testing compares baseline vs treatment configurations to prove system superiority through statistical comparison. This replaces traditional integration tests with empirical evidence.

## Purpose

**Prove Configuration Superiority**: Demonstrate that one configuration outperforms another through rigorous statistical analysis of performance and quality metrics.

## A/B Test Runner (Rust)

The `ab_test_runner` binary compares two configurations:

```bash
cargo run --bin ab_test_runner -- \
    --baseline-name baseline \
    --treatment-name treatment \
    --baseline-config configs/baseline.json \
    --treatment-config configs/treatment.json \
    --concurrent-users 16 \
    --duration-secs 60 \
    --output-dir ab_test_results
```

## Python A/B Test Framework

The `ab_test_comprehensive.py` script provides Python-based A/B testing:

```bash
python3 scripts/ab_test_comprehensive.py
```

### Custom Configuration

```python
from scripts.ab_test_comprehensive import ABTestFramework

framework = ABTestFramework()

baseline_config = {
    "TOPOLOGY_MODE": "hybrid",
    "RCE_ENABLED": "1",
    "N_TOKENS_BYPASS": "0",
}

treatment_config = {
    "TOPOLOGY_MODE": "hybrid",
    "RCE_ENABLED": "0",  # Disabled
    "N_TOKENS_BYPASS": "0",
}

prompts = [
    "Explain quantum computing",
    "Describe machine learning",
]

result = framework.run_ab_test(
    baseline_config, 
    treatment_config, 
    prompts,
    duration_secs=60
)
framework.print_report(result)
```

## Metrics Collected

- **Latency**: P50, P95, P99, mean
- **Throughput**: Requests per second
- **Error Rate**: Percentage of failed requests
- **Request Count**: Total requests processed

## Statistical Analysis

Each A/B test produces:

- **P-values**: Statistical significance for latency and throughput
- **Cohen's d**: Effect sizes for latency and throughput
- **Confidence Intervals**: 95% bootstrap confidence intervals
- **Effect Size Categories**: Small, Medium, Large, Very Large
- **Winner Determination**: Automated winner based on statistical significance

## Interpreting Results

### Statistically Significant Winner

If `statistical_significance == true`:
- Winner is determined with confidence (p < 0.05)
- Effect sizes indicate magnitude of difference
- Recommendations can be made based on results

### Inconclusive Results

If `winner == "Inconclusive"`:
- No statistically significant difference detected
- May need longer test duration or more samples
- Consider running additional iterations

## Example Output

```json
{
  "baseline_name": "baseline",
  "treatment_name": "treatment",
  "baseline_metrics": {
    "latency_p99": 450.2,
    "throughput": 12.5
  },
  "treatment_metrics": {
    "latency_p99": 380.5,
    "throughput": 15.2
  },
  "comparison": {
    "latency_difference_ms": -69.7,
    "latency_difference_pct": -15.5,
    "throughput_difference_pct": 21.6,
    "p_value_latency": 0.0021,
    "p_value_throughput": 0.0015,
    "cohens_d_latency": -0.92,
    "cohens_d_throughput": 0.85,
    "effect_size_latency": "Very Large",
    "effect_size_throughput": "Very Large",
    "winner_metrics": ["Lower Latency", "Higher Throughput"]
  },
  "winner": "Treatment",
  "statistical_significance": true
}
```

## Common Test Scenarios

### 1. Component Enable/Disable

```bash
# Test impact of disabling RCE
cargo run --bin ab_test_runner -- \
    --baseline-name "with_rce" \
    --treatment-name "without_rce" \
    --baseline-config <(echo '{"RCE_ENABLED":"1"}') \
    --treatment-config <(echo '{"RCE_ENABLED":"0"}')
```

### 2. Configuration Optimization

```bash
# Test different topology modes
cargo run --bin ab_test_runner -- \
    --baseline-name "baseline_mode" \
    --treatment-name "hybrid_mode" \
    --baseline-config <(echo '{"TOPOLOGY_MODE":"baseline"}') \
    --treatment-config <(echo '{"TOPOLOGY_MODE":"hybrid"}')
```

### 3. Performance Tuning

```bash
# Test GPU acceleration impact
cargo run --bin ab_test_runner -- \
    --baseline-name "cpu_only" \
    --treatment-name "gpu_accelerated" \
    --baseline-config <(echo '{"TCS_ENABLE_GPU":"0"}') \
    --treatment-config <(echo '{"TCS_ENABLE_GPU":"1"}')
```

## Best Practices

1. **Run sufficient duration**: At least 60 seconds for stable results
2. **Use concurrent users**: 16+ concurrent users for realistic load
3. **Check prerequisites**: Ensure vLLM and Qdrant are running
4. **Verify statistical significance**: Only trust results with p < 0.05
5. **Consider effect sizes**: Large effect sizes (|d| > 0.8) indicate meaningful differences
6. **Run multiple iterations**: A/B test results can vary; run 3-5 times for confidence

## Integration with Superiority Proof

A/B test results are automatically aggregated by the superiority proof generator:

```bash
./scripts/run_superiority_proof.sh
```

This aggregates all A/B test results and generates comprehensive superiority reports.

## Troubleshooting

### vLLM Not Available
```
Error: Failed to connect to vLLM endpoint
```
**Solution**: Ensure vLLM is running on port 5001

### Qdrant Not Available
```
Error: Failed to connect to Qdrant
```
**Solution**: Ensure Qdrant is running on port 6333

### Inconclusive Results
**Solution**: 
- Increase test duration (`--duration-secs 120`)
- Increase concurrent users (`--concurrent-users 32`)
- Run multiple iterations and aggregate results


