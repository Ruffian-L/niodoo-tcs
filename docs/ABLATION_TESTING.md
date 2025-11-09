# Ablation Testing Guide

## Overview

Ablation testing systematically disables components to quantify their contribution to system performance and cognitive capabilities. This replaces traditional unit/integration tests with empirical evidence of component value.

## Purpose

**Prove System Superiority**: Demonstrate that each component provides measurable value through statistical comparison of system behavior with and without the component.

## Ablation Runner

The `ablation_runner` binary runs systematic ablation experiments:

```bash
cargo run --bin ablation_runner -- \
    --experiment DisableRce \
    --baseline baselines/baseline-latest.json \
    --concurrent-users 16 \
    --duration-secs 60 \
    --output-dir ablation_results
```

## Available Experiments

### Single Component Ablations

- **ABL-001: DisableRce** - Disables RCE (Recursive Connectome Engine) layer
- **ABL-002: BypassNTokens** - Bypasses nTokens topology feature extraction
- **ABL-003: DisableTcsGpu** - Disables GPU acceleration for TCS analysis
- **ABL-004: DisableGpuFitness** - Disables GPU fitness calculation
- **ABL-005: DisableCurator** - Disables Curator quality assessment
- **ABL-006: BypassErag** - Bypasses ERAG memory retrieval (zero-shot mode)
- **ABL-007: DisableCompass** - Disables Compass engine
- **ABL-008: DisableLearning** - Disables Learning loop
- **ABL-009: DisableTcs** - Disables TCS topology analysis
- **ABL-010: DisableTokenizer** - Disables Tokenizer promotion

### Multi-Component Ablations

- **ABL-011: DisableRceAndNTokens** - Disables both RCE and nTokens
- **ABL-012: DisableCuratorAndErag** - Disables both Curator and ERAG

## Statistical Analysis

Each ablation experiment produces:

- **P-value**: Statistical significance (p < 0.05 = significant)
- **Cohen's d**: Effect size (Small < 0.2, Medium < 0.5, Large < 0.8, Very Large >= 0.8)
- **95% Confidence Interval**: Bootstrap confidence interval for metrics
- **Component Contribution Score**: Combined effect size and significance
- **Superiority Proof**: Automated recommendation based on statistical evidence

## Interpreting Results

### Critical Components
If `regression_detected == true` and `p_value < 0.05`:
- Component is **ESSENTIAL**
- Removing it causes significant degradation
- Recommendation: "CRITICAL: Component is essential"

### Important Components
If `cohens_d.abs() > 0.5` and `p_value < 0.05`:
- Component provides **SUBSTANTIAL VALUE**
- Removing it measurably degrades performance
- Recommendation: "IMPORTANT: Component provides substantial value"

### Optional Components
If `cohens_d.abs() < 0.5` or `p_value >= 0.05`:
- Component has **MINIMAL MEASURABLE IMPACT**
- May be optimized or removed if complexity cost is high
- Recommendation: "OPTIONAL: Component has minimal measurable impact"

## Example Output

```json
{
  "experiment": "DisableRce",
  "comparison": {
    "latency_change_p99_ms": 45.2,
    "latency_change_pct": 12.5,
    "p_value": 0.0032,
    "cohens_d_latency": 0.85,
    "regression_detected": true,
    "component_contribution_score": 0.82,
    "superiority_proof": {
      "effect_size_category": "Very Large",
      "recommendation": "CRITICAL: Component is essential. Removing it causes significant degradation (p=0.0032, d=0.85)"
    }
  }
}
```

## Running All Experiments

```bash
# Run all single-component ablations
for exp in DisableRce BypassNTokens DisableTcsGpu DisableGpuFitness DisableCurator BypassErag DisableCompass DisableLearning DisableTcs DisableTokenizer; do
    cargo run --bin ablation_runner -- \
        --experiment $exp \
        --baseline baselines/baseline-latest.json \
        --output-dir ablation_results
done

# Generate superiority proof
./scripts/run_superiority_proof.sh
```

## Best Practices

1. **Always compare against baseline**: Use `--baseline` to compare with known good configuration
2. **Run multiple iterations**: Ablation results can vary; run 3-5 times for confidence
3. **Check statistical significance**: Only trust results with p < 0.05
4. **Consider effect size**: Large effect sizes (|d| > 0.8) indicate critical components
5. **Review recommendations**: The superiority proof provides actionable insights

## Integration with CI/CD

Ablation tests can be integrated into CI/CD pipelines:

```yaml
- name: Run Ablation Tests
  run: |
    cargo run --bin ablation_runner -- \
      --experiment DisableRce \
      --baseline baselines/baseline-latest.json \
      --output-dir ablation_results
    # Fail if critical component shows regression
    jq -e '.comparison.regression_detected == false' ablation_results/*.json
```






