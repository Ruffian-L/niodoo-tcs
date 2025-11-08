# Pull Request

## Description

Brief description of changes...

## Changes Made

- [ ] Change 1
- [ ] Change 2

## Validation Impact

**Required**: All PRs modifying performance-sensitive or cognitive components must include quantitative metrics.

### Performance Impact

Run metrics_runner and include results:

```bash
cargo run --bin metrics_runner -- --scenario load_test --output pr_metrics.json
./scripts/compare_baseline.sh pr_metrics.json
```

**Metrics Summary**:
- p99 latency: X ms (baseline: Y ms, diff: +Z ms)
- Throughput: X tokens/sec (baseline: Y tokens/sec)
- Quality SLIs:
  - TCS stability CV: X (baseline: Y)
  - RCE β_meta compliance: X (baseline: Y)

### Statistical Analysis

- Bootstrap CI for p99: [lower, upper]
- Cohen's d effect size: X
- Mann-Whitney U p-value: X

### Regression Status

- [ ] No regression detected
- [ ] Regression detected but within acceptable tolerance
- [ ] Regression requires investigation (statistically significant + medium+ effect size)

### Cognitive Impact

If modifying cognitive components (ERAG, TCS, RCE, Curator, nTokens):

- [ ] LoCoMo benchmark scores: (if applicable)
- [ ] AQA-Bench success rate: (if applicable)
- [ ] Topological signature changes: (if applicable)

## Testing

- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Metrics runner baseline comparison passes
- [ ] Manual testing completed

## Related Issues

Fixes #X

## Checklist

- [ ] Code follows project style guidelines
- [ ] CHANGELOG.md updated
- [ ] Documentation updated (if applicable)
- [ ] Validation Impact section completed
- [ ] No new linter errors

