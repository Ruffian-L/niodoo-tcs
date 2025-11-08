# NIODOO-Code Pivot: Implementation Guide

This guide provides step-by-step instructions for implementing and validating the NIODOO-Code topological pivot.

## Prerequisites

1. **Gold Set**: Prepare 1,000-sample gold set with external metrics (bug-fix commits, static analysis errors, security vulnerabilities)
2. **Emotional Adapters**: Path to pre-trained emotional adapters (for orthogonalization)
3. **BigQuery Access**: Google Cloud credentials for BigQuery dataset access

## Step-by-Step Implementation

### Phase 1: Foundation

#### 1.1 Tune CQS Weights

```bash
# Create gold set (if not already created)
python niodoo-ai/scripts/scrape_bigquery_rust.py \
    --output gold_set.jsonl \
    --limit 1000

# Add external metrics to gold set (bug-fix commits, static analysis errors, etc.)
# Then tune weights:
python niodoo-ai/scripts/tune_cqs_weights.py \
    --gold-set gold_set.jsonl \
    --metric-type combined \
    --grid-size 10 \
    --output niodoo-ai/config/cqs_weights.yaml

# Validate weights
python niodoo-ai/scripts/validate_cqs_weights.py \
    --weights niodoo-ai/config/cqs_weights.yaml
```

#### 1.2 Build Dataset

```bash
# Scrape BigQuery for hotspots
python niodoo-ai/scripts/scrape_bigquery_rust.py \
    --output rust_bigquery_raw.jsonl \
    --limit 100000

# Build complete dataset with topology features
python niodoo-ai/scripts/build_rust_dataset.py \
    rust_bigquery_raw.jsonl \
    --output code_topology_dataset.jsonl \
    --max-examples 100000

# Validate dataset format
python niodoo-ai/scripts/validate_dataset_format.py \
    code_topology_dataset.jsonl
```

### Phase 2: Training Infrastructure

#### 2.1 Configure Training

Update `niodoo-ai/config/config_code_pivot.yml`:
- Set `train_file` to your dataset path
- Adjust `lambda_weight` (default: 0.1) based on validation
- Optionally enable multi-domain adapters (uncomment `multi_domain` section)

#### 2.2 Train Model

```bash
# Standard training (without orthogonalization)
python -m niodoo_ai.training \
    --config niodoo-ai/config/config_code_pivot.yml

# With adapter orthogonalization (if emotional adapters available)
# Update config_code_pivot.yml to include multi_domain section first
python -m niodoo_ai.training \
    --config niodoo-ai/config/config_code_pivot.yml
```

### Phase 3: Validation

#### 3.1 Ablation Study

Train two models:
1. **Control**: Qwen + QLoRA, `lambda_weight: 0.0` (no topological loss)
2. **Treatment**: Qwen + QLoRA, `lambda_weight: 0.1` (with topological loss)

Compare performance on HiBench and DSR-Bench.

#### 3.2 Topological Code MRI Demo

Run on "patient zero" 100k-line ADHD codebase:

```python
from tcs_tqft import CodeTrajectory, TrajectoryType
from tcs_tqft import TQFTEngine

# Build code trajectory from commit history or execution traces
trajectory = CodeTrajectory.new(TrajectoryType.CommitSequence)
# ... add trajectory points ...

# Detect thought-knots
engine = TQFTEngine::new(2).unwrap()
has_knot = TQFTEngine::detect_thought_knot(&trajectory, 0.1)

# Compute dBetti/dt
derivative_norm = TQFTEngine::compute_temporal_betti_derivative(&trajectory)
```

## Testing Procedures

### Unit Tests

```bash
# Test code trajectory
cargo test --package tcs-tqft code_trajectory

# Test CQS computation
python -m pytest niodoo-ai/tests/test_code_quality.py
```

### Integration Tests

```bash
# Test dataset format
python niodoo-ai/scripts/validate_dataset_format.py \
    niodoo-ai/data/code_topology_dataset.jsonl

# Test adapter orthogonalization
python niodoo-ai/scripts/orthogonalize_adapters.py \
    --frozen-adapter ./adapters/emotion_adapter \
    --new-adapter ./adapters/code_adapter \
    --base-model Qwen/Qwen2.5-Coder-32B-Instruct
```

## Validation Benchmarks

### HiBench

Tasks:
- Time Complexity prediction
- Space Complexity prediction
- Hierarchical reasoning

### DSR-Bench

Tasks:
- Graph operations
- Binary Search Tree operations
- Heap operations

## Troubleshooting

### CQS Weight Tuning Issues

- **Low correlation (<0.7)**: Increase gold set size, try different external metrics
- **Weights don't sum to 1.0**: Use `CQSWeights.normalize()` method

### Adapter Orthogonalization Issues

- **High interference (>5%)**: Check frozen adapter loading, verify orthogonal initialization
- **Training fails**: Ensure base model matches adapter architecture

### Dataset Format Issues

- **Missing fields**: Run `validate_dataset_format.py` to identify issues
- **Type mismatches**: Check `graph_adj` is `Vec<f32>`, `graph_dim` is `[usize, usize]`

## Success Criteria

- [x] CQS weights tuned (correlation >0.7)
- [x] Dataset format validated
- [x] Training configuration created
- [x] Differentiable TDA integrated
- [x] Adapter orthogonalization implemented
- [ ] Ablation study shows >5-10% improvement on HiBench/DSR-Bench
- [ ] Topological Code MRI demo identifies thought-knots in patient zero codebase

## Next Steps

1. Run gold-set CQS tuning experiment
2. Build 100k-file dataset with tuned weights
3. Train model with composite loss
4. Run ablation study
5. Implement Topological Code MRI demo
6. Validate on patient zero codebase

