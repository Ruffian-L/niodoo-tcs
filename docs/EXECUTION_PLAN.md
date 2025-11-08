# NIODOO-Code Pivot: Phased Execution Plan

This document outlines the step-by-step execution plan for the topological pivot pipeline, from CQS weight tuning to final validation.

## Prerequisites

- **Hardware**: 4x A100 GPUs (or equivalent) for training
- **Storage**: ~100GB free space for datasets
- **Credentials**: Google Cloud BigQuery access (for Phase 1-2)
- **Environment**: Python 3.10+, Rust 1.70+, CUDA 11.8+

## Phase 1: Tune CQS Weights on Gold Set

**Goal**: Ground proxy labels empirically using 1k-sample gold set.

### 1.1 Prep Gold Set

```bash
cd niodoo-ai

# Query BigQuery for 1k hotspot files
python scripts/scrape_bigquery_rust.py \
    --output data/gold_set_files.jsonl \
    --limit 1000 \
    --min-churn 5 \
    --credentials $GOOGLE_APPLICATION_CREDENTIALS

# Download actual code files (if not included in BigQuery response)
# Note: BigQuery contents table may not include full code, may need GitHub API
python scripts/download_github_files.py \
    --input data/gold_set_files.jsonl \
    --output data/gold_set_code/ \
    --rate-limit 5000  # GitHub API rate limit
```

**Expected output**: `data/gold_set_files.jsonl` with ~1,000 records containing:
- `code_string`: Source code
- `churn_count`: Commit churn metric
- `repo_name`, `path`: Repository identifiers
- External metrics (if available): `bug_fix_commits`, `static_analysis_errors`, `security_vulnerabilities`

### 1.2 Run Weight Tuning

```bash
# Initialize weights config (optional - script will use defaults if not present)
cat > config/cqs_weights.yaml << EOF
w_cc: 0.4
w_cog: 0.4
w_churn: 0.2
EOF

# Run tuning experiment
python scripts/tune_cqs_weights.py \
    --gold-set data/gold_set_files.jsonl \
    --metric-type combined \
    --grid-size 10 \
    --output config/tuned_cqs_weights.yaml \
    --report data/cqs_tuning_report.json

# Expected runtime: 1-2 hours on CPU/GPU
# Output: Optimal weights + correlation report
```

**Success criteria**: Pearson correlation >0.7 with external metrics

### 1.3 Validate Tuned Weights

```bash
python scripts/validate_cqs_weights.py \
    --weights config/tuned_cqs_weights.yaml \
    --gold-set data/gold_set_files.jsonl \
    --output data/validation_report.json

# Check correlation metrics
cat data/validation_report.json | jq '.correlation'
```

**If correlation <0.7**: Iterate by:
- Adding more samples to gold set
- Adjusting external metric definitions
- Expanding grid search space

---

## Phase 2: Construct Full Dataset with Tuned Weights

**Goal**: Build 50k-100k topological corpus using tuned CQS weights.

### 2.1 Source Full Hotspots

```bash
# Expand BigQuery query to 100k files
python scripts/scrape_bigquery_rust.py \
    --output data/hotspot_files.jsonl \
    --limit 100000 \
    --min-churn 3 \
    --credentials $GOOGLE_APPLICATION_CREDENTIALS

# Batch download code files (handle rate limits)
python scripts/download_github_files.py \
    --input data/hotspot_files.jsonl \
    --output data/hotspot_code/ \
    --batch-size 1000 \
    --rate-limit 5000 \
    --retry-attempts 3
```

**Expected output**: `data/hotspot_files.jsonl` with 100k records

### 2.2 Compute Labels and Topology

```bash
# Ensure Rust pipeline is built
cd ../tcs-parser  # or wherever Rust code ingestion lives
cargo build --release

cd ../niodoo-ai

# Build dataset with tuned weights
python scripts/build_rust_dataset.py \
    data/hotspot_files.jsonl \
    --output data/code_topology_dataset.jsonl \
    --weights config/tuned_cqs_weights.yaml \
    --max-examples 100000 \
    --batch-size 100

# This will:
# 1. Parse code via tree-sitter (Rust FFI)
# 2. Compute CQS with tuned weights
# 3. Generate adjacency matrices
# 4. Compute persistence diagrams (giotto-tda)
# 5. Serialize to CodeTopologicalData format
```

**Expected runtime**: 4-8 hours for 100k files
**Expected output**: `data/code_topology_dataset.jsonl` (~50GB)

### 2.3 Validate Dataset Format

```bash
# Validate format matches CodeTopologicalData struct
python scripts/validate_dataset_format.py \
    --dataset data/code_topology_dataset.jsonl \
    --output data/validation_report.json

# Check for outliers
python scripts/validate_dataset_format.py \
    --dataset data/code_topology_dataset.jsonl \
    --check-outliers \
    --outlier-threshold 3.0

# Split train/val (80/20)
python scripts/split_dataset.py \
    --input data/code_topology_dataset.jsonl \
    --train-output data/train.jsonl \
    --val-output data/val.jsonl \
    --split-ratio 0.8 \
    --seed 42
```

**Success criteria**:
- All records have required fields: `code_str`, `graph_adj`, `graph_dim`, `topology_signature`, `label_cqs`
- No outliers (CQS >3σ from mean)
- Train/val split preserves distribution

---

## Phase 3: Train with Composite Loss

**Goal**: Fine-tune Qwen2.5-Coder-32B with topology-aware QLoRA.

### 3.1 Setup Environment

```bash
cd niodoo-ai

# Install dependencies
pip install -r requirements.txt

# Verify GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

# Update config paths
# Edit config/config_code_pivot.yml:
# - data.train_file: ./data/train.jsonl
# - data.eval_file: ./data/val.jsonl
# - runtime.output_dir: ./outputs/qwen25-coder-code-pivot
```

### 3.2 Run Training

```bash
# Train with composite loss (differentiable TDA enabled)
python scripts/train_topology.py \
    config/config_code_pivot.yml \
    --use-differentiable-tda \
    --lambda-topo 0.1 \
    --multi-domain false  # Set true if loading emotional adapters

# Monitor with TensorBoard (in separate terminal)
tensorboard --logdir ./outputs/qwen25-coder-code-pivot/logs

# Expected runtime: 1-2 days on 4x A100s
# Monitor for:
# - Loss convergence (should decrease steadily)
# - Topological loss component (should correlate with CQS)
# - No divergence (if loss spikes, reduce lambda_topo)
```

**Training checkpoints**: Saved to `./outputs/qwen25-coder-code-pivot/checkpoint-*/`

### 3.3 Test Inference

```bash
# Quick inference test
python scripts/test_inference.py \
    --model ./outputs/qwen25-coder-code-pivot/checkpoint-2000 \
    --input "def complex_function(x, y):\n    # Complex nested logic\n    if x > 0:\n        if y < 0:\n            return x * y\n        else:\n            return x + y\n    else:\n        return 0" \
    --output data/inference_test.json

# Verify generated code has:
# - Lower CQS than input
# - Simpler topology (fewer cycles)
# - Preserved functionality
```

---

## Phase 4: Validate on Specialized Benchmarks

**Goal**: Prove topological edge with ablation + demos.

### 4.1 Run Ablation Study

```bash
# Train control model (no topological loss)
python scripts/train_topology.py \
    config/config_code_pivot.yml \
    --use-differentiable-tda false \
    --lambda-topo 0.0 \
    --output-dir ./outputs/qwen25-coder-control

# Evaluate both models on benchmarks
python scripts/eval_benchmarks.py \
    --treatment-model ./outputs/qwen25-coder-code-pivot/checkpoint-2000 \
    --control-model ./outputs/qwen25-coder-control/checkpoint-2000 \
    --benchmarks hibench,dsr-bench,bigcodebench \
    --output data/ablation_results.json

# Compute statistical significance (Cohen's d)
python scripts/analyze_ablation.py \
    --results data/ablation_results.json \
    --output data/ablation_analysis.json
```

**Success criteria**: >5-10% improvement in treatment group, Cohen's d >0.5

### 4.2 Killer Demo: Patient Zero

```bash
# Ingest ADHD codebase
cd ../tcs-parser
cargo run --bin analyzer -- \
    --input /path/to/adhd_codebase \
    --output ../niodoo-ai/data/adhd_analysis.json

cd ../niodoo-ai

# Detect thought-knots
python scripts/detect_thought_knots.py \
    --input data/adhd_analysis.json \
    --output data/adhd_thought_knots.json \
    --threshold 0.1

# Visualize Betti loops
python scripts/visualize_topology.py \
    --input data/adhd_thought_knots.json \
    --output data/adhd_visualization.html \
    --format plotly

# Generate refactor suggestions
python scripts/refactor_with_topology.py \
    --model ./outputs/qwen25-coder-code-pivot/checkpoint-2000 \
    --input data/adhd_thought_knots.json \
    --output data/adhd_refactors.json \
    --top-k 10
```

### 4.3 Full Validation

```bash
# Test on legacy language (COBOL)
# First, add COBOL grammar to tree-sitter
cd ../tcs-parser
# (Add COBOL grammar support)

# Re-run ingestion
python scripts/build_rust_dataset.py \
    data/cobol_samples.jsonl \
    --output data/cobol_topology.jsonl \
    --language cobol

# Evaluate on COBOL
python scripts/eval_benchmarks.py \
    --model ./outputs/qwen25-coder-code-pivot/checkpoint-2000 \
    --benchmarks cobol-refactor \
    --output data/cobol_results.json

# Multi-domain test (emotional PR feedback + code)
python scripts/test_multi_domain.py \
    --code-adapter ./outputs/qwen25-coder-code-pivot/checkpoint-2000 \
    --emotion-adapter ./adapters/emotion_adapter \
    --input data/multi_domain_test.jsonl \
    --output data/multi_domain_results.json
```

**Success criteria**:
- Structural simplicity: low dBetti/dt in generated code
- Multi-domain capability: emotional + code reasoning both work
- Legacy language support: COBOL refactoring successful

---

## Troubleshooting

### Common Issues

1. **BigQuery rate limits**: Use `--batch-size` and `--rate-limit` flags
2. **GPU OOM**: Reduce `batch_size` or `max_seq_length` in config
3. **Training divergence**: Reduce `lambda_topo` (try 0.05 or 0.01)
4. **Low correlation in CQS tuning**: Expand gold set or adjust external metrics
5. **Dataset validation failures**: Check `graph_dim` matches `graph_adj` size

### Debug Commands

```bash
# Check dataset statistics
python scripts/analyze_dataset.py --input data/code_topology_dataset.jsonl

# Test differentiable TDA in isolation
python -c "from niodoo_ai.differentiable_tda import DifferentiableTopologicalLoss; import torch; print('TDA OK')"

# Verify adapter loading
python scripts/test_adapters.py --adapter-path ./outputs/qwen25-coder-code-pivot
```

---

## Success Metrics

- **Phase 1**: CQS correlation >0.7 with external metrics
- **Phase 2**: 50k+ examples, format validation passes
- **Phase 3**: Training loss converges, inference generates simpler code
- **Phase 4**: >5-10% improvement over control, thought-knot detection works

---

## Next Steps After Validation

1. **Deploy as agent**: Integrate into CI/CD pipelines
2. **Enterprise pitch**: Legacy code refactoring use cases
3. **Scale up**: Expand to more languages, larger datasets
4. **Research**: Publish results, contribute to TDA+LLM literature

