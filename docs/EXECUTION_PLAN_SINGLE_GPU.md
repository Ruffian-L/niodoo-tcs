# NIODOO-Code Pivot: Execution Plan (Single A100 GPU)

**Hardware**: 1x A100 GPU (80GB VRAM)  
**Storage**: ~100GB free space  
**Runtime Estimates**: Adjusted for single GPU

## Prerequisites Setup

### 1. Google Cloud Authentication for BigQuery

```bash
# Install Google Cloud SDK (if not already installed)
# On Ubuntu/Debian:
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# On macOS:
brew install google-cloud-sdk

# Authenticate
gcloud auth login
gcloud auth application-default login

# Set your project (replace with your project ID)
gcloud config set project YOUR_PROJECT_ID

# Create service account (optional, for programmatic access)
gcloud iam service-accounts create bigquery-reader \
    --display-name="BigQuery Reader"

# Grant BigQuery Data Viewer role
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:bigquery-reader@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/bigquery.dataViewer"

# Create and download key
gcloud iam service-accounts keys create ~/bigquery-key.json \
    --iam-account=bigquery-reader@YOUR_PROJECT_ID.iam.gserviceaccount.com

# Set environment variable
export GOOGLE_APPLICATION_CREDENTIALS=~/bigquery-key.json

# Verify access
python -c "from google.cloud import bigquery; client = bigquery.Client(); print('✅ BigQuery access OK')"
```

**Alternative: Use Public Dataset (No Auth Required)**
```bash
# BigQuery public dataset doesn't require authentication for read-only queries
# Just install the library:
pip install google-cloud-bigquery

# The script will use default credentials or public access
```

### 2. Install Dependencies

```bash
cd /workspace/Niodoo-Final/niodoo-ai

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Install Google Cloud BigQuery (if not in requirements.txt)
pip install google-cloud-bigquery google-auth

# Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

---

## Phase 1: Tune CQS Weights on Gold Set

**Goal**: Find optimal weights (w_cc, w_cog, w_churn) using 1k-sample gold set  
**Runtime**: ~1-2 hours (CPU/GPU)

### Step 1.1: Query BigQuery for Gold Set

```bash
cd /workspace/Niodoo-Final/niodoo-ai

# Create data directory
mkdir -p data

# Run BigQuery scraper (1k files)
python scripts/scrape_bigquery_rust.py \
    --output data/gold_set_files.jsonl \
    --limit 1000 \
    --min-churn 5 \
    --credentials "$GOOGLE_APPLICATION_CREDENTIALS"  # Optional if using public dataset

# Verify output
wc -l data/gold_set_files.jsonl  # Should show ~1000 lines
head -n 1 data/gold_set_files.jsonl | python -m json.tool  # Inspect first record
```

**Expected Output**: `data/gold_set_files.jsonl` with records like:
```json
{
  "repo_name": "owner/repo",
  "path": "src/main.rs",
  "code_string": "fn main() { ... }",
  "churn_count": 15,
  "bug_fix_commits": 2,  # If available
  "static_analysis_errors": 5  # If available
}
```

### Step 1.2: Run Weight Tuning

```bash
# Initialize default weights (optional)
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

# Monitor progress (will show grid search iterations)
# Expected runtime: 1-2 hours
```

**Success Criteria**: 
- Output file `config/tuned_cqs_weights.yaml` created
- Correlation >0.7 in `data/cqs_tuning_report.json`
- Check report:
  ```bash
  python -c "import json; r=json.load(open('data/cqs_tuning_report.json')); print(f\"Best correlation: {r.get('best_correlation', 0):.3f}\")"
  ```

### Step 1.3: Validate Tuned Weights

```bash
python scripts/validate_cqs_weights.py \
    --weights config/tuned_cqs_weights.yaml \
    --gold-set data/gold_set_files.jsonl \
    --output data/validation_report.json

# Check correlation
cat data/validation_report.json | python -m json.tool | grep -A 5 correlation
```

**If correlation <0.7**: 
- Expand gold set to 2k samples
- Adjust external metric definitions
- Check for data quality issues

---

## Phase 2: Construct Full Dataset

**Goal**: Build 50k-100k topological corpus  
**Runtime**: ~4-8 hours (CPU-bound, can run overnight)

### Step 2.1: Source Full Hotspots

```bash
# Expand to 50k (start smaller, can scale up)
python scripts/scrape_bigquery_rust.py \
    --output data/hotspot_files.jsonl \
    --limit 50000 \
    --min-churn 3 \
    --credentials "$GOOGLE_APPLICATION_CREDENTIALS"

# Verify
wc -l data/hotspot_files.jsonl  # Should show ~50k lines
du -h data/hotspot_files.jsonl  # Check file size (~500MB-1GB)
```

**Note**: If BigQuery doesn't include full code, you may need to download via GitHub API:
```bash
# Create download script (if needed)
python scripts/download_github_files.py \
    --input data/hotspot_files.jsonl \
    --output data/hotspot_code/ \
    --batch-size 1000 \
    --rate-limit 5000  # GitHub API: 5000 requests/hour
```

### Step 2.2: Compute Labels and Topology

```bash
# Ensure Rust pipeline is built (if using FFI)
cd /workspace/Niodoo-Final/tcs-parser  # Adjust path as needed
cargo build --release
cd ../niodoo-ai

# Build dataset with tuned weights
python scripts/build_rust_dataset.py \
    data/hotspot_files.jsonl \
    --output data/code_topology_dataset.jsonl \
    --weights config/tuned_cqs_weights.yaml \
    --max-examples 50000 \
    --batch-size 100 \
    --min-code-length 50 \
    --max-code-length 50000

# Monitor progress (will show progress bar)
# Expected runtime: 4-8 hours for 50k files
```

**Expected Output**: 
- `data/code_topology_dataset.jsonl` (~20-50GB)
- Each record includes: `code_str`, `graph_adj`, `graph_dim`, `topology_signature`, `label_cqs`

### Step 2.3: Validate Dataset Format

```bash
# Validate format
python scripts/validate_dataset_format.py \
    --dataset data/code_topology_dataset.jsonl \
    --output data/dataset_validation.json

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

# Verify split
wc -l data/train.jsonl data/val.jsonl  # Should be ~80/20 ratio
```

**Success Criteria**:
- All records have required fields
- No outliers (CQS within 3σ)
- Train/val split preserves distribution

---

## Phase 3: Train with Composite Loss (Single A100)

**Goal**: Fine-tune Qwen2.5-Coder-32B with topology-aware QLoRA  
**Runtime**: ~3-5 days (single GPU, can pause/resume)

### Step 3.1: Setup Environment

```bash
cd /workspace/Niodoo-Final/niodoo-ai

# Verify GPU
nvidia-smi  # Should show A100 with ~80GB VRAM

# Check CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"

# Update config for single GPU
cp config/config_code_pivot.yml config/config_code_pivot_single_gpu.yml

# Edit config (reduce batch size, increase gradient accumulation)
cat > config/config_code_pivot_single_gpu.yml << 'EOF'
# Single A100 GPU Configuration
model:
  base_model: Qwen/Qwen2.5-Coder-32B-Instruct
  load_in_4bit: true
  compute_dtype: bfloat16
  quant_type: nf4
  trust_remote_code: false

lora:
  r: 64
  alpha: 128
  dropout: 0.05
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj

data:
  train_file: ./data/train.jsonl
  eval_file: ./data/val.jsonl
  eval_split_ratio: 0.1
  max_seq_length: 2048  # Reduced from 4096 for single GPU
  prompt_template: "[INST] {instruction}\n{input}\n{topology} [/INST]"
  feature_prefix: "[TOPOLOGY]"
  num_workers: 4

optimizer:
  learning_rate: 0.0002
  weight_decay: 0.01
  warmup_ratio: 0.03
  num_train_epochs: 3.0
  gradient_accumulation_steps: 16  # Increased for single GPU
  max_grad_norm: 1.0
  optim: paged_adamw_8bit

runtime:
  output_dir: ./outputs/qwen25-coder-code-pivot-single-gpu
  logging_steps: 10
  evaluation_strategy: steps
  eval_steps: 100
  save_steps: 200
  seed: 42
  bf16: true
  per_device_train_batch_size: 1  # Single GPU: batch size 1
  per_device_eval_batch_size: 1
  dataloader_num_workers: 2

topology:
  lambda_weight: 0.1
  lambda_teacher: 0.0
  lambda_sinkhorn: 0.0
  projection: mean
  teacher_cache: null
  teacher_match_field: source_path
  sinkhorn_p: 1.0
  sinkhorn_blur: 0.05
  sinkhorn_scaling: 0.8
  max_sinkhorn_points: 128

differentiable_tda:
  enabled: true
  tda_backend: torch-tda
  topology_lambda: 0.1
  wasserstein_p: 1.0
EOF
```

### Step 3.2: Start Training

```bash
# Start training (run in screen/tmux for long-running)
screen -S training  # or: tmux new -s training

# Train with composite loss
python scripts/train_topology.py \
    config/config_code_pivot_single_gpu.yml \
    --use-differentiable-tda \
    --lambda-topo 0.1

# Detach: Ctrl+A, then D
# Reattach: screen -r training
```

**Monitor Training**:
```bash
# In another terminal, watch logs
tail -f outputs/qwen25-coder-code-pivot-single-gpu/logs/training.log

# Or use TensorBoard
tensorboard --logdir outputs/qwen25-coder-code-pivot-single-gpu/logs --port 6006
# Open http://localhost:6006 in browser
```

**Expected Behavior**:
- Loss should decrease steadily
- Topological loss component should correlate with CQS
- If loss spikes: reduce `lambda_topo` to 0.05 or 0.01
- Checkpoints saved every 200 steps

**Resume Training** (if interrupted):
```bash
python scripts/train_topology.py \
    config/config_code_pivot_single_gpu.yml \
    --use-differentiable-tda \
    --lambda-topo 0.1
# HuggingFace Trainer will auto-resume from latest checkpoint
```

### Step 3.3: Test Inference

```bash
# Quick inference test
python scripts/test_inference.py \
    --model ./outputs/qwen25-coder-code-pivot-single-gpu/checkpoint-2000 \
    --input "def complex_function(x, y):\n    if x > 0:\n        if y < 0:\n            return x * y\n        else:\n            return x + y\n    else:\n        return 0" \
    --output data/inference_test.json

# Verify generated code has lower CQS
cat data/inference_test.json | python -m json.tool
```

---

## Phase 4: Validation (Single GPU Constraints)

**Goal**: Prove topological edge with ablation + demos  
**Runtime**: ~1-2 days (can run in parallel with other tasks)

### Step 4.1: Run Ablation Study

```bash
# Train control model (no topological loss) - smaller run
python scripts/train_topology.py \
    config/config_code_pivot_single_gpu.yml \
    --use-differentiable-tda false \
    --lambda-topo 0.0 \
    --output-dir ./outputs/qwen25-coder-control

# Note: This will take another 3-5 days. Consider:
# - Using fewer epochs (1 instead of 3)
# - Using smaller dataset subset
# - Or skip and compare with baseline from literature

# Evaluate both models (when ready)
python scripts/eval_benchmarks.py \
    --treatment-model ./outputs/qwen25-coder-code-pivot-single-gpu/checkpoint-2000 \
    --control-model ./outputs/qwen25-coder-control/checkpoint-2000 \
    --benchmarks hibench,dsr-bench \
    --output data/ablation_results.json
```

**Alternative: Quick Validation** (skip full ablation):
```bash
# Test on small sample
python scripts/quick_validate.py \
    --model ./outputs/qwen25-coder-code-pivot-single-gpu/checkpoint-2000 \
    --test-set data/val.jsonl \
    --samples 100 \
    --output data/quick_validation.json
```

### Step 4.2: Killer Demo (Patient Zero)

```bash
# Ingest ADHD codebase (if available)
cd /workspace/Niodoo-Final/tcs-parser
cargo run --bin analyzer -- \
    --input /path/to/adhd_codebase \
    --output ../niodoo-ai/data/adhd_analysis.json

cd ../niodoo-ai

# Detect thought-knots
python scripts/detect_thought_knots.py \
    --input data/adhd_analysis.json \
    --output data/adhd_thought_knots.json \
    --threshold 0.1

# Visualize (if plotly available)
python scripts/visualize_topology.py \
    --input data/adhd_thought_knots.json \
    --output data/adhd_visualization.html \
    --format plotly
```

### Step 4.3: Full Validation

```bash
# Test inference on validation set
python scripts/eval_benchmarks.py \
    --model ./outputs/qwen25-coder-code-pivot-single-gpu/checkpoint-2000 \
    --benchmarks code-quality,structural-simplicity \
    --output data/full_validation.json

# Check metrics
cat data/full_validation.json | python -m json.tool
```

---

## Troubleshooting (Single GPU)

### Common Issues

1. **OOM (Out of Memory)**:
   ```bash
   # Reduce batch size further
   # Edit config: per_device_train_batch_size: 1
   # Increase gradient_accumulation_steps: 32
   # Reduce max_seq_length: 1024
   ```

2. **Training too slow**:
   ```bash
   # Use gradient checkpointing (already enabled)
   # Reduce dataset size for initial tests
   # Use mixed precision (bf16) - already enabled
   ```

3. **BigQuery rate limits**:
   ```bash
   # Use smaller batches
   # Add delays between requests
   # Use public dataset (no auth needed)
   ```

4. **Low correlation in CQS tuning**:
   ```bash
   # Check data quality
   python scripts/analyze_gold_set.py --input data/gold_set_files.jsonl
   
   # Expand gold set
   python scripts/scrape_bigquery_rust.py --limit 2000 --output data/gold_set_files_large.jsonl
   ```

### Debug Commands

```bash
# Check dataset statistics
python scripts/analyze_dataset.py --input data/code_topology_dataset.jsonl

# Test differentiable TDA
python -c "from niodoo_ai.differentiable_tda import DifferentiableTopologicalLoss; import torch; print('✅ TDA OK')"

# Verify GPU memory
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# Check training progress
ls -lh outputs/qwen25-coder-code-pivot-single-gpu/checkpoint-*/
```

---

## Success Metrics (Single GPU)

- **Phase 1**: CQS correlation >0.7 ✅
- **Phase 2**: 50k+ examples, format validation passes ✅
- **Phase 3**: Training loss converges, checkpoints saved ✅
- **Phase 4**: Inference generates simpler code (lower CQS) ✅

---

## Time Estimates (Single A100)

- **Phase 1**: 2-3 hours (CQS tuning)
- **Phase 2**: 4-8 hours (dataset construction, can run overnight)
- **Phase 3**: 3-5 days (training, can pause/resume)
- **Phase 4**: 1-2 days (validation, can run in parallel)

**Total**: ~1-2 weeks (with pauses/resumes)

---

## Next Steps After Validation

1. **Deploy**: Integrate into CI/CD
2. **Scale**: Add more GPUs or use cloud training
3. **Research**: Publish results
4. **Enterprise**: Pitch legacy refactoring use cases

