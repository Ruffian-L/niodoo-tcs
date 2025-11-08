# Quick Start Guide: Single A100 GPU

**Step-by-step walkthrough** for running the code pivot pipeline on a single A100 GPU.

---

## 🔐 Step 0: Google Cloud Authentication

### Option A: Use Public Dataset (Easiest - No Auth Needed!)

```bash
# Just install the library - public dataset doesn't require auth
pip install google-cloud-bigquery

# That's it! The script will work without credentials
```

### Option B: Full Authentication (For Private Projects)

```bash
# 1. Install Google Cloud SDK
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# 2. Login
gcloud auth login
gcloud auth application-default login

# 3. Set project (replace YOUR_PROJECT_ID)
gcloud config set project YOUR_PROJECT_ID

# 4. Test access
python -c "from google.cloud import bigquery; print('✅ BigQuery OK')"
```

**That's it!** The scripts will automatically use your credentials.

---

## 📋 Phase 1: Tune CQS Weights (2-3 hours)

### Step 1: Get 1,000 code files from BigQuery

```bash
cd /workspace/Niodoo-Final/niodoo-ai

# Create data folder
mkdir -p data

# Download 1k files (no auth needed for public dataset!)
python scripts/scrape_bigquery_rust.py \
    --output data/gold_set_files.jsonl \
    --limit 1000 \
    --min-churn 5

# Check it worked
wc -l data/gold_set_files.jsonl  # Should show ~1000
```

**What this does**: Queries BigQuery public GitHub dataset for Rust files with high churn (frequently changed files = likely problematic code).

### Step 2: Find best weights

```bash
# Run tuning (takes 1-2 hours)
python scripts/tune_cqs_weights.py \
    --gold-set data/gold_set_files.jsonl \
    --metric-type combined \
    --grid-size 10 \
    --output config/tuned_cqs_weights.yaml

# Check results
cat config/tuned_cqs_weights.yaml
```

**What this does**: Tests different weight combinations (w_cc, w_cog, w_churn) to find which best predicts "bad code" (measured by bug fixes, static analysis errors, etc.).

**Success**: You should see `config/tuned_cqs_weights.yaml` with optimal weights.

---

## 📦 Phase 2: Build Full Dataset (4-8 hours, can run overnight)

### Step 1: Get 50k code files

```bash
# Download 50k files (bigger dataset)
python scripts/scrape_bigquery_rust.py \
    --output data/hotspot_files.jsonl \
    --limit 50000 \
    --min-churn 3

# Check size
wc -l data/hotspot_files.jsonl  # Should show ~50k
```

### Step 2: Compute topology + labels

```bash
# Build dataset (takes 4-8 hours)
python scripts/build_rust_dataset.py \
    data/hotspot_files.jsonl \
    --output data/code_topology_dataset.jsonl \
    --weights config/tuned_cqs_weights.yaml \
    --max-examples 50000 \
    --batch-size 100

# Monitor progress (shows progress bar)
```

**What this does**: 
- Parses each code file → AST → Graph
- Computes topological features (Betti numbers, persistence diagrams)
- Calculates CQS labels using tuned weights
- Saves everything in training format

**Success**: `data/code_topology_dataset.jsonl` (~20-50GB file)

### Step 3: Split train/val

```bash
# Split 80/20
python scripts/split_dataset.py \
    --input data/code_topology_dataset.jsonl \
    --train-output data/train.jsonl \
    --val-output data/val.jsonl \
    --split-ratio 0.8 \
    --seed 42

# Verify
wc -l data/train.jsonl data/val.jsonl
```

---

## 🚀 Phase 3: Train Model (3-5 days, can pause/resume)

### Step 1: Update config for single GPU

```bash
# Copy and edit config
cp config/config_code_pivot.yml config/config_code_pivot_single_gpu.yml

# Edit these settings:
# - per_device_train_batch_size: 1
# - gradient_accumulation_steps: 16
# - max_seq_length: 2048 (reduced from 4096)
```

Or use the pre-made config from `EXECUTION_PLAN_SINGLE_GPU.md`.

### Step 2: Start training

```bash
# Run in screen/tmux (so it keeps running)
screen -S training

# Start training
python scripts/train_topology.py \
    config/config_code_pivot_single_gpu.yml \
    --use-differentiable-tda \
    --lambda-topo 0.1

# Detach: Press Ctrl+A, then D
# Reattach later: screen -r training
```

**What this does**: Fine-tunes Qwen2.5-Coder-32B to understand code topology. Uses composite loss (normal loss + topological loss).

**Monitor**:
```bash
# In another terminal
tail -f outputs/qwen25-coder-code-pivot-single-gpu/logs/training.log

# Or TensorBoard
tensorboard --logdir outputs/qwen25-coder-code-pivot-single-gpu/logs
```

**Expected**: Loss decreases over time. Checkpoints saved every 200 steps.

**If OOM**: Reduce `max_seq_length` to 1024 or increase `gradient_accumulation_steps` to 32.

### Step 3: Test inference

```bash
# After training (or during, using a checkpoint)
python scripts/test_inference.py \
    --model ./outputs/qwen25-coder-code-pivot-single-gpu/checkpoint-2000 \
    --input "def complex_function(x, y):\n    if x > 0:\n        if y < 0:\n            return x * y\n        else:\n            return x + y\n    else:\n        return 0" \
    --output data/inference_test.json

# Check output
cat data/inference_test.json
```

**Success**: Generated code should be simpler (lower CQS) than input.

---

## ✅ Phase 4: Validate (1-2 days)

### Quick validation

```bash
# Test on validation set
python scripts/quick_validate.py \
    --model ./outputs/qwen25-coder-code-pivot-single-gpu/checkpoint-2000 \
    --test-set data/val.jsonl \
    --samples 100 \
    --output data/validation_results.json

# Check metrics
cat data/validation_results.json | python -m json.tool
```

**Success**: Model generates code with lower complexity (measured by CQS and topological metrics).

---

## 🐛 Troubleshooting

### BigQuery Issues

```bash
# Test BigQuery access
python -c "from google.cloud import bigquery; client = bigquery.Client(); print('✅ OK')"

# If error: Install library
pip install google-cloud-bigquery
```

### GPU OOM (Out of Memory)

```bash
# Edit config: reduce batch size, increase gradient accumulation
# Or reduce max_seq_length: 1024
```

### Training Too Slow

```bash
# Use smaller dataset for testing (1000 examples)
# Or reduce epochs: num_train_epochs: 1.0
```

### Low CQS Correlation

```bash
# Check gold set quality
head -n 5 data/gold_set_files.jsonl | python -m json.tool

# Try larger gold set (2000 samples)
python scripts/scrape_bigquery_rust.py --limit 2000 --output data/gold_set_large.jsonl
```

---

## 📊 Expected Timeline

- **Phase 1**: 2-3 hours (CQS tuning)
- **Phase 2**: 4-8 hours (dataset build, can run overnight)
- **Phase 3**: 3-5 days (training, can pause/resume)
- **Phase 4**: 1-2 days (validation)

**Total**: ~1-2 weeks (with pauses)

---

## 🎯 Success Checklist

- [ ] Phase 1: `config/tuned_cqs_weights.yaml` exists, correlation >0.7
- [ ] Phase 2: `data/train.jsonl` and `data/val.jsonl` exist, ~50k examples
- [ ] Phase 3: Training checkpoints saved, loss decreasing
- [ ] Phase 4: Inference generates simpler code

---

## 💡 Pro Tips

1. **Run Phase 2 overnight** - Dataset building is CPU-bound, doesn't need GPU
2. **Use screen/tmux** - Training takes days, keep it running
3. **Monitor GPU memory** - `watch -n 1 nvidia-smi`
4. **Start small** - Test with 1k examples first, then scale up
5. **Save checkpoints** - Training can be paused/resumed automatically

---

## 🆘 Need Help?

- Check `docs/EXECUTION_PLAN_SINGLE_GPU.md` for detailed commands
- Check `docs/CODE_PIVOT_IMPLEMENTATION_GUIDE.md` for technical details
- Review logs: `outputs/qwen25-coder-code-pivot-single-gpu/logs/`

