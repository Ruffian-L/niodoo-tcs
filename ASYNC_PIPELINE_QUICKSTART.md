# Async-Friendly Pipeline — Quick Start

**Status**: ✅ Ready to Use  
**Build**: Complete  
**Tests**: All Passing (3/3)  
**Binary**: Ready at `/workspace/Niodoo-Final/target/release/niodoo_real_integrated`

---

## 30-Second Setup

```bash
cd /workspace/Niodoo-Final/niodoo_real_integrated
source ../tcs_runtime.env
```

---

## Three Ways to Run

### 1️⃣ Run Library Tests (Mock Services)

No external services required. Tests use synthetic Ollama server.

```bash
cargo test --lib -- --nocapture
```

**Expected**:
```
running 3 tests
test eval::synthetic::tests::generates_consistent_prompt_cycles ... ok
test tests::mock_pipeline_embed_stage ... ok
test tests::test_process_prompt_with_mock_clients ... ok

test result: ok. 3 passed; 0 failed
```

---

### 2️⃣ Run Single Prompt (Real Services)

Requires: Ollama, Qdrant, vLLM running

```bash
cargo run --release --bin niodoo_real_integrated -- \
  --prompt "What is async-friendly pipeline architecture?" \
  --output json
```

---

### 3️⃣ Run Swarm Mode (4 Pipelines in Parallel)

Requires: Ollama, Qdrant, vLLM running

```bash
cargo run --release --bin niodoo_real_integrated -- \
  --batch ../prompts/test_prompts.txt \
  --swarm 4 \
  --output csv
```

**What happens**:
- Loads 5 test prompts
- Creates 4 independent pipelines
- Partitions prompts across pipelines using Rayon
- Joins results back in original order
- Outputs CSV with all metrics

---

## Key Commands

```bash
# Build library
cargo build --lib

# Build release binary
cargo build --release --bin niodoo_real_integrated

# Run with debug logging
RUST_LOG=debug cargo test --lib -- --nocapture

# Check binary help
/workspace/Niodoo-Final/target/release/niodoo_real_integrated --help
```

---

## Service Status

```bash
# Check all services
curl http://localhost:11434/api/tags     # Ollama
curl http://localhost:6333/health        # Qdrant
curl http://localhost:5001/v1/models     # vLLM
```

---

## Documentation

- **Full Build Guide**: `ASYNC_PIPELINE_BUILD_GUIDE.md`
- **Validation Report**: `ASYNC_PIPELINE_VALIDATION_REPORT.md`
- **Automation Script**: `bash build_and_test_async_pipeline.sh`

---

## What's Implemented

✅ **Arc<RwLock<>>**: Shared config, non-blocking reads  
✅ **spawn_blocking**: Compass + Q-learning offloaded  
✅ **Swarm Mode**: Multi-pipeline parallelism  
✅ **CompassRuntimeParams**: Tuned thresholds per run  

---

## Architecture at a Glance

```
Input Prompts
    ↓
[rayon::par_iter splits across cores]
    ↓
Pipeline Pool (Arc<AsyncMutex>)
    ├─ Pipeline 0
    ├─ Pipeline 1
    ├─ Pipeline 2
    └─ Pipeline 3
    ↓
[futures::join_all collects results]
    ↓
[sort by original index]
    ↓
CSV/JSON Output
```

---

**That's it. Everything is compiled, tested, and ready to go.**

Run `cargo test --lib` to verify—you should see 3 tests pass immediately.

