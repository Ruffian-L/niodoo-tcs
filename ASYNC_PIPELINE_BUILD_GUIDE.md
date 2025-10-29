# Async-Friendly Pipeline Build & Test Guide

## Overview

The Niodoo Real Integrated pipeline has been hardened with async-friendly concurrency primitives:
- Shared config behind `Arc<RwLock<...>>` for read-heavy operations
- Compass evaluation in `spawn_blocking` with owned guards
- Learning loop Q-table guarded by `Arc<RwLock<...>>` with heavyweight updates in `spawn_blocking`
- Multi-pipeline swarm mode via `--swarm` CLI flag with Rayon parallelism

## Prerequisites

### 1. Environment Setup
```bash
cd /workspace/Niodoo-Final

# Load runtime environment
source tcs_runtime.env

# Verify environment
echo "PROJECT_ROOT: $PROJECT_ROOT"
echo "MODELS_DIR: $MODELS_DIR"
```

### 2. Start External Services

```bash
# Start Ollama (used for embeddings via API)
ollama serve &
sleep 2
ollama pull qwen2:0.5b

# Start Qdrant (vector database for ERAG memory)
docker run -d -p 6333:6333 qdrant/qdrant:latest

# Optional: Start vLLM for generation (if configured)
# Uses vLLM HTTP endpoint for high-throughput text generation
python -m vllm.entrypoints.openai.api_server \
  --model /workspace/models/Qwen2.5-7B-Instruct-AWQ \
  --port 5001 \
  --tensor-parallel-size 1 \
  &
```

### 3. Verify Services

```bash
# Check Ollama
curl http://localhost:11434/api/tags

# Check Qdrant
curl http://localhost:6333/health

# Check vLLM (if running)
curl -s http://localhost:5001/v1/models | jq .
```

## Building & Testing

### Option A: Build & Test from Package Directory (Recommended)

This avoids virtual workspace issues:

```bash
cd /workspace/Niodoo-Final/niodoo_real_integrated

# Clean previous builds
cargo clean

# Run tests (skips those requiring ONNX if not present)
cargo test --lib -- --test-threads=1 --nocapture

# Specific test: mock_pipeline_embed_stage
cargo test --lib mock_pipeline_embed_stage -- --nocapture
```

### Option B: Build Binary

```bash
cd /workspace/Niodoo-Final/niodoo_real_integrated

# Release build
cargo build --release --bin niodoo_real_integrated

# Binary location
ls -lh target/release/niodoo_real_integrated
```

### Option C: Run Binary with Single Prompt

```bash
cd /workspace/Niodoo-Final/niodoo_real_integrated

cargo run --release --bin niodoo_real_integrated -- \
  --prompt "parallel test with single pipeline" \
  --output-format json
```

### Option D: Run with Swarm Mode (Multiple Pipelines)

This demonstrates the async-friendly concurrent architecture:

```bash
cd /workspace/Niodoo-Final/niodoo_real_integrated

cargo run --release --bin niodoo_real_integrated -- \
  --batch /workspace/Niodoo-Final/prompts/test_prompts.txt \
  --swarm 4 \
  --output-format csv
```

**What `--swarm 4` does:**
1. Creates 4 independent pipeline instances
2. Uses `rayon::par_iter()` to partition prompts across cores
3. Each pipeline is wrapped in `Arc<AsyncMutex<...>>` for safe async access
4. Futures are joined back in order using `futures::join_all()`
5. Results maintain original prompt order

### Option E: Run Full Test Suite

```bash
cd /workspace/Niodoo-Final/niodoo_real_integrated

# All tests
cargo test --lib -- --nocapture

# With logging
RUST_LOG=debug,niodoo_real_integrated=trace \
cargo test --lib -- --nocapture --test-threads=1
```

## Environment Variables

Key configuration for async pipeline:

```bash
# Embedding service (Ollama API endpoint)
export OLLAMA_URL="http://127.0.0.1:11434"
export OLLAMA_EMBED_MODEL="qwen2:0.5b"

# Embedding concurrency (controls semaphore limit)
export EMBEDDING_MAX_CONCURRENCY=8

# Generation service (vLLM endpoint)
export VLLM_ENDPOINT="http://127.0.0.1:5001"
export VLLM_MODEL="Qwen2.5-7B-Instruct-AWQ"
export VLLM_MAX_BATCH_SIZE=32

# Vector database (Qdrant)
export QDRANT_URL="http://127.0.0.1:6333"
export QDRANT_VECTOR_DIM=896

# Learning loop (async Q-table updates)
export DQN_EPSILON=0.1
export DQN_ALPHA=0.01
export DQN_GAMMA=0.99

# Compass engine (concurrent evaluation)
export MCTS_C=1.414
export VARIANCE_SPIKE=2.0
export VARIANCE_STAGNATION=0.1
```

## Validation Checklist

After building:

- [ ] Services running: `ollama`, `qdrant`, optionally `vllm`
- [ ] Environment loaded: `source tcs_runtime.env`
- [ ] Single-prompt execution works: `cargo run -- --prompt "test"`
- [ ] Tests pass: `cargo test --lib`
- [ ] Swarm mode works: `cargo run -- --swarm 4 --batch prompts.txt`

## Architecture Overview

### Arc<RwLock<>> Pattern

The async-friendly architecture uses:
- **Readers**: Shared guards for config/thresholds (hot path, non-blocking)
- **Writers**: Rare config updates acquire write lock
- **Spawn-blocking tasks**: Heavy computation (Compass, Q-updates) inside `spawn_blocking` with owned guards

```rust
// Example: Compass evaluation in spawn_blocking
let compass_guard = self.compass.clone().lock_owned().await;
let compass_task = tokio::task::spawn_blocking(move || {
    let mut engine = compass_guard;
    engine.evaluate_with_params(params, &state, topology)
});
```

### Multi-Pipeline Swarm

```
CLI Input (prompts)
    ↓
[rayon::par_iter() partitions]
    ↓
┌─────────────────────────────────┐
│ Pipeline 0 (Arc<AsyncMutex>)   │ ← Slot 0, 4, 8, ...
│ Pipeline 1 (Arc<AsyncMutex>)   │ ← Slot 1, 5, 9, ...
│ Pipeline 2 (Arc<AsyncMutex>)   │ ← Slot 2, 6, 10, ...
│ Pipeline 3 (Arc<AsyncMutex>)   │ ← Slot 3, 7, 11, ...
└─────────────────────────────────┘
    ↓
[join_all() futures::join()]
    ↓
[sort_by_key(original_index)]
    ↓
Output (in order)
```

## Troubleshooting

### Test Fails: "Qwen ONNX model missing"

**Solution**: The test has been fixed to handle missing ONNX gracefully. Re-run:
```bash
cargo test --lib -- --nocapture
```

### Binary Won't Build: Virtual Workspace Issue

**Solution**: Always build from the package directory:
```bash
cd /workspace/Niodoo-Final/niodoo_real_integrated
cargo build --release
```

### Ollama Connection Refused

**Check**:
```bash
ps aux | grep ollama
curl -i http://localhost:11434/api/tags
```

**Fix**:
```bash
killall ollama
ollama serve &
sleep 2
ollama pull qwen2:0.5b
```

### Qdrant Connection Refused

**Check**:
```bash
docker ps | grep qdrant
curl -i http://localhost:6333/health
```

**Fix**:
```bash
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant:latest
```

### Swarm Mode Hangs

**Debug**:
```bash
RUST_LOG=debug cargo run --release -- --swarm 4 --batch prompts.txt --output-format json
```

**Check**: All 4 pipelines initialized successfully; look for "Pipeline initialized" logs 4 times.

## Next Steps

1. **Performance**: Monitor latency with `--output-format csv` and analyze per-swarm impact
2. **Scale**: Increase `--swarm` to match CPU cores; measure throughput
3. **Learning**: Enable learning loop to validate Q-table concurrent updates
4. **Monitoring**: Use Prometheus metrics at `/metrics` (metrics_server binary)

## References

- **Async architecture**: `src/pipeline.rs` lines 92–111 (config/compass wiring)
- **Swarm implementation**: `src/main.rs` lines 67–101 (rayon + futures orchestration)
- **Learning loop**: `src/learning.rs` lines 120–142 (Q-table Arc<RwLock>)
- **Compass params**: `src/compass.rs` lines 33–108 (runtime params injection)

