# Endpoint Testing & A/B Test Plan

## Mission
Get all endpoints online, smoke test them with real requests, then run A/B test to prove topology understanding.

## Current Status

### Endpoints Status
- ✅ **Qdrant** (6333/6334): ONLINE
- ✅ **RL-Server** (8080): ONLINE  
- ❌ **vLLM Generation** (5001): OFFLINE
- ❌ **vLLM Curator** (5002): OFFLINE
- ❌ **Main Pipeline** (9090): OFFLINE

## Phase 1: Get All Endpoints Online

### 1.1 Start vLLM Generation (Port 5001)
**Purpose**: Main generation model (Qwen 3 Coder)

**Command**:
```bash
# Check if model path exists
export VLLM_MODEL_ID="${VLLM_MODEL_ID:-/workspace/models/Qwen3-Coder}"
export VLLM_PORT=5001

# Start vLLM server
source venv/bin/activate 2>/dev/null || true
nohup python3 -m vllm.entrypoints.openai.api_server \
    --model "$VLLM_MODEL_ID" \
    --host 127.0.0.1 \
    --port ${VLLM_PORT} \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 32768 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 64 \
    --trust-remote-code \
    > /tmp/vllm_coder.log 2>&1 &
```

**Verification**: `curl http://127.0.0.1:5001/v1/models`

### 1.2 Start vLLM Curator (Port 5002)
**Purpose**: Curator model (Qwen 2.5 Topology)

**Command**:
```bash
export CURATOR_MODEL="${CURATOR_MODEL:-/workspace/models/Qwen2.5-Topology}"
export CURATOR_VLLM_PORT=5002

source venv/bin/activate 2>/dev/null || true
nohup python3 -m vllm.entrypoints.openai.api_server \
    --model "$CURATOR_MODEL" \
    --host 127.0.0.1 \
    --port ${CURATOR_VLLM_PORT} \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.15 \
    --max-model-len 2048 \
    --max-num-batched-tokens 4096 \
    --max-num-seqs 32 \
    --trust-remote-code \
    > /tmp/vllm_curator.log 2>&1 &
```

**Verification**: `curl http://127.0.0.1:5002/v1/models`

### 1.3 Start Main Pipeline Server (Port 9090)
**Purpose**: Main application server with health endpoints

**Command**:
```bash
cd /workspace/Niodoo-Final
source tcs_runtime.env

# Build if needed
cargo build -p niodoo_real_integrated --release --features svc

# Start server
cargo run -p niodoo_real_integrated --release --features svc > /tmp/niodoo_main.log 2>&1 &
```

**Verification**: `curl http://localhost:9090/health`

## Phase 2: Smoke Test All Endpoints

### 2.1 Smoke Test Requirements
Each endpoint must pass:
1. **Health Check**: Endpoint responds to basic request
2. **Functional Test**: Endpoint performs actual work (not just returns 200)
3. **Error Handling**: Endpoint handles invalid requests gracefully

### 2.2 Smoke Test Matrix

| Endpoint | Health Check | Functional Test | Error Handling |
|----------|-------------|-----------------|----------------|
| Qdrant HTTP (6333) | GET /collections | Create/delete collection | Invalid collection name |
| Qdrant gRPC (6334) | Health check | Vector upsert/search | Invalid vector dimension |
| vLLM Generation (5001) | GET /v1/models | POST /v1/completions | Invalid model name |
| vLLM Curator (5002) | GET /v1/models | POST /v1/completions | Invalid prompt |
| Main Pipeline (9090) | GET /health | POST /v1/prompt (if exists) | Invalid JSON |
| RL Server (8080) | GET /health | POST /train (if exists) | Invalid request |

### 2.3 Smoke Test Script
See `scripts/smoke_test_all_endpoints.sh` for implementation.

## Phase 3: A/B Test - Proving Topology Understanding

### 3.1 Hypothesis
**Null Hypothesis**: Topology does NOT improve understanding
**Alternative Hypothesis**: Topology DOES improve understanding

### 3.2 Test Design
- **Baseline**: Topology-enabled (Hybrid mode, RCE enabled, nTokens enabled)
- **Treatment**: Topology-disabled (Baseline mode, RCE disabled, nTokens bypassed)

### 3.3 Success Criteria
Topology understanding is proven if:
1. **Persistence Entropy**: Higher in topology-enabled (richer structural understanding)
2. **Quality Scores**: Higher in topology-enabled (better understanding → better quality)
3. **β_meta**: More spikes in topology-enabled (RCE breakthrough detection)
4. **Statistical Significance**: p < 0.05 with medium+ effect size (Cohen's d > 0.5)

### 3.4 Metrics Collected
- Persistence entropy (mean, std)
- Spectral gap (mean)
- β_meta (current, peak)
- Quality scores (mean, std)
- Consonance scores
- Latency (P50, P95, P99, mean)
- Throughput (requests/second)
- Error rates

### 3.5 Test Execution
```bash
# Run A/B test
./scripts/run_topology_ab_test.sh

# Or manually:
cargo run --bin ab_test_runner --release -- \
    --baseline-name "topology_enabled" \
    --treatment-name "topology_disabled" \
    --baseline-config configs/topology_enabled.json \
    --treatment-config configs/topology_disabled.json \
    --concurrent-users 16 \
    --duration-secs 120 \
    --output-dir ab_test_results/topology_understanding
```

## Phase 4: Real Tests (Not Stubs)

### 4.1 Requirements
- **No Mock Mode**: All tests use real services
- **Real Requests**: Actual prompts sent to pipeline
- **Real Responses**: Full pipeline execution
- **Real Metrics**: Prometheus metrics from actual runs

### 4.2 Test Scenarios
1. **Simple Query**: Basic factual query
2. **Complex Reasoning**: Multi-step reasoning with context
3. **Emotional Context**: Emotional prompts for compass/PAD
4. **Code Generation**: Code generation tasks
5. **Topology-Aware**: Prompts that benefit from topology understanding

### 4.3 Validation
- All tests must complete without errors
- All metrics must be collected
- Statistical analysis must be performed
- Results must be documented

## Execution Order

1. ✅ Check current endpoint status
2. ⏳ Start all offline endpoints
3. ⏳ Wait for endpoints to be ready (vLLM takes 2-5 minutes)
4. ⏳ Run smoke tests on all endpoints
5. ⏳ Verify smoke tests pass
6. ⏳ Run A/B test
7. ⏳ Analyze A/B test results
8. ⏳ Document findings

## Troubleshooting

### vLLM Not Starting
- Check GPU memory: `nvidia-smi`
- Check logs: `tail -f /tmp/vllm_coder.log`
- Reduce GPU memory utilization if needed

### Pipeline Not Starting
- Check all dependencies are running
- Check logs: `tail -f /tmp/niodoo_main.log`
- Verify config: Check `tcs_runtime.env`

### Smoke Tests Failing
- Verify endpoints are actually responding
- Check service logs for errors
- Verify model paths are correct

## Success Criteria

✅ All endpoints online and responding
✅ All smoke tests pass
✅ A/B test completes successfully
✅ Statistical analysis shows topology improves understanding (if true)
✅ All results documented in CHANGELOG.md


