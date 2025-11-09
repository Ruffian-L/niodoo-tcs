# 🎯 COMPREHENSIVE PLAN: Endpoints → Smoke Tests → A/B Test

**Goal**: Get all endpoints online, smoke test them all, then run A/B test to prove AI uses topology for understanding.

## Phase 1: Get All Endpoints Online

### Required Endpoints

#### External Services (Dependencies)
1. **Qdrant** (Vector Database)
   - HTTP: `http://127.0.0.1:6333`
   - gRPC: `http://127.0.0.1:6334` (via HTTP health check)
   - Status: ❓ Check first

2. **vLLM Generation** (Qwen 3 Coder)
   - Endpoint: `http://127.0.0.1:5001/v1/models`
   - Status: ❓ Check first

3. **vLLM Curator** (Qwen 2.5 Topology)
   - Endpoint: `http://127.0.0.1:5002/v1/models` (or same as generation if shared)
   - Status: ❓ Check first

#### NIODOO Services
4. **Main Pipeline Server** (Health Endpoints)
   - Health: `http://localhost:9090/health`
   - Ready: `http://localhost:9090/ready`
   - Metrics: `http://localhost:9090/metrics`
   - Status: ❓ Check first

5. **RL Server** (Reinforcement Learning)
   - Health: `http://localhost:8080/health`
   - Evaluate: `http://localhost:8080/rl/evaluate` (POST)
   - Status: ❓ Check first

### Startup Commands

```bash
# 1. Start Qdrant
docker run -d \
    --name qdrant \
    --restart unless-stopped \
    -p 6333:6333 \
    -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage \
    qdrant/qdrant:latest

# 2. Start vLLM Generation (Qwen 3 Coder)
# Check if already running first
export VLLM_MODEL_ID="${VLLM_MODEL_ID:-/workspace/models/Qwen3-Coder}"
export VLLM_PORT=5001
python3 -m vllm.entrypoints.openai.api_server \
    --model "$VLLM_MODEL_ID" \
    --host 127.0.0.1 \
    --port ${VLLM_PORT} \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 32768 \
    --trust-remote-code \
    > /tmp/vllm_coder.log 2>&1 &

# 3. Start vLLM Curator (Qwen 2.5 Topology)
export CURATOR_MODEL="${CURATOR_MODEL:-/workspace/models/Qwen2.5-Topology}"
export CURATOR_VLLM_PORT=5002
python3 -m vllm.entrypoints.openai.api_server \
    --model "$CURATOR_MODEL" \
    --host 127.0.0.1 \
    --port ${CURATOR_VLLM_PORT} \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.15 \
    --max-model-len 2048 \
    --trust-remote-code \
    > /tmp/vllm_curator.log 2>&1 &

# 4. Start Main Pipeline Server
cd /workspace/Niodoo-Final
source tcs_runtime.env
cargo build -p niodoo_real_integrated --release --features svc
cargo run -p niodoo_real_integrated --release --features svc > /tmp/niodoo_main.log 2>&1 &

# 5. Start RL Server
cargo run --bin rl_server --release --features svc > /tmp/niodoo_rl.log 2>&1 &
```

## Phase 2: Smoke Test All Endpoints

### Smoke Test Checklist

Run comprehensive smoke tests using existing scripts:

```bash
# Option 1: Use verify script (quick check)
./scripts/verify_all_endpoints.sh

# Option 2: Use comprehensive test script (full smoke test)
./scripts/test_all_endpoints.sh

# Option 3: Manual smoke test
curl http://127.0.0.1:6333/collections  # Qdrant
curl http://127.0.0.1:6333/health        # Qdrant health
curl http://127.0.0.1:5001/v1/models    # vLLM Generation
curl http://127.0.0.1:5002/v1/models    # vLLM Curator
curl http://localhost:9090/health       # Main Pipeline
curl http://localhost:9090/ready        # Main Pipeline Ready
curl http://localhost:9090/metrics      # Main Pipeline Metrics
curl http://localhost:8080/health       # RL Server
```

### Expected Results
- ✅ All endpoints return HTTP 200/201
- ✅ Qdrant returns JSON with collections
- ✅ vLLM returns model information
- ✅ Health endpoints return healthy status
- ✅ Metrics endpoint returns Prometheus format

## Phase 3: A/B Test - Prove Topology Understanding

### Test Hypothesis
**H0**: AI does NOT use topology for understanding (no difference between topology-enabled and disabled)
**H1**: AI DOES use topology for understanding (topology-enabled performs better)

### Test Configuration

**Baseline (Topology-Enabled)**:
- `configs/topology_enabled.json`
- Hybrid mode, RCE enabled, nTokens enabled, GPU acceleration

**Treatment (Topology-Disabled)**:
- `configs/topology_disabled.json`
- Baseline mode, RCE disabled, nTokens bypassed, CPU only

### Metrics to Collect

1. **Topology Metrics** (KEY PROOF):
   - Persistence entropy (mean, std) - structural understanding
   - Spectral gap (mean) - exploration quality
   - β_meta (current, peak) - RCE breakthrough detection

2. **Quality Metrics**:
   - Quality scores (mean, std) - curator assessments
   - Consonance scores - coherence measurements

3. **Performance Metrics**:
   - Latency (P50, P95, P99, mean)
   - Throughput (requests/second)
   - Error rates

### Success Criteria

**Topology Impact Assessment**:
- **POSITIVE**: Higher persistence entropy AND higher quality scores → Topology helps understanding ✅
- **NEGATIVE**: Lower persistence entropy AND lower quality scores → Topology hurts understanding ❌
- **NEUTRAL**: Minimal differences → Topology has no effect
- **INCONCLUSIVE**: Mixed signals → Need more data

**Statistical Significance**:
- P-value < 0.05 for topology metrics
- Cohen's d > 0.5 (medium effect size)
- 95% confidence intervals don't overlap

### Execution Commands

```bash
# Step 1: Verify all endpoints are online
./scripts/verify_all_endpoints.sh

# Step 2: Run A/B test
./scripts/run_topology_ab_test.sh

# OR manually:
cargo run --bin ab_test_runner --release -- \
    --baseline-name "topology_enabled" \
    --treatment-name "topology_disabled" \
    --baseline-config configs/topology_enabled.json \
    --treatment-config configs/topology_disabled.json \
    --concurrent-users 16 \
    --duration-secs 120 \
    --output-dir ab_test_results/topology_understanding

# Step 3: Analyze results
cat ab_test_results/topology_understanding/ab_test_topology_enabled_vs_topology_disabled.json | jq '.'
```

### Expected Output

Results JSON will contain:
- `topology_impact`: "positive" | "negative" | "neutral" | "inconclusive"
- `persistence_entropy_difference`: Higher = richer structural understanding
- `quality_difference_pct`: Higher = better understanding
- `beta_meta_difference`: RCE breakthrough detection difference
- Statistical significance tests (p-values, effect sizes)

## Phase 4: Real Test Execution

### Pre-Flight Checklist
- [ ] All endpoints verified online
- [ ] Smoke tests passed
- [ ] Config files exist (`configs/topology_enabled.json`, `configs/topology_disabled.json`)
- [ ] Sufficient GPU memory available
- [ ] Environment variables set (`source tcs_runtime.env`)

### Execution Steps

1. **Verify Endpoints**:
   ```bash
   ./scripts/verify_all_endpoints.sh
   ```

2. **Smoke Test**:
   ```bash
   ./scripts/test_all_endpoints.sh
   ```

3. **Run A/B Test**:
   ```bash
   ./scripts/run_topology_ab_test.sh
   ```

4. **Review Results**:
   ```bash
   # Check topology impact
   cat ab_test_results/topology_understanding/ab_test_topology_enabled_vs_topology_disabled.json | jq '.topology_impact'
   
   # Check persistence entropy difference
   cat ab_test_results/topology_understanding/ab_test_topology_enabled_vs_topology_disabled.json | jq '.comparison.persistence_entropy_difference'
   
   # Check quality difference
   cat ab_test_results/topology_understanding/ab_test_topology_enabled_vs_topology_disabled.json | jq '.comparison.quality_difference_pct'
   ```

## Troubleshooting

### Endpoints Not Starting
- Check logs: `/tmp/vllm_coder.log`, `/tmp/vllm_curator.log`, `/tmp/niodoo_main.log`, `/tmp/niodoo_rl.log`
- Check ports: `lsof -i :6333`, `lsof -i :5001`, `lsof -i :5002`, `lsof -i :9090`, `lsof -i :8080`
- Check GPU: `nvidia-smi`

### Smoke Tests Failing
- Verify services are actually running (not just ports listening)
- Check service logs for errors
- Verify environment variables are set correctly

### A/B Test Failing
- Ensure all endpoints are online first
- Check config files exist and are valid JSON
- Verify sufficient resources (GPU memory, disk space)
- Check logs in `ab_test_results/topology_understanding/ab_test.log`

## Success Indicators

✅ **All Endpoints Online**: All 5 services responding correctly
✅ **Smoke Tests Pass**: All endpoints return expected responses
✅ **A/B Test Complete**: Results JSON generated successfully
✅ **Topology Impact = "positive"**: Proves AI uses topology for understanding
✅ **Statistical Significance**: P < 0.05, effect size > 0.5

---

**Created**: 2025-01-XX
**Status**: Ready for execution
