# 🧪 Complete Test Plan: Endpoints → Smoke Tests → A/B Test

**Created:** 2025-01-27  
**Purpose:** Get all endpoints online, smoke test them, then run A/B test to prove topology understanding

## Overview

This plan provides a complete end-to-end testing pipeline that:
1. ✅ Starts all required endpoints (Qdrant, vLLM Generation, vLLM Curator)
2. ✅ Smoke tests all endpoints to verify they're functional
3. ✅ Runs topology A/B test to prove if AI uses topology for understanding

## Master Script

**Location:** `scripts/start_smoke_and_ab_test.sh`

This script automates the entire process:

```bash
cd /workspace/Niodoo-Final
bash scripts/start_smoke_and_ab_test.sh
```

### What It Does

#### Phase 1: Start All Endpoints
- **Qdrant** (ports 6333 HTTP, 6334 gRPC)
  - Starts Docker container if not running
  - Verifies collections endpoint is accessible
- **vLLM Generation** (port 5001)
  - Starts Qwen 3 Coder model server
  - Waits up to 10 minutes for model to load
  - Verifies `/v1/models` endpoint
- **vLLM Curator** (port 5002)
  - Starts Qwen 2.5 Topology model server (optional)
  - Falls back to shared port if model not found
  - Verifies `/v1/models` endpoint

#### Phase 2: Smoke Tests
- Tests all critical endpoints:
  - Qdrant HTTP and health
  - vLLM Generation
  - vLLM Curator (or shared port)
  - Main pipeline health/ready/metrics (optional)
- **Fails fast** if critical endpoints are unavailable
- Reports detailed status for each endpoint

#### Phase 3: A/B Test
- Loads configurations:
  - Baseline: `configs/topology_enabled.json` (Hybrid mode, RCE enabled, nTokens enabled)
  - Treatment: `configs/topology_disabled.json` (Baseline mode, RCE disabled, nTokens bypassed)
- Runs statistical comparison:
  - Latency (P50, P95, P99, mean)
  - Throughput (requests/second)
  - Quality scores
  - **Topology metrics**: persistence entropy, spectral gap, β_meta
- Determines topology impact:
  - **Positive**: Higher persistence entropy AND higher quality scores
  - **Negative**: Lower persistence entropy AND lower quality scores
  - **Neutral**: Minimal differences
  - **Inconclusive**: Mixed signals

### Configuration

Set environment variables before running:

```bash
export CONCURRENT_USERS=16        # Number of concurrent users (default: 16)
export DURATION_SECS=120          # Test duration in seconds (default: 120)
export VLLM_MODEL_ID=/path/to/model    # Override default model path
export CURATOR_MODEL=/path/to/model     # Override curator model path
```

### Output

Results are saved to timestamped directory:
```
ab_test_results/topology_understanding_YYYYMMDD_HHMMSS/
├── ab_test_topology_enabled_vs_topology_disabled.json  # Full results
├── ab_test.log                                          # Execution log
```

### Key Metrics to Check

After test completes, check these fields in the JSON results:

- **`topology_impact`**: `"positive"`, `"negative"`, `"neutral"`, or `"inconclusive"`
- **`persistence_entropy_difference`**: Higher = richer structural understanding
- **`quality_difference_pct`**: Higher = better understanding/performance
- **`beta_meta_difference`**: RCE breakthrough detection difference
- **`cohens_d_latency`**: Effect size for latency (|d| > 0.8 = large effect)
- **`p_value_latency`**: Statistical significance (p < 0.05 = significant)

## Manual Steps (If Needed)

If you prefer to run steps manually:

### 1. Start Qdrant
```bash
docker run -d --name qdrant --restart unless-stopped \
    -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage \
    qdrant/qdrant:latest
```

### 2. Start vLLM Generation
```bash
source venv/bin/activate
python3 -m vllm.entrypoints.openai.api_server \
    --model /workspace/models/Qwen3-Coder \
    --host 127.0.0.1 \
    --port 5001 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 32768 \
    --trust-remote-code \
    > /tmp/vllm_generation.log 2>&1 &
```

### 3. Start vLLM Curator (Optional)
```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model /workspace/models/Qwen2.5-7B-Instruct-AWQ \
    --host 127.0.0.1 \
    --port 5002 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.15 \
    --max-model-len 2048 \
    --trust-remote-code \
    > /tmp/vllm_curator.log 2>&1 &
```

### 4. Verify Endpoints
```bash
bash scripts/verify_all_endpoints.sh
```

### 5. Run A/B Test
```bash
cargo run --bin ab_test_runner --release -- \
    --baseline-name "topology_enabled" \
    --treatment-name "topology_disabled" \
    --baseline-config configs/topology_enabled.json \
    --treatment-config configs/topology_disabled.json \
    --concurrent-users 16 \
    --duration-secs 120 \
    --output-dir ab_test_results/topology_understanding
```

## Troubleshooting

### vLLM Not Starting
- Check GPU memory: `nvidia-smi`
- Check logs: `tail -f /tmp/vllm_generation.log`
- Reduce GPU memory utilization if needed

### Qdrant Not Starting
- Check if port is in use: `lsof -i :6333`
- Remove old container: `docker rm -f qdrant`
- Check Docker logs: `docker logs qdrant`

### A/B Test Failing
- Ensure all endpoints are online (run smoke tests first)
- Check that config files exist: `ls configs/topology_*.json`
- Verify models are accessible: `ls -la /workspace/models/`
- Check A/B test logs in output directory

## Expected Results

If topology is helping understanding, you should see:
- **`topology_impact: "positive"`**
- **`persistence_entropy_difference > 0`** (topology-enabled has higher entropy)
- **`quality_difference_pct > 0`** (topology-enabled has better quality)
- **`p_value_latency < 0.05`** (statistically significant difference)

## Next Steps

After A/B test completes:
1. Review results JSON for topology impact assessment
2. Check logs for any errors or warnings
3. If topology impact is positive, document findings
4. If inconclusive, consider longer test duration or more concurrent users
5. Update CHANGELOG.md with results

---

**See Also:**
- `HOW_TO_START.md` - Detailed service startup guide
- `scripts/verify_all_endpoints.sh` - Endpoint verification script
- `scripts/run_topology_ab_test.sh` - Alternative A/B test wrapper
- `CHANGELOG.md` - Recent changes and updates


