# Running Test Suites - Quick Guide

## Endpoints Required

The ablation and A/B test suites require these services:

1. **vLLM** - `http://127.0.0.1:5001`
2. **Qdrant** - `http://127.0.0.1:6333` (HTTP) and `http://127.0.0.1:6334` (gRPC)

## Quick Check

```bash
# Check endpoints
./scripts/check_and_run_tests.sh
```

Or manually:
```bash
curl http://127.0.0.1:5001/v1/models  # vLLM
curl http://127.0.0.1:6333/collections  # Qdrant
```

## Starting Services

### Start Qdrant
```bash
docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

### Start vLLM
```bash
vllm serve /workspace/models/Qwen2.5-0.5B-Instruct --port 5001
```

## Running Test Suites

### 1. Ablation Test Suite
```bash
./scripts/run_ablation_suite.sh
```

Runs all ablation experiments:
- DisableRce
- BypassNTokens
- DisableTcsGpu
- DisableGpuFitness
- DisableCurator
- BypassErag

Results: `ablation_results/`

### 2. A/B Test Suite
```bash
./scripts/run_ab_test_suite.sh
```

Compares baseline vs treatment configurations.

Results: `ab_test_results/`

### 3. All Tests + Superiority Proof
```bash
./scripts/check_and_run_tests.sh
```

Runs both suites and generates superiority proof report.

## Using Rust Binaries Directly

### Ablation Runner
```bash
cargo run --release --bin ablation_runner -- \
    --experiment DisableRce \
    --baseline baselines/baseline-latest.json \
    --concurrent-users 4 \
    --duration-secs 30 \
    --output-dir ablation_results
```

### A/B Test Runner
```bash
cargo run --release --bin ab_test_runner -- \
    --baseline-name baseline \
    --treatment-name treatment \
    --baseline-config configs/baseline.json \
    --treatment-config configs/treatment.json \
    --concurrent-users 4 \
    --duration-secs 30 \
    --output-dir ab_test_results
```

## Python A/B Test Framework

```bash
python3 scripts/ab_test_comprehensive.py
```

## Available Test Scripts

- `scripts/check_and_run_tests.sh` - Main entry point (checks endpoints + runs all)
- `scripts/run_ablation_suite.sh` - Ablation test suite
- `scripts/run_ab_test_suite.sh` - A/B test suite
- `scripts/run_superiority_proof.sh` - Generate superiority proof report

## Troubleshooting

### Services Not Available
If endpoints are offline, tests will fail gracefully. Start services first:
1. Start Qdrant: `docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant`
2. Start vLLM: `vllm serve /path/to/model --port 5001`
3. Verify: `./scripts/check_and_run_tests.sh`

### Tests Failing
- Check service logs
- Verify endpoints are accessible
- Check environment variables (VLLM_ENDPOINT, QDRANT_URL)
- Review test logs in output directories

## Output Directories

- `ablation_results/` - Ablation test results
- `ab_test_results/` - A/B test results
- `superiority_proofs/` - Superiority proof reports


