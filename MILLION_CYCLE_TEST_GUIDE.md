# Million-Cycle NIODOO Test on H200

## Overview

This guide covers running 1M pipeline cycles on NVIDIA H200 hardware to validate NIODOO's operational torque at scale.

## Architecture

- **Pipeline**: Embedding → Torus Projection → TCS Analysis → ERAG Retrieval → Generation → Curator → Learning
- **Parallelization**: Rayon-based with configurable worker pools
- **Hardware**: NVIDIA H200 (141GB HBM3e, CUDA 12.8+)
- **Test Scale**: 1,000,000 cycles with dynamic prompt generation

## Quick Start

### Option 1: Automated Script (Recommended)

```bash
# Run with defaults (1M tests, 128 workers, 100 batch size)
./scripts/run_million_cycle_h200.sh

# Custom configuration
TEST_COUNT=1000000 WORKERS=256 BATCH_SIZE=100 ./scripts/run_million_cycle_h200.sh
```

### Option 2: Direct Cargo Execution

```bash
cargo run --release --bin million_cycle_test -- \
    --count 1000000 \
    --workers 128 \
    --batch-size 100 \
    --output-dir logs/million_cycle_test \
    --hardware h200
```

## Configuration Parameters

### Test Parameters

- `--count`: Total number of tests to run (default: 1,000,000)
- `--workers`: Number of parallel pipeline workers (default: 128)
- `--batch-size`: Batch size per worker (default: 100)
- `--output-dir`: Output directory for results (default: logs/million_cycle_test)
- `--hardware`: Hardware profile - `beelink`, `5080q`, or `h200` (default: h200)

### Hardware Profiles

| Profile | Batch Size | Latency Budget | KV Cache Tokens |
|---------|------------|----------------|-----------------|
| Beelink | 8          | 100ms          | 128K            |
| 5080q   | 4          | 180ms          | 256K            |
| H200    | 32         | Backfill       | 512K            |

## Expected Performance

On H200 hardware with optimized configuration:

- **Throughput**: 2000-5000 tests/sec
- **Average Latency**: 50-100ms per cycle
- **P95 Latency**: <150ms
- **GPU Utilization**: >90%
- **Memory Usage**: 60-80GB (out of 141GB)

## Output Files

After completion, check the output directory:

```
logs/million_cycle_test_YYYYMMDD_HHMMSS/
├── summary.json              # Test summary statistics
├── sample_results.csv        # First 1000 results
├── test_output.log           # Full test log
└── gpu_monitor.log           # GPU utilization tracking
```

## Monitoring

### GPU Monitoring

The script automatically monitors GPU stats:
- GPU utilization percentage
- Memory usage
- Power consumption
- Temperature

Check `gpu_monitor.log` for detailed metrics.

### Real-time Monitoring

In another terminal:
```bash
watch -n 1 nvidia-smi
```

## Bottleneck Analysis

The test reports timing for each pipeline stage:

- **Torus Projection**: Embedding to torus pad state
- **TCS Analysis**: Topological signature computation
- **ERAG Retrieval**: Memory lookup and collapse
- **Generation**: Hybrid vLLM/Ollama generation

Bottlenecks >50ms warrant investigation.

## Troubleshooting

### Out of Memory (OOM)

Reduce workers or batch size:
```bash
cargo run --release --bin million_cycle_test -- \
    --count 100000 \
    --workers 64 \
    --batch-size 50
```

### Slow Performance

1. Check GPU utilization: `nvidia-smi dmon -s pucvgt`
2. Verify vLLM endpoint is responding
3. Check Qdrant connection
4. Review generation source distribution

### Test Failures

Check logs:
```bash
tail -f logs/million_cycle_test_*/test_output.log
```

Common issues:
- vLLM endpoint unreachable → Check `VLLM_ENDPOINT` env var
- Qdrant connection failed → Check `QDRANT_URL` env var
- Torus seed issues → Check `TORUS_SEED` env var

## Benchmarking Different Configurations

### Conservative (Safe for Initial Testing)
```bash
cargo run --release --bin million_cycle_test -- \
    --count 10000 \
    --workers 32 \
    --batch-size 50
```

### Aggressive (Maximum Throughput)
```bash
cargo run --release --bin million_cycle_test -- \
    --count 1000000 \
    --workers 256 \
    --batch-size 200
```

### Balanced (Recommended for Production)
```bash
cargo run --release --bin million_cycle_test -- \
    --count 1000000 \
    --workers 128 \
    --batch-size 100
```

## Success Criteria

✅ **Operational Torque Validation**:
- Entropy stability (< 0.3 std)
- Emotional activation (>20% threat/healing)
- Average latency (<100ms on H200)
- Throughput (>1000 tests/sec)

✅ **Hardware Utilization**:
- GPU utilization >90%
- Memory usage 60-80%
- Temperature <70°C

✅ **Pipeline Stability**:
- Failure rate <1%
- Cache hit rate >30%
- Balanced stage timings

## Next Steps

After successful million-cycle test:

1. **Analyze Results**: Review `summary.json` for insights
2. **Optimize Bottlenecks**: Focus on slowest stages
3. **Scale Up**: Increase batch sizes for higher throughput
4. **Production Deployment**: Use validated configuration

## References

- NIODOO Architecture: `NIODOO_TCS_ARCHITECTURE.md`
- Pipeline Details: `niodoo_real_integrated/src/pipeline.rs`
- Configuration: `niodoo_real_integrated/src/config.rs`

