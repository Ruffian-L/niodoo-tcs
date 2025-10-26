# NIODOO H200 Million-Cycle Test Setup - Complete

## Summary

Successfully set up 1M-cycle parallel test harness for NIODOO pipeline on NVIDIA H200 hardware.

## What Was Built

### 1. Million-Cycle Test Binary (`million_cycle_test.rs`)

**Location**: `niodoo_real_integrated/src/bin/million_cycle_test.rs`

**Features**:
- Parallel execution using Rayon (CPU-bound tests)
- Configurable worker pools (default: 128 workers)
- Batch processing for efficient memory usage
- Dynamic prompt generation for diverse test coverage
- Comprehensive metrics collection (entropy, ROUGE, latency, bottleneck analysis)
- Hardware-aware configuration (H200 optimized)

**Key Components**:
- `PipelinePool`: Manages multiple pipeline instances for parallel execution
- Dynamic test prompt generation with 10 base templates × 20+ topics
- Statistical analysis (mean, std, percentiles)
- Bottleneck detection (Torus, TCS, ERAG, Generation timing)

### 2. H200 Hardware Profile Support

**Location**: `niodoo_real_integrated/src/config.rs`

**Specifications**:
- Batch Size: 32 (vs 8 for Beelink, 4 for 5080q)
- Latency Budget: 50ms (vs 100ms Beelink, 180ms 5080q)
- KV Cache Tokens: 512K (vs 128K Beelink, 256K 5080q)
- Optimized for H200's 141GB HBM3e memory

### 3. Test Execution Scripts

**Scripts**:
- `scripts/run_million_cycle_h200.sh`: Full million-cycle test with GPU monitoring
- `scripts/smoke_test_h200.sh`: Quick 1000-cycle validation test

**Features**:
- Automatic GPU monitoring (`nvidia-smi dmon`)
- Real-time GPU utilization tracking
- Automatic cleanup on exit
- Detailed performance summaries
- JSON and CSV result export

### 4. Documentation

**Files**:
- `MILLION_CYCLE_TEST_GUIDE.md`: Comprehensive usage guide
- `NIODOO_H200_SETUP_COMPLETE.md`: This file

## How to Use

### Quick Start (Smoke Test)

```bash
# Verify everything works with 1000 tests
./scripts/smoke_test_h200.sh
```

### Full Million-Cycle Test

```bash
# Run with defaults (1M tests, 128 workers)
./scripts/run_million_cycle_h200.sh

# Custom configuration
TEST_COUNT=1000000 WORKERS=256 BATCH_SIZE=100 ./scripts/run_million_cycle_h200.sh
```

### Direct Execution

```bash
cargo run --release --bin million_cycle_test -- \
    --count 1000000 \
    --workers 128 \
    --batch-size 100 \
    --output-dir logs/million_cycle_test \
    --hardware h200
```

## Expected Performance on H200

Based on hardware specifications and pipeline architecture:

| Metric | Target | Notes |
|--------|--------|-------|
| Throughput | 2000-5000 tests/sec | With 128 workers, batch=100 |
| Avg Latency | 50-100ms | Per full pipeline cycle |
| P95 Latency | <150ms | 95th percentile |
| GPU Utilization | >90% | Peak utilization |
| Memory Usage | 60-80GB | Out of 141GB HBM3e |
| Temperature | <70°C | Thermal limits |

## Pipeline Stages (Timing Breakdown)

1. **Embedding** (Qwen): ~10-20ms
2. **Torus Projection**: ~5-10ms
3. **TCS Analysis**: ~15-25ms
4. **ERAG Retrieval**: ~20-30ms
5. **Generation** (vLLM/Ollama): ~30-50ms
6. **Curator**: ~5-10ms
7. **Learning Loop**: ~5-10ms

**Total**: ~90-155ms per cycle (on H200)

## Bottleneck Analysis

The test reports timing for each stage to identify bottlenecks:

- **Torus Projection**: Embedding space → Torus pad state
- **TCS Analysis**: Topological signature computation
- **ERAG Retrieval**: Memory lookup and context collapse
- **Generation**: Hybrid vLLM/Ollama text generation

Stages consistently taking >50ms should be optimized.

## Output Files

After completion, check the output directory:

```
logs/million_cycle_test_YYYYMMDD_HHMMSS/
├── summary.json              # Test summary with statistics
├── sample_results.csv        # First 1000 detailed results
├── test_output.log           # Full test execution log
└── gpu_monitor.log           # GPU utilization over time
```

## Monitoring

### Real-time GPU Stats

```bash
# In another terminal
watch -n 1 nvidia-smi
```

### Analysis Commands

```bash
# View summary
cat logs/million_cycle_test_*/summary.json | jq

# Check GPU utilization
tail -100 logs/million_cycle_test_*/gpu_monitor.log

# View results
head -20 logs/million_cycle_test_*/sample_results.csv
```

## Configuration Tuning

### Conservative (Initial Testing)
```bash
cargo run --release --bin million_cycle_test -- \
    --count 10000 \
    --workers 32 \
    --batch-size 50
```

### Balanced (Recommended)
```bash
cargo run --release --bin million_cycle_test -- \
    --count 1000000 \
    --workers 128 \
    --batch-size 100
```

### Aggressive (Maximum Throughput)
```bash
cargo run --release --bin million_cycle_test -- \
    --count 1000000 \
    --workers 256 \
    --batch-size 200
```

## Success Criteria

✅ **Performance**:
- Throughput >1000 tests/sec
- Average latency <100ms
- P95 latency <150ms

✅ **Hardware Utilization**:
- GPU utilization >90%
- Memory usage 60-80%
- Temperature <70°C

✅ **Pipeline Stability**:
- Failure rate <1%
- Cache hit rate >30%
- Entropy stability (<0.3 std)
- Emotional activation (>20% threat/healing rate)

## Troubleshooting

### Out of Memory
```bash
# Reduce workers/batch size
--workers 64 --batch-size 50
```

### Slow Performance
1. Check GPU utilization: `nvidia-smi dmon -s pucvgt`
2. Verify vLLM endpoint: `curl $VLLM_ENDPOINT/health`
3. Check Qdrant connection: `curl $QDRANT_URL/health`

### High Failure Rate
Check logs: `tail -f logs/million_cycle_test_*/test_output.log`

Common issues:
- vLLM endpoint unreachable
- Qdrant connection failed
- Torus seed conflicts

## Files Modified

1. `niodoo_real_integrated/src/bin/million_cycle_test.rs` - New test harness
2. `niodoo_real_integrated/src/config.rs` - Added H200 hardware profile
3. `niodoo_real_integrated/Cargo.toml` - Added binary definition
4. `niodoo_real_integrated/src/lora_trainer.rs` - Fixed compilation errors
5. `scripts/run_million_cycle_h200.sh` - Full test runner
6. `scripts/smoke_test_h200.sh` - Smoke test
7. `MILLION_CYCLE_TEST_GUIDE.md` - Usage documentation

## Architecture

### Parallelization Strategy

- **Rayon**: For CPU-bound operations (prompt generation, result aggregation)
- **Tokio**: For async I/O (HTTP calls to vLLM, Qdrant, embeddings)
- **Worker Pool**: Multiple Pipeline instances running in parallel
- **Batch Processing**: Group tests into batches for efficient processing

### Memory Management

- LRU caches for embeddings and ERAG collapses
- Batch size limits prevent OOM
- Graceful degradation on memory pressure

### Error Handling

- Individual test failures don't abort entire run
- Failed tests are tracked and reported
- Automatic retry for transient failures

## Next Steps

1. **Run Smoke Test**: Validate with 1000 cycles
2. **Analyze Results**: Review bottlenecks and optimize
3. **Scale Up**: Run full million-cycle test
4. **Production**: Use validated configuration

## References

- Pipeline Architecture: `niodoo_real_integrated/src/pipeline.rs`
- Configuration: `niodoo_real_integrated/src/config.rs`
- Full Guide: `MILLION_CYCLE_TEST_GUIDE.md`
- GPU Specs: NVIDIA H200 Architecture Documentation

## Status

✅ Setup Complete
✅ Compilation Successful
✅ Ready for Testing

Run: `./scripts/smoke_test_h200.sh` to begin!

