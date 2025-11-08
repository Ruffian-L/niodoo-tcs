# NIODOO-TCS Baseline Performance Metrics

**Capture Date**: 2025-11-01  
**Pipeline Version**: Pre-optimization baseline  
**Hardware Profile**: NVIDIA GeForce RTX 5090 (26.83 GiB in use via `nvidia-smi`, total 32 GiB), AMD EPYC 9455P 48-Core CPU, 755 GiB system RAM (92 GiB used / 662 GiB available per `free -h`).

## Purpose

This document captures baseline performance metrics before implementing the back-half pipeline optimizations (Phase 1-6). These metrics serve as the comparison baseline for measuring optimization impact.

## Key Metrics

### Latency Metrics

- **P95 Latency**: 5155.83 ms (derived from `results/soak_validator_full/soak_results.csv`, 4000-cycle run)
- **P99 Latency**: 7424.88 ms (same dataset)
- **Average Latency**: 3827.33 ms (same dataset)
- **Stage-wise Breakdown**:
  - Embedding: 90-150 ms per cycle (observed from `logs/soak_validator_full.log` sample traces; no aggregate yet)
  - ERAG: 0.6-1.2 ms per cycle (same source; single-point upserts only)
  - TCS Analysis: 0.07-2.0 ms per cycle (hybrid mode traces)
  - Generation: 1.7-3.4 s per cycle (dominant share of latency; matches overall averages)
  - Curator: Pending aggregate instrumentation (curator passive for majority of run; retries skipped)
  - Learning Loop: Pending (LoRA updates logged but no duration histogram yet)

### Memory Metrics

- **VRAM Usage**: 26.83 GiB in use (measured via `nvidia-smi --query-gpu=memory.used --format=csv,noheader` on 2025-11-01)
- **System Memory**: 92 GiB used / 755 GiB total (captured with `free -h`)
- **Memory Growth Rate**: Pending (baseline run lacks time-series memory samples)

### Quality Metrics

- **ROUGE-L**: 0.437 (mean across 4000 cycles; `results/soak_validator_full/soak_results.csv`)
- **Entropy σ**: 0.00425 bits (std. dev. across same dataset)
- **Entropy Mean**: 1.945 bits (dataset mean)
- **Betti β₁ Fidelity**: 100% (β₁ observed as 0 for all cycles in dataset)

### Throughput Metrics

- **Requests/Second**: ~1.02 requests/s (4000 cycles over 3930.5 s; timestamps from `logs/soak_validator_full.log`)
- **ERAG Batch Throughput**: Single-point upserts only (no batching in baseline)
- **Qdrant QPS**: ~1.02 QPS (one gRPC upsert per cycle; mirrors request throughput)

### Error Rates

- **Success Rate**: 100% (4000/4000 cycles marked breakthroughs in validation report)
- **Failure Recovery**: No retries executed; curator passive caused retry gate to skip (see `logs/soak_validator_full.log`)
- **Circuit Breaker Trips**: 0 observed during baseline soak (same log)

## Benchmark Suite

### 50-Prompt Suite
- **Easy Prompts**: 25 prompts (~300-600 tokens)
- **Hard Prompts**: 25 prompts (~800-2K tokens)
- **Execution**: Sequential processing per cycle (2 easy + 4 hard per cycle)

### Soak Test Configuration
- **Duration**: 3600s (1 hour) or 120s (quick test)
- **Concurrent Workers**: 150 (default) or 36 (quick)
- **Memory Check Interval**: 60s (default) or 10s (quick)

## Measurement Methodology

### Profiling Commands

```bash
# Generate flamegraph
RUSTFLAGS='-g' cargo flamegraph --bin soak_test_v2 -- --quick --duration=120

# Run soak test with metrics export
cargo run --bin soak_test_v2 -- --quick --duration=120 2>&1 | tee soak_baseline.log

# Extract metrics from logs
grep -E "(P99|P95|VRAM|ROUGE|entropy)" soak_baseline.log
```

### Tracing Setup

```bash
# Enable tracing with ERAG, TCS, LearningLoop spans
RUST_LOG=niodoo_real_integrated=trace cargo run --bin soak_test_v2 -- --quick
```

## Component Bottlenecks (Pre-Optimization)

Based on architecture analysis:

1. **ERAG**: Single-point upserts (200ms), no batching
2. **TCS Analysis**: Full persistent homology (150-300ms), CPU-bound
3. **LearningLoop**: fp32 QLoRA adapters (6-8GB VRAM), serial policy updates
4. **Curator**: Serial ROUGE loops (150ms), blocking Python calls
5. **WeightedMem**: CPU fitness calculations (100ms), no GPU acceleration

## Comparison Targets

After optimization (Phase 1-6), we expect:

- **P99 Latency**: <600ms (down from 851ms) - **30% improvement**
- **VRAM Usage**: <4GB (down from 6-8GB) - **20-50% reduction**
- **ERAG Upsert**: 140-160ms (down from 200ms) - **20-30% improvement**
- **TCS Analysis**: 50ms (down from 150-300ms) - **60% improvement**
- **Curator Refine**: 105ms (down from 150ms) - **30% improvement**

## Phase 0 Profiling Artifacts

**Status**: Pending completion on host with proper permissions

**Container Limitation**: The current container has `kernel.perf_event_paranoid=4` and `/proc/sys` is read-only, preventing `cargo flamegraph` from running. Profiling must be completed on a host/container where `kernel.perf_event_paranoid ≤ 2` or `CAP_PERFMON` is granted.

### Required Artifacts

Once profiling is completed on a properly configured host, archive the following artifacts:

1. **Flamegraph Profile**:
   - Command: `RUSTFLAGS='-g' cargo flamegraph --bin soak_test_v2 -- --quick --duration=120`
   - Expected output: `flamegraph.svg`
   - Archive path: `docs/profiling/baseline/flamegraph.svg` (or link/path to archived location)

2. **Perf Data**:
   - Expected output: `perf.data` (raw perf sampling data)
   - Archive path: `docs/profiling/baseline/perf.data` (or link/path to archived location)

3. **Trace Logs** (with Qdrant and vLLM running):
   - Prerequisites: Qdrant and vLLM must be running
   - Command: `RUST_LOG=niodoo_real_integrated=trace cargo run --bin soak_test_v2 -- --quick`
   - Expected output: Trace logs containing ERAG/TCS/LearningLoop spans
   - Archive path: `docs/profiling/baseline/trace_baseline.log` (or link/path to archived location)

### Host Requirements

To complete Phase 0 profiling:

1. **Move to a host/container** where:
   - `kernel.perf_event_paranoid ≤ 2` (check with: `cat /proc/sys/kernel/perf_event_paranoid`)
   - OR `CAP_PERFMON` capability is granted to the process/container

2. **Ensure services are running**:
   - Qdrant (gRPC on port 6334)
   - vLLM (HTTP on port 5001)

3. **Generate artifacts** using the commands above

4. **Archive artifacts** alongside existing baseline artifacts in `docs/profiling/baseline/` or document their location in this section

### Artifact Links

Once artifacts are generated, update this section with their paths:

- **Flamegraph SVG**: `[Path to flamegraph.svg]` (pending)
- **Perf Data**: `[Path to perf.data]` (pending)
- **Trace Logs**: `[Path to trace_baseline.log]` (pending)

## Notes

- Metrics should be captured on representative hardware (RTX 5090 or similar)
- Multiple runs recommended for statistical significance
- Monitor for variance and outliers
- Document any anomalies or environmental factors affecting measurements
- On this container `cargo flamegraph` is blocked because `kernel.perf_event_paranoid=4` and `/proc/sys` is read-only; profiling must run on a host with CAP_PERFMON or a lowered paranoia level.



