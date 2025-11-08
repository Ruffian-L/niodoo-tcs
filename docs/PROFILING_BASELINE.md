# Profiling Baseline Setup Guide

## Overview

This guide documents the profiling baseline setup for measuring NIODOO-TCS performance before and after back-half pipeline optimizations (Phase 1-6).

## Prerequisites

### Required Tools

```bash
# Install flamegraph
cargo install flamegraph

# Install perf (Linux only)
sudo apt-get install linux-perf  # Debian/Ubuntu
# or
sudo yum install perf            # RHEL/CentOS

# Ensure debugging symbols are enabled
export RUSTFLAGS='-g'
```

### Environment Setup

```bash
# Enable tracing for detailed spans
export RUST_LOG=niodoo_real_integrated=trace,erag=debug,tcs_analysis=debug,learning=debug

# Set performance profiling environment
export CARGO_PROFILE_RELEASE_DEBUG=1
```

## Baseline Profiling Steps

### 1. Generate Flamegraph

```bash
# Quick test (120s, 36 workers)
RUSTFLAGS='-g' cargo flamegraph --bin soak_test_v2 -- --quick --duration=120

# Full soak test (3600s, 150 workers)
RUSTFLAGS='-g' cargo flamegraph --bin soak_test_v2 -- --duration=3600

# Output: flamegraph.svg
```

### 2. Capture Tracing Sessions

```bash
# Run with tracing enabled
RUST_LOG=niodoo_real_integrated=trace cargo run --bin soak_test_v2 -- --quick 2>&1 | tee trace_baseline.log

# Extract ERAG metrics
grep -E "erag|ERAG" trace_baseline.log | grep -E "latency|duration|ms" > erag_baseline.log

# Extract TCS metrics
grep -E "tcs|TCS|topology" trace_baseline.log | grep -E "latency|duration|ms" > tcs_baseline.log

# Extract LearningLoop metrics
grep -E "learning|LearningLoop" trace_baseline.log | grep -E "latency|duration|ms" > learning_baseline.log
```

### 3. Run Benchmark Suite

```bash
# Run 50-prompt suite with metrics export
cargo run --bin soak_test_v2 -- --quick --duration=120 2>&1 | tee soak_baseline_$(date +%Y%m%d_%H%M%S).log

# Extract key metrics
grep -E "(P99|P95|VRAM|ROUGE|entropy|betti)" soak_baseline_*.log | head -100
```

### 4. Measure Component Contributions

#### ERAG Bottleneck Analysis

```bash
# Filter ERAG-related spans
grep "erag\|ERAG\|collapse\|upsert" trace_baseline.log | \
  awk '{print $NF}' | \
  sort -n | \
  awk '{
    sum+=$1; sumsq+=$1*$1; count++
  } END {
    mean=sum/count
    stddev=sqrt((sumsq/count) - (mean*mean))
    print "Mean:", mean, "ms"
    print "StdDev:", stddev, "ms"
    print "P95:", mean + 1.96*stddev, "ms"
    print "P99:", mean + 2.58*stddev, "ms"
  }'
```

#### TCS Analysis Bottleneck

```bash
# Filter TCS-related spans
grep "tcs\|TCS\|topology\|betti\|persistence" trace_baseline.log | \
  grep -E "latency|duration|ms" | \
  awk '{print $NF}' | \
  sort -n | \
  awk '{
    sum+=$1; sumsq+=$1*$1; count++
  } END {
    mean=sum/count
    stddev=sqrt((sumsq/count) - (mean*mean))
    print "Mean:", mean, "ms"
    print "StdDev:", stddev, "ms"
    print "P95:", mean + 1.96*stddev, "ms"
    print "P99:", mean + 2.58*stddev, "ms"
  }'
```

#### LearningLoop Bottleneck

```bash
# Filter LearningLoop-related spans
grep "learning\|LearningLoop\|qlora\|DQN" trace_baseline.log | \
  grep -E "latency|duration|ms" | \
  awk '{print $NF}' | \
  sort -n | \
  awk '{
    sum+=$1; sumsq+=$1*$1; count++
  } END {
    mean=sum/count
    stddev=sqrt((sumsq/count) - (mean*mean))
    print "Mean:", mean, "ms"
    print "StdDev:", stddev, "ms"
    print "P95:", mean + 1.96*stddev, "ms"
    print "P99:", mean + 2.58*stddev, "ms"
  }'
```

## Identifying Bottlenecks in P99 Latency Path

### Flamegraph Analysis

1. **Open flamegraph.svg** in browser
2. **Identify hot paths**:
   - Look for wide bars (high CPU time)
   - Focus on ERAG, TCS, LearningLoop, Curator sections
   - Note function call frequencies

3. **Key areas to examine**:
   - `EragClient::upsert_memory_with_cascade` - should show single-point upserts
   - `TCSAnalyzer::compute_betti_numbers` - should show full persistent homology
   - `LearningLoop::update` - should show serial policy updates
   - `Curator::curate_with_consonance` - should show serial ROUGE loops

### Trace Analysis

```bash
# Generate stage-wise latency breakdown
cat trace_baseline.log | \
  grep -E "(embedding|erag|tcs|compass|generation|curator|learning)" | \
  grep -E "latency|duration|ms" | \
  awk '{
    if ($0 ~ /embedding/) stage="embedding"
    else if ($0 ~ /erag/) stage="erag"
    else if ($0 ~ /tcs/) stage="tcs"
    else if ($0 ~ /compass/) stage="compass"
    else if ($0 ~ /generation/) stage="generation"
    else if ($0 ~ /curator/) stage="curator"
    else if ($0 ~ /learning/) stage="learning"
    print stage, $NF
  }' | \
  sort -k1,1 | \
  awk '{
    stage=$1; latency=$2; sum[stage]+=latency; count[stage]++
  } END {
    for (s in sum) {
      print s, "Mean:", sum[s]/count[s], "ms", "Count:", count[s]
    }
  }'
```

## Expected Baseline Observations

Based on architecture analysis, you should observe:

1. **ERAG**: 
   - Single-point upserts: ~200ms per operation
   - No batching visible in flamegraph
   - Sequential collapse operations

2. **TCS Analysis**:
   - Full persistent homology: 150-300ms per cycle
   - CPU-bound computation (no GPU acceleration)
   - High variance in computation time

3. **LearningLoop**:
   - fp32 QLoRA adapters: 6-8GB VRAM usage
   - Serial policy updates
   - Replay buffer updates not batched

4. **Curator**:
   - Serial ROUGE loops: ~150ms per refine
   - Blocking Python calls (no async bridges)
   - vLLM calls dominate latency

5. **WeightedMem**:
   - CPU fitness calculations: ~100ms
   - No GPU acceleration
   - Sequential consolidation

## Post-Optimization Comparison

After implementing Phase 1-6 optimizations, re-run profiling and compare:

- **ERAG**: Should show batched upserts (128 points), reduced latency
- **TCS**: Should show approximate TDA (giotto-tda), ~60% speedup
- **LearningLoop**: Should show fp16 adapters, reduced VRAM
- **Curator**: Should show parallel ROUGE scoring, async bridges
- **WeightedMem**: Should show GPU fitness calculations

## Documentation

All profiling outputs should be stored in:
- `docs/profiling/baseline/` - Baseline profiling data
- `docs/profiling/optimized/` - Post-optimization profiling data

## Notes

- Profiling adds overhead (typically 5-10%), factor this into measurements
- Multiple runs recommended for statistical significance
- Document hardware configuration (GPU model, CPU, memory)
- Note any environmental factors (other processes, thermal throttling)




