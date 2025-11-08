# NIODOO Pipeline - Performance Tuning Guide

## Overview

This guide covers performance optimization strategies for the NIODOO pipeline in production environments.

## Table of Contents

1. [Cache Optimization](#cache-optimization)
2. [Concurrency Tuning](#concurrency-tuning)
3. [Memory Management](#memory-management)
4. [GPU Acceleration](#gpu-acceleration)
5. [Network Optimization](#network-optimization)
6. [Pipeline Stage Optimization](#pipeline-stage-optimization)

## Cache Optimization

### Embedding Cache

The embedding cache stores computed embeddings to avoid redundant computation.

**Configuration:**
```toml
embedding_cache_ttl_secs = 3600  # 1 hour default
cache_capacity = 256             # LRU cache size
```

**Tuning Guidelines:**
- **High-throughput workloads**: Increase `cache_capacity` to 512-1024
- **Stable prompts**: Increase `embedding_cache_ttl_secs` to 7200 (2 hours)
- **Dynamic prompts**: Decrease `embedding_cache_ttl_secs` to 1800 (30 min)

**Metrics to Monitor:**
- `niodoo_cache_hits_total / niodoo_cache_requests_total` - Hit rate (target: >0.8)
- `niodoo_cache_size_bytes` - Memory usage

### Collapse Cache

The collapse cache stores ERAG retrieval results.

**Configuration:**
```toml
collapse_cache_ttl_secs = 1800  # 30 minutes default
```

**Tuning Guidelines:**
- **Frequent ERAG updates**: Decrease TTL to 900 (15 min)
- **Stable memory**: Increase TTL to 3600 (1 hour)

### Cache Compression

Enable LZ4 compression for large cache entries:

```rust
// In cache.rs
use lz4_flex::{compress, decompress};

struct CompressedCacheEntry<T> {
    compressed: Vec<u8>,
    _phantom: PhantomData<T>,
}
```

**Benefits:**
- 50-70% memory reduction for embeddings
- Trade-off: CPU overhead (~5-10ms per operation)

## Concurrency Tuning

### Tokio Runtime

Configure worker threads:

```bash
export TOKIO_WORKER_THREADS=8  # Default: CPU count
```

**Guidelines:**
- **CPU-bound workloads**: `worker_threads = CPU cores`
- **IO-bound workloads**: `worker_threads = CPU cores * 2`
- **Mixed workloads**: `worker_threads = CPU cores * 1.5`

### Pipeline Parallelism

Current parallel stages:
- Compass + ERAG collapse (already parallelized)

**Potential Optimizations:**
- Embedding + Topology analysis (independent operations)
- ERAG retrieval + Tokenizer enhancement (can overlap)

**Example:**
```rust
let (embedding, topology) = tokio::join!(
    async { embedder.embed(prompt).await },
    async { tcs_analyzer.analyze(pad_state).await }
);
```

### Request Concurrency

Limit concurrent requests per pod:

```toml
max_concurrent_requests = 10  # Default: unlimited
```

**Tuning:**
- **Memory-constrained**: Lower to 5-8
- **High-memory**: Increase to 20-30
- Monitor: `niodoo_pipeline_active_requests`

## Memory Management

### Memory Pools

Pre-allocate buffers for common sizes:

```rust
struct MemoryPool {
    embedding_buffers: Vec<Vec<f32>>,  // 896-dim embeddings
    pad_state_buffers: Vec<PadGhostState>,
}
```

**Benefits:**
- Reduces allocation overhead
- Prevents fragmentation
- Predictable memory usage

### Garbage Collection

Rust doesn't use GC, but you can influence memory behavior:

**Force cleanup:**
```rust
// Drop large structures explicitly
drop(large_cache);
std::alloc::dealloc(...);
```

**Monitor memory:**
```bash
# Via Prometheus
niodoo_memory_bytes
niodoo_memory_allocated_bytes
niodoo_memory_freed_bytes
```

### Memory Limits

Set appropriate Kubernetes limits:

```yaml
resources:
  requests:
    memory: "2Gi"
  limits:
    memory: "4Gi"
```

**Guidelines:**
- **Minimal**: 1Gi request, 2Gi limit
- **Production**: 2Gi request, 4Gi limit
- **High-throughput**: 4Gi request, 8Gi limit

## GPU Acceleration

### TDA Operations

Enable GPU acceleration for topological data analysis:

**Configuration:**
```bash
export CUDA_VISIBLE_DEVICES=0
export TDA_USE_GPU=1
```

**Performance Impact:**
- **CPU**: ~500ms per topology analysis
- **GPU**: ~50ms per topology analysis (10x speedup)

**Requirements:**
- CUDA-capable GPU
- `candle-core` with CUDA support
- `tcs-ml` feature: `cuda`

### Embedding Computation

GPU acceleration for embeddings:

**Current**: CPU-based (QwenStatefulEmbedder)
**Future**: GPU-based via ONNX Runtime

**Expected Speedup:**
- CPU: ~100ms per embedding
- GPU: ~10ms per embedding (10x speedup)

## Network Optimization

### Connection Pooling

Reuse HTTP connections:

```rust
let client = reqwest::Client::builder()
    .pool_max_idle_per_host(10)
    .timeout(Duration::from_secs(10))
    .build()?;
```

**Benefits:**
- Reduces connection overhead
- Lower latency
- Better resource utilization

### Timeout Configuration

```toml
qdrant_timeout_secs = 10
vllm_timeout_secs = 30
embedding_timeout_secs = 5
```

**Tuning:**
- **Local services**: Lower timeouts (5s)
- **Remote services**: Higher timeouts (30s)
- **Critical paths**: Shorter timeouts with retries

### Circuit Breaker Settings

```rust
CircuitBreakerConfig {
    failure_threshold: 5,      // Open after 5 failures
    timeout: Duration::from_secs(60),  // Wait 60s before retry
    base_delay: Duration::from_millis(100),
    max_delay: Duration::from_secs(30),
    backoff_exponent: 2.0,
}
```

**Tuning:**
- **Aggressive**: Lower failure_threshold (3), shorter timeout (30s)
- **Conservative**: Higher failure_threshold (10), longer timeout (120s)

## Pipeline Stage Optimization

### Stage Timings

Monitor stage latencies:

```bash
# Via metrics
niodoo_pipeline_stage_embedding_seconds
niodoo_pipeline_stage_torus_seconds
niodoo_pipeline_stage_tcs_seconds
niodoo_pipeline_stage_compass_seconds
niodoo_pipeline_stage_erag_seconds
niodoo_pipeline_stage_tokenizer_seconds
niodoo_pipeline_stage_generation_seconds
```

### Bottleneck Identification

1. **Find slowest stage:**
   ```bash
   curl http://localhost:8080/metrics | grep stage | sort -k2 -n
   ```

2. **Optimize accordingly:**
   - **Embedding**: Enable GPU, increase cache
   - **TDA**: Enable GPU acceleration
   - **ERAG**: Optimize Qdrant queries, increase cache
   - **Generation**: Optimize vLLM batch size

### Topology Analysis

**Optimization Strategies:**
- Cache Vietoris-Rips complexes for similar PAD states
- Incremental homology computation
- Approximate persistence (faster, less accurate)

**Configuration:**
```toml
topology_cache_enabled = true
topology_cache_ttl_secs = 3600
topology_approximate = false  # Set true for speed
```

## Benchmarking

### Performance Benchmarks

Run benchmarks:

```bash
cargo bench --bench niodoo_real_bench
```

### Load Testing

Use soak test:

```bash
./run_small_soak.sh  # 64 cycles
./run_big_soak.sh    # 1000+ cycles
```

**Metrics to collect:**
- Latency percentiles (p50, p95, p99)
- Throughput (requests/sec)
- Error rate
- Memory usage
- CPU usage

### Profiling

Use `perf` or `flamegraph`:

```bash
# Install perf
sudo apt-get install linux-perf

# Profile
perf record -g --target-pid $(pgrep niodoo)
perf report
```

## Best Practices

1. **Start Conservative**: Begin with default settings
2. **Monitor First**: Collect metrics before optimizing
3. **One Change at a Time**: Isolate changes for testing
4. **Test Under Load**: Use realistic workloads
5. **Document Changes**: Track what works and what doesn't

## Troubleshooting Performance Issues

### High Latency

1. Check stage timings
2. Identify bottleneck stage
3. Optimize that stage (cache, GPU, parallelism)
4. Verify external services (Qdrant, vLLM)

### High Memory Usage

1. Check cache sizes
2. Reduce cache capacity if needed
3. Enable compression
4. Review memory leaks (use valgrind)

### High CPU Usage

1. Check for busy loops
2. Optimize hot paths
3. Enable GPU acceleration
4. Increase parallelism

### Low Throughput

1. Check concurrency limits
2. Verify no bottlenecks
3. Increase worker threads
4. Scale horizontally
