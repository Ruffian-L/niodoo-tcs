# Prometheus Metrics Configuration

This document describes the Prometheus scraping configuration for all dependencies of the niodoo_real_integrated system.

## Service Dependencies and Metrics

### vLLM Generation Service

**Endpoint**: `http://127.0.0.1:5001/metrics`  
**Health Check Query**: `up{job="vllm"} == 1`

**Key Metrics** (exact names depend on vLLM version):
- `vllm_request_latency_seconds` (Histogram): End-to-end request latencies for P50/P95/P99 calculation
- `vllm_requests_total` (Counter, labeled by status): Request rate and error rate
- `vllm_prompt_tokens_total` (Counter): Rate of prompt token processing
- `vllm_generation_tokens_total` (Counter): Rate of generation token processing
- `vllm_gpu_cache_usage_perc` (Gauge): GPU KV cache utilization (critical for OOM prevention)

**Notes**: vLLM must be started with `--metrics` flag to expose this endpoint.

### Qdrant gRPC Vector Store

**Endpoint**: `http://127.0.0.1:6333/metrics` (HTTP port)  
**gRPC Health**: `grpc://127.0.0.1:6334` (requires gRPC health check protocol)

**Key Metrics**:
- `qdrant_grpc_responses_total` (Counter): Total gRPC API responses
- `qdrant_grpc_responses_fail_total` (Counter): Failed gRPC responses
- `qdrant_grpc_responses_avg_duration_seconds` (Gauge): Average gRPC response duration
- `collections_vector_total` (Gauge): Total vectors stored in collections
- `cluster_pending_operations_total` (Gauge): Pending cluster operations (distributed deployments)
- `cluster_peers_total` (Gauge): Number of cluster peers (distributed deployments)

**Health Checks**:
- HTTP: `/healthz`, `/livez`, `/readyz` endpoints
- gRPC: `grpc.health.v1.Health/Check` method (SERVING status)

### NVIDIA GPU Subsystem

**Endpoint**: `http://127.0.0.1:9400/metrics` (via nvidia-ml-py exporter)  
**Tool**: `nvidia-smi` for health monitoring

**Key Metrics**:
- `gpu_utilization` (Gauge): GPU utilization percentage
- `memory_used` (Gauge): GPU VRAM used in bytes
- `memory_total` (Gauge): Total GPU VRAM in bytes
- `power_draw` (Gauge): GPU power consumption in watts
- `temperature_gpu` (Gauge): GPU temperature in Celsius

**Advanced Profiling**: CUPTI (CUDA Profiling Tools Interface) for hardware performance counters:
- `instruction_throughput`
- `memory_throughput`

**Setup**: Requires nvidia-ml-py exporter or node_exporter with `--collector.nvidia_gpu` flag.

### ONNX Runtime

**Profiling**: JSON performance traces (not Prometheus metrics)  
**Location**: Generated via `sess_options.enable_profiling = True`

**Key Metrics** (from JSON traces):
- Operator-level latency (`dur` field in trace file)
- Latency distribution of key operators within Qwen embedding model

**Note**: If ONNX Runtime metrics endpoint is exposed, it will be added to prometheus.yml.

## Scrape Configuration

All scrape configs are defined in `prometheus.yml`:

- **vLLM**: 10s scrape interval
- **Qdrant**: 10s scrape interval
- **NVIDIA GPU**: 15s scrape interval
- **NIODOO Pipeline**: 5s scrape interval (existing)

## Health Check Strategy

### Multi-Tiered Health Checks

1. **Basic Liveness**: HTTP endpoint availability (`up` metric)
2. **Service-Level**: Prometheus metrics endpoint reachability
3. **Functional**: gRPC health protocol (for Qdrant)
4. **Cognitive**: Quality SLIs (TCS stability, RCE β_meta compliance)

### Alert Configuration

Prometheus alerting rules are defined in `prometheus-alerts.yml` (see VAL-01-alerts task).

## Environment Variables

Service endpoints can be configured via environment variables:

- `VLLM_ENDPOINT`: vLLM HTTP endpoint (default: `http://127.0.0.1:5001`)
- `QDRANT_URL`: Qdrant HTTP endpoint (default: `http://127.0.0.1:6333`)
- `NIODOO_HEALTH_PORT`: NIODOO pipeline metrics port (default: `9090`)

## Validation

To verify metrics are being scraped:

```bash
# Check Prometheus targets
curl http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | select(.health == "up")'

# Query vLLM metrics
curl http://127.0.0.1:5001/metrics | grep -E "vllm_|llamacpp_"

# Query Qdrant metrics
curl http://127.0.0.1:6333/metrics | grep -E "qdrant_"

# Query NIODOO pipeline metrics
curl http://127.0.0.1:9090/metrics | grep -E "niodoo_"
```

