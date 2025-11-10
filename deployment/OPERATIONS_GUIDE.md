# NIODOO Pipeline - Production Deployment Guide

## Overview

This guide covers production deployment, monitoring, and operations for the NIODOO pipeline.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Kubernetes Deployment](#kubernetes-deployment)
3. [Helm Deployment](#helm-deployment)
4. [Health Checks](#health-checks)
5. [Monitoring & Observability](#monitoring--observability)
6. [Circuit Breakers](#circuit-breakers)
7. [Scaling](#scaling)
8. [Troubleshooting](#troubleshooting)

## Prerequisites

- Kubernetes cluster (1.24+)
- kubectl configured
- Helm 3.x installed
- Prometheus operator (optional, for monitoring)
- Grafana (optional, for dashboards)

## Kubernetes Deployment

### Basic Deployment

```bash
kubectl apply -f deployment/k8s/deployment.yaml
```

### Verify Deployment

```bash
kubectl get pods -l app=niodoo-pipeline
kubectl get svc niodoo-pipeline
```

### Check Logs

```bash
kubectl logs -l app=niodoo-pipeline --tail=100 -f
```

## Helm Deployment

### Install with Helm

```bash
helm install niodoo ./deployment/helm/niodoo \
  --set config.qdrant_url=http://qdrant:6333 \
  --set config.vllm_endpoint=http://vllm:8000
```

### Upgrade Deployment

```bash
helm upgrade niodoo ./deployment/helm/niodoo \
  --set image.tag=v1.2.0
```

### Uninstall

```bash
helm uninstall niodoo
```

## Health Checks

The pipeline exposes three HTTP endpoints:

- `/health` - Liveness probe (200 = healthy, 503 = unhealthy)
- `/ready` - Readiness probe (200 = ready to accept traffic)
- `/metrics` - Prometheus metrics

### Manual Health Check

```bash
curl http://localhost:8080/health
curl http://localhost:8080/ready
curl http://localhost:8080/metrics
```

## Monitoring & Observability

### Prometheus Metrics

Metrics are exposed at `/metrics` endpoint:

- `niodoo_pipeline_latency_seconds` - Request latency histogram
- `niodoo_pipeline_requests_total` - Total requests counter
- `niodoo_pipeline_errors_total` - Error counter
- `niodoo_cache_hits_total` - Cache hits
- `niodoo_cache_requests_total` - Cache requests
- `niodoo_tokenizer_promotions_total` - Token promotions
- `niodoo_circuit_breaker_state` - Circuit breaker state (0=closed, 1=open, 2=half-open)
- `niodoo_qdrant_latency_seconds` - Qdrant latency histogram
- `niodoo_vllm_latency_seconds` - vLLM latency histogram

### Grafana Dashboard

Import the dashboard from `deployment/monitoring/grafana-dashboard.yaml`:

1. Open Grafana UI
2. Go to Dashboards → Import
3. Upload `grafana-dashboard.yaml`
4. Select Prometheus as data source

### Prometheus Alerts

Apply alerts:

```bash
kubectl apply -f deployment/monitoring/prometheus-alerts.yaml
```

Key alerts:

- **HighErrorRate**: Error rate > 0.1/sec for 5 minutes
- **HighLatency**: 95th percentile latency > 5s for 5 minutes
- **CircuitBreakerOpen**: Circuit breaker open for 2 minutes
- **LowCacheHitRate**: Cache hit rate < 50% for 10 minutes
- **ServiceDown**: Service unavailable for 1 minute

## Circuit Breakers

Circuit breakers protect against cascading failures:

### Configuration

Circuit breakers are configured per service:

- **Qdrant**: Failure threshold: 5, Timeout: 60s
- **vLLM**: Failure threshold: 5, Timeout: 60s

### Manual Reset

Circuit breakers auto-recover, but can be manually reset via health registry:

```rust
health_registry.register_component("qdrant".to_string(), HealthStatus::Healthy, None).await;
```

## Scaling

### Horizontal Pod Autoscaling

HPA is configured by default:

```bash
kubectl get hpa niodoo-pipeline-hpa
```

HPA scales based on:
- CPU utilization (target: 70%)
- Memory utilization (target: 80%)

### Manual Scaling

```bash
kubectl scale deployment niodoo-pipeline --replicas=5
```

### Vertical Scaling

Update resource limits in `deployment.yaml`:

```yaml
resources:
  requests:
    memory: "4Gi"
    cpu: "2000m"
  limits:
    memory: "8Gi"
    cpu: "4000m"
```

## Troubleshooting

### Pod Not Starting

```bash
kubectl describe pod <pod-name>
kubectl logs <pod-name>
```

Common issues:
- Image pull errors → Check image registry credentials
- ConfigMap not found → Verify ConfigMap exists
- PVC not bound → Check storage class

### High Latency

1. Check metrics: `curl http://localhost:8080/metrics | grep latency`
2. Check circuit breaker status: `grep circuit_breaker_state metrics`
3. Check external service health:
   - Qdrant: `curl http://qdrant:6333/health`
   - vLLM: `curl http://vllm:8000/health`

### Circuit Breaker Open

1. Check service logs for errors
2. Verify external services are healthy
3. Wait for timeout period (60s) or manually reset

### Cache Issues

```bash
# Check cache hit rate
curl http://localhost:8080/metrics | grep cache_hit_rate

# Clear cache (restart pod)
kubectl delete pod <pod-name>
```

## Performance Tuning

### Cache Configuration

- `embedding_cache_ttl_secs`: Increase for stable workloads
- `collapse_cache_ttl_secs`: Tune based on ERAG update frequency
- `cache_capacity`: Increase for high-throughput workloads

### Concurrency

- Adjust `tokio` runtime worker threads via `TOKIO_WORKER_THREADS`
- Tune `max_concurrent_requests` in pipeline config

### Memory

- Monitor memory usage via Prometheus
- Adjust `cache_capacity` to balance memory vs hit rate
- Enable compression for large caches (LZ4)

## Security

### Network Policies

Restrict pod-to-pod communication:

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: niodoo-network-policy
spec:
  podSelector:
    matchLabels:
      app: niodoo-pipeline
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: prometheus
    ports:
    - protocol: TCP
      port: 8080
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: qdrant
    ports:
    - protocol: TCP
      port: 6333
```

### Secrets Management

Use Kubernetes secrets for sensitive config:

```bash
kubectl create secret generic niodoo-secrets \
  --from-literal=qdrant-api-key=xxx \
  --from-literal=vllm-api-key=yyy
```

Reference in deployment:

```yaml
env:
- name: QDRANT_API_KEY
  valueFrom:
    secretKeyRef:
      name: niodoo-secrets
      key: qdrant-api-key
```

## Backup & Recovery

### Backup Configuration

```bash
kubectl get configmap niodoo-config -o yaml > backup-config.yaml
```

### Backup Persistent Storage

```bash
# Create snapshot
kubectl exec -it <pod-name> -- tar czf /backup/niodoo-data-$(date +%Y%m%d).tar.gz /var/lib/niodoo
```

## Support

For issues or questions:
- Check logs: `kubectl logs -l app=niodoo-pipeline`
- Check metrics: `curl http://localhost:8080/metrics`
- Review CHANGELOG.md for recent changes
