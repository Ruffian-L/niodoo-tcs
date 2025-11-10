# NIODOO RunPod Deployment Guide
## Stateful Pod Architecture - Solution Path 1

This guide implements the **Stateful Pod Architecture** to fix all startup issues identified in the RunPod endpoint investigation.

**Updated**: All model references now use Qwen3-Coder. Service manager (`unified_service_manager.sh`) handles proper startup orchestration.

## Prerequisites

1. **RunPod Account** with GPU pod access
2. **Network Volume** created in RunPod UI (for Qdrant persistence)
3. **HuggingFace Token** (if using private models)
4. **Docker Image** built and pushed to a registry

## Phase 1: Build and Push Docker Image

### Option A: Build Locally

```bash
# Set your HuggingFace token (if needed)
export HF_TOKEN="your_hf_token_here"

# Build the image
./scripts/build_runpod_image.sh

# Tag for your registry
docker tag niodoo-runpod:latest your-registry/niodoo-runpod:latest

# Push to registry
docker push your-registry/niodoo-runpod:latest
```

### Option B: Build on RunPod

```bash
# SSH into a RunPod pod with Docker
runpodctl ssh <pod_id>

# Clone repository
git clone <your-repo-url>
cd Niodoo-Final

# Build image
./scripts/build_runpod_image.sh

# Save image as tar
docker save niodoo-runpod:latest -o niodoo-runpod.tar

# Upload to RunPod volume or registry
```

## Phase 2: Create Network Volume (CRITICAL - Fixes Failure Pattern 2)

1. Go to RunPod Dashboard → **Network Volumes**
2. Click **Create Volume**
3. Select your data center region
4. Set size (minimum 50GB recommended for models + Qdrant data)
5. Note the Volume ID

**This volume persists independently of pod lifecycle and prevents Qdrant data loss.**

## Phase 3: Configure Secrets (Fixes Failure Pattern 5)

1. Go to RunPod Dashboard → **Secrets**
2. Create the following secrets:

   - **VLLM_API_KEY**: API key for vLLM authentication (optional but recommended)
   - **QDRANT_API_KEY**: API key for Qdrant authentication (optional but recommended)
   - **HF_TOKEN**: HuggingFace token for model access (if using private models)

**Note**: The entrypoint script automatically bridges these env vars to file-based secrets.

## Phase 4: Deploy Pod

### Using RunPod UI

1. Go to **Pods** → **Deploy Pod**
2. Select GPU template (24GB+ VRAM recommended)
3. **Container Image**: `your-registry/niodoo-runpod:latest`
4. **Container Disk**: 20GB (for logs and temp files)
5. **Network Volume**: Attach the volume created in Phase 2
   - **Mount Path**: `/data`
6. **Environment Variables**:
   ```
   VLLM_MODEL_PATH=/models/Qwen3-Coder
   VLLM_ENDPOINT=http://localhost:5001
   QDRANT_URL=http://localhost:6333
   QDRANT_STORAGE_PATH=/data/qdrant_storage
   NIODOO_RUNPOD_MODE=true
   RUST_LOG=info
   ```
7. **Ports** (expose for monitoring):
   - `5001` → vLLM API
   - `6333` → Qdrant HTTP
   - `6334` → Qdrant gRPC
   - `9090` → NIODOO health/metrics
8. **Secrets**: Attach secrets created in Phase 3
9. Deploy

### Using RunPod API/CLI

```bash
runpodctl pod create \
  --name niodoo-stateful \
  --image your-registry/niodoo-runpod:latest \
  --gpu-type RTX4090 \
  --container-disk-size 20 \
  --network-volume <volume-id>:/data \
  --env "VLLM_MODEL_PATH=/models/Qwen3-Coder" \
  --env "VLLM_ENDPOINT=http://localhost:5001" \
  --env "QDRANT_URL=http://localhost:6333" \
  --env "QDRANT_STORAGE_PATH=/data/qdrant_storage" \
  --env "NIODOO_RUNPOD_MODE=true" \
  --env "RUST_LOG=info" \
  --secret VLLM_API_KEY \
  --secret QDRANT_API_KEY \
  --secret HF_TOKEN \
  --ports "5001,6333,6334,9090"
```

## Phase 5: Verify Deployment

### Check Logs

```bash
# Get pod logs
runpodctl logs <pod_id>

# Or SSH into pod
runpodctl ssh <pod_id>

# Check supervisord logs
tail -f /app/logs/supervisord.log

# Check individual service logs
tail -f /app/logs/qdrant.log
tail -f /app/logs/vllm.log
tail -f /app/logs/niodoo_service.log
```

### Verify Services

```bash
# SSH into pod
runpodctl ssh <pod_id>

# Check Qdrant
curl http://localhost:6333/health

# Check vLLM
curl http://localhost:5001/v1/models

# Check NIODOO service
curl http://localhost:9090/health
curl http://localhost:9090/metrics
```

### Expected Startup Sequence

1. **Qdrant** starts first (5-10 seconds)
2. **vLLM** waits for Qdrant, then starts (2-5 minutes for cold start, model loading)
3. **Curator vLLM** starts (optional, 1-3 minutes if separate instance)
4. **NIODOO Service** waits for Qdrant + vLLM, then starts (10-20 seconds)

Total startup time: **2-3 minutes** (first boot), **30-60 seconds** (subsequent boots with cached CUDA graphs)

**Note**: The `unified_service_manager.sh` script handles proper startup order and dependency waiting automatically.

## Phase 6: Optimize Performance (Week 2)

### Fix vLLM Cold Start (Failure Pattern 3)

The Docker image already bakes model weights (saves 8 seconds). To optimize further:

1. **Cache CUDA Graphs** (recommended):
   - CUDA graphs are cached in `/data` (Network Volume)
   - First boot: ~14 seconds for graph capture
   - Subsequent boots: graphs loaded from cache

2. **Disable CUDA Graphs** (faster startup, slower inference):
   - Edit `deployment/runpod/supervisord.conf`
   - Add `--enforce-eager` to vLLM command
   - Saves 14 seconds but reduces inference throughput

### Fix VRAM Contention (Failure Pattern 4)

The supervisord config already pins services correctly:
- **vLLM**: GPU (all VRAM)
- **NIODOO embedding**: CPU (no VRAM)
- **Qdrant**: CPU (no VRAM)

Verify with:
```bash
nvidia-smi
# Should show only vLLM using GPU memory
```

## Troubleshooting

### Service Won't Start

1. **Check logs**: `runpodctl logs <pod_id>`
2. **Verify Network Volume**: Ensure `/data` is mounted
3. **Check GPU**: `nvidia-smi` inside pod
4. **Verify model path**: `ls -la /models/Qwen3-Coder`

### Qdrant Data Lost

- **Cause**: Not using Network Volume
- **Fix**: Ensure Network Volume is attached at `/data`
- **Verify**: `ls -la /data/qdrant_storage` should persist across pod restarts

### vLLM Timeout

- **Cause**: Cold start taking longer than health check timeout
- **Fix**: Increase timeout in `supervisord.conf` (currently 120 seconds)
- **Or**: Use `--enforce-eager` flag to skip CUDA graph capture

### Authentication Errors

- **Cause**: Secrets not bridged correctly
- **Fix**: Check `/run/secrets/vllm_api_key` and `/run/secrets/qdrant_api_key` exist
- **Verify**: `cat /run/secrets/vllm_api_key` should show your key

### Port Conflicts

- **Cause**: Ports already in use
- **Fix**: Change ports in `supervisord.conf` and update environment variables

## Monitoring

### Prometheus Metrics

Expose port `9090` and scrape:
```
http://<pod-ip>:9090/metrics
```

Key metrics:
- `niodoo_pipeline_latency_seconds`
- `niodoo_erag_cache_hit_rate`
- `vllm_gpu_cache_usage_perc` (if exposed)

### Log Aggregation

All logs are centralized in `/app/logs/`:
- `supervisord.log` - Process manager logs
- `qdrant.log` - Qdrant service logs
- `vllm.log` - vLLM service logs
- `niodoo_service.log` - NIODOO Rust service logs

## Success Criteria

✅ **All 5 failure patterns fixed**:
1. ✅ Service discovery uses localhost (no DNS issues)
2. ✅ Qdrant data persists on Network Volume
3. ✅ Startup order managed by supervisord (no race conditions)
4. ✅ VRAM contention eliminated (CPU/GPU pinning)
5. ✅ Secrets bridged correctly (env vars → files)

✅ **Pod starts reliably** in 2-3 minutes (first boot)
✅ **Services remain healthy** across pod restarts
✅ **Data persists** when pod is deleted/recreated

## Next Steps

- **Phase 4**: Implement Prometheus/Grafana dashboards
- **Phase 5**: Set up automated health checks and alerts
- **Future**: Consider Solution Path 2 (Decoupled Serverless) for scale



