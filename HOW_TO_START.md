# HOW TO START NIODOO - Complete End-to-End Guide

**This guide tells you EXACTLY how to start all services and run NIODOO end-to-end.**

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Service Startup Order](#service-startup-order)
3. [Step-by-Step Startup](#step-by-step-startup)
4. [Starting Main Application](#starting-main-application)
5. [Verification & Testing](#verification--testing)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

Before starting, ensure you have:

- **Rust 1.87+** installed (`rustup install 1.87 && rustup default 1.87`)
- **Docker** installed and running (`docker ps` should work)
- **Python 3.10+** with venv set up
- **vLLM** installed in Python environment
- **CUDA/GPU** available (for vLLM models)
- **Git submodules** initialized (`git submodule update --init --recursive`)

### Environment Setup

Source the runtime environment:

```bash
cd /workspace/Niodoo-Final
source tcs_runtime.env
```

Or for specific hardware profiles:

```bash
# For H200 GPU
source config/h200.env

# For RTX 5090 GPU
source config/rtx5090.env

# For A100 GPU
source config/a100.env
```

---

## Service Startup Order

**CRITICAL: Start services in this exact order:**

1. **Qdrant** (Vector Database) - Ports 6333 (HTTP), 6334 (gRPC)
2. **Qwen Embeddings** (ONNX Runtime) - Local, no service needed
3. **Qwen 3 Coder** (vLLM) - Port 5001 for generation
4. **Qwen 2.5 Topology** (vLLM Curator) - Port 5001 (same instance) or separate port
5. **Main Pipeline Server** - Port 9090 (health endpoints)

---

## Step-by-Step Startup

### Step 1: Start Qdrant (Vector Database)

Qdrant provides vector storage for ERAG memory. It uses **both HTTP (6333) and gRPC (6334)**.

```bash
# Check if Qdrant is already running
curl -s http://127.0.0.1:6333/collections > /dev/null 2>&1 && echo "✅ Qdrant already running" || {
    echo "🚀 Starting Qdrant..."
    
    # Start Qdrant via Docker
    docker run -d \
        --name qdrant \
        --restart unless-stopped \
        -p 6333:6333 \
        -p 6334:6334 \
        -v $(pwd)/qdrant_storage:/qdrant/storage \
        qdrant/qdrant:latest
    
    # Wait for Qdrant to be ready
    echo "⏳ Waiting for Qdrant to initialize..."
    for i in {1..30}; do
        sleep 2
        if curl -s http://127.0.0.1:6333/collections > /dev/null 2>&1; then
            echo "✅ Qdrant is ready!"
            break
        fi
        printf "."
    done
    echo ""
}

# Verify Qdrant is accessible
curl -s http://127.0.0.1:6333/collections | jq '.' || echo "⚠️  Qdrant HTTP endpoint check failed"
```

**Verification:**
```bash
# HTTP endpoint
curl http://127.0.0.1:6333/collections

# Should return JSON with collections (may be empty initially)
```

---

### Step 2: Qwen Embeddings (ONNX Runtime)

**NO SERVICE NEEDED** - Embeddings run locally via ONNX runtime.

The system uses `QwenStatefulEmbedder` which loads ONNX models directly. Just ensure:

1. ONNX runtime libraries are in `LD_LIBRARY_PATH` (set in `tcs_runtime.env`)
2. Model files exist at the configured path

**Verification:**
```bash
# Check ONNX runtime is available
ls -la third_party/onnxruntime-linux-x64-gpu-*/lib/libonnxruntime.so || echo "⚠️  ONNX runtime not found"

# Check embedding model path (if configured)
echo "Embedding model: ${EMBEDDING_MODEL_NAME:-not set}"
```

---

### Step 3: Start Qwen 3 Coder (vLLM - Generation Model)

Qwen 3 Coder is the main generation model, served via vLLM on **port 5001**.

```bash
# Set model path (adjust to your actual Qwen 3 Coder model location)
export VLLM_MODEL_ID="${VLLM_MODEL_ID:-/workspace/models/Qwen3-Coder}"
export VLLM_PORT=5001

# Check if vLLM is already running
curl -s http://127.0.0.1:${VLLM_PORT}/v1/models > /dev/null 2>&1 && echo "✅ vLLM already running" || {
    echo "🚀 Starting Qwen 3 Coder (vLLM) on port ${VLLM_PORT}..."
    
    # Activate Python environment with vLLM
    if [ -f venv/bin/activate ]; then
        source venv/bin/activate
    fi
    
    # Start vLLM server
    nohup python3 -m vllm.entrypoints.openai.api_server \
        --model "$VLLM_MODEL_ID" \
        --host 127.0.0.1 \
        --port ${VLLM_PORT} \
        --dtype bfloat16 \
        --gpu-memory-utilization 0.85 \
        --max-model-len 32768 \
        --max-num-batched-tokens 8192 \
        --max-num-seqs 64 \
        --trust-remote-code \
        > /tmp/vllm_coder.log 2>&1 &
    
    VLLM_PID=$!
    echo "vLLM started (PID: $VLLM_PID)"
    echo "Logs: tail -f /tmp/vllm_coder.log"
    
    # Wait for vLLM to load (this takes 2-5 minutes)
    echo "⏳ Waiting for Qwen 3 Coder to load (this may take 2-5 minutes)..."
    for i in {1..120}; do
        sleep 5
        if curl -s http://127.0.0.1:${VLLM_PORT}/v1/models > /dev/null 2>&1; then
            echo ""
            echo "✅✅✅ Qwen 3 Coder is READY! ✅✅✅"
            break
        fi
        if [ $((i % 10)) -eq 0 ]; then
            printf "\n   Still loading... ($i/120)\n"
        else
            printf "."
        fi
    done
    echo ""
}

# Verify vLLM is accessible
curl -s http://127.0.0.1:${VLLM_PORT}/v1/models | jq '.' || echo "⚠️  vLLM endpoint check failed"
```

**Verification:**
```bash
# Check vLLM models endpoint
curl http://127.0.0.1:5001/v1/models

# Should return JSON with model information
```

**Alternative: Using the provided script**

```bash
cd /workspace/Niodoo-Final/niodoo-ai
./scripts/start_vllm.sh /path/to/qwen3-coder-model 5001 0.85
```

---

### Step 4: Start Qwen 2.5 Topology (vLLM - Curator Model)

Qwen 2.5 Topology is the curator model. It can run on the **same vLLM instance (port 5001)** or a **separate port**.

**Option A: Same vLLM instance (port 5001)**

If using the same instance, the curator will use the same endpoint. Just ensure the model supports both generation and curation tasks.

**Option B: Separate vLLM instance (recommended)**

```bash
# Set curator model path
export CURATOR_MODEL="${CURATOR_MODEL:-/workspace/models/Qwen2.5-Topology}"
export CURATOR_VLLM_PORT=5002

# Check if curator vLLM is already running
curl -s http://127.0.0.1:${CURATOR_VLLM_PORT}/v1/models > /dev/null 2>&1 && echo "✅ Curator vLLM already running" || {
    echo "🚀 Starting Qwen 2.5 Topology Curator (vLLM) on port ${CURATOR_VLLM_PORT}..."
    
    # Activate Python environment
    if [ -f venv/bin/activate ]; then
        source venv/bin/activate
    fi
    
    # Start curator vLLM server
    nohup python3 -m vllm.entrypoints.openai.api_server \
        --model "$CURATOR_MODEL" \
        --host 127.0.0.1 \
        --port ${CURATOR_VLLM_PORT} \
        --dtype bfloat16 \
        --gpu-memory-utilization 0.15 \
        --max-model-len 2048 \
        --max-num-batched-tokens 4096 \
        --max-num-seqs 32 \
        --trust-remote-code \
        > /tmp/vllm_curator.log 2>&1 &
    
    CURATOR_PID=$!
    echo "Curator vLLM started (PID: $CURATOR_PID)"
    echo "Logs: tail -f /tmp/vllm_curator.log"
    
    # Wait for curator to load
    echo "⏳ Waiting for Qwen 2.5 Topology Curator to load..."
    for i in {1..60}; do
        sleep 5
        if curl -s http://127.0.0.1:${CURATOR_VLLM_PORT}/v1/models > /dev/null 2>&1; then
            echo ""
            echo "✅ Qwen 2.5 Topology Curator is READY!"
            break
        fi
        printf "."
    done
    echo ""
}

# Set curator endpoint
export CURATOR_VLLM_ENDPOINT="http://127.0.0.1:${CURATOR_VLLM_PORT}"
```

**Configuration:**

If using separate ports, set environment variables:

```bash
export VLLM_ENDPOINT="http://127.0.0.1:5001"  # Qwen 3 Coder
export CURATOR_VLLM_ENDPOINT="http://127.0.0.1:5002"  # Qwen 2.5 Topology
```

**Verification:**
```bash
# Check curator endpoint
curl http://127.0.0.1:5002/v1/models
```

---

### Step 5: Verify All Services Are Running

Before starting the main application, verify all services:

```bash
echo "🔍 Verifying all services..."

# Qdrant
echo -n "Qdrant (HTTP 6333): "
curl -s http://127.0.0.1:6333/collections > /dev/null && echo "✅" || echo "❌"

# Qdrant gRPC (6334) - check via HTTP health
echo -n "Qdrant (gRPC 6334): "
curl -s http://127.0.0.1:6333/health > /dev/null && echo "✅" || echo "❌"

# Qwen 3 Coder (vLLM)
echo -n "Qwen 3 Coder (vLLM 5001): "
curl -s http://127.0.0.1:5001/v1/models > /dev/null && echo "✅" || echo "❌"

# Qwen 2.5 Topology Curator (vLLM 5002, if separate)
if [ -n "${CURATOR_VLLM_PORT:-}" ]; then
    echo -n "Qwen 2.5 Topology Curator (vLLM ${CURATOR_VLLM_PORT}): "
    curl -s http://127.0.0.1:${CURATOR_VLLM_PORT}/v1/models > /dev/null && echo "✅" || echo "❌"
fi

echo ""
echo "✅ All services verified!"
```

---

## Starting Main Application

Once all services are running, start the main NIODOO pipeline server.

### Service Mode (with HTTP endpoints)

```bash
cd /workspace/Niodoo-Final

# Source environment
source tcs_runtime.env

# Build if needed
cargo build -p niodoo_real_integrated --release --features svc

# Start in service mode (no --prompt flag = starts health server)
cargo run -p niodoo_real_integrated --release --features svc > /tmp/niodoo_main.log 2>&1 &
MAIN_PID=$!
echo $MAIN_PID > /tmp/niodoo_main.pid

echo "Main pipeline server started (PID: $MAIN_PID)"
echo "Logs: tail -f /tmp/niodoo_main.log"
echo "Health endpoint: http://localhost:9090/health"
```

**Wait for server to start:**

```bash
# Wait for health endpoint to be ready
echo "⏳ Waiting for main server to start..."
for i in {1..60}; do
    sleep 2
    if curl -s http://localhost:9090/health > /dev/null 2>&1; then
        echo ""
        echo "✅ Main pipeline server is READY!"
        break
    fi
    printf "."
done
echo ""
```

### CLI Mode (single prompt)

```bash
cd /workspace/Niodoo-Final
source tcs_runtime.env

# Process a single prompt
cargo run -p niodoo_real_integrated --release -- --prompt "Your prompt here"
```

---

## Verification & Testing

### Health Endpoints

```bash
# Health check (liveness probe)
curl http://localhost:9090/health | jq '.'

# Readiness check
curl http://localhost:9090/ready | jq '.'

# Prometheus metrics
curl http://localhost:9090/metrics
```

### Test Pipeline End-to-End

```bash
# Using the provided test script
cd /workspace/Niodoo-Final
./scripts/start_and_test_all_endpoints.sh
```

Or manually test:

```bash
# Test health
curl http://localhost:9090/health

# Test readiness
curl http://localhost:9090/ready

# Test metrics
curl http://localhost:9090/metrics | head -20
```

### Process a Test Prompt

```bash
# Via CLI
cargo run -p niodoo_real_integrated --release -- --prompt "What is topological data analysis?"

# Or use the service endpoint (if implemented)
curl -X POST http://localhost:9090/v1/prompt \
    -H "Content-Type: application/json" \
    -d '{"prompt": "What is topological data analysis?"}'
```

---

## Troubleshooting

### Qdrant Not Starting

**Problem:** Qdrant Docker container fails to start

**Solutions:**
```bash
# Check if port is already in use
lsof -i :6333
lsof -i :6334

# Remove old container
docker rm -f qdrant

# Check Docker logs
docker logs qdrant

# Start with verbose logging
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest
docker logs -f qdrant
```

### vLLM Not Loading

**Problem:** vLLM takes too long or fails to load

**Solutions:**
```bash
# Check GPU memory
nvidia-smi

# Check vLLM logs
tail -f /tmp/vllm_coder.log
tail -f /tmp/vllm_curator.log

# Reduce GPU memory utilization
export VLLM_GPU_MEMORY_UTILIZATION=0.70  # Lower from 0.85

# Check if model path is correct
ls -la "$VLLM_MODEL_ID"

# Verify vLLM is installed
python3 -c "import vllm; print(vllm.__version__)"
```

### Main Application Fails to Start

**Problem:** Main pipeline server fails or health endpoint doesn't respond

**Solutions:**
```bash
# Check logs
tail -f /tmp/niodoo_main.log

# Check if port 9090 is available
lsof -i :9090

# Verify all services are running (run Step 5 verification)

# Check environment variables
env | grep -E "VLLM|QDRANT|CURATOR"

# Try rebuilding
cargo clean
cargo build -p niodoo_real_integrated --release --features svc
```

### Embeddings Not Working

**Problem:** ONNX runtime errors

**Solutions:**
```bash
# Check LD_LIBRARY_PATH
echo $LD_LIBRARY_PATH

# Verify ONNX runtime library exists
ls -la third_party/onnxruntime-linux-x64-gpu-*/lib/libonnxruntime.so

# Check embedding model path
echo $EMBEDDING_MODEL_NAME
ls -la "$EMBEDDING_MODEL_NAME" 2>/dev/null || echo "Model not found"
```

### Curator Not Responding

**Problem:** Curator endpoint not accessible

**Solutions:**
```bash
# Check if curator vLLM is running
curl http://127.0.0.1:5002/v1/models  # or 5001 if same instance

# Check curator logs
tail -f /tmp/vllm_curator.log

# Verify CURATOR_VLLM_ENDPOINT is set correctly
echo $CURATOR_VLLM_ENDPOINT

# Check config
grep -i curator niodoo_real_integrated/src/config.rs | head -20
```

---

## Quick Start Script

For convenience, use the automated startup script:

```bash
cd /workspace/Niodoo-Final
./start_all_services.sh

# Then start main application
cargo run -p niodoo_real_integrated --release --features svc
```

---

## Service Ports Summary

| Service | Port | Protocol | Purpose |
|---------|------|----------|---------|
| Qdrant HTTP | 6333 | HTTP | Vector DB HTTP API |
| Qdrant gRPC | 6334 | gRPC | Vector DB gRPC API |
| Qwen 3 Coder | 5001 | HTTP | Main generation model |
| Qwen 2.5 Topology Curator | 5001 or 5002 | HTTP | Curator model |
| Main Pipeline Health | 9090 | HTTP | Health/ready/metrics endpoints |

---

## Environment Variables Reference

Key environment variables (set in `tcs_runtime.env` or hardware-specific configs):

```bash
# Service endpoints
VLLM_ENDPOINT=http://127.0.0.1:5001
CURATOR_VLLM_ENDPOINT=http://127.0.0.1:5001  # or 5002 if separate
QDRANT_URL=http://127.0.0.1:6333

# Model paths
VLLM_MODEL_ID=/workspace/models/Qwen3-Coder
CURATOR_MODEL=/workspace/models/Qwen2.5-Topology
EMBEDDING_MODEL_NAME=/path/to/qwen/embedding/onnx

# Qdrant config
QDRANT_COLLECTION=experiences
QDRANT_USE_GRPC=false  # Set to true to use gRPC (port 6334)

# ONNX runtime
LD_LIBRARY_PATH=/workspace/Niodoo-Final/third_party/onnxruntime-linux-x64-gpu-1.24.0/lib:${LD_LIBRARY_PATH}
```

---

## Next Steps

Once everything is running:

1. **Read AI_SETUP_GUIDE.md** for understanding the codebase architecture
2. **Check SYSTEM_ARCHITECTURE.md** for component details
3. **Review RUNTIME_FLOW.md** for pipeline execution flow
4. **Run validation tests** using `scripts/start_and_test_all_endpoints.sh`

---

## Need Help?

- Check logs: `/tmp/vllm_coder.log`, `/tmp/vllm_curator.log`, `/tmp/niodoo_main.log`
- Verify services: Run Step 5 verification commands
- Review documentation: `AI_SETUP_GUIDE.md`, `SYSTEM_ARCHITECTURE.md`
- Check CHANGELOG.md for recent changes

---

**Last Updated:** 2025-01-XX  
**Maintained by:** NIODOO Development Team






