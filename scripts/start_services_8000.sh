#!/bin/bash
# Start Qdrant and vLLM on port 8000 with Granite and Topological Qwen Curator models
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

# Source environment
if [ -f "$ROOT_DIR/tcs_runtime.env" ]; then
    source "$ROOT_DIR/tcs_runtime.env"
fi

# Model paths
GRANITE_MODEL="/workspace/.cache/huggingface/hub/models--ibm-granite--granite-3b-code-instruct/snapshots/7bac3cddc929b4a80e1e3136a5db7a3f21ac431e"
QWEN_CURATOR_MODEL="/workspace/Niodoo-AI/outputs/qwen25-coder-topology-20251105/merged"

# Verify models exist
if [ ! -d "$GRANITE_MODEL" ]; then
    echo "ERROR: Granite model not found at $GRANITE_MODEL"
    exit 1
fi

if [ ! -d "$QWEN_CURATOR_MODEL" ]; then
    echo "ERROR: Topological Qwen Curator model not found at $QWEN_CURATOR_MODEL"
    exit 1
fi

echo "✅ Models verified:"
echo "   Granite: $GRANITE_MODEL"
echo "   Qwen Curator: $QWEN_CURATOR_MODEL"

# Python with vLLM
PYTHON_CMD="/workspace/Niodoo-Final/venv/bin/python3"
if [ ! -f "$PYTHON_CMD" ] || ! "$PYTHON_CMD" -c "import vllm" 2>/dev/null; then
    echo "ERROR: vLLM not found in venv"
    exit 1
fi

# Start Qdrant
echo ""
echo "1️⃣ Starting Qdrant on port 6333..."
QDRANT_BIN="$ROOT_DIR/third_party/qdrant/qdrant"
QDRANT_STORAGE="$ROOT_DIR/qdrant_storage"
QDRANT_CONFIG="$ROOT_DIR/qdrant_config.yaml"

# Create qdrant config if it doesn't exist
if [ ! -f "$QDRANT_CONFIG" ]; then
    cat > "$QDRANT_CONFIG" <<EOF
log_level: INFO
storage:
  storage_path: $QDRANT_STORAGE
service:
  http_port: 6333
  grpc_port: 6334
EOF
fi

# Check if Qdrant is already running
if curl -s http://127.0.0.1:6333/health > /dev/null 2>&1; then
    echo "   ✅ Qdrant already running"
else
    echo "   Starting Qdrant..."
    mkdir -p "$QDRANT_STORAGE"
    nohup "$QDRANT_BIN" --config-path "$QDRANT_CONFIG" > /tmp/qdrant.log 2>&1 &
    QDRANT_PID=$!
    echo "   Qdrant started (PID: $QDRANT_PID)"
    
    # Wait for Qdrant to be ready
    echo "   Waiting for Qdrant to be ready..."
    for i in {1..30}; do
        if curl -s http://127.0.0.1:6333/health > /dev/null 2>&1; then
            echo "   ✅ Qdrant is ready!"
            break
        fi
        sleep 1
    done
    
    if ! curl -s http://127.0.0.1:6333/health > /dev/null 2>&1; then
        echo "   ⚠️  Qdrant may not be ready yet, check /tmp/qdrant.log"
    fi
fi

# Start vLLM on port 8002 (internal) with Granite model
echo ""
echo "2️⃣ Starting vLLM on port 8002 (internal) with Granite model..."
VLLM_GRANITE_PORT=8002
VLLM_GRANITE_LOG="/tmp/vllm_granite_8002.log"

# Check if vLLM granite is already running
if curl -s http://127.0.0.1:$VLLM_GRANITE_PORT/v1/models > /dev/null 2>&1; then
    echo "   ✅ vLLM Granite already running on port $VLLM_GRANITE_PORT"
else
    # Kill any existing vLLM on this port
    pkill -9 -f "vllm.*$VLLM_GRANITE_PORT" || true
    sleep 2
    
    echo "   Starting vLLM with Granite model..."
    cd "$ROOT_DIR"
    
    # vLLM configuration for Granite (max_model_len must match model's max_position_embeddings)
    export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
    VLLM_GRANITE_ARGS=(
        -m vllm.entrypoints.openai.api_server
        --model "$GRANITE_MODEL"
        --host 127.0.0.1
        --port "$VLLM_GRANITE_PORT"
        --dtype bfloat16
        --gpu-memory-utilization 0.3
        --max-model-len 2048
        --trust-remote-code
    )
    
    nohup "$PYTHON_CMD" "${VLLM_GRANITE_ARGS[@]}" > "$VLLM_GRANITE_LOG" 2>&1 &
    VLLM_GRANITE_PID=$!
    echo "   vLLM Granite started (PID: $VLLM_GRANITE_PID)"
    echo "   Log: $VLLM_GRANITE_LOG"
    
    # Wait for vLLM granite to be ready
    echo "   Waiting for vLLM Granite to be ready (this may take 2-5 minutes)..."
    for i in {1..120}; do
        if curl -s http://127.0.0.1:$VLLM_GRANITE_PORT/v1/models > /dev/null 2>&1; then
            echo "   ✅ vLLM Granite is ready!"
            break
        fi
        if [ $((i % 10)) -eq 0 ]; then
            echo "   Still loading... ($i/120)"
        fi
        sleep 2
    done
fi

# Start second vLLM instance for Qwen Curator on port 8003 (internal)
echo ""
echo "3️⃣ Starting vLLM on port 8003 (internal) with Topological Qwen Curator model..."
VLLM_CURATOR_PORT=8003
VLLM_CURATOR_LOG="/tmp/vllm_curator_8003.log"

# Check if vLLM curator is already running
if curl -s http://127.0.0.1:$VLLM_CURATOR_PORT/v1/models > /dev/null 2>&1; then
    echo "   ✅ vLLM Curator already running on port $VLLM_CURATOR_PORT"
else
    # Kill any existing vLLM on this port
    pkill -9 -f "vllm.*$VLLM_CURATOR_PORT" || true
    sleep 2
    
    echo "   Starting vLLM with Topological Qwen Curator model..."
    
    VLLM_CURATOR_ARGS=(
        -m vllm.entrypoints.openai.api_server
        --model "$QWEN_CURATOR_MODEL"
        --host 127.0.0.1
        --port "$VLLM_CURATOR_PORT"
        --dtype bfloat16
        --gpu-memory-utilization 0.3
        --max-model-len 4096
        --trust-remote-code
    )
    
    nohup "$PYTHON_CMD" "${VLLM_CURATOR_ARGS[@]}" > "$VLLM_CURATOR_LOG" 2>&1 &
    VLLM_CURATOR_PID=$!
    echo "   vLLM Curator started (PID: $VLLM_CURATOR_PID)"
    echo "   Log: $VLLM_CURATOR_LOG"
    
    # Wait for vLLM curator to be ready
    echo "   Waiting for vLLM Curator to be ready (this may take 2-5 minutes)..."
    for i in {1..120}; do
        if curl -s http://127.0.0.1:$VLLM_CURATOR_PORT/v1/models > /dev/null 2>&1; then
            echo "   ✅ vLLM Curator is ready!"
            break
        fi
        if [ $((i % 10)) -eq 0 ]; then
            echo "   Still loading... ($i/120)"
        fi
        sleep 2
    done
fi

# Start proxy on port 8000 to route requests
echo ""
echo "4️⃣ Starting vLLM Proxy on port 8000 (public) to route to both models..."
PROXY_PORT=8000
PROXY_LOG="/tmp/vllm_proxy_8000.log"

# Check if proxy is already running
if curl -s http://127.0.0.1:$PROXY_PORT/v1/models > /dev/null 2>&1; then
    echo "   ✅ vLLM Proxy already running on port $PROXY_PORT"
else
    # Kill any existing proxy on this port
    pkill -9 -f "vllm_proxy.*$PROXY_PORT" || true
    sleep 1
    
    echo "   Starting vLLM Proxy..."
    cd "$ROOT_DIR"
    
    nohup "$PYTHON_CMD" "$ROOT_DIR/scripts/vllm_proxy.py" > "$PROXY_LOG" 2>&1 &
    PROXY_PID=$!
    echo "   vLLM Proxy started (PID: $PROXY_PID)"
    echo "   Log: $PROXY_LOG"
    
    # Wait for proxy to be ready
    echo "   Waiting for proxy to be ready..."
    sleep 2
    for i in {1..10}; do
        if curl -s http://127.0.0.1:$PROXY_PORT/v1/models > /dev/null 2>&1; then
            echo "   ✅ vLLM Proxy is ready!"
            break
        fi
        sleep 1
    done
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ ALL SERVICES STARTED"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Services:"
echo "  Qdrant:        http://127.0.0.1:6333"
echo "  vLLM Proxy:    http://127.0.0.1:8000 (routes to both models)"
echo "  vLLM Granite:  http://127.0.0.1:8002 (internal)"
echo "  vLLM Curator:  http://127.0.0.1:8003 (internal)"
echo ""
echo "Logs:"
echo "  Qdrant:        /tmp/qdrant.log"
echo "  vLLM Granite:  $VLLM_GRANITE_LOG"
echo "  vLLM Curator:  $VLLM_CURATOR_LOG"
echo "  vLLM Proxy:    $PROXY_LOG"

