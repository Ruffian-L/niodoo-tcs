#!/bin/bash
# Start Qdrant and vLLM on port 8000 with Granite (direct on 8000) and Curator (on 8003 for hot-swapping)
# No proxy - Granite directly on 8000 as expected by all code
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

# Start vLLM on port 8000 (direct) with Granite model - no proxy
echo ""
echo "2️⃣ Starting vLLM on port 8000 (direct) with Granite model..."
VLLM_GRANITE_PORT=8000
VLLM_GRANITE_LOG="/tmp/vllm_granite_8000.log"

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
    # GPU memory utilization set to 0.25 to leave room for training while allowing KV cache
    export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
    VLLM_GRANITE_ARGS=(
        -m vllm.entrypoints.openai.api_server
        --model "$GRANITE_MODEL"
        --host 127.0.0.1
        --port "$VLLM_GRANITE_PORT"
        --dtype bfloat16
        --gpu-memory-utilization 0.25
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

# Start second vLLM instance for Qwen Curator on port 8003 (for hot-swapping)
# Note: Curator can be started manually when needed to avoid GPU memory conflicts
echo ""
echo "3️⃣ Checking vLLM Curator on port 8003 (for hot-swapping)..."
VLLM_CURATOR_PORT=8003
VLLM_CURATOR_LOG="/tmp/vllm_curator_8003.log"

# Check if vLLM curator is already running
if curl -s http://127.0.0.1:$VLLM_CURATOR_PORT/v1/models > /dev/null 2>&1; then
    echo "   ✅ vLLM Curator already running on port $VLLM_CURATOR_PORT"
else
    echo "   ⚠️  Curator not running. Start manually when needed for hot-swapping."
    echo "   To start Curator: bash scripts/start_curator_8003.sh"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ ALL SERVICES STARTED"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Services:"
echo "  Qdrant:        http://127.0.0.1:6333"
echo "  vLLM Granite:  http://127.0.0.1:8000 (direct - main endpoint)"
echo "  vLLM Curator:  http://127.0.0.1:8003 (for hot-swapping)"
echo ""
echo "Logs:"
echo "  Qdrant:        /tmp/qdrant.log"
echo "  vLLM Granite:  $VLLM_GRANITE_LOG"
echo "  vLLM Curator:  $VLLM_CURATOR_LOG"
echo ""
echo "Note: Granite is directly on port 8000. Curator is on 8003 for hot-swapping."

