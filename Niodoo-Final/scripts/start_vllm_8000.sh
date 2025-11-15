#!/bin/bash
# Start vLLM service on port 8000 with Qwen2.5-7B-Instruct-AWQ
# GPU memory utilization set to 0.6 (60%) to prevent OOM errors in niodoo_real_integrated pipeline
# Using AWQ quantization for better performance and memory efficiency
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

# Source environment
if [ -f "$ROOT_DIR/niodoo_real_integrated.env" ]; then
    source "$ROOT_DIR/niodoo_real_integrated.env"
fi

# Python with vLLM
PYTHON_CMD="$ROOT_DIR/venv/bin/python3"
if [ ! -f "$PYTHON_CMD" ] || ! "$PYTHON_CMD" -c "import vllm" 2>/dev/null; then
    echo "ERROR: vLLM not found in venv"
    exit 1
fi

VLLM_PORT=8000
VLLM_LOG="/tmp/vllm_8000.log"
# Use 7B AWQ model for better structured JSON generation in TopoCoT
VLLM_MODEL="${VLLM_MODEL_ID:-/home/beelink/niodoo-tcs/models/Qwen2.5-7B-Instruct-AWQ}"

# Check if vLLM is already running
if curl -s http://127.0.0.1:$VLLM_PORT/v1/models > /dev/null 2>&1; then
    echo "✅ vLLM already running on port $VLLM_PORT"
    exit 0
fi

# Kill any existing vLLM on this port
pkill -9 -f "vllm.*$VLLM_PORT" || true
sleep 2

echo "Starting vLLM on port $VLLM_PORT with GPU memory utilization 0.95 (GPU-optimized)..."
echo "Model: $VLLM_MODEL"
echo "Log: $VLLM_LOG"

cd "$ROOT_DIR"
# Verify model path exists
if [ ! -d "$VLLM_MODEL" ]; then
    echo "ERROR: Model not found at $VLLM_MODEL"
    exit 1
fi

# Ensure CUDA is visible
export CUDA_VISIBLE_DEVICES=0

echo "Starting vLLM with AWQ quantization for 7B model (GPU-optimized settings)..."
nohup "$PYTHON_CMD" -m vllm.entrypoints.openai.api_server \
    --model "$VLLM_MODEL" \
    --host 127.0.0.1 \
    --port "$VLLM_PORT" \
    --quantization awq \
    --dtype auto \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.95 \
    --trust-remote-code \
    --max-model-len 8192 \
    > "$VLLM_LOG" 2>&1 &

VLLM_PID=$!
echo "vLLM started (PID: $VLLM_PID)"
echo "Waiting for vLLM to be ready (this may take 1-3 minutes)..."

# Wait for vLLM to be ready
for i in {1..120}; do
    if curl -s http://127.0.0.1:$VLLM_PORT/v1/models > /dev/null 2>&1; then
        echo "✅ vLLM is ready!"
        echo "Endpoint: http://127.0.0.1:$VLLM_PORT"
        exit 0
    fi
    if [ $((i % 10)) -eq 0 ]; then
        echo "Still loading... ($i/120)"
    fi
    sleep 2
done

echo "⚠️  vLLM may not be ready yet, check $VLLM_LOG"
exit 1

