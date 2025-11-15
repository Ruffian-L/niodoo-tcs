#!/bin/bash
# Run NiDoo Euler test harness with OOM mitigations and vLLM checks
# Usage: ./run_test_harness.sh [niodoo_path] [extra_args]

set -e

NIODOO_PATH="${1:-/home/beelink/niodoo-tcs/Niodoo-Final/niodoo_real_integrated}"
EXTRA_ARGS="${@:2}"
ENV_FILE="${NIODOO_PATH}/../niodoo_real_integrated.env"

echo "🔍 Pre-flight checks..."

# Check vLLM
if ! curl -s --max-time 2 "http://127.0.0.1:8000/v1/models" > /dev/null 2>&1; then
    echo "⚠️  vLLM not responding on port 8000"
    echo "   Attempting to start vLLM..."
    if [ -f "${NIODOO_PATH}/../scripts/start_vllm_8000.sh" ]; then
        bash "${NIODOO_PATH}/../scripts/start_vllm_8000.sh" || true
        sleep 10
    fi
fi

# Verify vLLM model
MODEL_ID=$(curl -s "http://127.0.0.1:8000/v1/models" 2>/dev/null | python3 -c "import sys, json; d=json.load(sys.stdin); print(d['data'][0]['id'] if d.get('data') else '')" 2>/dev/null || echo "")
if [ -z "$MODEL_ID" ]; then
    echo "❌ vLLM model not available"
    exit 1
fi
echo "✅ vLLM model: $MODEL_ID"

# Source environment
if [ -f "$ENV_FILE" ]; then
    source "$ENV_FILE"
    echo "✅ Loaded environment from $ENV_FILE"
else
    echo "⚠️  Environment file not found: $ENV_FILE"
fi

# Set OOM mitigations (GPU-optimized: removed ORT_DISABLE_ALL to enable CUDA)
export QWEN_CHUNK_TOKENS=512
export NIODOO_SKIP_SMOKE=1
# Enable GPU execution providers for ONNX
export ONNXRUNTIME_EXECUTION_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider
export CUDA_VISIBLE_DEVICES=0

echo ""
echo "🛡️  Configuration Active:"
echo "   QWEN_CHUNK_TOKENS=$QWEN_CHUNK_TOKENS"
echo "   ONNXRUNTIME_EXECUTION_PROVIDERS=$ONNXRUNTIME_EXECUTION_PROVIDERS"
echo "   CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "   NIODOO_SKIP_SMOKE=$NIODOO_SKIP_SMOKE"
echo ""

# Check memory
FREE_MB=$(free -m | awk 'NR==2{print $7}')
echo "💾 Available RAM: ${FREE_MB} MB"
if [ "$FREE_MB" -lt 2048 ]; then
    echo "⚠️  Low memory warning - consider closing other applications"
fi

# Run test
echo ""
echo "🚀 Running Euler test harness..."
echo "   Path: $NIODOO_PATH"
echo "   Args: $EXTRA_ARGS"
echo ""

cd "$NIODOO_PATH" || exit 1

# Run with memory monitoring in background if htop available
if command -v htop > /dev/null 2>&1; then
    echo "   (Run 'htop' in another terminal to monitor memory)"
fi

cargo run --release --bin euler_test -- $EXTRA_ARGS

echo ""
echo "✅ Test harness completed"

