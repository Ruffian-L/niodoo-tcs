#!/bin/bash
set -euo pipefail

# Run the Curator-Executor Learning Loop on your beelink server
# Make sure vLLM and Qdrant are running

# Load central runtime env if available
if [ -f "/workspace/Niodoo-Final/tcs_runtime.env" ]; then
    set -a
    . "/workspace/Niodoo-Final/tcs_runtime.env"
    set +a
fi

# Default endpoints align with supervisor (127.0.0.1:5001 / 127.0.0.1:6333)
VLLM_ENDPOINT="${VLLM_ENDPOINT:-http://127.0.0.1:5001}"
QDRANT_URL="${QDRANT_URL:-http://127.0.0.1:6333}"

echo "🚀 Starting Curator-Executor Persistent Learning Loop..."
echo "📍 vLLM server: ${VLLM_ENDPOINT}"
echo "📍 Qdrant server: ${QDRANT_URL}"
echo ""

# Check if vLLM is accessible
if ! curl -f -s --max-time 5 "${VLLM_ENDPOINT%/}/health" > /dev/null; then
    echo "❌ Cannot reach vLLM at ${VLLM_ENDPOINT}"
    echo "Make sure vLLM is running on your beelink server!"
    exit 1
fi

# Check if Qdrant is accessible
if ! curl -f -s --max-time 5 "${QDRANT_URL%/}/health" > /dev/null; then
    echo "❌ Cannot reach Qdrant at ${QDRANT_URL}"
    echo "Make sure Qdrant is running on your beelink server!"
    exit 1
fi

echo "✅ Services are running!"
echo ""

# Set environment variables
export RUST_LOG=${RUST_LOG:-info}
export VLLM_ENDPOINT
export QDRANT_URL

# Build if needed
if [ ! -f "target/debug/curator_executor" ]; then
    echo "Building curator_executor..."
    cargo build
fi

# Run the learning loop
echo "🧠 Starting learning loop..."
cargo run --bin curator_executor -- \
    --curator-model "Qwen/Qwen2.5-0.5B-Instruct" \
    --executor-model "Qwen/Qwen2.5-Coder-7B-Instruct" \
    --vllm-endpoint "${VLLM_ENDPOINT}" \
    --qdrant-url "${QDRANT_URL}" \
    --max-concurrent-tasks 4