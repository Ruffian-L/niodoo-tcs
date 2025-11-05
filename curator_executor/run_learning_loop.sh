#!/bin/bash
set -euo pipefail

# Run the Curator-Executor Learning Loop on your beelink server
# Make sure vLLM is running at http://beelink:8000
# Make sure Qdrant is running at http://beelink:6333

echo "🚀 Starting Curator-Executor Persistent Learning Loop..."
echo "📍 vLLM server: http://beelink:8000"
echo "📍 Qdrant server: http://beelink:6333"
echo ""

# Check if vLLM is accessible
if ! curl -f -s --max-time 5 http://beelink:8000/health > /dev/null; then
    echo "❌ Cannot reach vLLM at http://beelink:8000"
    echo "Make sure vLLM is running on your beelink server!"
    exit 1
fi

# Check if Qdrant is accessible
if ! curl -f -s --max-time 5 http://beelink:6333/health > /dev/null; then
    echo "❌ Cannot reach Qdrant at http://beelink:6333"
    echo "Make sure Qdrant is running on your beelink server!"
    exit 1
fi

echo "✅ Services are running!"
echo ""

# Set environment variables
export RUST_LOG=info
export VLLM_ENDPOINT=http://beelink:8000
export QDRANT_URL=http://beelink:6333

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
    --vllm-endpoint "http://beelink:8000" \
    --qdrant-url "http://beelink:6333" \
    --max-concurrent-tasks 4