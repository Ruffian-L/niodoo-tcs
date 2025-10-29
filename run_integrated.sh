
#!/bin/bash
# NIODOO-TCS INTEGRATED PIPELINE RUNNER
# Sets all env vars and runs the full pipeline

cd /home/beelink/Niodoo-Final

# Load environment from central runtime file if present
if [ -f "tcs_runtime.env" ]; then
    set -a
    . "tcs_runtime.env"
    set +a
fi

# Base model/library paths
export QWEN_MODEL_PATH=/home/beelink/models/Qwen2.5-7B-Instruct-AWQ
export LD_LIBRARY_PATH=/home/beelink/Niodoo-Final/onnxruntime-linux-x64-1.16.3/lib:$LD_LIBRARY_PATH

# Standardize endpoints to supervisor configuration (vLLM:5001, Qdrant:6333)
export VLLM_ENDPOINT="${VLLM_ENDPOINT:-http://127.0.0.1:5001}"
export VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
export VLLM_PORT="${VLLM_PORT:-5001}"
export QDRANT_URL="${QDRANT_URL:-http://127.0.0.1:6333}"
export QDRANT_COLLECTION="${QDRANT_COLLECTION:-experiences}"
export QDRANT_VECTOR_SIZE="${QDRANT_VECTOR_SIZE:-896}"
export RUST_LOG="${RUST_LOG:-info}"

# Run the binary
./target/release/niodoo_real_integrated "$@"

