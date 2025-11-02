#!/bin/bash
# H200 Bootstrapper
#
# Prepares the NIODOO stack to take advantage of an NVIDIA H200 for intensive runs.
# - Validates CUDA availability
# - Exports high-throughput runtime overrides to config/h200.env
# - Wires in CUDA-enabled ONNX Runtime libraries if present
# - Builds the workspace with GPU features enabled

set -euo pipefail

ROOT_DIR="/workspace/Niodoo-Final"
cd "${ROOT_DIR}"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 NIODOO H200 Bootstrap"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found. Install CUDA drivers before running this script." >&2
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 | tr -d '\r')
echo "✓ Detected GPU: ${GPU_NAME}"

if ! echo "${GPU_NAME}" | grep -qi "H200"; then
    echo "⚠️  GPU name does not mention H200. Continuing anyway." >&2
fi

echo ""
echo "🔧 Resolving CUDA-enabled ONNX Runtime libraries"
GPU_LIB_CANDIDATES=(
    "/workspace/Niodoo-Final/third_party/onnxruntime-linux-x64-gpu-1.24.0/lib"
    "/workspace/onnxruntime-linux-x64-gpu-1.24.0/lib"
    "/workspace/Niodoo-Final/third_party/onnxruntime-linux-x64-gpu-1.18.1/lib"
    "/workspace/onnxruntime-linux-x64-gpu-1.16.3/lib"
)

ONNX_GPU_PATH=""
for path in "${GPU_LIB_CANDIDATES[@]}"; do
    if [ -d "${path}" ] && [ -f "${path}/libonnxruntime_providers_cuda.so" ]; then
        ONNX_GPU_PATH="${path}"
        break
    fi
done

if [ -n "${ONNX_GPU_PATH}" ]; then
    COMPAT_PATH="${ONNX_GPU_PATH}/cuda_compat"
    CUDNN_PATH="/tmp/cudnn8_extract/cudnn-linux-x86_64-8.9.7.29_cuda11-archive/lib"
    CUDA11="/usr/local/cuda-11.8/lib64"
    CUDA12="/usr/local/cuda-12.8/lib64"
    CUDA13="/usr/local/cuda-13.0/lib64"

    LIB_PATH_COMPONENTS=("${ONNX_GPU_PATH}")
    [ -d "${COMPAT_PATH}" ] && LIB_PATH_COMPONENTS+=("${COMPAT_PATH}")
    [ -d "${CUDNN_PATH}" ] && LIB_PATH_COMPONENTS+=("${CUDNN_PATH}")
    [ -d "${CUDA11}" ] && LIB_PATH_COMPONENTS+=("${CUDA11}")
    [ -d "${CUDA12}" ] && LIB_PATH_COMPONENTS+=("${CUDA12}")
    [ -d "${CUDA13}" ] && LIB_PATH_COMPONENTS+=("${CUDA13}")

    LIB_PATH_STRING=$(IFS=:; echo "${LIB_PATH_COMPONENTS[*]}")
    export LD_LIBRARY_PATH="${LIB_PATH_STRING}:${LD_LIBRARY_PATH:-}"
    echo "✓ LD_LIBRARY_PATH primed for CUDA ONNX Runtime"
else
    echo "⚠️  Could not find CUDA-enabled ONNX Runtime libs. Embeddings may run on CPU." >&2
fi

ENV_OUTPUT="${ROOT_DIR}/config/h200.env"
mkdir -p "${ROOT_DIR}/config"

echo ""
echo "📝 Writing high-throughput overrides to ${ENV_OUTPUT}"
cat >"${ENV_OUTPUT}" <<'EOF'
# NIODOO H200 runtime overrides
USE_GPU_FITNESS=1
OPTIMIZED_ERAG=1
CACHE_PREFETCH_ENABLED=1
ERAG_BATCH_SIZE=256
ERAG_BATCH_FLUSH_MS=150
CACHE_PREFETCH_PROMPTS=16
CACHE_PREFETCH_TOP_HITS=8
CACHE_PREFETCH_PARALLELISM=12
GENERATION_MAX_TOKENS=4096
DYNAMIC_TOKEN_MAX=1024
TOKEN_PROMOTION_INTERVAL=30
ENABLE_CURATOR=1
CURATOR_BACKEND=vllm
HARDWARE=h200

# vLLM tuning
VLLM_HOST=127.0.0.1
VLLM_PORT=5001
VLLM_ENDPOINT=http://127.0.0.1:5001
CURATOR_VLLM_ENDPOINT=http://127.0.0.1:5001
VLLM_MODEL_ID=/workspace/models/hf_cache/models--Qwen--Qwen2.5-7B-Instruct-AWQ
VLLM_MODEL_PATH=/workspace/models/hf_cache/models--Qwen--Qwen2.5-7B-Instruct-AWQ
VLLM_DTYPE=bfloat16
VLLM_GPU_MEMORY_UTILIZATION=0.92
VLLM_MAX_MODEL_LEN=128000
VLLM_MAX_NUM_BATCHED_TOKENS=8192
VLLM_MAX_NUM_SEQS=64
VLLM_ATTENTION_BACKEND=flashinfer
VLLM_KV_CACHE_DTYPE=fp8
VLLM_USE_DEEP_GEMM=1
VLLM_ENABLE_CHUNKED_PREFILL=1
VLLM_ALL2ALL_BACKEND=pplx
EOF

echo "✓ Runtime overrides captured"

echo ""
echo "🏗️  Building workspace with GPU features"
cargo build --release --features gpu

echo ""
echo "✅ H200 bootstrap complete"
echo "To use the tuned profile, run:"
echo "  source config/h200.env"
echo "  cargo run --release --features gpu -- --hardware h200 --prompt 'test prompt'"
echo ""
echo "You can also reuse config/h200.env with scripts/start_all_services.sh"

