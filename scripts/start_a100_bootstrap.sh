#!/bin/bash
# A100 bootstrap script for RunPod environment
# Optimized for NVIDIA A100-SXM4-80GB (80GB VRAM)

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "🔧 NIODOO A100 Bootstrap"
echo "=========================="
echo ""

# Check GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ ERROR: nvidia-smi not found. Are you on a GPU instance?" >&2
    exit 1
fi

GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -n1)
echo "📊 GPU: $GPU_INFO"

DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)
echo "📊 Driver: $DRIVER_VERSION"

if [[ "$GPU_INFO" != *"A100"* ]]; then
    echo "⚠️  WARNING: This script is optimized for A100. Current GPU: $GPU_INFO" >&2
fi

# CUDA detection
CUDA_HOME=""
for candidate in /usr/local/cuda-13.0 /usr/local/cuda-12.8 /usr/local/cuda-12 /usr/local/cuda-11.8 /usr/local/cuda; do
    if [ -d "${candidate}" ] && [ -f "${candidate}/bin/nvcc" ]; then
        CUDA_HOME="${candidate}"
        break
    fi
done

if [ -z "$CUDA_HOME" ]; then
    echo "⚠️  WARNING: CUDA not found. Some GPU features may be unavailable." >&2
else
    echo "✓ CUDA: $CUDA_HOME"
    export CUDA_HOME
    export PATH="${CUDA_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi

# ONNX Runtime GPU libs
ONNX_GPU_PATH=""
for path in \
    "/workspace/Niodoo-Final/third_party/onnxruntime-linux-x64-gpu-1.24.0/lib" \
    "/workspace/Niodoo-Final/third_party/onnxruntime-linux-x64-gpu-1.23.2/lib" \
    "/workspace/onnxruntime-linux-x64-gpu-1.24.0/lib" \
    "/workspace/onnxruntime-linux-x64-gpu-1.23.2/lib"; do
    if [ -d "${path}" ]; then
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

ENV_OUTPUT="${ROOT_DIR}/config/a100.env"
mkdir -p "${ROOT_DIR}/config"

echo ""
echo "📝 Writing A100 runtime overrides to ${ENV_OUTPUT}"
cat >"${ENV_OUTPUT}" <<'EOF'
# NIODOO A100 runtime overrides (80GB VRAM)
USE_GPU_FITNESS=1
TCS_ENABLE_GPU=1
OPTIMIZED_ERAG=1
CACHE_PREFETCH_ENABLED=1
ERAG_BATCH_SIZE=384
ERAG_BATCH_FLUSH_MS=120
CACHE_PREFETCH_PROMPTS=24
CACHE_PREFETCH_TOP_HITS=12
CACHE_PREFETCH_PARALLELISM=12
GENERATION_MAX_TOKENS=8192
DYNAMIC_TOKEN_MAX=2048
TOKEN_PROMOTION_INTERVAL=25
ENABLE_CURATOR=1
CURATOR_BACKEND=vllm
HARDWARE=a100

# ONNX Runtime GPU memory limit for A100 (6GB for embeddings, leaves room for training)
QWEN_CUDA_MEM_LIMIT_MB=6144

# vLLM tuning for A100-SXM4-80GB
VLLM_HOST=127.0.0.1
VLLM_PORT=5001
VLLM_ENDPOINT=http://127.0.0.1:5001
CURATOR_VLLM_ENDPOINT=http://127.0.0.1:5001
VLLM_MODEL_ID=/workspace/models/Qwen2.5-7B-Instruct-AWQ
VLLM_MODEL_PATH=/workspace/models/Qwen2.5-7B-Instruct-AWQ
VLLM_DTYPE=bfloat16
VLLM_GPU_MEMORY_UTILIZATION=0.85
VLLM_MAX_MODEL_LEN=32768
VLLM_MAX_NUM_BATCHED_TOKENS=16384
VLLM_MAX_NUM_SEQS=128
VLLM_ATTENTION_BACKEND=FLASH_ATTN
VLLM_KV_CACHE_DTYPE=fp16
VLLM_ENABLE_CHUNKED_PREFILL=1
VLLM_USE_DEEP_GEMM=0
VLLM_ALL2ALL_BACKEND=

# Runtime controls
RUSTONIG_SYSTEM_LIBONIG=1
LD_LIBRARY_PATH=/workspace/Niodoo-Final/third_party/onnxruntime-linux-x64-gpu-1.24.0/lib:/workspace/onnxruntime-linux-x64-gpu-1.24.0/lib:${LD_LIBRARY_PATH}

# Logging defaults
RUST_LOG=info
EOF

echo "✓ Runtime overrides captured"

echo ""
echo "🔨 Building workspace with GPU features..."
if command -v cargo &> /dev/null; then
    if [ -f "${ROOT_DIR}/.runpod_env.sh" ]; then
        source "${ROOT_DIR}/.runpod_env.sh"
    fi
    cargo build --release --features gpu || {
        echo "⚠️  GPU build failed. Falling back to CPU-only build." >&2
        cargo build --release
    }
else
    echo "⚠️  Cargo not found. Skipping build." >&2
fi

echo ""
echo "✅ A100 bootstrap complete!"
echo ""
echo "Next steps:"
echo "  1. Source the environment: source config/a100.env"
echo "  2. Start services: ./start_all_services.sh --hardware a100"
echo ""
