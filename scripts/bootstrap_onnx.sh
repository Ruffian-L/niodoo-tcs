#!/bin/bash
# Universal ONNX Runtime Bootstrap
# Auto-detects and configures ONNX Runtime library paths
# Usage: source scripts/bootstrap_onnx.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Auto-detect ONNX Runtime GPU libraries
ONNX_GPU_PATH=""
GPU_LIB_CANDIDATES=(
    "${ROOT_DIR}/third_party/onnxruntime-linux-x64-gpu-1.24.0/lib"
    "${ROOT_DIR}/third_party/onnxruntime-linux-x64-gpu-1.23.2/lib"
    "${ROOT_DIR}/third_party/onnxruntime-linux-x64-gpu-1.18.1/lib"
    "${ROOT_DIR}/third_party/onnxruntime-linux-x64-gpu-1.16.3/lib"
    "/workspace/onnxruntime-linux-x64-gpu-1.24.0/lib"
    "/workspace/onnxruntime-linux-x64-gpu-1.23.2/lib"
    "/workspace/onnxruntime-linux-x64-gpu-1.18.1/lib"
)

for path in "${GPU_LIB_CANDIDATES[@]}"; do
    if [ -d "${path}" ] && [ -f "${path}/libonnxruntime_providers_cuda.so" ]; then
        ONNX_GPU_PATH="${path}"
        break
    fi
done

# Fallback to CPU-only ONNX Runtime
if [ -z "${ONNX_GPU_PATH}" ]; then
    CPU_LIB_CANDIDATES=(
        "${ROOT_DIR}/third_party/onnxruntime-linux-x64-1.18.1/lib"
        "${ROOT_DIR}/third_party/onnxruntime-linux-x64-1.16.3/lib"
    )
    for path in "${CPU_LIB_CANDIDATES[@]}"; do
        if [ -d "${path}" ] && [ -f "${path}/libonnxruntime.so" ]; then
            ONNX_GPU_PATH="${path}"
            break
        fi
    done
fi

if [ -n "${ONNX_GPU_PATH}" ]; then
    # Build LD_LIBRARY_PATH components
    LIB_PATH_COMPONENTS=("${ONNX_GPU_PATH}")
    
    # Add CUDA compat directory if it exists
    COMPAT_PATH="${ONNX_GPU_PATH}/cuda_compat"
    [ -d "${COMPAT_PATH}" ] && LIB_PATH_COMPONENTS+=("${COMPAT_PATH}")
    
    # Add cuDNN extract directory if it exists
    CUDNN_PATH="/tmp/cudnn8_extract/cudnn-linux-x86_64-8.9.7.29_cuda11-archive/lib"
    [ -d "${CUDNN_PATH}" ] && LIB_PATH_COMPONENTS+=("${CUDNN_PATH}")
    
    # Add CUDA library paths
    for cuda_path in /usr/local/cuda-11.8/lib64 /usr/local/cuda-12.8/lib64 /usr/local/cuda-12/lib64 /usr/local/cuda/lib64; do
        [ -d "${cuda_path}" ] && LIB_PATH_COMPONENTS+=("${cuda_path}")
    done
    
    # Build final LD_LIBRARY_PATH
    LIB_PATH_STRING=$(IFS=:; echo "${LIB_PATH_COMPONENTS[*]}")
    export LD_LIBRARY_PATH="${LIB_PATH_STRING}:${LD_LIBRARY_PATH:-}"
    
    # Set ORT environment variables
    export ORT_DYLIB_PATH="${ONNX_GPU_PATH}/libonnxruntime.so"
    export ORT_DYLIB_DEFAULT_PATH="${ONNX_GPU_PATH}"
    export ORT_STRICT_VERSION_CHECK=0
    
    echo "✓ ONNX Runtime configured: ${ONNX_GPU_PATH}"
    echo "✓ LD_LIBRARY_PATH updated"
else
    echo "⚠️  WARNING: ONNX Runtime libraries not found. Embeddings may fail." >&2
    echo "   Searched paths:" >&2
    for path in "${GPU_LIB_CANDIDATES[@]}"; do
        echo "     - ${path}" >&2
    done
fi


