#!/bin/bash
# Build script for NIODOO RunPod Stateful Pod Docker image
# Implements Solution Path 1: All-in-One Stateful Pod Architecture

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Configuration
IMAGE_NAME="${IMAGE_NAME:-niodoo-runpod}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
HF_TOKEN="${HF_TOKEN:-}"
VLLM_MODEL_REPO="${VLLM_MODEL_REPO:-Qwen/Qwen2.5-7B-Instruct-AWQ}"

echo "=========================================="
echo "Building NIODOO RunPod Stateful Pod Image"
echo "=========================================="
echo "Image: $IMAGE_NAME:$IMAGE_TAG"
echo "Model: $VLLM_MODEL_REPO"
echo ""

# Check prerequisites
command -v docker >/dev/null 2>&1 || { echo "❌ Docker not found"; exit 1; }

# Build arguments
BUILD_ARGS=(
    --build-arg "VLLM_MODEL_REPO=$VLLM_MODEL_REPO"
    --build-arg "VLLM_MODEL_PATH=/models/qwen2.5-7b-instruct-awq"
)

if [ -n "$HF_TOKEN" ]; then
    BUILD_ARGS+=(--build-arg "HF_TOKEN=$HF_TOKEN")
    echo "✓ Using HuggingFace token for model download"
else
    echo "⚠ No HF_TOKEN provided - model must be publicly accessible"
fi

# Build the image
echo "Building Docker image (this may take 10-20 minutes)..."
docker build \
    "${BUILD_ARGS[@]}" \
    -f Dockerfile.runpod \
    -t "$IMAGE_NAME:$IMAGE_TAG" \
    -t "$IMAGE_NAME:latest" \
    .

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Build successful!"
    echo ""
    echo "Image: $IMAGE_NAME:$IMAGE_TAG"
    echo ""
    echo "Next steps:"
    echo "1. Push to registry: docker push $IMAGE_NAME:$IMAGE_TAG"
    echo "2. Deploy on RunPod using deployment/runpod/DEPLOYMENT_GUIDE.md"
    echo ""
else
    echo ""
    echo "❌ Build failed"
    exit 1
fi





