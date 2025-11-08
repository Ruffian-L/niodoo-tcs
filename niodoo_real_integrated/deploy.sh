#!/bin/bash
# NIODOO Production Deployment Script
# Usage: ./deploy.sh [environment]

set -euo pipefail

ENVIRONMENT="${1:-production}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "🚀 Deploying NIODOO Real Integrated (Environment: $ENVIRONMENT)"

# Validate environment
if [[ ! "$ENVIRONMENT" =~ ^(dev|staging|production)$ ]]; then
    echo "❌ Invalid environment: $ENVIRONMENT"
    echo "Usage: $0 [dev|staging|production]"
    exit 1
fi

# Check prerequisites
command -v docker >/dev/null 2>&1 || { echo "❌ Docker not found"; exit 1; }
command -v cargo >/dev/null 2>&1 || { echo "❌ Cargo not found"; exit 1; }

# Create necessary directories
mkdir -p "$PROJECT_ROOT/logs"
mkdir -p "$PROJECT_ROOT/data"
mkdir -p "$PROJECT_ROOT/niodoo_real_integrated/logs"

# Set environment-specific variables
case "$ENVIRONMENT" in
    dev)
        RUST_LOG="debug"
        BUILD_PROFILE="dev"
        ;;
    staging)
        RUST_LOG="info"
        BUILD_PROFILE="release"
        ;;
    production)
        RUST_LOG="warn"
        BUILD_PROFILE="release"
        ;;
esac

echo "📦 Building Docker image..."
cd "$PROJECT_ROOT/niodoo_real_integrated"
docker build \
    --build-arg BUILD_PROFILE="$BUILD_PROFILE" \
    --tag "niodoo-real-integrated:${ENVIRONMENT}" \
    --tag "niodoo-real-integrated:latest" \
    -f Dockerfile .

echo "✅ Build complete"
echo ""
echo "📋 Deployment Summary:"
echo "   Environment: $ENVIRONMENT"
echo "   Image: niodoo-real-integrated:${ENVIRONMENT}"
echo "   Log Level: $RUST_LOG"
echo ""
echo "🚀 To run the container:"
echo "   docker run -d \\"
echo "     --name niodoo-${ENVIRONMENT} \\"
echo "     -v $PROJECT_ROOT/logs:/app/logs \\"
echo "     -v $PROJECT_ROOT/data:/app/data \\"
echo "     -e RUST_LOG=$RUST_LOG \\"
echo "     -e VLLM_ENDPOINT=\${VLLM_ENDPOINT} \\"
echo "     -e QDRANT_URL=\${QDRANT_URL} \\"
echo "     -e OLLAMA_URL=\${OLLAMA_URL} \\"
echo "     niodoo-real-integrated:${ENVIRONMENT}"
echo ""
echo "📊 To check health:"
echo "   docker inspect --format='{{.State.Health.Status}}' niodoo-${ENVIRONMENT}"
echo ""
echo "📝 To view logs:"
echo "   docker logs -f niodoo-${ENVIRONMENT}"

