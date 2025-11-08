#!/bin/bash
# Verify ALL endpoints are using CORRECT URLs that the code expects
set -e
cd "$(dirname "$0")/.."
source tcs_runtime.env 2>/dev/null || true
VLLM_ENDPOINT=${VLLM_ENDPOINT:-http://127.0.0.1:5001}
QDRANT_URL=${QDRANT_URL:-http://127.0.0.1:6333}
echo "CORRECT Endpoints (as code expects):"
echo "vLLM: $VLLM_ENDPOINT/v1/chat/completions"
echo "Qdrant: $QDRANT_URL/collections"
curl -s "$VLLM_ENDPOINT/v1/models" > /dev/null && echo "✅ vLLM OK" || echo "❌ vLLM FAIL"
curl -s "$QDRANT_URL/collections" > /dev/null && echo "✅ Qdrant OK" || echo "❌ Qdrant FAIL"
