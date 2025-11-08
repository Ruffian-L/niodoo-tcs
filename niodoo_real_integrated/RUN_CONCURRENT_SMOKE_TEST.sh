#!/bin/bash
# Run concurrent cascade smoke test with environment setup

set -e

echo "=== Setting up environment for Concurrent Cascade Smoke Test ==="

# Set defaults if not already set
export TOKENIZER_JSON="${TOKENIZER_JSON:-../tokenizer.json}"
export VLLM_ENDPOINT="${VLLM_ENDPOINT:-http://127.0.0.1:5001}"
export OLLAMA_ENDPOINT="${OLLAMA_ENDPOINT:-http://127.0.0.1:11434}"
export QDRANT_URL="${QDRANT_URL:-http://127.0.0.1:6333}"
export MAX_CONCURRENT="${MAX_CONCURRENT:-20}"

echo "Configuration:"
echo "  TOKENIZER_JSON: $TOKENIZER_JSON"
echo "  VLLM_ENDPOINT: $VLLM_ENDPOINT"
echo "  OLLAMA_ENDPOINT: $OLLAMA_ENDPOINT"
echo "  QDRANT_URL: $QDRANT_URL"
echo "  MAX_CONCURRENT: $MAX_CONCURRENT"
echo ""

echo "=== Running Concurrent Cascade Smoke Test ==="
cargo test --test concurrent_cascade_smoke_test concurrent_smoke_test -- --nocapture

echo ""
echo "=== Test Complete ==="
