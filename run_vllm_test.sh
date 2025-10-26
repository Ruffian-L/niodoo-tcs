#!/bin/bash
# Simple vLLM test runner

set -e

echo "🔍 Testing vLLM..."

# Wait for vLLM to be ready
echo "⏳ Waiting for vLLM to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo "✅ vLLM is ready!"
        break
    fi
    echo "  Still loading... ($i/30)"
    sleep 5
done

# Test the connection
echo ""
echo "🧪 Testing vLLM connection..."
cd /workspace/Niodoo-Final
cargo test --lib vllm_bridge::tests::test_generate_if_available -- --nocapture

echo ""
echo "✅ vLLM test complete!"

