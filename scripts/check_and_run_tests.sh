#!/bin/bash
# Check Endpoints and Run Test Suites

set -e

echo "🔍 NIODOO Test Suite Runner"
echo "============================"
echo ""

# Check endpoints
VLLM_ENDPOINT="${VLLM_ENDPOINT:-http://127.0.0.1:5001}"
QDRANT_URL="${QDRANT_URL:-http://127.0.0.1:6333}"

echo "📡 Checking Endpoints..."
echo ""

VLLM_OK=false
QDRANT_OK=false

# Check vLLM
if curl -s --max-time 2 "${VLLM_ENDPOINT}/v1/models" > /dev/null 2>&1; then
    echo "✅ vLLM: ${VLLM_ENDPOINT} - ONLINE"
    VLLM_MODELS=$(curl -s "${VLLM_ENDPOINT}/v1/models" | jq -r '.data[].id' 2>/dev/null | head -3 | tr '\n' ' ' || echo "unknown")
    echo "   Models: ${VLLM_MODELS}"
    VLLM_OK=true
else
    echo "❌ vLLM: ${VLLM_ENDPOINT} - OFFLINE"
    echo "   To start: vllm serve /path/to/model --port 5001"
fi

# Check Qdrant
if curl -s --max-time 2 "${QDRANT_URL}/collections" > /dev/null 2>&1; then
    echo "✅ Qdrant: ${QDRANT_URL} - ONLINE"
    QDRANT_COLLECTIONS=$(curl -s "${QDRANT_URL}/collections" | jq -r '.result.collections[].name' 2>/dev/null | head -3 | tr '\n' ' ' || echo "unknown")
    echo "   Collections: ${QDRANT_COLLECTIONS}"
    QDRANT_OK=true
else
    echo "❌ Qdrant: ${QDRANT_URL} - OFFLINE"
    echo "   To start: docker run -p 6333:6333 qdrant/qdrant"
fi

echo ""

# Determine what we can run
if [ "$VLLM_OK" = true ] && [ "$QDRANT_OK" = true ]; then
    echo "✅ All endpoints available! Running full test suites..."
    echo ""
    
    # Run ablation suite
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🧪 Running Ablation Test Suite"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    ./scripts/run_ablation_suite.sh
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🔬 Running A/B Test Suite"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    ./scripts/run_ab_test_suite.sh
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📊 Generating Superiority Proof"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    ./scripts/run_superiority_proof.sh
    
elif [ "$VLLM_OK" = false ] && [ "$QDRANT_OK" = false ]; then
    echo "⚠️  Both services offline. Cannot run tests."
    echo ""
    echo "Quick Start Guide:"
    echo "1. Start Qdrant:"
    echo "   docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant"
    echo ""
    echo "2. Start vLLM:"
    echo "   vllm serve /workspace/models/Qwen2.5-0.5B-Instruct --port 5001"
    echo ""
    echo "3. Then run this script again:"
    echo "   ./scripts/check_and_run_tests.sh"
    exit 1
else
    echo "⚠️  Partial service availability. Some tests may fail."
    echo ""
    echo "Running tests anyway (will fail gracefully)..."
    ./scripts/run_ablation_suite.sh || echo "Ablation tests failed (expected with missing services)"
fi

echo ""
echo "✅ Test suite execution complete!"
echo "📁 Check results in: ablation_results/ and ab_test_results/"


