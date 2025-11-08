#!/bin/bash
# Run A/B Test Suite
# Compares baseline vs treatment configurations

set -e

echo "🔬 NIODOO A/B Test Suite"
echo "======================="
echo ""

# Check endpoints
VLLM_ENDPOINT="${VLLM_ENDPOINT:-http://127.0.0.1:5001}"
QDRANT_URL="${QDRANT_URL:-http://127.0.0.1:6333}"

echo "📡 Checking endpoints..."
VLLM_OK=false
QDRANT_OK=false

if curl -s --max-time 2 "${VLLM_ENDPOINT}/v1/models" > /dev/null 2>&1; then
    echo "✅ vLLM: ${VLLM_ENDPOINT} - ONLINE"
    VLLM_OK=true
else
    echo "❌ vLLM: ${VLLM_ENDPOINT} - OFFLINE"
fi

if curl -s --max-time 2 "${QDRANT_URL}/collections" > /dev/null 2>&1; then
    echo "✅ Qdrant: ${QDRANT_URL} - ONLINE"
    QDRANT_OK=true
else
    echo "❌ Qdrant: ${QDRANT_URL} - OFFLINE"
fi

echo ""

if [ "$VLLM_OK" = false ] || [ "$QDRANT_OK" = false ]; then
    echo "⚠️  Services not available. Cannot run A/B tests."
    echo "   Start services:"
    echo "   - vLLM: vllm serve /path/to/model --port 5001"
    echo "   - Qdrant: docker run -p 6333:6333 qdrant/qdrant"
    exit 1
fi

# Create output directory
OUTPUT_DIR="${OUTPUT_DIR:-ab_test_results}"
mkdir -p "${OUTPUT_DIR}"

# Create test configurations
BASELINE_CONFIG="${OUTPUT_DIR}/baseline_config.json"
TREATMENT_CONFIG="${OUTPUT_DIR}/treatment_config.json"

cat > "${BASELINE_CONFIG}" << EOF
{
  "TOPOLOGY_MODE": "hybrid",
  "RCE_ENABLED": "1",
  "N_TOKENS_BYPASS": "0",
  "ENABLE_CURATOR": "1",
  "TCS_ENABLE_GPU": "1"
}
EOF

cat > "${TREATMENT_CONFIG}" << EOF
{
  "TOPOLOGY_MODE": "hybrid",
  "RCE_ENABLED": "0",
  "N_TOKENS_BYPASS": "0",
  "ENABLE_CURATOR": "1",
  "TCS_ENABLE_GPU": "1"
}
EOF

echo "🧪 Running A/B test: Baseline vs Treatment (RCE disabled)"
echo ""

if cargo run --release --bin ab_test_runner -- \
    --baseline-name "baseline" \
    --treatment-name "treatment_no_rce" \
    --baseline-config "${BASELINE_CONFIG}" \
    --treatment-config "${TREATMENT_CONFIG}" \
    --concurrent-users 4 \
    --duration-secs 30 \
    --output-dir "${OUTPUT_DIR}" 2>&1 | tee "${OUTPUT_DIR}/ab_test.log"; then
    echo ""
    echo "✅ A/B test completed successfully!"
    echo "📁 Results: ${OUTPUT_DIR}/"
    echo ""
    echo "View results:"
    echo "  cat ${OUTPUT_DIR}/ab_test_baseline_vs_treatment_no_rce.json | jq"
    exit 0
else
    echo ""
    echo "❌ A/B test failed. Check logs: ${OUTPUT_DIR}/ab_test.log"
    exit 1
fi


