#!/bin/bash
# Run Ablation Test Suite
# Checks endpoints and runs ablation experiments

set -e

echo "🔬 NIODOO Ablation Test Suite"
echo "=============================="
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
    echo "   Start vLLM: vllm serve /path/to/model --port 5001"
fi

if curl -s --max-time 2 "${QDRANT_URL}/collections" > /dev/null 2>&1; then
    echo "✅ Qdrant: ${QDRANT_URL} - ONLINE"
    QDRANT_OK=true
else
    echo "❌ Qdrant: ${QDRANT_URL} - OFFLINE"
    echo "   Start Qdrant: docker run -p 6333:6333 qdrant/qdrant"
fi

echo ""

if [ "$VLLM_OK" = false ] || [ "$QDRANT_OK" = false ]; then
    echo "⚠️  Services not available. Running in mock mode..."
    export MOCK_MODE=true
fi

# Create output directory
OUTPUT_DIR="${OUTPUT_DIR:-ablation_results}"
mkdir -p "${OUTPUT_DIR}"

# Baseline file
BASELINE="${BASELINE:-baselines/baseline-latest.json}"

# Experiments to run
EXPERIMENTS=(
    "DisableRce"
    "BypassNTokens"
    "DisableTcsGpu"
    "DisableGpuFitness"
    "DisableCurator"
    "BypassErag"
)

echo "🧪 Running ablation experiments..."
echo ""

SUCCESS_COUNT=0
FAIL_COUNT=0

for exp in "${EXPERIMENTS[@]}"; do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Experiment: ${exp}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    if [ -f "${BASELINE}" ]; then
        BASELINE_ARG="--baseline ${BASELINE}"
    else
        BASELINE_ARG=""
        echo "⚠️  No baseline file found, running without comparison"
    fi
    
    if cargo run --release --bin ablation_runner -- \
        --experiment "${exp}" \
        ${BASELINE_ARG} \
        --concurrent-users 4 \
        --duration-secs 30 \
        --output-dir "${OUTPUT_DIR}" 2>&1 | tee "${OUTPUT_DIR}/${exp}.log"; then
        echo "✅ ${exp} - SUCCESS"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo "❌ ${exp} - FAILED"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
    echo ""
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Successful: ${SUCCESS_COUNT}"
echo "❌ Failed: ${FAIL_COUNT}"
echo "📁 Results: ${OUTPUT_DIR}/"
echo ""

if [ ${FAIL_COUNT} -eq 0 ]; then
    echo "🎉 All experiments completed successfully!"
    echo ""
    echo "Generate superiority proof:"
    echo "  ./scripts/run_superiority_proof.sh"
    exit 0
else
    echo "⚠️  Some experiments failed. Check logs in ${OUTPUT_DIR}/"
    exit 1
fi


