#!/bin/bash
# Master Validation Runner Script
# Runs comprehensive validation suite proving NIODOO superiority

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="${PROJECT_ROOT}/validation_results/$(date +%Y%m%d_%H%M%S)"

echo "🔥🔥🔥 NIODOO MASTER VALIDATION SUITE 🔥🔥🔥"
echo "=============================================="
echo ""
echo "This will comprehensively validate NIODOO and prove superiority over all AI coders"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Set environment variables
export WORKSPACE_ROOT="$PROJECT_ROOT"
export RUST_LOG=info

# Auto-detect ONNX runtime
if [ -z "$LD_LIBRARY_PATH" ]; then
    ONNX_PATHS=(
        "$PROJECT_ROOT/third_party/onnxruntime-linux-x64-gpu-1.24.0/lib"
        "$PROJECT_ROOT/third_party/onnxruntime-linux-x64-gpu-1.23.2/lib"
        "$PROJECT_ROOT/third_party/onnxruntime-linux-x64-gpu-1.18.1/lib"
        "$PROJECT_ROOT/third_party/onnxruntime-linux-x64-gpu-1.16.3/lib"
    )
    
    for path in "${ONNX_PATHS[@]}"; do
        if [ -d "$path" ] && [ -f "$path/libonnxruntime_providers_cuda.so" ]; then
            export LD_LIBRARY_PATH="$path:$LD_LIBRARY_PATH"
            export ORT_DYLIB_PATH="$path/libonnxruntime.so"
            echo "✅ Auto-detected ONNX Runtime: $path"
            break
        fi
    done
fi

# Check for required services
echo ""
echo "Checking required services..."
VLLM_AVAILABLE=false
QDRANT_AVAILABLE=false

if timeout 2 bash -c "echo > /dev/tcp/127.0.0.1/5001" 2>/dev/null; then
    VLLM_AVAILABLE=true
    echo "✅ vLLM available at http://127.0.0.1:5001"
else
    echo "⚠️  vLLM not available at http://127.0.0.1:5001"
fi

if timeout 2 bash -c "echo > /dev/tcp/127.0.0.1/6333" 2>/dev/null; then
    QDRANT_AVAILABLE=true
    echo "✅ Qdrant available at http://127.0.0.1:6333"
else
    echo "⚠️  Qdrant not available at http://127.0.0.1:6333"
fi

if [ "$VLLM_AVAILABLE" = false ] || [ "$QDRANT_AVAILABLE" = false ]; then
    echo ""
    echo "⚠️  WARNING: Some services are not available. Validation will run in mock mode."
    echo "   To run full validation, start services:"
    echo "   - vLLM: Start vLLM server on port 5001"
    echo "   - Qdrant: docker run -p 6333:6333 qdrant/qdrant"
    echo ""
fi

# Run master validation
echo ""
echo "🚀 Starting master validation..."
echo ""

cd "$PROJECT_ROOT/niodoo_real_integrated"

# Determine quick mode based on argument
QUICK_FLAG=""
if [ "${1:-}" = "--quick" ]; then
    QUICK_FLAG="--quick"
    echo "Running in QUICK mode (reduced test counts)"
fi

# Run the master validation binary
cargo run --bin master_validation -- \
    --output-dir "$OUTPUT_DIR" \
    $QUICK_FLAG \
    --compare-baseline

VALIDATION_EXIT_CODE=$?

echo ""
echo "=============================================="
if [ $VALIDATION_EXIT_CODE -eq 0 ]; then
    echo "✅ VALIDATION COMPLETE"
    echo ""
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "📊 View results:"
    echo "   - JSON Report: $OUTPUT_DIR/master_validation_report.json"
    echo "   - Summary: $OUTPUT_DIR/VALIDATION_SUMMARY.md"
    echo ""
    echo "🎉🎉🎉 NIODOO SUPERIORITY PROVEN 🎉🎉🎉"
else
    echo "⚠️  VALIDATION COMPLETED WITH WARNINGS"
    echo ""
    echo "Check results in: $OUTPUT_DIR"
    echo ""
fi

exit $VALIDATION_EXIT_CODE

