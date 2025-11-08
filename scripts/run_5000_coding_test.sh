#!/bin/bash
# Convenience script to run the 5000 coding prompts test suite
# Verifies services, runs A/B comparison, and generates reports

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR/.."
cd "$ROOT_DIR"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

log_warn() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 5000 CODING PROMPTS TEST SUITE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check if services are running
log_info "Checking required services..."

# Check vLLM
if curl -s http://localhost:5001/v1/models > /dev/null 2>&1; then
    log_success "vLLM is running"
else
    log_error "vLLM is not running on port 5001"
    log_info "Start vLLM: python -m vllm.entrypoints.openai.api_server --model <model_path> --port 5001"
    exit 1
fi

# Check Qdrant
if curl -s http://localhost:6333/collections > /dev/null 2>&1; then
    log_success "Qdrant is running"
else
    log_error "Qdrant is not running on port 6333"
    log_info "Start Qdrant: docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant"
    exit 1
fi

# Optional: Check health endpoints (if svc feature is enabled)
if curl -s http://localhost:9090/health > /dev/null 2>&1; then
    log_success "Main pipeline health endpoint is available"
else
    log_warn "Main pipeline health endpoint not available (svc feature may not be enabled)"
fi

if curl -s http://localhost:8080/health > /dev/null 2>&1; then
    log_success "RL server health endpoint is available"
else
    log_warn "RL server health endpoint not available"
fi

echo ""
log_info "Starting test suite..."
echo ""

# Run the test suite
# Default: 500 conversations (10-20 turns each = 5000+ prompts)
NUM_CONVERSATIONS=${NUM_CONVERSATIONS:-500}
OUTPUT_DIR=${OUTPUT_DIR:-test_results_5000_coding}

log_info "Configuration:"
log_info "  Conversations: $NUM_CONVERSATIONS"
log_info "  Output directory: $OUTPUT_DIR"
log_info "  Total prompts: ~$((NUM_CONVERSATIONS * 10))-$(($NUM_CONVERSATIONS * 20))"
log_info "  Total executions: ~$((NUM_CONVERSATIONS * 10 * 2))-$(($NUM_CONVERSATIONS * 20 * 2)) (A/B)"
echo ""

# Run the test
cargo run --bin test_5000_coding_prompts -- \
    --num-conversations "$NUM_CONVERSATIONS" \
    --output-dir "$OUTPUT_DIR" \
    --baseline-name "baseline" \
    --treatment-name "treatment"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    log_success "Test suite completed successfully!"
    log_info "Results saved to: $OUTPUT_DIR/"
    log_info "  - conversations.json: Generated conversation flows"
    log_info "  - test_report_ab_<timestamp>.json: Full A/B comparison report"
    echo ""
else
    echo ""
    log_error "Test suite failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi


