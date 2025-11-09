#!/bin/bash
# Run A/B test comparing topology-enabled vs topology-disabled configurations
# to prove if AI uses topology for understanding

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
echo "🔬 TOPOLOGY UNDERSTANDING A/B TEST"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check if configs exist
BASELINE_CONFIG="$ROOT_DIR/configs/topology_enabled.json"
TREATMENT_CONFIG="$ROOT_DIR/configs/topology_disabled.json"

if [ ! -f "$BASELINE_CONFIG" ]; then
    log_error "Baseline config not found: $BASELINE_CONFIG"
    exit 1
fi

if [ ! -f "$TREATMENT_CONFIG" ]; then
    log_error "Treatment config not found: $TREATMENT_CONFIG"
    exit 1
fi

# Verify endpoints are online
log_info "Verifying all endpoints are online..."
if ! bash "$ROOT_DIR/scripts/verify_all_endpoints.sh"; then
    log_error "Some endpoints are not responding. Please start all services first."
    exit 1
fi

echo ""
log_info "Starting A/B test..."
log_info "  Baseline: Topology-Enabled (hybrid mode, RCE enabled, nTokens enabled)"
log_info "  Treatment: Topology-Disabled (baseline mode, RCE disabled, nTokens bypassed)"
echo ""

# Run A/B test
OUTPUT_DIR="$ROOT_DIR/ab_test_results/topology_understanding"
mkdir -p "$OUTPUT_DIR"

log_info "Running A/B test (this may take several minutes)..."
log_info "  Concurrent users: ${CONCURRENT_USERS:-16}"
log_info "  Duration: ${DURATION_SECS:-120} seconds"
echo ""

cd "$ROOT_DIR"
cargo run --bin ab_test_runner --release -- \
    --baseline-name "topology_enabled" \
    --treatment-name "topology_disabled" \
    --baseline-config "$BASELINE_CONFIG" \
    --treatment-config "$TREATMENT_CONFIG" \
    --concurrent-users "${CONCURRENT_USERS:-16}" \
    --duration-secs "${DURATION_SECS:-120}" \
    --output-dir "$OUTPUT_DIR" \
    2>&1 | tee "$OUTPUT_DIR/ab_test.log"

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    log_success "A/B test completed successfully!"
    log_info "Results saved to: $OUTPUT_DIR"
    echo ""
    log_info "Key files:"
    log_info "  - ab_test_topology_enabled_vs_topology_disabled.json (full results)"
    log_info "  - ab_test.log (execution log)"
    echo ""
    log_info "To analyze results, check:"
    log_info "  - topology_impact field (positive/negative/neutral/inconclusive)"
    log_info "  - persistence_entropy_difference (higher = richer structure)"
    log_info "  - quality_difference_pct (higher = better understanding)"
    log_info "  - beta_meta_difference (RCE breakthrough detection)"
else
    log_error "A/B test failed. Check logs: $OUTPUT_DIR/ab_test.log"
    exit 1
fi

