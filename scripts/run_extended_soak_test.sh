#!/bin/bash
# 🌊 EXTENDED SOAK TEST - PROVES SYSTEM STABILITY OVER HOURS 🌊
#
# This runs the Rust-based soak test for extended periods to prove
# the system can handle continuous load without degradation.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

echo -e "${BOLD}${BLUE}"
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  🌊 EXTENDED SOAK TEST - RUST IMPLEMENTATION 🌊              ║"
echo "║  Testing system stability under continuous load               ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Parse arguments
DURATION="${1:-3600}"  # Default 1 hour
PROMPTS="${2:-1000}"    # Default 1000 prompts
QUICK="${3:-false}"

if [ "$QUICK" = "true" ] || [ "$QUICK" = "--quick" ] || [ "$QUICK" = "-q" ]; then
    echo -e "${GREEN}Quick mode: 60 seconds, 100 prompts${NC}"
    cargo run --release --bin soak_test -- --quick --prompts=100 2>&1 | tee soak_test_quick_$(date +%Y%m%d_%H%M%S).log
else
    echo -e "${GREEN}Full soak test: ${DURATION}s, ${PROMPTS} prompts${NC}"
    cargo run --release --bin soak_test -- --duration=$DURATION --prompts=$PROMPTS 2>&1 | tee soak_test_full_$(date +%Y%m%d_%H%M%S).log
fi

echo ""
echo -e "${BOLD}${GREEN}✅ Soak test completed!${NC}"
echo "Results saved to: soak_test_results.json"
echo "Learning metrics saved to: learning_metrics_soak.json"

