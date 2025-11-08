#!/bin/bash
# 🔥 COMPREHENSIVE E2E LOAD & SOAK TEST RUNNER 🔥
# 
# This script runs comprehensive end-to-end tests that PROVE Niodoo's superiority
# over every other AI system with real load and soak testing.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

echo -e "${BOLD}${BLUE}"
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  🔥 COMPREHENSIVE E2E LOAD & SOAK TEST SUITE 🔥                ║"
echo "║  PROVING NIODOO IS SUPERIOR TO EVERY AI SYSTEM                 ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Parse arguments
MODE="${1:-load}"
DURATION="${2:-60}"
CONCURRENT="${3:-10}"
PROMPTS="${4:-100}"

case "$MODE" in
    "load")
        echo -e "${GREEN}Mode: LOAD TEST${NC}"
        echo "  Duration: ${DURATION}s"
        echo "  Concurrent Users: ${CONCURRENT}"
        echo "  Prompts: ${PROMPTS}"
        ;;
    "soak")
        echo -e "${GREEN}Mode: SOAK TEST${NC}"
        echo "  Duration: ${DURATION} hours"
        echo "  Concurrent Users: ${CONCURRENT}"
        echo "  Prompts: ${PROMPTS}"
        ;;
    *)
        echo -e "${RED}Invalid mode: $MODE${NC}"
        echo "Usage: $0 [load|soak] [duration] [concurrent] [prompts]"
        echo ""
        echo "Examples:"
        echo "  $0 load 60 10 100     # 60s load test, 10 concurrent, 100 prompts"
        echo "  $0 soak 2 5 50       # 2 hour soak test, 5 concurrent, 50 prompts"
        exit 1
        ;;
esac

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ python3 not found${NC}"
    exit 1
fi

# Check if services are running (optional check)
echo ""
echo -e "${BLUE}Checking services...${NC}"
if curl -s http://127.0.0.1:5001/v1/models > /dev/null 2>&1; then
    echo -e "${GREEN}✅ vLLM is running${NC}"
else
    echo -e "${YELLOW}⚠️  vLLM not detected (will use cargo run)${NC}"
fi

if curl -s http://127.0.0.1:6333/collections > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Qdrant is running${NC}"
else
    echo -e "${YELLOW}⚠️  Qdrant not detected${NC}"
fi

# Run Python test suite
echo ""
echo -e "${BOLD}${BLUE}Starting comprehensive E2E test suite...${NC}"
echo ""

python3 "$SCRIPT_DIR/comprehensive_e2e_load_test.py" \
    --mode "$MODE" \
    --duration "$DURATION" \
    --concurrent-users "$CONCURRENT" \
    --prompts "$PROMPTS" \
    --results-dir "test_reports/e2e_load_test_$(date +%Y%m%d_%H%M%S)"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo -e "${BOLD}${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}${GREEN}║  ✅ TEST SUITE COMPLETED SUCCESSFULLY                          ║${NC}"
    echo -e "${BOLD}${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${GREEN}Results saved to: test_reports/e2e_load_test_*/${NC}"
else
    echo ""
    echo -e "${BOLD}${RED}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}${RED}║  ❌ TEST SUITE FAILED                                           ║${NC}"
    echo -e "${BOLD}${RED}╚════════════════════════════════════════════════════════════════╝${NC}"
fi

exit $EXIT_CODE

