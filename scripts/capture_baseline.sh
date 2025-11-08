#!/bin/bash
# Baseline Capture Script
# Runs metrics_runner Baseline scenario and saves timestamped JSON baseline

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR/.."
cd "$ROOT_DIR"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}📊 Capturing Baseline Metrics${NC}"
echo "=================================="

# Create baselines directory if it doesn't exist
mkdir -p baselines

# Generate timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASELINE_FILE="baselines/baseline-${TIMESTAMP}.json"
LATEST_SYMLINK="baselines/baseline-latest.json"

echo "Timestamp: $TIMESTAMP"
echo "Baseline file: $BASELINE_FILE"
echo ""

# Check if services are available (unless MOCK_MODE is set)
if [ -z "$MOCK_MODE" ] || [ "$MOCK_MODE" != "true" ]; then
    echo "Checking service dependencies..."
    
    # Check vLLM
    if curl -s -f "http://127.0.0.1:5001/health" > /dev/null 2>&1; then
        echo "✓ vLLM service available"
    else
        echo -e "${YELLOW}⚠️  vLLM service not available (will use mock mode if enabled)${NC}"
    fi
    
    # Check Qdrant
    if curl -s -f "http://127.0.0.1:6333/health" > /dev/null 2>&1; then
        echo "✓ Qdrant service available"
    else
        echo -e "${YELLOW}⚠️  Qdrant service not available (will use mock mode if enabled)${NC}"
    fi
fi

echo ""
echo "Running metrics_runner Baseline scenario..."
echo "This may take several minutes..."

# Run metrics_runner with Baseline scenario
if cargo run --bin metrics_runner -- \
    --scenario Baseline \
    --concurrent-users 16 \
    --duration-secs 60 \
    --target-tokens 2048 \
    --output "$BASELINE_FILE" 2>&1 | tee "baselines/capture-${TIMESTAMP}.log"; then
    
    if [ -f "$BASELINE_FILE" ]; then
        echo ""
        echo -e "${GREEN}✅ Baseline captured successfully${NC}"
        echo "File: $BASELINE_FILE"
        
        # Create symlink to latest
        ln -sf "baseline-${TIMESTAMP}.json" "$LATEST_SYMLINK"
        echo "Symlink created: $LATEST_SYMLINK"
        
        # Display key metrics
        echo ""
        echo "Key Metrics:"
        if command -v jq > /dev/null 2>&1; then
            echo "  Latency (p99): $(jq -r '.latency.p99_ms' "$BASELINE_FILE") ms"
            echo "  Throughput: $(jq -r '.throughput.requests_per_sec' "$BASELINE_FILE") req/s"
            echo "  TCS Stability CV: $(jq -r '.quality_slis.tcs_stability_cv // "N/A"' "$BASELINE_FILE")"
            echo "  RCE β_meta Compliance: $(jq -r '.quality_slis.rce_beta_meta_compliance // "N/A"' "$BASELINE_FILE")"
        else
            echo "  (Install jq for detailed metrics display)"
        fi
        
        echo ""
        echo "Baseline capture complete!"
    else
        echo -e "${YELLOW}⚠️  Baseline file not found after execution${NC}"
        exit 1
    fi
else
    echo ""
    echo -e "${YELLOW}❌ Baseline capture failed${NC}"
    echo "Check logs: baselines/capture-${TIMESTAMP}.log"
    exit 1
fi

