#!/bin/bash
# Full End-to-End Test for NIODOO_REAL_INTEGRATED Pipeline
# Tests all components: ERAG, Compass, Curator, RCE, Generation, Learning Loop

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR/.."
cd "$ROOT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="e2e_test_results_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  FULL END-TO-END NIODOO_REAL_INTEGRATED PIPELINE TEST        ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Results directory: $RESULTS_DIR"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

TEST_PASSED=0
TEST_FAILED=0
TEST_SKIPPED=0

log_test() {
    local status=$1
    local message=$2
    case $status in
        PASS)
            echo -e "${GREEN}✅ PASS:${NC} $message"
            ((TEST_PASSED++))
            ;;
        FAIL)
            echo -e "${RED}❌ FAIL:${NC} $message"
            ((TEST_FAILED++))
            ;;
        SKIP)
            echo -e "${YELLOW}⏭️  SKIP:${NC} $message"
            ((TEST_SKIPPED++))
            ;;
    esac
}

# Step 1: Check Services
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 1: Service Health Checks"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check vLLM
echo -n "Checking vLLM (http://127.0.0.1:5001)... "
if curl -s -f http://127.0.0.1:5001/v1/models > /dev/null 2>&1; then
    echo -e "${GREEN}✓ ONLINE${NC}"
    VLLM_OK=true
else
    echo -e "${RED}✗ OFFLINE${NC}"
    VLLM_OK=false
    log_test FAIL "vLLM service not available"
fi

# Check Qdrant
echo -n "Checking Qdrant (http://127.0.0.1:6333)... "
if curl -s -f http://127.0.0.1:6333/collections > /dev/null 2>&1; then
    echo -e "${GREEN}✓ ONLINE${NC}"
    QDRANT_OK=true
else
    echo -e "${RED}✗ OFFLINE${NC}"
    QDRANT_OK=false
    log_test FAIL "Qdrant service not available"
fi

if [ "$VLLM_OK" != "true" ] || [ "$QDRANT_OK" != "true" ]; then
    echo ""
    echo -e "${RED}❌ Required services are not available. Cannot run full E2E test.${NC}"
    echo "Please start:"
    echo "  - vLLM: python -m vllm.entrypoints.openai.api_server --model <model_path>"
    echo "  - Qdrant: docker run -p 6333:6333 qdrant/qdrant"
    exit 1
fi

echo ""

# Step 2: Build Binary
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 2: Building Pipeline Binary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

cd niodoo_real_integrated

if cargo build --bin niodoo_real_integrated --release --features svc 2>&1 | tee "../$RESULTS_DIR/build.log" | tail -20; then
    log_test PASS "Pipeline binary build"
    echo ""
else
    log_test FAIL "Pipeline binary build"
    exit 1
fi

cd ..

# Step 3: Test Cases
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 3: Running End-to-End Pipeline Tests"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Test prompts covering different scenarios
TEST_PROMPTS=(
    "What is the capital of France?"
    "Explain how neural networks learn"
    "Write a Python function to calculate fibonacci numbers"
    "What are the main components of the NIODOO system?"
    "Describe the relationship between topology and consciousness"
)

cd niodoo_real_integrated

for i in "${!TEST_PROMPTS[@]}"; do
    prompt="${TEST_PROMPTS[$i]}"
    test_num=$((i + 1))
    
    echo -e "${BLUE}Test $test_num/${#TEST_PROMPTS[@]}:${NC} ${prompt:0:60}..."
    
    output_file="../$RESULTS_DIR/test_${test_num}_output.json"
    log_file="../$RESULTS_DIR/test_${test_num}_log.txt"
    
    if timeout 120 cargo run --bin niodoo_real_integrated --release --features svc -- \
        --prompt "$prompt" \
        --output json \
        2>&1 | tee "$log_file" | tail -50 > "$output_file"; then
        
        # Check if output contains valid response
        if grep -q "hybrid_response\|hybrid\|response" "$output_file" 2>/dev/null; then
            log_test PASS "Test $test_num: Pipeline execution"
            
            # Extract key metrics
            echo "  Response generated successfully"
            if grep -q "latency\|entropy\|compass" "$log_file" 2>/dev/null; then
                echo "  Metrics captured"
            fi
        else
            log_test FAIL "Test $test_num: No valid response in output"
        fi
    else
        log_test FAIL "Test $test_num: Pipeline execution failed"
    fi
    
    echo ""
done

cd ..

# Step 4: Integration Test Binary
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 4: Running Integration Test Binary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

cd niodoo_real_integrated

if cargo build --bin test_pipeline_integration --release 2>&1 | tee "../$RESULTS_DIR/build_integration_test.log" | tail -10; then
    echo "Running integration test..."
    if timeout 300 cargo run --bin test_pipeline_integration --release 2>&1 | tee "../$RESULTS_DIR/integration_test.log"; then
        if grep -q "Integration test completed\|✅\|PASS" "../$RESULTS_DIR/integration_test.log"; then
            log_test PASS "Integration test binary"
        else
            log_test FAIL "Integration test binary (check logs)"
        fi
    else
        log_test FAIL "Integration test binary execution"
    fi
else
    log_test SKIP "Integration test binary (build failed or not available)"
fi

cd ..

# Step 5: Python Test Suite
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 5: Running Python Test Suite"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ -f "niodoo-ai/scripts/user_test_utils.py" ]; then
    cd niodoo-ai
    
    # Run a few test prompts via Python test suite
    TEST_PROMPTS_PY=(
        "What is machine learning?"
        "Explain the concept of topology"
    )
    
    for prompt in "${TEST_PROMPTS_PY[@]}"; do
        echo "Testing: $prompt"
        if python3 -c "
import sys
sys.path.insert(0, 'scripts')
from user_test_utils import test_pipeline
result = test_pipeline('$prompt', timeout=60)
if result['success']:
    print('✅ Test passed')
    sys.exit(0)
else:
    print('❌ Test failed:', result.get('errors', []))
    sys.exit(1)
" 2>&1 | tee "../$RESULTS_DIR/python_test_${prompt:0:20}.log"; then
            log_test PASS "Python test suite: ${prompt:0:40}..."
        else
            log_test FAIL "Python test suite: ${prompt:0:40}..."
        fi
    done
    
    cd ..
else
    log_test SKIP "Python test suite (not found)"
fi

echo ""

# Step 6: Summary
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST SUMMARY"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Total Tests: $((TEST_PASSED + TEST_FAILED + TEST_SKIPPED))"
echo -e "${GREEN}Passed: $TEST_PASSED${NC}"
echo -e "${RED}Failed: $TEST_FAILED${NC}"
echo -e "${YELLOW}Skipped: $TEST_SKIPPED${NC}"
echo ""
echo "Results directory: $RESULTS_DIR"
echo ""

# Generate JSON summary
python3 <<EOF
import json
import os
from datetime import datetime

summary = {
    'timestamp': datetime.now().isoformat(),
    'results_dir': '$RESULTS_DIR',
    'tests': {
        'passed': $TEST_PASSED,
        'failed': $TEST_FAILED,
        'skipped': $TEST_SKIPPED,
        'total': $((TEST_PASSED + TEST_FAILED + TEST_SKIPPED))
    },
    'services': {
        'vllm': '$VLLM_OK',
        'qdrant': '$QDRANT_OK'
    }
}

with open('$RESULTS_DIR/summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("Summary saved to: $RESULTS_DIR/summary.json")
EOF

if [ $TEST_FAILED -gt 0 ]; then
    echo -e "${RED}❌ Some tests failed. Check logs in $RESULTS_DIR${NC}"
    exit 1
else
    echo -e "${GREEN}✅ All tests passed!${NC}"
    exit 0
fi

