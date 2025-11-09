#!/bin/bash
# Comprehensive smoke test for all endpoints
# Tests each endpoint with real requests to ensure they're actually working

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

# Ports
QDRANT_PORT=6333
VLLM_PORT=5001
VLLM_CURATOR_PORT=5002
HEALTH_PORT=9090
RL_SERVER_PORT=8080

# Test results
TESTS_PASSED=0
TESTS_FAILED=0

test_passed() {
    TESTS_PASSED=$((TESTS_PASSED + 1))
    log_success "$1"
}

test_failed() {
    TESTS_FAILED=$((TESTS_FAILED + 1))
    log_error "$1"
}

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔥 SMOKE TESTING ALL ENDPOINTS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 1. Qdrant HTTP (6333)
echo "📡 Testing Qdrant HTTP (port $QDRANT_PORT)..."
echo ""

# Health check
log_info "  Health check..."
if curl -s --max-time 10 "http://127.0.0.1:$QDRANT_PORT/collections" > /dev/null 2>&1; then
    test_passed "Qdrant HTTP health check"
else
    test_failed "Qdrant HTTP health check"
fi

# Functional test: Create collection
log_info "  Functional test: Create/delete collection..."
TEST_COLLECTION="smoke_test_$(date +%s)"
CREATE_RESPONSE=$(curl -s --max-time 10 -X PUT "http://127.0.0.1:$QDRANT_PORT/collections/$TEST_COLLECTION" \
    -H "Content-Type: application/json" \
    -d '{"vectors": {"size": 128, "distance": "Cosine"}}' 2>&1)

if echo "$CREATE_RESPONSE" | grep -q "ok\|true\|status.*200"; then
    # Delete collection
    curl -s --max-time 10 -X DELETE "http://127.0.0.1:$QDRANT_PORT/collections/$TEST_COLLECTION" > /dev/null 2>&1 || true
    test_passed "Qdrant HTTP functional test (create/delete collection)"
else
    test_failed "Qdrant HTTP functional test (create collection failed)"
    echo "    Response: $CREATE_RESPONSE"
fi

# Error handling: Invalid collection name
log_info "  Error handling test..."
ERROR_RESPONSE=$(curl -s --max-time 10 -X PUT "http://127.0.0.1:$QDRANT_PORT/collections/" \
    -H "Content-Type: application/json" \
    -d '{"vectors": {"size": 128, "distance": "Cosine"}}' 2>&1)

if echo "$ERROR_RESPONSE" | grep -qi "error\|not found\|400\|404"; then
    test_passed "Qdrant HTTP error handling"
else
    test_failed "Qdrant HTTP error handling (should reject invalid request)"
fi

echo ""

# 2. vLLM Generation (5001)
echo "📡 Testing vLLM Generation (port $VLLM_PORT)..."
echo ""

# Health check
log_info "  Health check..."
MODELS_RESPONSE=$(curl -s --max-time 30 "http://127.0.0.1:$VLLM_PORT/v1/models" 2>&1)
if echo "$MODELS_RESPONSE" | grep -q "data\|id"; then
    MODEL_ID=$(echo "$MODELS_RESPONSE" | jq -r '.data[0].id' 2>/dev/null || echo "unknown")
    test_passed "vLLM Generation health check (model: $MODEL_ID)"
else
    test_failed "vLLM Generation health check"
    echo "    Response: $MODELS_RESPONSE"
fi

# Functional test: Completion
log_info "  Functional test: Generate completion..."
COMPLETION_RESPONSE=$(curl -s --max-time 60 -X POST "http://127.0.0.1:$VLLM_PORT/v1/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'"${MODEL_ID:-default}"'",
        "prompt": "def fibonacci(n):",
        "max_tokens": 50,
        "temperature": 0.7
    }' 2>&1)

if echo "$COMPLETION_RESPONSE" | grep -q "choices\|text"; then
    GENERATED_TEXT=$(echo "$COMPLETION_RESPONSE" | jq -r '.choices[0].text' 2>/dev/null | head -c 100 || echo "")
    test_passed "vLLM Generation functional test (generated: ${GENERATED_TEXT}...)"
else
    test_failed "vLLM Generation functional test"
    echo "    Response: $COMPLETION_RESPONSE"
fi

# Error handling: Invalid model
log_info "  Error handling test..."
ERROR_RESPONSE=$(curl -s --max-time 10 -X POST "http://127.0.0.1:$VLLM_PORT/v1/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "invalid_model_name_12345",
        "prompt": "test",
        "max_tokens": 10
    }' 2>&1)

if echo "$ERROR_RESPONSE" | grep -qi "error\|not found\|400\|404"; then
    test_passed "vLLM Generation error handling"
else
    test_failed "vLLM Generation error handling (should reject invalid model)"
fi

echo ""

# 3. vLLM Curator (5002) - may use same port as generation
echo "📡 Testing vLLM Curator (port $VLLM_CURATOR_PORT)..."
echo ""

CURATOR_URL="http://127.0.0.1:$VLLM_CURATOR_PORT"
if ! curl -s --max-time 5 "$CURATOR_URL/v1/models" > /dev/null 2>&1; then
    log_warn "  Curator not on separate port, checking if using same port as generation..."
    CURATOR_URL="http://127.0.0.1:$VLLM_PORT"
fi

# Health check
log_info "  Health check..."
CURATOR_MODELS_RESPONSE=$(curl -s --max-time 30 "$CURATOR_URL/v1/models" 2>&1)
if echo "$CURATOR_MODELS_RESPONSE" | grep -q "data\|id"; then
    CURATOR_MODEL_ID=$(echo "$CURATOR_MODELS_RESPONSE" | jq -r '.data[0].id' 2>/dev/null || echo "unknown")
    test_passed "vLLM Curator health check (model: $CURATOR_MODEL_ID)"
else
    test_failed "vLLM Curator health check"
    echo "    Response: $CURATOR_MODELS_RESPONSE"
fi

# Functional test: Completion
log_info "  Functional test: Generate completion..."
CURATOR_COMPLETION_RESPONSE=$(curl -s --max-time 60 -X POST "$CURATOR_URL/v1/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'"${CURATOR_MODEL_ID:-default}"'",
        "prompt": "Evaluate this code: def test(): pass",
        "max_tokens": 50,
        "temperature": 0.7
    }' 2>&1)

if echo "$CURATOR_COMPLETION_RESPONSE" | grep -q "choices\|text"; then
    CURATOR_TEXT=$(echo "$CURATOR_COMPLETION_RESPONSE" | jq -r '.choices[0].text' 2>/dev/null | head -c 100 || echo "")
    test_passed "vLLM Curator functional test (generated: ${CURATOR_TEXT}...)"
else
    test_failed "vLLM Curator functional test"
    echo "    Response: $CURATOR_COMPLETION_RESPONSE"
fi

echo ""

# 4. Main Pipeline Server (9090)
echo "📡 Testing Main Pipeline Server (port $HEALTH_PORT)..."
echo ""

# Health check
log_info "  Health check..."
HEALTH_RESPONSE=$(curl -s --max-time 10 "http://localhost:$HEALTH_PORT/health" 2>&1)
if echo "$HEALTH_RESPONSE" | grep -q "healthy\|ok\|status.*200"; then
    test_passed "Main Pipeline health check"
else
    test_failed "Main Pipeline health check"
    echo "    Response: $HEALTH_RESPONSE"
fi

# Readiness check
log_info "  Readiness check..."
READY_RESPONSE=$(curl -s --max-time 10 "http://localhost:$HEALTH_PORT/ready" 2>&1)
if echo "$READY_RESPONSE" | grep -q "ready\|ok\|status.*200"; then
    test_passed "Main Pipeline readiness check"
else
    test_failed "Main Pipeline readiness check"
    echo "    Response: $READY_RESPONSE"
fi

# Metrics check
log_info "  Metrics check..."
METRICS_RESPONSE=$(curl -s --max-time 10 "http://localhost:$HEALTH_PORT/metrics" 2>&1)
if echo "$METRICS_RESPONSE" | grep -q "niodoo\|prometheus\|# HELP\|# TYPE"; then
    test_passed "Main Pipeline metrics endpoint"
else
    test_failed "Main Pipeline metrics endpoint"
    echo "    Response: ${METRICS_RESPONSE:0:200}..."
fi

# Error handling: Invalid endpoint
log_info "  Error handling test..."
ERROR_RESPONSE=$(curl -s --max-time 10 "http://localhost:$HEALTH_PORT/invalid_endpoint" 2>&1)
if echo "$ERROR_RESPONSE" | grep -qi "404\|not found\|error"; then
    test_passed "Main Pipeline error handling"
else
    test_failed "Main Pipeline error handling (should reject invalid endpoint)"
fi

echo ""

# 5. RL Server (8080) - optional
echo "📡 Testing RL Server (port $RL_SERVER_PORT)..."
echo ""

# Health check
log_info "  Health check..."
RL_HEALTH_RESPONSE=$(curl -s --max-time 10 "http://localhost:$RL_SERVER_PORT/health" 2>&1)
if echo "$RL_HEALTH_RESPONSE" | grep -q "healthy\|ok\|status.*200"; then
    test_passed "RL Server health check"
else
    log_warn "RL Server health check (optional service)"
fi

echo ""

# Summary
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 SMOKE TEST SUMMARY"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Tests Passed: $TESTS_PASSED"
echo "Tests Failed: $TESTS_FAILED"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    log_success "All smoke tests passed! ✅"
    echo ""
    echo "Ready for A/B testing."
    exit 0
else
    log_error "Some smoke tests failed! ❌"
    echo ""
    echo "Please fix failing endpoints before running A/B tests."
    exit 1
fi
