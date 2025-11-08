#!/bin/bash
# Comprehensive script to start all endpoints and test everything end-to-end

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR/.."
cd "$ROOT_DIR"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Ports
HEALTH_PORT=${NIODOO_HEALTH_PORT:-9090}
RL_SERVER_PORT=8080
VLLM_PORT=5001
QDRANT_PORT=6333

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

# Test HTTP endpoint
test_endpoint() {
    local url=$1
    local name=$2
    local method=${3:-GET}
    local data=${4:-}
    
    log_info "Testing $name: $method $url"
    
    if [ "$method" = "GET" ]; then
        response=$(curl -s -w "\n%{http_code}" "$url" --max-time 10 2>&1 || echo "000")
    else
        response=$(curl -s -w "\n%{http_code}" -X "$method" -H "Content-Type: application/json" -d "$data" "$url" --max-time 30 2>&1 || echo "000")
    fi
    
    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | sed '$d')
    
    if [ "$http_code" = "200" ] || [ "$http_code" = "201" ]; then
        log_success "$name returned $http_code"
        return 0
    else
        log_error "$name returned $http_code"
        if [ -n "$body" ]; then
            echo "   Error: $(echo "$body" | head -c 200)"
        fi
        return 1
    fi
}

# Wait for service to be ready
wait_for_service() {
    local url=$1
    local name=$2
    local max_attempts=${3:-60}
    
    log_info "Waiting for $name to be ready..."
    for i in $(seq 1 $max_attempts); do
        if curl -s "$url" > /dev/null 2>&1; then
            log_success "$name is ready"
            return 0
        fi
        sleep 2
        if [ $((i % 10)) -eq 0 ]; then
            printf "\n   Still waiting... ($i/$max_attempts)\n"
        else
            printf "."
        fi
    done
    echo ""
    log_error "$name failed to start after $((max_attempts * 2)) seconds"
    return 1
}

# Check if service is running
check_service_running() {
    local port=$1
    local name=$2
    
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        log_success "$name is running on port $port"
        return 0
    else
        log_warn "$name is not running on port $port"
        return 1
    fi
}

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 STARTING ALL ENDPOINTS AND TESTING"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Step 1: Check external services
log_info "Step 1: Checking external services..."

# Check Qdrant
if wait_for_service "http://localhost:$QDRANT_PORT/collections" "Qdrant" 10; then
    test_endpoint "http://localhost:$QDRANT_PORT/collections" "Qdrant Collections API"
fi

# Check vLLM (wait longer, it takes time to load)
log_info "Waiting for vLLM to load (this may take 2-5 minutes)..."
if wait_for_service "http://localhost:$VLLM_PORT/v1/models" "vLLM" 120; then
    test_endpoint "http://localhost:$VLLM_PORT/v1/models" "vLLM Models API"
fi

# Step 2: Start main pipeline server
echo ""
log_info "Step 2: Starting main pipeline server (health endpoints on port $HEALTH_PORT)..."

if check_service_running $HEALTH_PORT "Main Pipeline Server"; then
    log_warn "Main pipeline server already running"
else
    log_info "Launching main pipeline server..."
    cd "$ROOT_DIR"
    cargo run --bin niodoo_real_integrated --features svc > /tmp/niodoo_main.log 2>&1 &
    MAIN_PID=$!
    echo $MAIN_PID > /tmp/niodoo_main.pid
    log_info "Main pipeline server started (PID: $MAIN_PID)"
    log_info "Logs: tail -f /tmp/niodoo_main.log"
    log_info "Waiting for compilation to finish (this may take a few minutes)..."
    
    # Wait for server to start
    sleep 5
    if wait_for_service "http://localhost:$HEALTH_PORT/health" "Main Pipeline Server" 30; then
        log_success "Main pipeline server is ready"
    fi
fi

# Step 3: Start RL server
echo ""
log_info "Step 3: Starting RL server (port $RL_SERVER_PORT)..."

if check_service_running $RL_SERVER_PORT "RL Server"; then
    log_warn "RL server already running"
else
    log_info "Launching RL server..."
    cd "$ROOT_DIR"
    cargo run --bin rl_server --features svc > /tmp/niodoo_rl.log 2>&1 &
    RL_PID=$!
    echo $RL_PID > /tmp/niodoo_rl.pid
    log_info "RL server started (PID: $RL_PID)"
    log_info "Logs: tail -f /tmp/niodoo_rl.log"
    
    # Wait for server to start
    sleep 5
    if wait_for_service "http://localhost:$RL_SERVER_PORT/health" "RL Server" 30; then
        log_success "RL server is ready"
    fi
fi

# Step 4: Test all endpoints
echo ""
log_info "Step 4: Testing all HTTP endpoints..."

TESTS_PASSED=0
TESTS_FAILED=0

# Main pipeline endpoints
if test_endpoint "http://localhost:$HEALTH_PORT/health" "Health Check"; then
    TESTS_PASSED=$((TESTS_PASSED + 1))
else
    TESTS_FAILED=$((TESTS_FAILED + 1))
fi

if test_endpoint "http://localhost:$HEALTH_PORT/ready" "Readiness Check"; then
    TESTS_PASSED=$((TESTS_PASSED + 1))
else
    TESTS_FAILED=$((TESTS_FAILED + 1))
fi

if test_endpoint "http://localhost:$HEALTH_PORT/metrics" "Prometheus Metrics"; then
    TESTS_PASSED=$((TESTS_PASSED + 1))
else
    TESTS_FAILED=$((TESTS_FAILED + 1))
fi

# RL server endpoints
if test_endpoint "http://localhost:$RL_SERVER_PORT/health" "RL Server Health"; then
    TESTS_PASSED=$((TESTS_PASSED + 1))
else
    TESTS_FAILED=$((TESTS_FAILED + 1))
fi

EVAL_PAYLOAD='{"code":"def add(a, b):\n    return a + b","language":"python","problem":{"id":"test_1","description":"Add two numbers","test_cases":["add(1, 2) == 3"]}}'
if test_endpoint "http://localhost:$RL_SERVER_PORT/rl/evaluate" "RL Evaluate" "POST" "$EVAL_PAYLOAD"; then
    TESTS_PASSED=$((TESTS_PASSED + 1))
else
    TESTS_FAILED=$((TESTS_FAILED + 1))
fi

# Step 5: Run end-to-end pipeline tests
echo ""
log_info "Step 5: Running end-to-end pipeline tests..."

if [ -f "$ROOT_DIR/niodoo-ai/scripts/test_pipeline_e2e.py" ]; then
    log_info "Running Python E2E test script..."
    cd "$ROOT_DIR"
    if python3 niodoo-ai/scripts/test_pipeline_e2e.py --timeout 180 2>&1; then
        log_success "E2E pipeline tests passed"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        log_error "E2E pipeline tests failed"
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
else
    log_warn "E2E test script not found, skipping"
fi

# Summary
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 TEST SUMMARY"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "✅ Tests Passed: $TESTS_PASSED"
echo "❌ Tests Failed: $TESTS_FAILED"
echo ""

if [ $TESTS_FAILED -gt 0 ]; then
    echo "Available endpoints:"
    echo ""
    echo "Main Pipeline Server (port $HEALTH_PORT):"
    echo "  - GET  /health    - Health check"
    echo "  - GET  /ready     - Readiness probe"
    echo "  - GET  /metrics   - Prometheus metrics"
    echo ""
    echo "RL Server (port $RL_SERVER_PORT):"
    echo "  - GET  /health         - Health check"
    echo "  - POST /rl/evaluate    - Code evaluation"
    echo ""
    echo "External Services:"
    echo "  - vLLM: http://localhost:$VLLM_PORT/v1/models"
    echo "  - Qdrant: http://localhost:$QDRANT_PORT/collections"
    echo ""
    echo "Check logs:"
    echo "  - Main server: tail -f /tmp/niodoo_main.log"
    echo "  - RL server: tail -f /tmp/niodoo_rl.log"
    echo "  - vLLM: tail -f /tmp/vllm_service.log"
    exit 1
else
    log_success "🎉 All endpoints tested successfully!"
    echo ""
    echo "Available endpoints:"
    echo ""
    echo "Main Pipeline Server (port $HEALTH_PORT):"
    echo "  - GET  /health    - Health check"
    echo "  - GET  /ready     - Readiness probe"
    echo "  - GET  /metrics   - Prometheus metrics"
    echo ""
    echo "RL Server (port $RL_SERVER_PORT):"
    echo "  - GET  /health         - Health check"
    echo "  - POST /rl/evaluate    - Code evaluation"
    echo ""
    echo "External Services:"
    echo "  - vLLM: http://localhost:$VLLM_PORT/v1/models"
    echo "  - Qdrant: http://localhost:$QDRANT_PORT/collections"
    exit 0
fi

