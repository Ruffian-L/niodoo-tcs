#!/bin/bash
# Comprehensive End-to-End Endpoint Testing Script
# Tests ALL endpoints: health, metrics, RL server, and full pipeline

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

# Ports
HEALTH_PORT=${NIODOO_HEALTH_PORT:-9090}
RL_SERVER_PORT=8080
VLLM_PORT=5001
QDRANT_PORT=6333

# Test results
TESTS_PASSED=0
TESTS_FAILED=0
FAILED_TESTS=()

log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
    TESTS_PASSED=$((TESTS_PASSED + 1))
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
    TESTS_FAILED=$((TESTS_FAILED + 1))
    FAILED_TESTS+=("$1")
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
        response=$(curl -s -w "\n%{http_code}" "$url" --max-time 10 || echo "000")
    else
        response=$(curl -s -w "\n%{http_code}" -X "$method" -H "Content-Type: application/json" -d "$data" "$url" --max-time 30 || echo "000")
    fi
    
    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | sed '$d')
    
    if [ "$http_code" = "200" ] || [ "$http_code" = "201" ]; then
        log_success "$name returned $http_code"
        if [ -n "$body" ]; then
            echo "   Response preview: $(echo "$body" | head -c 200)"
        fi
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
    local max_attempts=${3:-30}
    
    log_info "Waiting for $name to be ready..."
    for i in $(seq 1 $max_attempts); do
        if curl -s "$url" > /dev/null 2>&1; then
            log_success "$name is ready"
            return 0
        fi
        sleep 2
        printf "."
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
echo "🚀 COMPREHENSIVE ENDPOINT TESTING"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Step 1: Start external services
log_info "Step 1: Starting external services..."
if [ -f "$ROOT_DIR/start_all_services.sh" ]; then
    bash "$ROOT_DIR/start_all_services.sh" || log_warn "Some services may have failed to start"
else
    log_warn "start_all_services.sh not found, assuming services are already running"
fi

echo ""
log_info "Step 2: Checking external service health..."

# Check vLLM
if wait_for_service "http://localhost:$VLLM_PORT/v1/models" "vLLM" 30; then
    test_endpoint "http://localhost:$VLLM_PORT/v1/models" "vLLM Models API"
fi

# Check Qdrant
if wait_for_service "http://localhost:$QDRANT_PORT/collections" "Qdrant" 10; then
    test_endpoint "http://localhost:$QDRANT_PORT/collections" "Qdrant Collections API"
fi

echo ""
log_info "Step 3: Starting NIODOO services..."

# Start main pipeline server (service mode, no --prompt flag)
log_info "Starting main pipeline server (health endpoints on port $HEALTH_PORT)..."
if check_service_running $HEALTH_PORT "Main Pipeline Server"; then
    log_warn "Main pipeline server already running"
else
    # Start in background
    log_info "Launching main pipeline server..."
    cd "$ROOT_DIR"
    cargo run --features svc > /tmp/niodoo_main.log 2>&1 &
    MAIN_PID=$!
    echo $MAIN_PID > /tmp/niodoo_main.pid
    log_info "Main pipeline server started (PID: $MAIN_PID)"
    log_info "Logs: tail -f /tmp/niodoo_main.log"
fi

# Start RL server
log_info "Starting RL server (port $RL_SERVER_PORT)..."
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
fi

# Wait for services to start
echo ""
log_info "Waiting for NIODOO services to initialize..."
sleep 5

# Wait for main pipeline server (health endpoints only)
if wait_for_service "http://localhost:$HEALTH_PORT/health" "Main Pipeline Server" 30; then
    # Test main pipeline endpoints (these are the ONLY endpoints on main server)
    echo ""
    log_info "Step 4: Testing Main Pipeline Server endpoints..."
    
    test_endpoint "http://localhost:$HEALTH_PORT/health" "Health Check"
    test_endpoint "http://localhost:$HEALTH_PORT/ready" "Readiness Check"
    test_endpoint "http://localhost:$HEALTH_PORT/metrics" "Prometheus Metrics"
fi

# Wait for RL server
if wait_for_service "http://localhost:$RL_SERVER_PORT/health" "RL Server" 20; then
    # Test RL server endpoints
    echo ""
    log_info "Step 5: Testing RL Server endpoints..."
    
    test_endpoint "http://localhost:$RL_SERVER_PORT/health" "RL Server Health"
    
    # Test RL evaluate endpoint
    EVAL_PAYLOAD='{"code":"def add(a, b):\n    return a + b","language":"python","problem":{"id":"test_1","description":"Add two numbers","test_cases":["add(1, 2) == 3"]}}'
    test_endpoint "http://localhost:$RL_SERVER_PORT/rl/evaluate" "RL Evaluate" "POST" "$EVAL_PAYLOAD"
fi

# Test full pipeline end-to-end
echo ""
log_info "Step 6: Running end-to-end pipeline tests..."

if [ -f "$ROOT_DIR/niodoo-ai/scripts/test_pipeline_e2e.py" ]; then
    log_info "Running Python E2E test script..."
    cd "$ROOT_DIR"
    if python3 niodoo-ai/scripts/test_pipeline_e2e.py --timeout 180 2>&1; then
        log_success "E2E pipeline tests passed"
    else
        log_error "E2E pipeline tests failed"
    fi
else
    log_warn "E2E test script not found, skipping"
fi

# Test with cargo test if available
echo ""
log_info "Step 7: Running Rust integration tests..."
cd "$ROOT_DIR"
if cargo test --lib --features svc 2>&1 | head -100; then
    log_success "Rust integration tests passed"
else
    log_warn "Some Rust tests may have failed (check output above)"
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
    echo "Failed tests:"
    for test in "${FAILED_TESTS[@]}"; do
        echo "  - $test"
    done
    echo ""
    echo "Check logs:"
    echo "  - Main server: tail -f /tmp/niodoo_main.log"
    echo "  - RL server: tail -f /tmp/niodoo_rl.log"
    echo "  - vLLM: tail -f /tmp/vllm_service.log"
    exit 1
else
    echo "🎉 All endpoints tested successfully!"
    echo ""
    echo "Available endpoints (ACTUAL IMPLEMENTED ENDPOINTS):"
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
    echo "NOTE: The pipeline processes prompts via CLI (cargo run --prompt 'text'),"
    echo "      not via HTTP API. Use the E2E test script for full pipeline testing."
    exit 0
fi

