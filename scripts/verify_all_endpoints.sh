#!/bin/bash
# Verify all endpoints are online and responding correctly
# Used before running smoke tests and A/B tests

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
VLLM_CURATOR_PORT=${CURATOR_VLLM_PORT:-5002}
QDRANT_PORT=6333
QDRANT_GRPC_PORT=6334

# Status tracking
ALL_OK=true

log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
    ALL_OK=false
}

log_warn() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Check HTTP endpoint
check_endpoint() {
    local url=$1
    local name=$2
    local timeout=${3:-10}
    
    if curl -s --max-time "$timeout" "$url" > /dev/null 2>&1; then
        log_success "$name is responding"
        return 0
    else
        log_error "$name is not responding ($url)"
        return 1
    fi
}

# Check if port is listening
check_port() {
    local port=$1
    local name=$2
    
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        log_success "$name is listening on port $port"
        return 0
    else
        log_error "$name is not listening on port $port"
        return 1
    fi
}

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔍 VERIFYING ALL ENDPOINTS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# External Services
echo "📡 External Services:"
echo ""

# Qdrant HTTP
check_endpoint "http://127.0.0.1:$QDRANT_PORT/collections" "Qdrant HTTP (port $QDRANT_PORT)"

# Qdrant gRPC (check via HTTP health endpoint)
check_endpoint "http://127.0.0.1:$QDRANT_PORT/health" "Qdrant Health"

# vLLM Generation (Qwen 3 Coder - executor)
log_info "Checking vLLM Generation (Qwen 3 Coder - executor)..."
check_endpoint "http://127.0.0.1:$VLLM_PORT/v1/models" "vLLM Generation (port $VLLM_PORT)" 30

# vLLM Curator (Qwen 2.5 Topology) - optional, may use same port
log_info "Checking vLLM Curator (Qwen 2.5 Topology)..."
if check_endpoint "http://127.0.0.1:$VLLM_CURATOR_PORT/v1/models" "vLLM Curator (port $VLLM_CURATOR_PORT)" 10; then
    log_success "Curator using separate port $VLLM_CURATOR_PORT"
elif check_endpoint "http://127.0.0.1:$VLLM_PORT/v1/models" "vLLM Curator (shared port $VLLM_PORT)" 10; then
    log_warn "Curator using same port as generation ($VLLM_PORT)"
else
    log_warn "Curator endpoint not accessible (may be optional)"
fi

echo ""
echo "🔧 NIODOO Services:"
echo ""

# Main Pipeline Server
log_info "Checking Main Pipeline Server..."
check_port $HEALTH_PORT "Main Pipeline Server"
check_endpoint "http://localhost:$HEALTH_PORT/health" "Main Pipeline Health"
check_endpoint "http://localhost:$HEALTH_PORT/ready" "Main Pipeline Ready"
check_endpoint "http://localhost:$HEALTH_PORT/metrics" "Main Pipeline Metrics"

# RL Server
log_info "Checking RL Server..."
check_port $RL_SERVER_PORT "RL Server"
check_endpoint "http://localhost:$RL_SERVER_PORT/health" "RL Server Health"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 VERIFICATION SUMMARY"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ "$ALL_OK" = true ]; then
    log_success "All critical endpoints are online!"
    echo ""
    echo "Ready for smoke testing and A/B tests."
    exit 0
else
    log_error "Some endpoints are not responding!"
    echo ""
    echo "Please ensure all services are started:"
    echo "  - Qdrant: docker run -d -p $QDRANT_PORT:$QDRANT_PORT -p $QDRANT_GRPC_PORT:$QDRANT_GRPC_PORT qdrant/qdrant"
    echo "  - vLLM Generation: python3 -m vllm.entrypoints.openai.api_server --model <qwen3-model> --port $VLLM_PORT"
    echo "  - vLLM Curator: python3 -m vllm.entrypoints.openai.api_server --model <qwen2.5-model> --port $VLLM_CURATOR_PORT"
    echo "  - Main Pipeline: cargo run -p niodoo_real_integrated --release --features svc"
    echo "  - RL Server: cargo run --bin rl_server --features svc"
    exit 1
fi

