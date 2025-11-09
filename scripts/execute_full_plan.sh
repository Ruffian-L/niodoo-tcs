#!/bin/bash
# Master execution script: Get all endpoints online → Smoke test → A/B test
# This script executes the full plan to prove topology understanding

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR/.."
cd "$ROOT_DIR"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Ports
HEALTH_PORT=${NIODOO_HEALTH_PORT:-9090}
RL_SERVER_PORT=8080
VLLM_PORT=5001
VLLM_CURATOR_PORT=${CURATOR_VLLM_PORT:-5002}
QDRANT_PORT=6333
QDRANT_GRPC_PORT=6334

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

log_section() {
    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Wait for endpoint with timeout
wait_for_endpoint() {
    local url=$1
    local name=$2
    local max_attempts=${3:-60}
    local timeout=${4:-10}
    
    log_info "Waiting for $name to be ready..."
    for i in $(seq 1 $max_attempts); do
        if curl -s --max-time "$timeout" "$url" > /dev/null 2>&1; then
            log_success "$name is ready!"
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

# Check if port is listening
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Start Qdrant
start_qdrant() {
    log_section "PHASE 1.1: Starting Qdrant"
    
    if check_port $QDRANT_PORT; then
        log_success "Qdrant already running on port $QDRANT_PORT"
        return 0
    fi
    
    if ! command_exists docker; then
        log_error "Docker not found. Please install Docker first."
        return 1
    fi
    
    log_info "Starting Qdrant container..."
    docker run -d \
        --name qdrant \
        --restart unless-stopped \
        -p $QDRANT_PORT:$QDRANT_PORT \
        -p $QDRANT_GRPC_PORT:$QDRANT_GRPC_PORT \
        -v "$ROOT_DIR/qdrant_storage:/qdrant/storage" \
        qdrant/qdrant:latest 2>&1 | grep -v "already in use" || true
    
    wait_for_endpoint "http://127.0.0.1:$QDRANT_PORT/collections" "Qdrant" 30
}

# Start vLLM Generation
start_vllm_generation() {
    log_section "PHASE 1.2: Starting vLLM Generation (Qwen 3 Coder)"
    
    if check_port $VLLM_PORT; then
        log_success "vLLM Generation already running on port $VLLM_PORT"
        return 0
    fi
    
    if ! command_exists python3; then
        log_error "Python3 not found. Please install Python3 first."
        return 1
    fi
    
    export VLLM_MODEL_ID="${VLLM_MODEL_ID:-/workspace/models/Qwen3-Coder}"
    export VLLM_PORT=5001
    
    log_info "Starting vLLM Generation server..."
    log_info "Model: $VLLM_MODEL_ID"
    log_info "Port: $VLLM_PORT"
    log_warn "This may take 2-5 minutes to load..."
    
    # Activate venv if exists
    if [ -f "$ROOT_DIR/venv/bin/activate" ]; then
        source "$ROOT_DIR/venv/bin/activate"
    fi
    
    nohup python3 -m vllm.entrypoints.openai.api_server \
        --model "$VLLM_MODEL_ID" \
        --host 127.0.0.1 \
        --port ${VLLM_PORT} \
        --dtype bfloat16 \
        --gpu-memory-utilization 0.85 \
        --max-model-len 32768 \
        --trust-remote-code \
        > /tmp/vllm_coder.log 2>&1 &
    
    VLLM_PID=$!
    echo $VLLM_PID > /tmp/vllm_coder.pid
    log_info "vLLM Generation started (PID: $VLLM_PID)"
    log_info "Logs: tail -f /tmp/vllm_coder.log"
    
    wait_for_endpoint "http://127.0.0.1:$VLLM_PORT/v1/models" "vLLM Generation" 120 30
}

# Start vLLM Curator
start_vllm_curator() {
    log_section "PHASE 1.3: Starting vLLM Curator (Qwen 2.5 Topology)"
    
    if check_port $VLLM_CURATOR_PORT; then
        log_success "vLLM Curator already running on port $VLLM_CURATOR_PORT"
        return 0
    fi
    
    export CURATOR_MODEL="${CURATOR_MODEL:-/workspace/models/Qwen2.5-Topology}"
    export CURATOR_VLLM_PORT=5002
    
    log_info "Starting vLLM Curator server..."
    log_info "Model: $CURATOR_MODEL"
    log_info "Port: $VLLM_CURATOR_PORT"
    log_warn "This may take 2-5 minutes to load..."
    
    # Activate venv if exists
    if [ -f "$ROOT_DIR/venv/bin/activate" ]; then
        source "$ROOT_DIR/venv/bin/activate"
    fi
    
    nohup python3 -m vllm.entrypoints.openai.api_server \
        --model "$CURATOR_MODEL" \
        --host 127.0.0.1 \
        --port ${CURATOR_VLLM_PORT} \
        --dtype bfloat16 \
        --gpu-memory-utilization 0.15 \
        --max-model-len 2048 \
        --trust-remote-code \
        > /tmp/vllm_curator.log 2>&1 &
    
    CURATOR_PID=$!
    echo $CURATOR_PID > /tmp/vllm_curator.pid
    log_info "vLLM Curator started (PID: $CURATOR_PID)"
    log_info "Logs: tail -f /tmp/vllm_curator.log"
    
    wait_for_endpoint "http://127.0.0.1:$VLLM_CURATOR_PORT/v1/models" "vLLM Curator" 60 30
}

# Start Main Pipeline Server
start_main_pipeline() {
    log_section "PHASE 1.4: Starting Main Pipeline Server"
    
    if check_port $HEALTH_PORT; then
        log_success "Main Pipeline Server already running on port $HEALTH_PORT"
        return 0
    fi
    
    if ! command_exists cargo; then
        log_error "Cargo not found. Please install Rust first."
        return 1
    fi
    
    log_info "Building main pipeline server..."
    cd "$ROOT_DIR"
    
    # Source environment if exists
    if [ -f "$ROOT_DIR/tcs_runtime.env" ]; then
        source "$ROOT_DIR/tcs_runtime.env"
    fi
    
    # Build if needed
    if [ ! -f "$ROOT_DIR/target/release/niodoo_real_integrated" ]; then
        log_info "Building release binary (this may take a while)..."
        cargo build -p niodoo_real_integrated --release --features svc
    fi
    
    log_info "Starting main pipeline server..."
    nohup cargo run -p niodoo_real_integrated --release --features svc \
        > /tmp/niodoo_main.log 2>&1 &
    
    MAIN_PID=$!
    echo $MAIN_PID > /tmp/niodoo_main.pid
    log_info "Main Pipeline Server started (PID: $MAIN_PID)"
    log_info "Logs: tail -f /tmp/niodoo_main.log"
    
    wait_for_endpoint "http://localhost:$HEALTH_PORT/health" "Main Pipeline Server" 60
}

# Start RL Server
start_rl_server() {
    log_section "PHASE 1.5: Starting RL Server"
    
    if check_port $RL_SERVER_PORT; then
        log_success "RL Server already running on port $RL_SERVER_PORT"
        return 0
    fi
    
    log_info "Starting RL server..."
    cd "$ROOT_DIR"
    
    nohup cargo run --bin rl_server --release --features svc \
        > /tmp/niodoo_rl.log 2>&1 &
    
    RL_PID=$!
    echo $RL_PID > /tmp/niodoo_rl.pid
    log_info "RL Server started (PID: $RL_PID)"
    log_info "Logs: tail -f /tmp/niodoo_rl.log"
    
    wait_for_endpoint "http://localhost:$RL_SERVER_PORT/health" "RL Server" 30
}

# Phase 2: Smoke Test
run_smoke_tests() {
    log_section "PHASE 2: Running Smoke Tests"
    
    if [ -f "$ROOT_DIR/scripts/test_all_endpoints.sh" ]; then
        log_info "Running comprehensive smoke tests..."
        bash "$ROOT_DIR/scripts/test_all_endpoints.sh"
    elif [ -f "$ROOT_DIR/scripts/verify_all_endpoints.sh" ]; then
        log_info "Running endpoint verification..."
        bash "$ROOT_DIR/scripts/verify_all_endpoints.sh"
    else
        log_warn "No smoke test script found, running manual checks..."
        
        # Manual checks
        curl -s http://127.0.0.1:$QDRANT_PORT/collections > /dev/null && log_success "Qdrant OK" || log_error "Qdrant FAILED"
        curl -s http://127.0.0.1:$VLLM_PORT/v1/models > /dev/null && log_success "vLLM Generation OK" || log_error "vLLM Generation FAILED"
        curl -s http://127.0.0.1:$VLLM_CURATOR_PORT/v1/models > /dev/null && log_success "vLLM Curator OK" || log_error "vLLM Curator FAILED"
        curl -s http://localhost:$HEALTH_PORT/health > /dev/null && log_success "Main Pipeline OK" || log_error "Main Pipeline FAILED"
        curl -s http://localhost:$RL_SERVER_PORT/health > /dev/null && log_success "RL Server OK" || log_error "RL Server FAILED"
    fi
}

# Phase 3: A/B Test
run_ab_test() {
    log_section "PHASE 3: Running A/B Test - Proving Topology Understanding"
    
    if [ -f "$ROOT_DIR/scripts/run_topology_ab_test.sh" ]; then
        log_info "Running topology understanding A/B test..."
        bash "$ROOT_DIR/scripts/run_topology_ab_test.sh"
    else
        log_error "A/B test script not found: scripts/run_topology_ab_test.sh"
        return 1
    fi
}

# Main execution
main() {
    log_section "🚀 EXECUTING FULL PLAN: Endpoints → Smoke Tests → A/B Test"
    
    # Phase 1: Get all endpoints online
    log_section "PHASE 1: Getting All Endpoints Online"
    
    start_qdrant || log_warn "Qdrant startup had issues (may already be running)"
    start_vllm_generation || log_warn "vLLM Generation startup had issues (may already be running)"
    start_vllm_curator || log_warn "vLLM Curator startup had issues (may already be running)"
    start_main_pipeline || log_warn "Main Pipeline startup had issues (may already be running)"
    start_rl_server || log_warn "RL Server startup had issues (may already be running)"
    
    # Phase 2: Smoke test
    run_smoke_tests
    
    # Phase 3: A/B test
    run_ab_test
    
    log_section "✅ PLAN EXECUTION COMPLETE"
    log_success "Check results in: ab_test_results/topology_understanding/"
}

# Run main
main "$@"


