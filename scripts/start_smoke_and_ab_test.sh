#!/bin/bash
# Master script: Start all endpoints, smoke test them, then run topology A/B test
# This script ensures all services are online and functional before running tests

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

log_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
log_success() { echo -e "${GREEN}✅ $1${NC}"; }
log_error() { echo -e "${RED}❌ $1${NC}"; }
log_warn() { echo -e "${YELLOW}⚠️  $1${NC}"; }

# Source environment
if [ -f "$ROOT_DIR/tcs_runtime.env" ]; then
    source "$ROOT_DIR/tcs_runtime.env"
fi

# Ports
QDRANT_PORT=6333
VLLM_PORT=${VLLM_PORT:-5001}
CURATOR_VLLM_PORT=${CURATOR_VLLM_PORT:-5002}
HEALTH_PORT=${NIODOO_HEALTH_PORT:-9090}
RL_SERVER_PORT=8080

# Model paths
VLLM_MODEL_ID=${VLLM_MODEL_ID:-/workspace/models/Qwen3-Coder}
CURATOR_MODEL=${CURATOR_MODEL:-/workspace/models/Qwen2.5-7B-Instruct-AWQ}

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 MASTER TEST PLAN: START → SMOKE → A/B TEST"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ============================================================================
# PHASE 1: START ALL ENDPOINTS
# ============================================================================
log_info "PHASE 1: Starting all endpoints..."

# 1. Qdrant
log_info "Starting Qdrant..."
if curl -s "http://127.0.0.1:$QDRANT_PORT/collections" > /dev/null 2>&1; then
    log_success "Qdrant already running"
else
    log_info "Starting Qdrant via Docker..."
    docker run -d --name qdrant --restart unless-stopped \
        -p $QDRANT_PORT:$QDRANT_PORT -p 6334:6334 \
        -v "$ROOT_DIR/qdrant_storage:/qdrant/storage" \
        qdrant/qdrant:latest 2>&1 || log_warn "Qdrant may already exist"
    
    for i in {1..30}; do
        sleep 2
        if curl -s "http://127.0.0.1:$QDRANT_PORT/collections" > /dev/null 2>&1; then
            log_success "Qdrant is ready!"
            break
        fi
        [ $i -eq 30 ] && log_error "Qdrant failed to start"
    done
fi

# 2. vLLM Generation (Qwen 3 Coder)
log_info "Starting vLLM Generation (port $VLLM_PORT)..."
if curl -s "http://127.0.0.1:$VLLM_PORT/v1/models" > /dev/null 2>&1; then
    log_success "vLLM Generation already running"
else
    if [ ! -d "$VLLM_MODEL_ID" ] && [ ! -f "$VLLM_MODEL_ID" ]; then
        log_error "Model not found: $VLLM_MODEL_ID"
        exit 1
    fi
    
    log_info "Starting vLLM server (this takes 2-5 minutes)..."
    if [ -f "$ROOT_DIR/venv/bin/activate" ]; then
        source "$ROOT_DIR/venv/bin/activate"
    fi
    
    nohup python3 -m vllm.entrypoints.openai.api_server \
        --model "$VLLM_MODEL_ID" \
        --host 127.0.0.1 \
        --port $VLLM_PORT \
        --dtype bfloat16 \
        --gpu-memory-utilization 0.85 \
        --max-model-len 32768 \
        --max-num-batched-tokens 8192 \
        --max-num-seqs 64 \
        --trust-remote-code \
        > /tmp/vllm_generation.log 2>&1 &
    
    log_info "Waiting for vLLM to load..."
    for i in {1..120}; do
        sleep 5
        if curl -s "http://127.0.0.1:$VLLM_PORT/v1/models" > /dev/null 2>&1; then
            log_success "vLLM Generation is ready!"
            break
        fi
        [ $((i % 10)) -eq 0 ] && log_info "Still loading... ($i/120)"
    done
fi

# 3. vLLM Curator (Qwen 2.5 Topology)
log_info "Starting vLLM Curator (port $CURATOR_VLLM_PORT)..."
if curl -s "http://127.0.0.1:$CURATOR_VLLM_PORT/v1/models" > /dev/null 2>&1; then
    log_success "vLLM Curator already running"
elif [ -d "$CURATOR_MODEL" ] || [ -f "$CURATOR_MODEL" ]; then
    log_info "Starting Curator vLLM server..."
    nohup python3 -m vllm.entrypoints.openai.api_server \
        --model "$CURATOR_MODEL" \
        --host 127.0.0.1 \
        --port $CURATOR_VLLM_PORT \
        --dtype bfloat16 \
        --gpu-memory-utilization 0.15 \
        --max-model-len 2048 \
        --max-num-batched-tokens 4096 \
        --max-num-seqs 32 \
        --trust-remote-code \
        > /tmp/vllm_curator.log 2>&1 &
    
    log_info "Waiting for Curator to load..."
    for i in {1..60}; do
        sleep 5
        if curl -s "http://127.0.0.1:$CURATOR_VLLM_PORT/v1/models" > /dev/null 2>&1; then
            log_success "vLLM Curator is ready!"
            break
        fi
    done
else
    log_warn "Curator model not found, skipping (will use same port as generation)"
fi

# 4. Main Pipeline Server (optional - A/B test creates its own instances)
log_info "Checking Main Pipeline Server (port $HEALTH_PORT)..."
if curl -s "http://localhost:$HEALTH_PORT/health" > /dev/null 2>&1; then
    log_success "Main Pipeline Server already running"
else
    log_warn "Main Pipeline Server not running (optional - A/B test creates own instances)"
fi

echo ""
log_success "All endpoints started!"
echo ""

# ============================================================================
# PHASE 2: SMOKE TESTS
# ============================================================================
log_info "PHASE 2: Running smoke tests on all endpoints..."

SMOKE_FAILED=0

check_endpoint() {
    local url=$1
    local name=$2
    local timeout=${3:-10}
    
    if curl -s --max-time "$timeout" "$url" > /dev/null 2>&1; then
        log_success "$name: OK"
        return 0
    else
        log_error "$name: FAILED"
        SMOKE_FAILED=$((SMOKE_FAILED + 1))
        return 1
    fi
}

# Smoke test all endpoints
check_endpoint "http://127.0.0.1:$QDRANT_PORT/collections" "Qdrant HTTP"
check_endpoint "http://127.0.0.1:$QDRANT_PORT/health" "Qdrant Health"
check_endpoint "http://127.0.0.1:$VLLM_PORT/v1/models" "vLLM Generation" 30
check_endpoint "http://127.0.0.1:$CURATOR_VLLM_PORT/v1/models" "vLLM Curator" 10 || \
    check_endpoint "http://127.0.0.1:$VLLM_PORT/v1/models" "vLLM Curator (shared)" 10
# Main pipeline server is optional (A/B test creates own instances)
if curl -s "http://localhost:$HEALTH_PORT/health" > /dev/null 2>&1; then
    check_endpoint "http://localhost:$HEALTH_PORT/health" "Main Pipeline Health"
    check_endpoint "http://localhost:$HEALTH_PORT/ready" "Main Pipeline Ready"
    check_endpoint "http://localhost:$HEALTH_PORT/metrics" "Main Pipeline Metrics"
else
    log_warn "Main Pipeline Server not running (skipping - A/B test creates own instances)"
fi

if [ $SMOKE_FAILED -gt 0 ]; then
    log_error "Smoke tests failed! Fix issues before running A/B test."
    exit 1
fi

echo ""
log_success "All smoke tests passed!"
echo ""

# ============================================================================
# PHASE 3: A/B TEST
# ============================================================================
log_info "PHASE 3: Running topology understanding A/B test..."

# Verify configs exist
BASELINE_CONFIG="$ROOT_DIR/configs/topology_enabled.json"
TREATMENT_CONFIG="$ROOT_DIR/configs/topology_disabled.json"

if [ ! -f "$BASELINE_CONFIG" ] || [ ! -f "$TREATMENT_CONFIG" ]; then
    log_error "A/B test configs not found!"
    exit 1
fi

# Run A/B test
OUTPUT_DIR="$ROOT_DIR/ab_test_results/topology_understanding_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

log_info "Running A/B test..."
log_info "  Baseline: Topology-Enabled"
log_info "  Treatment: Topology-Disabled"
log_info "  Concurrent users: ${CONCURRENT_USERS:-16}"
log_info "  Duration: ${DURATION_SECS:-120} seconds"
log_info "  Output: $OUTPUT_DIR"
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
    log_info "Results: $OUTPUT_DIR"
    echo ""
    log_info "Key metrics to check:"
    log_info "  - topology_impact (positive/negative/neutral/inconclusive)"
    log_info "  - persistence_entropy_difference"
    log_info "  - quality_difference_pct"
    log_info "  - beta_meta_difference"
else
    log_error "A/B test failed. Check logs: $OUTPUT_DIR/ab_test.log"
    exit 1
fi

echo ""
log_success "🎉 ALL TESTS COMPLETE!"
echo ""

