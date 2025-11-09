#!/bin/bash
# Start all endpoints required for smoke testing and A/B tests
# This script ensures all services are online before proceeding

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

# Source environment
if [ -f "$ROOT_DIR/tcs_runtime.env" ]; then
    source "$ROOT_DIR/tcs_runtime.env"
    log_info "Sourced runtime environment"
fi

# Ports
QDRANT_PORT=6333
QDRANT_GRPC_PORT=6334
VLLM_PORT=5001
VLLM_CURATOR_PORT=5002
MAIN_PIPELINE_PORT=9090
RL_SERVER_PORT=8080

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 STARTING ALL ENDPOINTS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Step 1: Start Qdrant
log_info "Step 1: Checking Qdrant (ports $QDRANT_PORT/$QDRANT_GRPC_PORT)..."
if curl -s --max-time 5 "http://127.0.0.1:$QDRANT_PORT/collections" > /dev/null 2>&1; then
    log_success "Qdrant is already running"
else
    log_info "Starting Qdrant..."
    docker run -d \
        --name qdrant \
        --restart unless-stopped \
        -p $QDRANT_PORT:$QDRANT_PORT \
        -p $QDRANT_GRPC_PORT:$QDRANT_GRPC_PORT \
        -v "$ROOT_DIR/qdrant_storage:/qdrant/storage" \
        qdrant/qdrant:latest 2>&1 || log_warn "Qdrant container may already exist"
    
    log_info "Waiting for Qdrant to initialize..."
    for i in {1..30}; do
        sleep 2
        if curl -s --max-time 5 "http://127.0.0.1:$QDRANT_PORT/collections" > /dev/null 2>&1; then
            log_success "Qdrant is ready!"
            break
        fi
        printf "."
    done
    echo ""
fi

# Step 2: Start vLLM Generation (port 5001)
log_info "Step 2: Checking vLLM Generation (port $VLLM_PORT)..."
if curl -s --max-time 10 "http://127.0.0.1:$VLLM_PORT/v1/models" > /dev/null 2>&1; then
    log_success "vLLM Generation is already running"
else
    log_info "Starting vLLM Generation..."
    
    # Check for Python venv
    if [ -f "$ROOT_DIR/venv/bin/activate" ]; then
        source "$ROOT_DIR/venv/bin/activate"
    fi
    
    # Set model path - use available model
    if [ -z "$VLLM_MODEL_ID" ]; then
        if [ -d "/workspace/models/Qwen2.5-7B-Instruct-AWQ" ]; then
            export VLLM_MODEL_ID="/workspace/models/Qwen2.5-7B-Instruct-AWQ"
        elif [ -d "/workspace/models/Qwen2.5-Coder-7B-Instruct" ]; then
            export VLLM_MODEL_ID="/workspace/models/Qwen2.5-Coder-7B-Instruct"
        else
            export VLLM_MODEL_ID="/workspace/models/Qwen2.5-0.5B-Instruct"
        fi
    fi
    export VLLM_PORT=$VLLM_PORT
    
    # Check if model exists
    if [ ! -d "$VLLM_MODEL_ID" ]; then
        log_error "Model not found: $VLLM_MODEL_ID"
        log_info "Available models:"
        ls -d /workspace/models/*/ 2>/dev/null | head -5
        exit 1
    fi
    
    log_info "Starting vLLM server with model: $VLLM_MODEL_ID"
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
        > /tmp/vllm_coder.log 2>&1 &
    
    VLLM_PID=$!
    echo $VLLM_PID > /tmp/vllm_coder.pid
    log_info "vLLM Generation started (PID: $VLLM_PID)"
    log_info "Logs: tail -f /tmp/vllm_coder.log"
    
    log_info "Waiting for vLLM to load (this may take 2-5 minutes)..."
    for i in {1..120}; do
        sleep 5
        if curl -s --max-time 10 "http://127.0.0.1:$VLLM_PORT/v1/models" > /dev/null 2>&1; then
            log_success "vLLM Generation is READY!"
            break
        fi
        if [ $((i % 10)) -eq 0 ]; then
            printf "\n   Still loading... ($i/120)\n"
        else
            printf "."
        fi
    done
    echo ""
fi

# Step 3: Start vLLM Curator (port 5002)
log_info "Step 3: Checking vLLM Curator (port $VLLM_CURATOR_PORT)..."
if curl -s --max-time 10 "http://127.0.0.1:$VLLM_CURATOR_PORT/v1/models" > /dev/null 2>&1; then
    log_success "vLLM Curator is already running"
else
    log_info "Starting vLLM Curator..."
    
    # Use same model or different model for curator
    export CURATOR_MODEL="${CURATOR_MODEL:-$VLLM_MODEL_ID}"
    export CURATOR_VLLM_PORT=$VLLM_CURATOR_PORT
    
    log_info "Starting curator vLLM server with model: $CURATOR_MODEL"
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
    
    CURATOR_PID=$!
    echo $CURATOR_PID > /tmp/vllm_curator.pid
    log_info "vLLM Curator started (PID: $CURATOR_PID)"
    log_info "Logs: tail -f /tmp/vllm_curator.log"
    
    log_info "Waiting for curator to load..."
    for i in {1..60}; do
        sleep 5
        if curl -s --max-time 10 "http://127.0.0.1:$CURATOR_VLLM_PORT/v1/models" > /dev/null 2>&1; then
            log_success "vLLM Curator is READY!"
            break
        fi
        printf "."
    done
    echo ""
fi

# Step 4: Start Main Pipeline Server
log_info "Step 4: Checking Main Pipeline Server (port $MAIN_PIPELINE_PORT)..."
if curl -s --max-time 5 "http://localhost:$MAIN_PIPELINE_PORT/health" > /dev/null 2>&1; then
    log_success "Main Pipeline Server is already running"
else
    log_info "Building Main Pipeline Server..."
    cd "$ROOT_DIR"
    cargo build -p niodoo_real_integrated --release --features svc 2>&1 | tail -5
    
    log_info "Starting Main Pipeline Server..."
    nohup cargo run -p niodoo_real_integrated --release --features svc \
        > /tmp/niodoo_main.log 2>&1 &
    
    MAIN_PID=$!
    echo $MAIN_PID > /tmp/niodoo_main.pid
    log_info "Main Pipeline Server started (PID: $MAIN_PID)"
    log_info "Logs: tail -f /tmp/niodoo_main.log"
    
    log_info "Waiting for server to start..."
    for i in {1..60}; do
        sleep 2
        if curl -s --max-time 5 "http://localhost:$MAIN_PIPELINE_PORT/health" > /dev/null 2>&1; then
            log_success "Main Pipeline Server is READY!"
            break
        fi
        printf "."
    done
    echo ""
fi

# Step 5: Start RL Server (optional)
log_info "Step 5: Checking RL Server (port $RL_SERVER_PORT)..."
if curl -s --max-time 5 "http://localhost:$RL_SERVER_PORT/health" > /dev/null 2>&1; then
    log_success "RL Server is already running"
else
    log_info "Building RL Server..."
    cd "$ROOT_DIR"
    cargo build --bin rl_server --release --features svc 2>&1 | tail -5
    
    log_info "Starting RL Server..."
    nohup cargo run --bin rl_server --release --features svc \
        > /tmp/niodoo_rl.log 2>&1 &
    
    RL_PID=$!
    echo $RL_PID > /tmp/niodoo_rl.pid
    log_info "RL Server started (PID: $RL_PID)"
    log_info "Logs: tail -f /tmp/niodoo_rl.log"
    
    log_info "Waiting for RL Server to start..."
    for i in {1..30}; do
        sleep 2
        if curl -s --max-time 5 "http://localhost:$RL_SERVER_PORT/health" > /dev/null 2>&1; then
            log_success "RL Server is READY!"
            break
        fi
        printf "."
    done
    echo ""
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ ALL ENDPOINTS STARTED"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
log_info "Next step: Run verify_all_endpoints.sh to verify all endpoints"
log_info "Then run smoke tests and A/B tests"
