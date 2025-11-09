#!/usr/bin/env bash

set -euo pipefail

ROOT="/workspace/Niodoo-Final"
PID_DIR="$ROOT/.service_pids"
LOG_DIR="$ROOT/logs"
mkdir -p "$PID_DIR" "$LOG_DIR"

# Colors
BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}➜${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
}

log_error() {
    echo -e "${RED}✖${NC} $1"
}

log_section() {
    echo -e "\n${BOLD}${BLUE}==> $1${NC}\n"
}

# Load environment
if [[ -f "$ROOT/tcs_runtime.env" ]]; then
    set -a
    source "$ROOT/tcs_runtime.env"
    set +a
fi

# Service configuration
QDRANT_BIN="${QDRANT_ROOT:-/workspace/qdrant}/qdrant"
QDRANT_CONFIG="${QDRANT_CONFIG_DIR:-/workspace/qdrant_config}/config.yaml"
QDRANT_STORAGE="${QDRANT_STORAGE_PATH:-/workspace/qdrant_storage}"
QDRANT_PID_FILE="$PID_DIR/qdrant.pid"
QDRANT_LOG="$LOG_DIR/qdrant.log"

VLLM_MODEL="${VLLM_MODEL_ID:-${VLLM_MODEL_PATH:-/workspace/models/Qwen3-Coder}}"
VLLM_PORT="${VLLM_PORT:-5001}"
VLLM_ENDPOINT="${VLLM_ENDPOINT:-http://127.0.0.1:5001}"
VLLM_PID_FILE="$PID_DIR/vllm.pid"
VLLM_LOG="$LOG_DIR/vllm.log"

CURATOR_MODEL="${CURATOR_MODEL:-/workspace/models/Qwen2.5-Topology}"
CURATOR_VLLM_PORT="${CURATOR_VLLM_PORT:-5002}"
CURATOR_VLLM_ENDPOINT="${CURATOR_VLLM_ENDPOINT:-http://127.0.0.1:5002}"
CURATOR_PID_FILE="$PID_DIR/vllm_curator.pid"
CURATOR_LOG="$LOG_DIR/vllm_curator.log"

MAIN_PID_FILE="$PID_DIR/niodoo_main.pid"
MAIN_LOG="$LOG_DIR/niodoo_main.log"
MAIN_PORT="${NIODOO_HEALTH_PORT:-9090}"

# Health check function
check_http_health() {
    local name=$1
    local url=$2
    local attempts=${3:-30}
    local delay=${4:-5}

    if ! command -v curl >/dev/null 2>&1; then
        log_warn "curl not available; skipping $name health probe"
        return 0
    fi

    for ((i=1; i<=attempts; i++)); do
        if curl -fsS "$url" >/dev/null 2>&1; then
            log_info "$name healthy at $url"
            return 0
        fi
        if [ $((i % 5)) -eq 0 ]; then
            log_warn "$name not ready yet (attempt $i/$attempts)"
        fi
        sleep "$delay"
    done

    log_error "$name failed health checks at $url after $attempts attempts"
    return 1
}

# Check if process is running
is_running() {
    local pid_file=$1
    if [[ ! -f "$pid_file" ]]; then
        return 1
    fi
    local pid=$(cat "$pid_file" 2>/dev/null || echo "")
    if [[ -z "$pid" ]]; then
        return 1
    fi
    if kill -0 "$pid" 2>/dev/null; then
        return 0
    else
        rm -f "$pid_file"
        return 1
    fi
}

# Start Qdrant
start_qdrant() {
    if is_running "$QDRANT_PID_FILE"; then
        log_info "Qdrant already running (PID: $(cat "$QDRANT_PID_FILE"))"
        return 0
    fi

    log_section "Starting Qdrant"

    # Check if Qdrant binary exists
    if [[ ! -x "$QDRANT_BIN" ]]; then
        log_error "Qdrant binary not found at $QDRANT_BIN"
        log_info "Run bootstrap script first: scripts/runpod_bootstrap.sh"
        return 1
    fi

    # Ensure storage directory exists
    mkdir -p "$QDRANT_STORAGE" "$QDRANT_STORAGE/wal"

    # Start Qdrant
    log_info "Starting Qdrant server..."
    nohup "$QDRANT_BIN" --config-path "$QDRANT_CONFIG" > "$QDRANT_LOG" 2>&1 &
    local pid=$!
    echo "$pid" > "$QDRANT_PID_FILE"
    log_info "Qdrant started (PID: $pid)"
    log_info "Logs: tail -f $QDRANT_LOG"

    # Wait for Qdrant to be ready
    log_info "Waiting for Qdrant to initialize..."
    if check_http_health "Qdrant" "http://127.0.0.1:6333/health" 30 2; then
        log_info "Qdrant is ready!"
        return 0
    else
        log_error "Qdrant failed to start"
        return 1
    fi
}

# Start vLLM (Qwen 3 Coder)
start_vllm() {
    if is_running "$VLLM_PID_FILE"; then
        log_info "vLLM already running (PID: $(cat "$VLLM_PID_FILE"))"
        return 0
    fi

    log_section "Starting vLLM (Qwen 3 Coder)"

    # Check if model exists
    if [[ ! -d "$VLLM_MODEL" ]] && [[ ! -f "$VLLM_MODEL" ]]; then
        log_error "Model not found at $VLLM_MODEL"
        log_info "Set VLLM_MODEL_ID or VLLM_MODEL_PATH environment variable"
        return 1
    fi

    # Activate Python environment
    if [[ -f "$ROOT/venv/bin/activate" ]]; then
        source "$ROOT/venv/bin/activate"
    fi

    # Check if vLLM is installed
    if ! python3 -c "import vllm" 2>/dev/null; then
        log_error "vLLM not installed in Python environment"
        log_info "Install vLLM: pip install vllm"
        return 1
    fi

    # Build vLLM command
    local vllm_cmd=(
        python3 -m vllm.entrypoints.openai.api_server
        --model "$VLLM_MODEL"
        --host 127.0.0.1
        --port "$VLLM_PORT"
        --dtype "${VLLM_DTYPE:-bfloat16}"
        --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
        --max-model-len "${VLLM_MAX_MODEL_LEN:-32768}"
        --max-num-batched-tokens "${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}"
        --max-num-seqs "${VLLM_MAX_NUM_SEQS:-64}"
        --trust-remote-code
    )

    # Add optional flags
    [[ -n "${VLLM_ATTENTION_BACKEND:-}" ]] && vllm_cmd+=(--attention-backend "$VLLM_ATTENTION_BACKEND")
    [[ -n "${VLLM_KV_CACHE_DTYPE:-}" ]] && vllm_cmd+=(--kv-cache-dtype "$VLLM_KV_CACHE_DTYPE")
    [[ "${VLLM_USE_DEEP_GEMM:-0}" == "1" ]] && vllm_cmd+=(--enable-deep-gemm)
    [[ "${VLLM_ENABLE_CHUNKED_PREFILL:-0}" == "1" ]] && vllm_cmd+=(--enable-chunked-prefill)

    log_info "Starting vLLM server (model: $VLLM_MODEL, port: $VLLM_PORT)..."
    log_info "Command: ${vllm_cmd[*]}"
    
    nohup "${vllm_cmd[@]}" > "$VLLM_LOG" 2>&1 &
    local pid=$!
    echo "$pid" > "$VLLM_PID_FILE"
    log_info "vLLM started (PID: $pid)"
    log_info "Logs: tail -f $VLLM_LOG"

    # Wait for vLLM to load (this takes 2-5 minutes)
    log_info "Waiting for vLLM to load model (this may take 2-5 minutes)..."
    if check_http_health "vLLM" "${VLLM_ENDPOINT%/}/v1/models" 120 5; then
        log_info "vLLM is ready!"
        return 0
    else
        log_error "vLLM failed to start or model loading timed out"
        log_info "Check logs: tail -f $VLLM_LOG"
        return 1
    fi
}

# Start Curator vLLM (Qwen 2.5 Topology) - optional
start_curator_vllm() {
    # Skip if curator uses same endpoint as main vLLM
    if [[ "${CURATOR_VLLM_ENDPOINT:-}" == "$VLLM_ENDPOINT" ]] || [[ "${CURATOR_VLLM_PORT:-5002}" == "$VLLM_PORT" ]]; then
        log_info "Curator uses same vLLM instance (port $VLLM_PORT)"
        return 0
    fi

    if is_running "$CURATOR_PID_FILE"; then
        log_info "Curator vLLM already running (PID: $(cat "$CURATOR_PID_FILE"))"
        return 0
    fi

    log_section "Starting Curator vLLM (Qwen 2.5 Topology)"

    # Check if model exists
    if [[ ! -d "$CURATOR_MODEL" ]] && [[ ! -f "$CURATOR_MODEL" ]]; then
        log_warn "Curator model not found at $CURATOR_MODEL, skipping curator vLLM"
        return 0
    fi

    # Activate Python environment
    if [[ -f "$ROOT/venv/bin/activate" ]]; then
        source "$ROOT/venv/bin/activate"
    fi

    local curator_cmd=(
        python3 -m vllm.entrypoints.openai.api_server
        --model "$CURATOR_MODEL"
        --host 127.0.0.1
        --port "$CURATOR_VLLM_PORT"
        --dtype "${VLLM_DTYPE:-bfloat16}"
        --gpu-memory-utilization "${CURATOR_VLLM_GPU_MEMORY_UTILIZATION:-0.15}"
        --max-model-len "${CURATOR_VLLM_MAX_MODEL_LEN:-2048}"
        --max-num-batched-tokens "${CURATOR_VLLM_MAX_NUM_BATCHED_TOKENS:-4096}"
        --max-num-seqs "${CURATOR_VLLM_MAX_NUM_SEQS:-32}"
        --trust-remote-code
    )

    log_info "Starting Curator vLLM server (model: $CURATOR_MODEL, port: $CURATOR_VLLM_PORT)..."
    
    nohup "${curator_cmd[@]}" > "$CURATOR_LOG" 2>&1 &
    local pid=$!
    echo "$pid" > "$CURATOR_PID_FILE"
    log_info "Curator vLLM started (PID: $pid)"
    log_info "Logs: tail -f $CURATOR_LOG"

    # Wait for curator to load
    log_info "Waiting for Curator vLLM to load..."
    if check_http_health "Curator vLLM" "${CURATOR_VLLM_ENDPOINT%/}/v1/models" 60 5; then
        log_info "Curator vLLM is ready!"
        return 0
    else
        log_warn "Curator vLLM failed to start (non-critical)"
        return 0
    fi
}

# Start Main Pipeline Server
start_main() {
    if is_running "$MAIN_PID_FILE"; then
        log_info "Main pipeline server already running (PID: $(cat "$MAIN_PID_FILE"))"
        return 0
    fi

    log_section "Starting Main Pipeline Server"

    # Check if binary exists
    local binary="$ROOT/target/release/niodoo_real_integrated"
    if [[ ! -x "$binary" ]]; then
        log_error "Main pipeline binary not found at $binary"
        log_info "Build first: cargo build -p niodoo_real_integrated --release --features svc"
        return 1
    fi

    log_info "Starting main pipeline server (port: $MAIN_PORT)..."
    cd "$ROOT"
    nohup "$binary" > "$MAIN_LOG" 2>&1 &
    local pid=$!
    echo "$pid" > "$MAIN_PID_FILE"
    log_info "Main pipeline server started (PID: $pid)"
    log_info "Logs: tail -f $MAIN_LOG"

    # Wait for server to be ready
    log_info "Waiting for main pipeline server to initialize..."
    if check_http_health "Main Pipeline" "http://127.0.0.1:$MAIN_PORT/health" 60 2; then
        log_info "Main pipeline server is ready!"
        return 0
    else
        log_error "Main pipeline server failed to start"
        log_info "Check logs: tail -f $MAIN_LOG"
        return 1
    fi
}

# Stop service
stop_service() {
    local name=$1
    local pid_file=$2

    if ! is_running "$pid_file"; then
        log_info "$name is not running"
        return 0
    fi

    local pid=$(cat "$pid_file")
    log_info "Stopping $name (PID: $pid)..."
    
    # Try graceful shutdown first
    if kill -TERM "$pid" 2>/dev/null; then
        # Wait up to 10 seconds for graceful shutdown
        for i in {1..10}; do
            if ! kill -0 "$pid" 2>/dev/null; then
                log_info "$name stopped gracefully"
                rm -f "$pid_file"
                return 0
            fi
            sleep 1
        done
    fi

    # Force kill if still running
    if kill -0 "$pid" 2>/dev/null; then
        log_warn "Force killing $name..."
        kill -KILL "$pid" 2>/dev/null || true
        sleep 1
    fi

    rm -f "$pid_file"
    log_info "$name stopped"
}

# Start all services
start() {
    log_section "Starting NIODOO Services"

    # Start services in dependency order
    start_qdrant || {
        log_error "Failed to start Qdrant"
        return 1
    }

    start_vllm || {
        log_error "Failed to start vLLM"
        return 1
    }

    start_curator_vllm || {
        log_warn "Curator vLLM failed (non-critical)"
    }

    start_main || {
        log_error "Failed to start main pipeline server"
        return 1
    }

    log_section "All services started successfully"
    log_info "Services:"
    log_info "  - Qdrant: http://127.0.0.1:6333"
    log_info "  - vLLM: $VLLM_ENDPOINT"
    [[ -f "$CURATOR_PID_FILE" ]] && log_info "  - Curator vLLM: $CURATOR_VLLM_ENDPOINT"
    log_info "  - Main Pipeline: http://127.0.0.1:$MAIN_PORT"
}

# Stop all services
stop() {
    log_section "Stopping NIODOO Services"

    stop_service "Main Pipeline Server" "$MAIN_PID_FILE"
    stop_service "Curator vLLM" "$CURATOR_PID_FILE"
    stop_service "vLLM" "$VLLM_PID_FILE"
    stop_service "Qdrant" "$QDRANT_PID_FILE"

    log_section "All services stopped"
}

# Status check
status() {
    log_section "Service Status"

    if is_running "$QDRANT_PID_FILE"; then
        log_info "Qdrant: RUNNING (PID: $(cat "$QDRANT_PID_FILE"))"
    else
        log_warn "Qdrant: NOT RUNNING"
    fi

    if is_running "$VLLM_PID_FILE"; then
        log_info "vLLM: RUNNING (PID: $(cat "$VLLM_PID_FILE"))"
    else
        log_warn "vLLM: NOT RUNNING"
    fi

    if is_running "$CURATOR_PID_FILE"; then
        log_info "Curator vLLM: RUNNING (PID: $(cat "$CURATOR_PID_FILE"))"
    else
        log_info "Curator vLLM: NOT RUNNING (optional)"
    fi

    if is_running "$MAIN_PID_FILE"; then
        log_info "Main Pipeline: RUNNING (PID: $(cat "$MAIN_PID_FILE"))"
    else
        log_warn "Main Pipeline: NOT RUNNING"
    fi
}

# Main command handler
case "${1:-start}" in
    start)
        start
        ;;
    stop)
        stop
        ;;
    restart)
        stop
        sleep 2
        start
        ;;
    status)
        status
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status}"
        exit 1
        ;;
esac
