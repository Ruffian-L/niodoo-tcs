#!/bin/bash
# Unified Service Manager with Health Checks
# Idempotent service lifecycle management for RunPod bootstrap

set -euo pipefail

ROOT="/workspace/Niodoo-Final"
PROJECT_ROOT="${NIODOO_ROOT:-$ROOT}"
PID_DIR="/tmp/niodoo_services"
LOG_DIR="/tmp/niodoo_logs"
HEALTH_CHECK_INTERVAL="${HEALTH_CHECK_INTERVAL:-30}"

mkdir -p "$PID_DIR" "$LOG_DIR"

BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

print_header() {
    echo -e "\n${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}${BLUE}  $1${NC}"
    echo -e "${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
}

print_step() {
    echo -e "${BOLD}${GREEN}$(timestamp) ▶ $1${NC}"
}

print_success() {
    echo -e "${GREEN}$(timestamp) ✓ $1${NC}"
}

print_error() {
    echo -e "${RED}$(timestamp) ✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}$(timestamp) ⚠ $1${NC}"
}

HAS_CURL=1
if ! command -v curl >/dev/null 2>&1; then
    HAS_CURL=0
    print_warning "curl not found; HTTP health checks will be skipped"
fi

normalize_url() {
    local url=$1
    [[ -z "$url" ]] && return
    echo "${url%/}"
}

parse_host_port() {
    local url=$1
    local fallback_port=$2

    if [[ -z "$url" ]]; then
        echo "127.0.0.1:$fallback_port"
        return
    fi

    local hostport=$url
    if [[ "$hostport" == *"://"* ]]; then
        hostport=${hostport#*://}
    fi
    hostport=${hostport%%/*}

    local host=${hostport%%:*}
    local port=${hostport##*:}

    if [[ -z "$host" || "$host" == "$hostport" ]]; then
        host="$hostport"
    fi
    if [[ -z "$host" ]]; then
        host="127.0.0.1"
    fi
    if [[ "$hostport" == "$host" || -z "$port" || "$port" == "$hostport" ]]; then
        port=$fallback_port
    fi

    echo "$host:$port"
}

if [[ -f "$PROJECT_ROOT/tcs_runtime.env" ]]; then
    # shellcheck disable=SC1091
    set -a
    source "$PROJECT_ROOT/tcs_runtime.env"
    set +a
fi

VLLM_ENDPOINT=${VLLM_ENDPOINT:-http://127.0.0.1:5001}
QDRANT_URL=${QDRANT_URL:-http://127.0.0.1:6333}
OLLAMA_ENDPOINT=${OLLAMA_ENDPOINT:-http://127.0.0.1:11434}
METRICS_ENDPOINT=${METRICS_ENDPOINT:-http://127.0.0.1:9093/metrics}

VLLM_MODEL_PATH=${VLLM_MODEL:-/workspace/models/Qwen2.5-7B-Instruct-AWQ}
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.85}
VLLM_TRUST_REMOTE_CODE=${VLLM_TRUST_REMOTE_CODE:-1}
HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-0}

IFS=: read -r VLLM_HOST VLLM_PORT <<< "$(parse_host_port "$VLLM_ENDPOINT" 5001)"
IFS=: read -r QDRANT_HOST QDRANT_PORT <<< "$(parse_host_port "$QDRANT_URL" 6333)"
IFS=: read -r OLLAMA_HOST OLLAMA_PORT <<< "$(parse_host_port "$OLLAMA_ENDPOINT" 11434)"

QDRANT_ROOT=${QDRANT_ROOT:-/workspace/qdrant}
QDRANT_BIN=${QDRANT_BIN:-$QDRANT_ROOT/qdrant}
QDRANT_CONFIG_DIR=${QDRANT_CONFIG_DIR:-/workspace/qdrant_config}
QDRANT_CONFIG_PATH=${QDRANT_CONFIG_PATH:-$QDRANT_CONFIG_DIR/config.yaml}

OLLAMA_ROOT=${OLLAMA_ROOT:-/workspace/ollama}
if command -v ollama >/dev/null 2>&1; then
    OLLAMA_BIN="$(command -v ollama)"
elif [[ -x "$OLLAMA_ROOT/bin/ollama" ]]; then
    OLLAMA_BIN="$OLLAMA_ROOT/bin/ollama"
else
    OLLAMA_BIN=""
fi

declare -A SERVICES=()
declare -A SERVICE_PORTS=()
declare -A SERVICE_HEALTH_ENDPOINTS=()
declare -a SERVICE_ORDER=()

if [[ -x "$PROJECT_ROOT/venv/bin/python" ]]; then
    VLLM_CMD="cd \"$PROJECT_ROOT\" && source venv/bin/activate && export HF_HUB_ENABLE_HF_TRANSFER=$HF_HUB_ENABLE_HF_TRANSFER && vllm serve \"$VLLM_MODEL_PATH\" --host \"$VLLM_HOST\" --port \"$VLLM_PORT\" --gpu-memory-utilization \"$VLLM_GPU_MEMORY_UTILIZATION\""
    if [[ "$VLLM_TRUST_REMOTE_CODE" == "1" ]]; then
        VLLM_CMD="$VLLM_CMD --trust-remote-code"
    fi
    SERVICES["vllm"]="$VLLM_CMD"
    SERVICE_PORTS["vllm"]="$VLLM_PORT"
    SERVICE_HEALTH_ENDPOINTS["vllm"]="$(normalize_url "$VLLM_ENDPOINT")/v1/models"
    SERVICE_ORDER+=("vllm")
else
    print_warning "Python virtualenv missing; skipping vLLM service configuration"
fi

if [[ -x "$QDRANT_BIN" ]]; then
    SERVICES["qdrant"]="cd \"$QDRANT_ROOT\" && \"$QDRANT_BIN\" --config-path \"$QDRANT_CONFIG_PATH\""
    SERVICE_PORTS["qdrant"]="$QDRANT_PORT"
    SERVICE_HEALTH_ENDPOINTS["qdrant"]="$(normalize_url "$QDRANT_URL")/health"
    SERVICE_ORDER+=("qdrant")
else
    print_warning "Qdrant binary not found at $QDRANT_BIN; skipping"
fi

if [[ -n "${OLLAMA_BIN:-}" ]]; then
    SERVICES["ollama"]="cd /workspace && OLLAMA_HOST=${OLLAMA_HOST}:${OLLAMA_PORT} \"$OLLAMA_BIN\" serve"
    SERVICE_PORTS["ollama"]="$OLLAMA_PORT"
    SERVICE_HEALTH_ENDPOINTS["ollama"]="$(normalize_url "$OLLAMA_ENDPOINT")/api/tags"
    SERVICE_ORDER+=("ollama")
else
    print_warning "Ollama binary not available; skipping"
fi

if [[ "${ENABLE_METRICS:-1}" != "0" ]]; then
    if [[ -f "$HOME/.cargo/env" ]]; then
        SERVICES["metrics"]="cd \"$PROJECT_ROOT/tcs-core\" && source \"$HOME/.cargo/env\" && cargo run --bin metrics_server --release 2>&1"
        SERVICE_PORTS["metrics"]="${METRICS_PORT:-9093}"
        SERVICE_HEALTH_ENDPOINTS["metrics"]="$(normalize_url "$METRICS_ENDPOINT")"
        SERVICE_ORDER+=("metrics")
    else
        print_warning "Cargo environment missing; metrics server disabled"
    fi
fi

if [[ ${#SERVICE_ORDER[@]} -eq 0 ]]; then
    print_error "No services configured. Ensure dependencies are installed."
    exit 1
fi

start_service() {
    local service_name=$1
    local cmd=$2
    local pid_file="$PID_DIR/${service_name}.pid"
    local log_file="$LOG_DIR/${service_name}.log"

    if [[ -f "$pid_file" ]]; then
        local pid
        pid=$(cat "$pid_file")
        if ps -p "$pid" >/dev/null 2>&1; then
            print_warning "$service_name already running (PID: $pid)"
            return 0
        fi
    fi

    print_step "Starting $service_name"
    nohup bash -lc "$cmd" >"$log_file" 2>&1 &
    local pid=$!
    echo "$pid" >"$pid_file"

    sleep 3

    if ps -p "$pid" >/dev/null 2>&1; then
        print_success "$service_name started (PID: $pid)"
        return 0
    else
        print_error "$service_name failed to start"
        rm -f "$pid_file"
        return 1
    fi
}

stop_service() {
    local service_name=$1
    local pid_file="$PID_DIR/${service_name}.pid"

    if [[ -f "$pid_file" ]]; then
        local pid
        pid=$(cat "$pid_file")
        if ps -p "$pid" >/dev/null 2>&1; then
            print_step "Stopping $service_name (PID: $pid)"
            kill "$pid" 2>/dev/null || true
            sleep 2
            kill -9 "$pid" 2>/dev/null || true
            print_success "$service_name stopped"
        fi
        rm -f "$pid_file"
    fi
}

is_service_running() {
    local service_name=$1
    local pid_file="$PID_DIR/${service_name}.pid"

    if [[ ! -f "$pid_file" ]]; then
        return 1
    fi

    local pid
    pid=$(cat "$pid_file")
    if ps -p "$pid" >/dev/null 2>&1; then
        return 0
    fi
    return 1
}

health_check_service() {
    local service_name=$1
    local endpoint="${SERVICE_HEALTH_ENDPOINTS[$service_name]}"

    if [[ -z "$endpoint" ]]; then
        print_warning "$service_name has no health endpoint configured"
        return 0
    fi

    if [[ $HAS_CURL -eq 0 ]]; then
        print_warning "$service_name health check skipped (curl unavailable)"
        return 0
    fi

    if curl -fsS "$endpoint" >/dev/null 2>&1; then
        print_success "$service_name is healthy"
        return 0
    fi

    print_error "$service_name health check failed ($endpoint)"
    return 1
}

start_all() {
    print_header "Starting All Services"

    local failed=0

    for service in "${SERVICE_ORDER[@]}"; do
        local cmd="${SERVICES[$service]}"
        if [[ -z "$cmd" ]]; then
            print_warning "No command configured for $service; skipping"
            continue
        fi
        if ! start_service "$service" "$cmd"; then
            ((failed++))
        fi
    done

    if [[ $failed -eq 0 ]]; then
        print_success "All services started successfully"
    else
        print_error "$failed service(s) failed to start"
    fi

    return $failed
}

stop_all() {
    print_header "Stopping All Services"

    for service in "${SERVICE_ORDER[@]}"; do
        stop_service "$service"
    done

    print_success "All services stopped"
}

status_all() {
    print_header "Service Status"

    for service in "${SERVICE_ORDER[@]}"; do
        if is_service_running "$service"; then
            local pid
            pid=$(cat "$PID_DIR/${service}.pid")
            print_success "$service: Running (PID: $pid)"
        else
            print_error "$service: Not running"
        fi
    done
}

health_check_all() {
    print_header "Health Check All Services"

    local healthy=0
    local unhealthy=0

    for service in "${SERVICE_ORDER[@]}"; do
        if is_service_running "$service"; then
            if health_check_service "$service"; then
                ((healthy++))
            else
                ((unhealthy++))
            fi
        else
            print_warning "$service is not running"
            ((unhealthy++))
        fi
    done

    echo ""
    echo "Healthy: $healthy"
    echo "Unhealthy: $unhealthy"
}

restart_service() {
    local service_name=$1
    if [[ -z "${SERVICES[$service_name]:-}" ]]; then
        print_error "Unknown service: $service_name"
        exit 1
    fi

    print_header "Restarting $service_name"
    stop_service "$service_name"
    sleep 2
    start_service "$service_name" "${SERVICES[$service_name]}"
}

monitor() {
    print_header "Service Monitor"
    echo "Monitoring services every ${HEALTH_CHECK_INTERVAL}s..."
    echo "Press Ctrl+C to stop"

    while true; do
        sleep "$HEALTH_CHECK_INTERVAL"

        for service in "${SERVICE_ORDER[@]}"; do
            if ! is_service_running "$service"; then
                print_warning "$service is down, restarting"
                start_service "$service" "${SERVICES[$service]}"
            else
                if ! health_check_service "$service" >/dev/null 2>&1; then
                    print_warning "$service health check failed, restarting"
                    restart_service "$service"
                fi
            fi
        done
    done
}

case "${1:-}" in
    start)
        start_all
        ;;
    stop)
        stop_all
        ;;
    restart)
        stop_all
        sleep 2
        start_all
        ;;
    status)
        status_all
        ;;
    health)
        health_check_all
        ;;
    monitor)
        monitor
        ;;
    start-service)
        if [[ -z "${2:-}" ]]; then
            echo "Usage: $0 start-service <service_name>"
            exit 1
        fi
        srv=$2
        if [[ -z "${SERVICES[$srv]:-}" ]]; then
            print_error "Unknown service: $srv"
            exit 1
        fi
        start_service "$srv" "${SERVICES[$srv]}"
        ;;
    stop-service)
        if [[ -z "${2:-}" ]]; then
            echo "Usage: $0 stop-service <service_name>"
            exit 1
        fi
        stop_service "$2"
        ;;
    restart-service)
        if [[ -z "${2:-}" ]]; then
            echo "Usage: $0 restart-service <service_name>"
            exit 1
        fi
        restart_service "$2"
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|health|monitor|start-service|stop-service|restart-service}"
        echo ""
        echo "Commands:"
        echo "  start              - Start all services"
        echo "  stop               - Stop all services"
        echo "  restart            - Restart all services"
        echo "  status             - Show status of all services"
        echo "  health             - Run health checks"
        echo "  monitor            - Auto-monitor and restart services"
        echo "  start-service NAME - Start specific service"
        echo "  stop-service NAME  - Stop specific service"
        echo "  restart-service NAME - Restart specific service"
        echo ""
        echo "Available services: ${SERVICE_ORDER[*]}"
        exit 1
        ;;
esac

exit 0


