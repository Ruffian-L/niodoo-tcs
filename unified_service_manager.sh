#!/bin/bash
# Unified Service Manager with Health Checks
# Comprehensive service lifecycle management with automatic health monitoring

set -e

PROJECT_ROOT="/workspace/Niodoo-Final"
PID_DIR="/tmp/niodoo_services"
LOG_DIR="/tmp/niodoo_logs"
HEALTH_CHECK_INTERVAL=30

mkdir -p "$PID_DIR" "$LOG_DIR"

BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo -e "\n${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}${BLUE}  $1${NC}"
    echo -e "${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
}

print_step() {
    echo -e "${BOLD}${GREEN}▶ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Service configurations
declare -A SERVICES=(
    ["vllm"]="cd /workspace/Niodoo-Final && source venv/bin/activate && export HF_HUB_ENABLE_HF_TRANSFER=0 && vllm serve /workspace/models/Qwen2.5-7B-Instruct-AWQ --host 127.0.0.1 --port 5001 --gpu-memory-utilization 0.85 --trust-remote-code"
    ["qdrant"]="cd /workspace/qdrant && /workspace/qdrant/qdrant --config-path /workspace/qdrant_config/config.yaml"
    ["ollama"]="cd /workspace && OLLAMA_HOST=127.0.0.1:11434 /workspace/ollama/bin/ollama serve"
    ["metrics"]="cd /workspace/Niodoo-Final/tcs-core && cargo run --bin metrics_server --release 2>&1"
)

declare -A SERVICE_PORTS=(
    ["vllm"]="5001"
    ["qdrant"]="6333"
    ["ollama"]="11434"
    ["metrics"]="9093"
)

declare -A SERVICE_HEALTH_ENDPOINTS=(
    ["vllm"]="http://127.0.0.1:5001/v1/models"
    ["qdrant"]="http://127.0.0.1:6333/health"
    ["ollama"]="http://127.0.0.1:11434/api/tags"
    ["metrics"]="http://127.0.0.1:9093/metrics"
)

# Start a service
start_service() {
    local service_name=$1
    local cmd="${SERVICES[$service_name]}"
    local pid_file="$PID_DIR/${service_name}.pid"
    local log_file="$LOG_DIR/${service_name}.log"
    
    if [ -f "$pid_file" ]; then
        local old_pid=$(cat "$pid_file")
        if ps -p $old_pid > /dev/null 2>&1; then
            print_warning "$service_name already running (PID: $old_pid)"
            return 0
        fi
    fi
    
    print_step "Starting $service_name..."
    nohup bash -c "$cmd" > "$log_file" 2>&1 &
    local pid=$!
    echo $pid > "$pid_file"
    
    sleep 3
    
    if ps -p $pid > /dev/null 2>&1; then
        print_success "$service_name started (PID: $pid)"
        return 0
    else
        print_error "$service_name failed to start"
        rm -f "$pid_file"
        return 1
    fi
}

# Stop a service
stop_service() {
    local service_name=$1
    local pid_file="$PID_DIR/${service_name}.pid"
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if ps -p $pid > /dev/null 2>&1; then
            print_step "Stopping $service_name (PID: $pid)..."
            kill $pid 2>/dev/null || true
            sleep 2
            kill -9 $pid 2>/dev/null || true
            print_success "$service_name stopped"
        fi
        rm -f "$pid_file"
    fi
}

# Check if service is running
is_service_running() {
    local service_name=$1
    local pid_file="$PID_DIR/${service_name}.pid"
    
    if [ ! -f "$pid_file" ]; then
        return 1
    fi
    
    local pid=$(cat "$pid_file")
    if ps -p $pid > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Health check for a service
health_check_service() {
    local service_name=$1
    local endpoint="${SERVICE_HEALTH_ENDPOINTS[$service_name]}"
    
    if curl -s "$endpoint" > /dev/null 2>&1; then
        print_success "$service_name is healthy"
        return 0
    else
        print_error "$service_name health check failed"
        return 1
    fi
}

# Start all services
start_all() {
    print_header "Starting All Services"
    
    local failed=0
    
    for service in "${!SERVICES[@]}"; do
        if ! start_service "$service"; then
            ((failed++))
        fi
    done
    
    if [ $failed -eq 0 ]; then
        print_success "All services started successfully"
    else
        print_error "$failed service(s) failed to start"
    fi
    
    return $failed
}

# Stop all services
stop_all() {
    print_header "Stopping All Services"
    
    for service in "${!SERVICES[@]}"; do
        stop_service "$service"
    done
    
    print_success "All services stopped"
}

# Status check
status_all() {
    print_header "Service Status"
    
    for service in "${!SERVICES[@]}"; do
        if is_service_running "$service"; then
            local pid=$(cat "$PID_DIR/${service}.pid")
            print_success "$service: Running (PID: $pid)"
        else
            print_error "$service: Not running"
        fi
    done
}

# Health check all services
health_check_all() {
    print_header "Health Check All Services"
    
    local healthy=0
    local unhealthy=0
    
    for service in "${!SERVICES[@]}"; do
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

# Restart a service
restart_service() {
    local service_name=$1
    print_header "Restarting $service_name"
    
    stop_service "$service_name"
    sleep 2
    start_service "$service_name"
}

# Monitor services
monitor() {
    print_header "Service Monitor"
    echo "Monitoring services every ${HEALTH_CHECK_INTERVAL}s..."
    echo "Press Ctrl+C to stop"
    
    while true; do
        sleep $HEALTH_CHECK_INTERVAL
        
        for service in "${!SERVICES[@]}"; do
            if ! is_service_running "$service"; then
                print_warning "$service is down, restarting..."
                start_service "$service"
            else
                if ! health_check_service "$service" > /dev/null 2>&1; then
                    print_warning "$service health check failed, restarting..."
                    restart_service "$service"
                fi
            fi
        done
    done
}

# Main menu
case "$1" in
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
        if [ -z "$2" ]; then
            echo "Usage: $0 start-service <service_name>"
            exit 1
        fi
        start_service "$2"
        ;;
    stop-service)
        if [ -z "$2" ]; then
            echo "Usage: $0 stop-service <service_name>"
            exit 1
        fi
        stop_service "$2"
        ;;
    restart-service)
        if [ -z "$2" ]; then
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
        echo "  health             - Run health checks on all services"
        echo "  monitor            - Monitor and auto-restart services"
        echo "  start-service NAME - Start a specific service"
        echo "  stop-service NAME  - Stop a specific service"
        echo "  restart-service NAME - Restart a specific service"
        echo ""
        echo "Available services: ${!SERVICES[@]}"
        exit 1
        ;;
esac

exit 0

