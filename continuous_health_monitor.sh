#!/bin/bash
# Continuous Health Monitoring System
# Monitors all services and records metrics over time

set -e

PROJECT_ROOT="/workspace/Niodoo-Final"
METRICS_DIR="$PROJECT_ROOT/results/health_metrics"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
METRICS_FILE="$METRICS_DIR/health_metrics_${TIMESTAMP}.csv"

mkdir -p "$METRICS_DIR"

# Create CSV header
echo "timestamp,service,status,response_time_ms,cpu_usage,memory_usage" > "$METRICS_FILE"

declare -A SERVICE_HEALTH_ENDPOINTS=(
    ["vllm"]="http://127.0.0.1:5001/v1/models"
    ["qdrant"]="http://127.0.0.1:6333/health"
    ["ollama"]="http://127.0.0.1:11434/api/tags"
    ["metrics"]="http://127.0.0.1:9093/metrics"
)

check_service_health() {
    local service=$1
    local endpoint="${SERVICE_HEALTH_ENDPOINTS[$service]}"
    
    local start_time=$(date +%s%N)
    local response=$(curl -s -w "%{http_code}" -o /dev/null "$endpoint" 2>/dev/null || echo "000")
    local end_time=$(date +%s%N)
    
    local response_time_ms=$(( (end_time - start_time) / 1000000 ))
    
    if [ "$response" = "200" ]; then
        echo "healthy,$response_time_ms"
    else
        echo "unhealthy,$response_time_ms"
    fi
}

record_metrics() {
    local timestamp=$(date +%Y-%m-%dT%H:%M:%S)
    
    for service in "${!SERVICE_HEALTH_ENDPOINTS[@]}"; do
        local health_result=$(check_service_health "$service")
        local status=$(echo "$health_result" | cut -d',' -f1)
        local response_time=$(echo "$health_result" | cut -d',' -f2)
        
        local cpu_usage=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1}')
        local memory_usage=$(free | grep Mem | awk '{printf("%.2f", $3/$2 * 100.0)}')
        
        echo "$timestamp,$service,$status,$response_time,$cpu_usage,$memory_usage" >> "$METRICS_FILE"
    done
}

print_metrics() {
    clear
    echo "════════════════════════════════════════════════════════════"
    echo "  Continuous Health Monitoring - $(date)"
    echo "════════════════════════════════════════════════════════════"
    echo ""
    
    for service in "${!SERVICE_HEALTH_ENDPOINTS[@]}"; do
        local health_result=$(check_service_health "$service")
        local status=$(echo "$health_result" | cut -d',' -f1)
        local response_time=$(echo "$health_result" | cut -d',' -f2)
        
        if [ "$status" = "healthy" ]; then
            echo "  ✓ $service: HEALTHY (${response_time}ms)"
        else
            echo "  ✗ $service: UNHEALTHY (${response_time}ms)"
        fi
    done
    
    echo ""
    echo "Metrics recorded to: $METRICS_FILE"
    echo "Press Ctrl+C to stop monitoring"
}

# Main monitoring loop
main() {
    echo "Starting continuous health monitoring..."
    echo "Metrics will be recorded to: $METRICS_FILE"
    echo ""
    
    while true; do
        print_metrics
        record_metrics
        sleep 10
    done
}

# Run main
main "$@"


