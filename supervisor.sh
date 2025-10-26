#!/bin/bash
# Supervisor script for vllm, qdrant, and ollama services
# This script monitors and restarts services if they crash

set -e

LOGDIR=/tmp
PIDFILE=$LOGDIR/supervisor.pid

# Function to start a service
start_service() {
    local service_name=$1
    local cmd=$2
    local logfile=$LOGDIR/${service_name}.log
    
    if [ -f "$LOGDIR/${service_name}.pid" ]; then
        local old_pid=$(cat "$LOGDIR/${service_name}.pid")
        if ps -p $old_pid > /dev/null 2>&1; then
            echo "Service $service_name already running (PID: $old_pid)"
            return 0
        fi
    fi
    
    echo "Starting $service_name..."
    nohup bash -c "$cmd" > "$logfile" 2>&1 &
    local pid=$!
    echo $pid > "$LOGDIR/${service_name}.pid"
    echo "Started $service_name with PID: $pid"
    sleep 2
    
    if ps -p $pid > /dev/null 2>&1; then
        echo "✅ $service_name started successfully"
        return 0
    else
        echo "❌ $service_name failed to start"
        return 1
    fi
}

# Function to stop a service
stop_service() {
    local service_name=$1
    local pidfile=$LOGDIR/${service_name}.pid
    
    if [ -f "$pidfile" ]; then
        local pid=$(cat "$pidfile")
        if ps -p $pid > /dev/null 2>&1; then
            echo "Stopping $service_name (PID: $pid)..."
            kill $pid
            wait $pid 2>/dev/null || true
        fi
        rm -f "$pidfile"
    fi
}

# Function to check if a service is running
check_service() {
    local service_name=$1
    local pidfile=$LOGDIR/${service_name}.pid
    
    if [ ! -f "$pidfile" ]; then
        return 1
    fi
    
    local pid=$(cat "$pidfile")
    if ps -p $pid > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Start all services
start_all() {
    echo "🚀 Starting all services..."
    
    # Start vLLM
    start_service "vllm" "cd /workspace/Niodoo-Final && source venv/bin/activate && vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ --host 127.0.0.1 --port 8000 --gpu-memory-utilization 0.85"
    
    # Start Qdrant
    start_service "qdrant" "cd /workspace/qdrant && /workspace/qdrant/qdrant --config-path /workspace/qdrant_config/config.yaml"
    
    # Start Ollama
    start_service "ollama" "cd /workspace && /workspace/ollama serve"
    
    echo "✅ All services started"
}

# Stop all services
stop_all() {
    echo "🛑 Stopping all services..."
    stop_service "vllm"
    stop_service "qdrant"
    stop_service "ollama"
    echo "✅ All services stopped"
}

# Monitor and restart services
monitor() {
    echo "👀 Starting supervisor monitor..."
    echo "Press Ctrl+C to stop"
    
    while true; do
        sleep 10
        
        # Check vLLM
        if ! check_service "vllm"; then
            echo "⚠️  vLLM down, restarting..."
            start_service "vllm" "cd /workspace/Niodoo-Final && source venv/bin/activate && vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ --host 127.0.0.1 --port 8000 --gpu-memory-utilization 0.85"
        fi
        
        # Check Qdrant
        if ! check_service "qdrant"; then
            echo "⚠️  Qdrant down, restarting..."
            start_service "qdrant" "cd /workspace/qdrant && /workspace/qdrant/qdrant --config-path /workspace/qdrant_config/config.yaml"
        fi
        
        # Check Ollama
        if ! check_service "ollama"; then
            echo "⚠️  Ollama down, restarting..."
            start_service "ollama" "cd /workspace && /workspace/ollama serve"
        fi
    done
}

# Status check
status() {
    echo "📊 Service Status:"
    echo "=================="
    
    if check_service "vllm"; then
        local pid=$(cat "$LOGDIR/vllm.pid")
        echo "✅ vLLM: Running (PID: $pid)"
    else
        echo "❌ vLLM: Not running"
    fi
    
    if check_service "qdrant"; then
        local pid=$(cat "$LOGDIR/qdrant.pid")
        echo "✅ Qdrant: Running (PID: $pid)"
    else
        echo "❌ Qdrant: Not running"
    fi
    
    if check_service "ollama"; then
        local pid=$(cat "$LOGDIR/ollama.pid")
        echo "✅ Ollama: Running (PID: $pid)"
    else
        echo "❌ Ollama: Not running"
    fi
}

# Main logic
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
        status
        ;;
    monitor)
        monitor
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|monitor}"
        exit 1
        ;;
esac

exit 0

