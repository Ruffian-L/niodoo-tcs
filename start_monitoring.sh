#!/bin/bash
# Start full monitoring stack for TCS

set -e

echo "🚀 Starting TCS Monitoring Stack..."

# Check if Docker is available
if command -v docker &> /dev/null; then
    echo "✅ Docker found"
    docker ps &> /dev/null || echo "⚠️  Docker daemon not running - trying to start..."
    
    # Try to start Docker
    sudo service docker start 2>/dev/null || sudo systemctl start docker 2>/dev/null || echo "⚠️  Could not start Docker"
    
    # Start metrics collection
    echo "📊 Starting Prometheus and Grafana..."
    docker-compose up -d
    
    echo "⏳ Waiting for services to start..."
    sleep 5
    
    # Check if services are up
    if curl -s http://localhost:9090/-/healthy > /dev/null; then
        echo "✅ Prometheus is running at http://localhost:9090"
    else
        echo "⚠️  Prometheus not responding"
    fi
    
    if curl -s http://localhost:3000/api/health > /dev/null; then
        echo "✅ Grafana is running at http://localhost:3000"
    else
        echo "⚠️  Grafana not responding"
    fi
else
    echo "⚠️  Docker not found - using podman or direct binaries"
fi

# Start metrics server
echo "🔧 Building metrics server..."
cd tcs-core
cargo build --bin metrics_server --release 2>/dev/null || cargo build --bin metrics_server

echo "🚀 Starting metrics server on :9093..."
TCS_METRICS_PORT=9093 cargo run --bin metrics_server > ../logs/metrics_server.log 2>&1 &
METRICS_PID=$!
echo $METRICS_PID > ../logs/metrics_server.pid

sleep 2

# Check if metrics endpoint is up
if curl -s http://localhost:9093/metrics > /dev/null; then
    echo "✅ Metrics server is running at http://localhost:9093/metrics"
else
    echo "⚠️  Metrics server not responding"
fi

echo ""
echo "📊 Dashboard URLs:"
echo "  Simple: http://localhost:3000/d/simple"
echo "  Advanced: http://localhost:3000/d/advanced"
echo "  Metrics: http://localhost:9093/metrics"
echo "  Prometheus: http://localhost:9090"
echo ""
echo "To stop: ./stop_monitoring.sh"

