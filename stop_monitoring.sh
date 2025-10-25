#!/bin/bash
# Stop monitoring stack

echo "🛑 Stopping TCS Monitoring Stack..."

# Stop metrics server
if [ -f logs/metrics_server.pid ]; then
    PID=$(cat logs/metrics_server.pid)
    kill $PID 2>/dev/null && echo "✅ Stopped metrics server"
    rm logs/metrics_server.pid
fi

# Stop Docker services
if command -v docker &> /dev/null; then
    docker-compose down
    echo "✅ Stopped Docker services"
fi

echo "✅ Monitoring stopped"

