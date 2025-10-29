#!/bin/bash
# Quick verification that all endpoints are running and accessible

set -e

echo "🔍 VERIFYING ALL ENDPOINTS"
echo "=========================="
echo ""

# Qdrant
echo "✅ Qdrant (port 6333):"
if curl -s http://127.0.0.1:6333/collections/experiences > /dev/null 2>&1; then
    echo "   Status: RUNNING"
    VEC_SIZE=$(curl -s http://127.0.0.1:6333/collections/experiences | grep -o '"size":[0-9]*' | grep -o '[0-9]*')
    echo "   Vector Size: $VEC_SIZE"
else
    echo "   Status: NOT RUNNING"
    exit 1
fi

echo ""

# Ollama
echo "✅ Ollama (port 11434):"
if curl -s http://127.0.0.1:11434/api/tags > /dev/null 2>&1; then
    echo "   Status: RUNNING"
    if curl -s http://127.0.0.1:11434/api/tags | grep -q "qwen2"; then
        echo "   Model: qwen2:0.5b found"
    fi
else
    echo "   Status: NOT RUNNING"
    exit 1
fi

echo ""

# vLLM
echo "✅ vLLM (port 5001):"
if curl -s http://127.0.0.1:5001/v1/models > /dev/null 2>&1; then
    echo "   Status: RUNNING"
    MODEL=$(curl -s http://127.0.0.1:5001/v1/models | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4)
    echo "   Model: $MODEL"
else
    echo "   Status: NOT RUNNING"
    exit 1
fi

echo ""

# Metrics
echo "✅ Metrics (port 9093):"
if curl -s http://127.0.0.1:9093/metrics > /dev/null 2>&1; then
    echo "   Status: RUNNING"
    METRIC_COUNT=$(curl -s http://127.0.0.1:9093/metrics | grep -c "^# HELP" || echo "0")
    echo "   Metrics Exported: $METRIC_COUNT"
else
    echo "   Status: NOT RUNNING"
    exit 1
fi

echo ""
echo "=========================="
echo "✅ ALL ENDPOINTS RUNNING"
echo "=========================="
echo ""
echo "All tests can now run successfully!"
echo ""
echo "To restart all services:"
echo "  ./supervisor.sh restart"
echo ""
echo "To check status:"
echo "  ./supervisor.sh status"

