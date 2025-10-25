#!/bin/bash
# Quick script to check continual learning progress

echo "📊 NIODOO CONTINUAL LEARNING PROGRESS CHECK"
echo "==========================================="
echo ""

# Check if running
if pgrep -f "continual_learning_runner.sh" > /dev/null; then
    echo "✅ Continual learning is RUNNING"
    PID=$(pgrep -f "continual_learning_runner.sh")
    echo "   PID: $PID"
else
    echo "❌ Continual learning is NOT running"
fi
echo ""

# Get latest metrics
if [ -d "continual_logs" ]; then
    LATEST_METRICS=$(ls -t continual_logs/metrics_*.csv 2>/dev/null | head -1)
    if [ -f "$LATEST_METRICS" ]; then
        echo "📈 Latest Metrics File: $LATEST_METRICS"
        ITERATIONS=$(wc -l < "$LATEST_METRICS")
        echo "   Iterations completed: $((ITERATIONS - 1))"
        
        # Show last 5 iterations
        echo ""
        echo "🔍 Last 5 iterations:"
        echo "   Iter | Entropy | ROUGE | Quality | Latency | Memories"
        tail -5 "$LATEST_METRICS" | while IFS=',' read -r iter ts prompt ent rouge qual lat mem resp; do
            printf "   %4s | %7s | %5s | %7s | %7s | %8s\n" "$iter" "$ent" "$rouge" "$qual" "${lat}ms" "$mem"
        done
        
        # Calculate averages
        echo ""
        echo "📊 Averages (last 10):"
        AVG_ENTROPY=$(tail -10 "$LATEST_METRICS" | cut -d',' -f4 | awk '{sum+=$1; count++} END {if(count>0) print sum/count; else print "N/A"}')
        AVG_ROUGE=$(tail -10 "$LATEST_METRICS" | cut -d',' -f5 | awk '{sum+=$1; count++} END {if(count>0) print sum/count; else print "N/A"}')
        AVG_QUALITY=$(tail -10 "$LATEST_METRICS" | cut -d',' -f6 | awk '{sum+=$1; count++} END {if(count>0) print sum/count; else print "N/A"}')
        echo "   Entropy: $AVG_ENTROPY"
        echo "   ROUGE: $AVG_ROUGE"
        echo "   Quality: $AVG_QUALITY"
    fi
fi

# Check memory growth
echo ""
echo "🧠 Memory System:"
MEM_COUNT=$(curl -s http://localhost:6333/collections/experiences 2>/dev/null | jq -r '.result.points_count' 2>/dev/null || echo "Error")
echo "   Total memories stored: $MEM_COUNT"

# Show recent activity
echo ""
echo "📜 Recent Activity (last 20 lines):"
tail -20 continual_learning.out 2>/dev/null | grep -E "Iteration|Entropy|ROUGE|LEARNING DETECTED" || echo "   No output yet..."

echo ""
echo "💡 Tips:"
echo "   - Check back in 30-60 minutes for patterns"
echo "   - Look for entropy dropping over time (learning!)"
echo "   - Watch memory count growing"
echo "   - To stop: pkill -f continual_learning_runner.sh"
echo "   - Full logs in: continual_logs/"
