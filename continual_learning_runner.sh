#!/bin/bash
# Continual Learning Runner - Let it run while you hang with family!
# Come back to cool learning patterns and data

echo "🚀 NIODOO CONTINUAL LEARNING SESSION"
echo "====================================="
echo "Started at: $(date)"
echo "This will run continuously, generating learning data"
echo "Go hang with your family - check back later for cool patterns!"
echo ""

# Configuration
export TOKENIZER_JSON=/workspace/Niodoo-Final/models/tokenizer.json
export RUST_LOG=info
LOG_DIR="continual_logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SESSION_LOG="$LOG_DIR/session_${TIMESTAMP}.log"
METRICS_LOG="$LOG_DIR/metrics_${TIMESTAMP}.csv"
MEMORY_LOG="$LOG_DIR/memory_growth_${TIMESTAMP}.log"

# Create log directory
mkdir -p "$LOG_DIR"

# Initialize metrics CSV
echo "iteration,timestamp,prompt,entropy,rouge,quality,latency,memory_count,response_preview" > "$METRICS_LOG"

# Read problems from hard_problems.txt
PROBLEMS=(
    "Implement a balanced binary tree in Rust that handles 1M nodes efficiently"
    "Create a distributed consensus algorithm using Raft protocol"
    "Build a lock-free concurrent hash map with memory ordering"
    "Design a neural network from scratch in Rust with backpropagation"
    "Implement a JIT compiler for a simple language"
    "Create a real-time ray tracer with global illumination"
    "Build a database storage engine with B+ trees and WAL"
    "Implement TCP from scratch over raw sockets"
    "Create a garbage collector with generational collection"
    "Build a type inference system for a functional language"
)

echo "📚 Loaded ${#PROBLEMS[@]} complex problems for learning"
echo ""

# Function to get memory count
get_memory_count() {
    curl -s http://localhost:6333/collections/experiences | jq -r '.result.points_count' 2>/dev/null || echo "0"
}

# Function to run single iteration
run_iteration() {
    local iteration=$1
    local prompt="$2"
    local start_time=$(date +%s)
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🔄 Iteration $iteration - $(date +%H:%M:%S)"
    echo "📝 Prompt: ${prompt:0:60}..."
    
    # Run the pipeline
    OUTPUT=$(./target/release/niodoo_real_integrated --prompt "$prompt" 2>&1)
    
    # Extract metrics
    ENTROPY=$(echo "$OUTPUT" | grep -oP "entropy[=:]\s*\K[0-9.]+" | tail -1)
    ROUGE=$(echo "$OUTPUT" | grep -oP "rouge[=:]\s*\K[0-9.]+" | tail -1)
    QUALITY=$(echo "$OUTPUT" | grep -oP "quality.*\K[0-9.]+" | tail -1)
    LATENCY=$(echo "$OUTPUT" | grep -oP "latency_ms[=:]\s*\K[0-9.]+" | tail -1)
    
    # Get response preview (first 100 chars of baseline)
    RESPONSE=$(echo "$OUTPUT" | grep -A1 "Baseline:" | tail -1 | cut -c1-100)
    
    # Get memory count
    MEM_COUNT=$(get_memory_count)
    
    # Log to CSV
    echo "$iteration,$(date +%s),\"$prompt\",$ENTROPY,$ROUGE,$QUALITY,$LATENCY,$MEM_COUNT,\"$RESPONSE\"" >> "$METRICS_LOG"
    
    # Display progress
    echo "  📊 Entropy: $ENTROPY"
    echo "  📈 ROUGE: $ROUGE"
    echo "  ⭐ Quality: $QUALITY"
    echo "  ⏱️  Latency: ${LATENCY}ms"
    echo "  🧠 Memories: $MEM_COUNT"
    echo "  💬 Response: ${RESPONSE}..."
    
    # Check for learning patterns
    if [ "$iteration" -gt 10 ]; then
        # Compare with earlier iterations
        EARLY_ENTROPY=$(head -15 "$METRICS_LOG" | tail -5 | cut -d',' -f4 | awk '{sum+=$1} END {print sum/NR}')
        RECENT_ENTROPY=$(tail -5 "$METRICS_LOG" | cut -d',' -f4 | awk '{sum+=$1} END {print sum/NR}')
        
        if (( $(echo "$RECENT_ENTROPY < $EARLY_ENTROPY" | bc -l 2>/dev/null) )); then
            echo "  ✨ LEARNING DETECTED! Entropy dropping over time"
        fi
    fi
    
    # Save full output
    echo "$OUTPUT" >> "$SESSION_LOG"
    
    # Small delay between iterations
    sleep 5
}

# Main loop
echo "🎯 Starting continual learning loop..."
echo "Press Ctrl+C to stop (progress is saved)"
echo ""

ITERATION=1
while true; do
    # Cycle through problems
    for problem in "${PROBLEMS[@]}"; do
        run_iteration "$ITERATION" "$problem"
        ITERATION=$((ITERATION + 1))
        
        # Every 10 iterations, run a repeat to test memory
        if [ $((ITERATION % 10)) -eq 0 ]; then
            echo ""
            echo "🔁 MEMORY TEST - Repeating earlier problem..."
            run_iteration "$ITERATION" "${PROBLEMS[0]}"
            ITERATION=$((ITERATION + 1))
            echo "Check if this was faster/better than iteration 1!"
            echo ""
        fi
        
        # Every 20 iterations, show summary
        if [ $((ITERATION % 20)) -eq 0 ]; then
            echo ""
            echo "📊 SUMMARY after $ITERATION iterations:"
            echo "  Total memories: $(get_memory_count)"
            echo "  Avg entropy: $(tail -20 "$METRICS_LOG" | cut -d',' -f4 | awk '{sum+=$1} END {print sum/NR}')"
            echo "  Avg quality: $(tail -20 "$METRICS_LOG" | cut -d',' -f5 | awk '{sum+=$1} END {print sum/NR}')"
            echo "  Session log: $SESSION_LOG"
            echo "  Metrics CSV: $METRICS_LOG"
            echo ""
        fi
    done
done
