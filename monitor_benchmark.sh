#!/bin/bash
# Monitor topology benchmark progress

LOG_FILE="/tmp/topology_bench_64.log"
RESULTS_DIR="results/benchmarks/topology"

echo "=== Topology Benchmark Monitor ==="
echo ""

# Check if benchmark is running
if pgrep -f "topology_bench.*--cycles 64" > /dev/null; then
    echo "✓ Benchmark process is running"
else
    echo "✗ Benchmark process not found"
fi

echo ""
echo "=== Latest Progress ==="
if [ -f "$LOG_FILE" ]; then
    strings "$LOG_FILE" 2>/dev/null | grep -E "cycle=|Processing prompt|Benchmark complete" | tail -5
else
    echo "Log file not found: $LOG_FILE"
fi

echo ""
echo "=== Latest Results ==="
if [ -d "$RESULTS_DIR" ]; then
    LATEST=$(ls -t "$RESULTS_DIR"/*.json 2>/dev/null | head -1)
    if [ -n "$LATEST" ]; then
        echo "Latest result: $(basename $LATEST)"
        echo ""
        python3 << EOF
import json
import sys
try:
    with open('$LATEST') as f:
        d = json.load(f)
        s = d['summary']
        print(f"Cycles completed: {d['total_cycles']}")
        print(f"Generated: {d['generated_at']}")
        print(f"\nLatency:")
        print(f"  Baseline: {s['avg_latency_baseline_ms']:.1f} ms")
        print(f"  Hybrid:   {s['avg_latency_hybrid_ms']:.1f} ms")
        print(f"  Overhead: {((s['avg_latency_hybrid_ms'] / s['avg_latency_baseline_ms'] - 1) * 100):+.1f}%")
        print(f"\nTopology Metrics:")
        print(f"  Knot Complexity: baseline={s['avg_knot_complexity_baseline']:.2f}, hybrid={s['avg_knot_complexity_hybrid']:.2f}")
        print(f"  Persistence Entropy: baseline={s['avg_persistence_entropy_baseline']:.2f}, hybrid={s['avg_persistence_entropy_hybrid']:.2f}")
        print(f"  Spectral Gap: baseline={s['avg_spectral_gap_baseline']:.2f}, hybrid={s['avg_spectral_gap_hybrid']:.2f}")
except Exception as e:
    print(f"Error reading results: {e}")
EOF
    else
        echo "No results found yet"
    fi
else
    echo "Results directory not found: $RESULTS_DIR"
fi

echo ""
echo "To watch live: tail -f $LOG_FILE | grep -E 'cycle=|complete'"

