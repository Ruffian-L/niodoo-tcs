#!/bin/bash
# Start long experiments - harness and sweep side by side

set -e

echo "=== Starting Long Experiments ==="
echo "Timestamp: $(date)"

# Start harness with stable binary
echo "Starting harness (stable binary)..."
nohup cargo run -p niodoo_real_integrated --bin rut_gauntlet --release -- --output-dir logs/harness_long 2>&1 > harness.log &
HARNESS_PID=$!
echo "Harness PID: ${HARNESS_PID}"

# Wait a bit for initialization
sleep 5

# Start sweep with experimental binary  
echo "Starting topology sweep (experimental binary)..."
nohup ./topology_stress_sweep.sh 2>&1 > sweep.log &
SWEEP_PID=$!
echo "Sweep PID: ${SWEEP_PID}"

# Start GPU monitoring
echo "Starting GPU monitoring..."
nohup ./monitor_gpu.sh 2>&1 > gpu_monitor.log &
MONITOR_PID=$!
echo "Monitor PID: ${MONITOR_PID}"

echo ""
echo "=== All Experiments Running ==="
echo "Harness PID: ${HARNESS_PID}"
echo "Sweep PID: ${SWEEP_PID}"
echo "Monitor PID: ${MONITOR_PID}"
echo ""
echo "To check status:"
echo "  ps aux | grep -E '${HARNESS_PID}|${SWEEP_PID}|${MONITOR_PID}'"
echo ""
echo "To tail logs:"
echo "  tail -f harness.log"
echo "  tail -f sweep.log"
echo "  tail -f gpu_monitor.log"
echo ""
echo "To check GPU:"
echo "  nvidia-smi"
echo ""
echo "To stop all:"
echo "  kill ${HARNESS_PID} ${SWEEP_PID} ${MONITOR_PID}"

# Wait a bit and show initial GPU status
sleep 10
echo ""
echo "=== Initial GPU Status ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv

