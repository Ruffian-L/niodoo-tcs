#!/bin/bash
# Master script to launch all experiments in parallel
# 1) Long harness run 2) Sweep 3) GPU monitoring

set -e

echo "=== Launching All Experiments ==="
echo "Timestamp: $(date)"

# Start GPU monitoring in background
echo "Starting GPU monitor..."
chmod +x monitor_gpu.sh
./monitor_gpu.sh &
MONITOR_PID=$!
echo "GPU monitor PID: ${MONITOR_PID}"

# Start 1000-iteration harness in background
echo "Starting 1000-iteration harness..."
chmod +x run_1000_iteration_harness.sh
nohup ./run_1000_iteration_harness.sh > harness_nohup.log 2>&1 &
HARNESS_PID=$!
echo "Harness PID: ${HARNESS_PID}"

# Wait a bit for harness to warm up
sleep 10

# Start topology sweep in background
echo "Starting topology stress sweep..."
chmod +x topology_stress_sweep.sh
nohup ./topology_stress_sweep.sh > sweep_nohup.log 2>&1 &
SWEEP_PID=$!
echo "Sweep PID: ${SWEEP_PID}"

echo ""
echo "=== All Experiments Running ==="
echo "Harness PID: ${HARNESS_PID}"
echo "Sweep PID: ${SWEEP_PID}"
echo "Monitor PID: ${MONITOR_PID}"
echo ""
echo "To check status:"
echo "  ps aux | grep -E '${HARNESS_PID}|${SWEEP_PID}|${MONITOR_PID}'"
echo ""
echo "To check GPU:"
echo "  nvidia-smi"
echo ""
echo "To tail logs:"
echo "  tail -f harness_nohup.log"
echo "  tail -f sweep_nohup.log"
echo ""
echo "To stop all:"
echo "  kill ${HARNESS_PID} ${SWEEP_PID} ${MONITOR_PID}"

