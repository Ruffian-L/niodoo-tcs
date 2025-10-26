#!/bin/bash
# 1000-iteration harness with rotating prompts
# Runs in background, streams metrics continuously

set -e

ITERATIONS=1000
PROMPT_ROTATION=100
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./harness_logs_${TIMESTAMP}"
METRICS_DIR="./metrics_${TIMESTAMP}"

mkdir -p "${LOG_DIR}"
mkdir -p "${METRICS_DIR}"

echo "Starting 1000-iteration harness at $(date)"
echo "Logs: ${LOG_DIR}"
echo "Metrics: ${METRICS_DIR}"

# Rotate prompts every ~100 runs
PROMPT_SEQUENCE=(
    "rut_gauntlet"
    "adversarial_code"
    "emotional_arcs"
    "rut_gauntlet"
    "adversarial_code"
    "emotional_arcs"
    "rut_gauntlet"
    "adversarial_code"
    "emotional_arcs"
    "rut_gauntlet"
)

PROMPT_INDEX=0
for i in $(seq 1 ${ITERATIONS}); do
    # Rotate prompt every PROMPT_ROTATION iterations
    if [ $((i % PROMPT_ROTATION)) -eq 1 ]; then
        PROMPT_TYPE="${PROMPT_SEQUENCE[$((PROMPT_INDEX % ${#PROMPT_SEQUENCE[@]}))]}"
        PROMPT_INDEX=$((PROMPT_INDEX + 1))
        echo "[$(date +%H:%M:%S)] Iteration ${i}: Switching to ${PROMPT_TYPE}"
    fi
    
    # Run single iteration using rut_gauntlet binary
    cargo run -p niodoo_real_integrated --bin rut_gauntlet --release -- \
        --output-dir "${METRICS_DIR}" \
        2>&1 | tee -a "${LOG_DIR}/iter_${i}.log"
    
    # Every 10 iterations, check GPU health
    if [ $((i % 10)) -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] GPU Status:" >> "${LOG_DIR}/gpu_health.log"
        nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader >> "${LOG_DIR}/gpu_health.log"
    fi
    
    # Every 50 iterations, dump cumulative metrics
    if [ $((i % 50)) -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] Iteration ${i}/${ITERATIONS} - Cumulative metrics:" >> "${LOG_DIR}/metrics_summary.log"
        find "${METRICS_DIR}" -name "*.prom" -exec cat {} \; | grep -E "entropy|latency|adjusted_params" >> "${LOG_DIR}/metrics_summary.log"
    fi
done

echo "Harness completed at $(date)" >> "${LOG_DIR}/completion.log"

