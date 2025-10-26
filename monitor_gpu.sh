#!/bin/bash
# Continuous GPU monitoring with nvidia-smi dmon
# Tracks utilization, memory, temperature in real-time

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MONITOR_DIR="./gpu_monitor_${TIMESTAMP}"
mkdir -p "${MONITOR_DIR}"

echo "Starting GPU monitoring at $(date)"
echo "Output: ${MONITOR_DIR}/gpu_stats.csv"

# CSV header
echo "timestamp,gpu_util,memory_used_mb,memory_total_mb,temp_c,power_w" > "${MONITOR_DIR}/gpu_stats.csv"

# Monitor with 1-second intervals
while true; do
    TIMESTAMP=$(date +%s)
    STATS=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits)
    
    echo "${TIMESTAMP},${STATS}" >> "${MONITOR_DIR}/gpu_stats.csv"
    
    # Watch for thermal throttling (temp > 80C)
    TEMP=$(echo "${STATS}" | cut -d',' -f4 | tr -d ' ')
    if [ "${TEMP}" -gt 80 ]; then
        echo "[WARNING] GPU temperature at ${TEMP}C" >> "${MONITOR_DIR}/thermal_warnings.log"
    fi
    
    sleep 1
done

