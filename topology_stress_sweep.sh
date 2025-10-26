#!/bin/bash
# Topology stress map: sweep parameters to find stabilization points
# CUDA-accelerated, runs 10-iteration probes per combination

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SWEEP_DIR="./sweep_${TIMESTAMP}"
mkdir -p "${SWEEP_DIR}"

# Parameter ranges
TORUS_SEEDS=(42 1337 9999 12345 54321)
ENTROPY_THRESHOLDS=(0.1 0.2 0.3 0.4 0.5)
PREDICTOR_WEIGHTS=(0.1 0.2 0.3 0.4 0.5)

PROBE_ITERATIONS=10

echo "Starting topology stress sweep at $(date)"
echo "Output: ${SWEEP_DIR}/sweep_results.csv"

# CSV header
echo "torus_seed,entropy_threshold,predictor_weight,avg_entropy,entropy_std,stabilized,avg_latency_ms" > "${SWEEP_DIR}/sweep_results.csv"

TOTAL_COMBOS=$((${#TORUS_SEEDS[@]} * ${#ENTROPY_THRESHOLDS[@]} * ${#PREDICTOR_WEIGHTS[@]}))
COMBO_COUNT=0

for seed in "${TORUS_SEEDS[@]}"; do
    for threshold in "${ENTROPY_THRESHOLDS[@]}"; do
        for weight in "${PREDICTOR_WEIGHTS[@]}"; do
            COMBO_COUNT=$((COMBO_COUNT + 1))
            echo "[${COMBO_COUNT}/${TOTAL_COMBOS}] seed=${seed}, threshold=${threshold}, weight=${weight}"
            
            # Run 10-iteration probe
            PROBE_LOG="${SWEEP_DIR}/probe_s${seed}_t${threshold}_w${weight}.log"
            
            ENTROPIES=()
            LATENCIES=()
            
            for iter in $(seq 1 ${PROBE_ITERATIONS}); do
                # Generate a test prompt with current parameters
                TEST_PROMPT="Test seed=${seed} threshold=${threshold} weight=${weight} iteration=${iter}"
                OUTPUT=$(./niodoo_real_integrated_experimental \
                    --prompt "${TEST_PROMPT}" \
                    2>&1 | tee -a "${PROBE_LOG}")
                
                # Extract entropy and latency from output
                ENTROPY=$(echo "${OUTPUT}" | grep -oP 'entropy[:\s]+\K[0-9.]+' | head -1 || echo "0")
                LATENCY=$(echo "${OUTPUT}" | grep -oP 'latency[:\s]+\K[0-9.]+' | head -1 || echo "0")
                
                ENTROPIES+=("${ENTROPY}")
                LATENCIES+=("${LATENCY}")
            done
            
            # Compute statistics
            AVG_ENTROPY=$(printf '%s\n' "${ENTROPIES[@]}" | awk '{sum+=$1; count++} END {print (count>0 ? sum/count : 0)}')
            ENTROPY_STD=$(printf '%s\n' "${ENTROPIES[@]}" | awk -v mean="${AVG_ENTROPY}" '{sum+=($1-mean)^2; count++} END {print (count>0 ? sqrt(sum/count) : 0)}')
            AVG_LATENCY=$(printf '%s\n' "${LATENCIES[@]}" | awk '{sum+=$1; count++} END {print (count>0 ? sum/count : 0)}')
            
            # Stabilized if entropy std is low (< 0.1)
            STABILIZED=$(( $(echo "${ENTROPY_STD} < 0.1" | bc -l) ))
            
            # Append to results
            echo "${seed},${threshold},${weight},${AVG_ENTROPY},${ENTROPY_STD},${STABILIZED},${AVG_LATENCY}" >> "${SWEEP_DIR}/sweep_results.csv"
            
            echo "  -> avg_entropy=${AVG_ENTROPY}, entropy_std=${ENTROPY_STD}, stabilized=${STABILIZED}"
        done
    done
done

echo "Sweep completed at $(date)"
echo "Results: ${SWEEP_DIR}/sweep_results.csv"

# Generate quick analysis
echo ""
echo "=== SWEEP ANALYSIS ==="
echo "Stabilized combinations:"
grep ",1," "${SWEEP_DIR}/sweep_results.csv" | cut -d',' -f1-3
echo ""
echo "Highest entropy combinations:"
sort -t',' -k4 -nr "${SWEEP_DIR}/sweep_results.csv" | head -5

