#!/bin/bash
# H200 Million-Cycle NIODOO Test Runner
# Runs 1M pipeline cycles on NVIDIA H200 with monitoring

set -e

# Configuration
TEST_COUNT=${TEST_COUNT:-1000000}
WORKERS=${WORKERS:-128}
BATCH_SIZE=${BATCH_SIZE:-100}
Instances=${Instances:-100}
OUTPUT_DIR="logs/million_cycle_test_$(date +%Y%m%d_%H%M%S)"
HARDWARE_PROFILE="h200"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}🚀 NIODOO Million-Cycle Test on H200${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo "Configuration:"
echo "  Test Count: ${TEST_COUNT}"
echo "  Workers: ${WORKERS}"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  Output Dir: ${OUTPUT_DIR}"
echo "  Hardware: ${HARDWARE_PROFILE}"
echo ""

# Check GPU availability
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}ERROR: nvidia-smi not found. Is CUDA installed?${NC}"
    exit 1
fi

GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
echo -e "${GREEN}✓${NC} Detected GPU: ${GPU_NAME}"
echo -e "${GREEN}✓${NC} GPU Count: ${GPU_COUNT}"
echo ""

# Start GPU monitoring in background
echo "Starting GPU monitoring..."
GPU_LOG="${OUTPUT_DIR}/gpu_monitor.log"
mkdir -p "${OUTPUT_DIR}"
nvidia-smi dmon -s pucvgt -d 1 > "${GPU_LOG}" 2>&1 &
GPU_MONITOR_PID=$!
echo -e "${GREEN}✓${NC} GPU monitoring started (PID: ${GPU_MONITOR_PID})"
echo ""

# Trap to cleanup monitoring on exit
cleanup() {
    echo ""
    echo "Cleaning up..."
    kill ${GPU_MONITOR_PID} 2>/dev/null || true
    echo -e "${GREEN}✓${NC} GPU monitoring stopped"
}
trap cleanup EXIT

# Run the test
echo "Starting million-cycle test..."
echo ""
cd /workspace/Niodoo-Final

cargo run --release --bin million_cycle_test -- \
    --count ${TEST_COUNT} \
    --workers ${WORKERS} \
    --batch-size ${BATCH_SIZE} \
    --output-dir "${OUTPUT_DIR}" \
    --hardware "${HARDWARE_PROFILE}" \
    2>&1 | tee "${OUTPUT_DIR}/test_output.log"

TEST_EXIT_CODE=$?

# Stop GPU monitoring
kill ${GPU_MONITOR_PID} 2>/dev/null || true

# Analyze GPU usage
if [ -f "${GPU_LOG}" ]; then
    echo ""
    echo "GPU Usage Summary:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Calculate average GPU utilization
    AVG_UTIL=$(awk 'NR>1 && NF>=3 {sum+=$3; count++} END {if(count>0) print sum/count; else print 0}' "${GPU_LOG}")
    MAX_UTIL=$(awk 'NR>1 && NF>=3 {if(max<$3) max=$3} END {print max}' "${GPU_LOG}")
    
    # Calculate average memory usage
    AVG_MEM=$(awk 'NR>1 && NF>=4 {sum+=$4; count++} END {if(count>0) print sum/count; else print 0}' "${GPU_LOG}")
    MAX_MEM=$(awk 'NR>1 && NF>=4 {if(max<$4) max=$4} END {print max}' "${GPU_LOG}")
    
    echo "Average GPU Utilization: ${AVG_UTIL}%"
    echo "Peak GPU Utilization: ${MAX_UTIL}%"
    echo "Average Memory Usage: ${AVG_MEM}%"
    echo "Peak Memory Usage: ${MAX_MEM}%"
    echo ""
fi

# Display results
if [ -f "${OUTPUT_DIR}/summary.json" ]; then
    echo ""
    echo "Test Results:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    cat "${OUTPUT_DIR}/summary.json" | jq -r '
        "Total Tests: \(.total_tests)",
        "Completed: \(.completed) (\(.completed / .total_tests * 100 | floor)%\)",
        "Failed: \(.failed)",
        "",
        "Performance:",
        "  Avg Entropy: \(.avg_entropy)",
        "  Avg ROUGE: \(.avg_rouge)",
        "  Avg Latency: \(.avg_latency_ms)ms",
        "  P95 Latency: \(.p95_latency_ms)ms",
        "",
        "Throughput:",
        "  Total Time: \(.total_time_secs)s",
        "  Throughput: \(.throughput_per_sec) tests/sec",
        "",
        "Consciousness:",
        "  Threat Rate: \(.threat_rate)%",
        "  Healing Rate: \(.healing_rate)%",
        "  Cache Hit Rate: \(.cache_hit_rate)%"
    '
fi

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
if [ ${TEST_EXIT_CODE} -eq 0 ]; then
    echo -e "${GREEN}✓ Test completed successfully${NC}"
else
    echo -e "${RED}✗ Test failed with exit code ${TEST_EXIT_CODE}${NC}"
fi
echo "Results saved to: ${OUTPUT_DIR}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

exit ${TEST_EXIT_CODE}

