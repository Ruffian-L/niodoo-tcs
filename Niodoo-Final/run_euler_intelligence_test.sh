#!/bin/bash
# EULER MATHEMATICAL INTELLIGENCE TEST - FULL SYSTEM SYNTHESIS
# Tests complete niodoo_real_integrated + autonomous gating + mathematical reasoning
# Created 2025-11-11 for validating true intelligence vs speed optimization

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

SMOKE_MODE=0
POSITIONAL=()
PROBLEMS_DEFAULT=10
TIMEOUT_DEFAULT=300

while (($#)); do
    case "$1" in
        --smoke)
            SMOKE_MODE=1
            shift
            ;;
        --problems)
            PROBLEMS="${2:?Missing value for --problems}"
            shift 2
            ;;
        --timeout)
            TIMEOUT="${2:?Missing value for --timeout}"
            shift 2
            ;;
        -h|--help)
            cat <<'EOF'
Usage: ./run_euler_intelligence_test.sh [--smoke] [--problems N] [--timeout SECONDS] [PROBLEMS] [TIMEOUT]

Positional arguments remain for backward compatibility (PROBLEMS first, TIMEOUT second).
EOF
            exit 0
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done

if [ ${#POSITIONAL[@]} -ge 1 ]; then
    PROBLEMS="${PROBLEMS:-${POSITIONAL[0]}}"
fi

if [ ${#POSITIONAL[@]} -ge 2 ]; then
    TIMEOUT="${TIMEOUT:-${POSITIONAL[1]}}"
fi

PROBLEMS="${PROBLEMS:-$PROBLEMS_DEFAULT}"
TIMEOUT="${TIMEOUT:-$TIMEOUT_DEFAULT}"

# Allow buffers to be tuned via environment without touching the script
EXTRA_TIMEOUT_BUFFER="${EULER_TIMEOUT_BUFFER:-60}"
KILL_AFTER_BUFFER="${EULER_TIMEOUT_KILL_AFTER:-15}"

# Fallback to sane defaults if the env vars are unset or invalid
if ! [[ "$EXTRA_TIMEOUT_BUFFER" =~ ^[0-9]+$ ]]; then
    EXTRA_TIMEOUT_BUFFER=60
fi

if ! [[ "$KILL_AFTER_BUFFER" =~ ^[0-9]+$ ]]; then
    KILL_AFTER_BUFFER=15
fi

TOTAL_TIMEOUT=$((TIMEOUT * PROBLEMS + EXTRA_TIMEOUT_BUFFER))

echo -e "${PURPLE}🧮 EULER MATHEMATICAL INTELLIGENCE TEST${NC}"
echo -e "${PURPLE}=====================================${NC}"
echo -e "${BLUE}Goal: Test FULL NIODOO system intelligence with Level 50 mathematical problems${NC}"
echo -e "${BLUE}Focus: Intelligence over Speed - Deep mathematical reasoning assessment${NC}"
echo ""

# Configuration
OUTPUT_DIR="euler_test_results_$(date +%Y%m%d_%H%M%S)"

echo -e "${YELLOW}📋 Test Configuration:${NC}"
echo "   Problems: $PROBLEMS/10 Euler Level 50 problems"
echo "   Timeout: ${TIMEOUT}s per problem"
echo "   Mode: $([ "$SMOKE_MODE" -eq 1 ] && echo SMOKE || echo FULL)"
echo "   Output: $OUTPUT_DIR/"
echo "   Command Timeout: ${TOTAL_TIMEOUT}s (buffer ${EXTRA_TIMEOUT_BUFFER}s)"
echo "   Kill-after grace: ${KILL_AFTER_BUFFER}s"
echo ""

# Step 1: Load unified environment (System2_loop proven patterns)
echo -e "${BLUE}🔧 Step 1: Loading Unified Environment${NC}"
if [ ! -f "niodoo_real_integrated.env" ]; then
    echo -e "${RED}❌ niodoo_real_integrated.env not found${NC}"
    echo "   Run the system synthesis plan first"
    exit 1
fi

source niodoo_real_integrated.env
echo -e "${GREEN}✅ Environment loaded (System2_loop + niodoo_real_integrated synthesis)${NC}"

# Step 2: Verify Critical Services  
echo -e "${BLUE}🔍 Step 2: Service Health Check${NC}"

# Check Cloud Qdrant (proven working)
if curl -H "api-key: $QDRANT_API_KEY" "$QDRANT_URL/health" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Cloud Qdrant accessible${NC}"
else
    echo -e "${RED}❌ Cloud Qdrant connection failed${NC}"
    echo "   URL: $QDRANT_URL"
    echo "   Check QDRANT_API_KEY configuration"
    exit 1
fi

# Check llama.cpp generation service (OpenAI-compatible)
if curl -s "http://127.0.0.1:8000/v1/models" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ llama.cpp generation service ready (port 8000)${NC}"
else
    echo -e "${YELLOW}⚠️ llama.cpp generation service not running - attempting startup${NC}"
    if [ -f "./start_full_niodoo_system.sh" ]; then
        echo "   Starting services with unified script..."
        STARTUP_LOG="$SCRIPT_DIR/start_full_niodoo_system.log"
        nohup ./start_full_niodoo_system.sh >"$STARTUP_LOG" 2>&1 &
        sleep 30 # Give services time to start
        echo "   Service startup logs: $STARTUP_LOG"
    else
        echo -e "${RED}❌ llama.cpp generation service required for testing${NC}"
        echo "   Run: ./start_full_niodoo_system.sh"
        exit 1
    fi
fi

# Step 3: Test Build (if cargo available)
echo -e "${BLUE}🔨 Step 3: Building Full System${NC}"
cd niodoo_real_integrated

if command -v cargo >/dev/null 2>&1; then
    echo "   Building euler_test binary (cli_bins feature)..."
    if cargo build --release --bin euler_test --features cli_bins 2>/tmp/build.log; then
        echo -e "${GREEN}✅ euler_test binary built successfully${NC}"
        EULER_BINARY="./target/release/euler_test"
        if [ ! -f "$EULER_BINARY" ] && [ -f "../target/release/euler_test" ]; then
            EULER_BINARY="../target/release/euler_test"
        fi
    else
        echo -e "${YELLOW}⚠️ Build failed - checking for existing binary${NC}"
        cat /tmp/build.log | tail -10
        if [ -f "./target/release/euler_test" ]; then
            echo -e "${GREEN}✅ Using existing euler_test binary${NC}"
            EULER_BINARY="./target/release/euler_test"
        elif [ -f "../target/release/euler_test" ]; then
            echo -e "${GREEN}✅ Using workspace euler_test binary${NC}"
            EULER_BINARY="../target/release/euler_test"
        else
            echo -e "${RED}❌ No euler_test binary available${NC}"
            cd ..
            exit 1
        fi
    fi
else
    echo -e "${YELLOW}⚠️ Cargo not available - checking for pre-built binary${NC}"
    if [ -f "./target/release/euler_test" ]; then
        echo -e "${GREEN}✅ Found pre-built euler_test binary${NC}"
        EULER_BINARY="./target/release/euler_test"
    elif [ -f "../target/release/euler_test" ]; then
        echo -e "${GREEN}✅ Found workspace euler_test binary${NC}"
        EULER_BINARY="../target/release/euler_test"
    else
        echo -e "${RED}❌ No binary available and cannot build${NC}"
        cd ..
        exit 1
    fi
fi

# Step 4: Run Euler Mathematical Intelligence Test
echo ""
echo -e "${PURPLE}🧮 Step 4: MATHEMATICAL INTELLIGENCE ASSESSMENT${NC}"
echo -e "${PURPLE}===============================================${NC}"

# Determine a writable output base (default parent repo, fallback to local results/)
OUTPUT_BASE="${EULER_OUTPUT_BASE:-..}"
if [ ! -w "$OUTPUT_BASE" ]; then
    OUTPUT_BASE="results"
    mkdir -p "$OUTPUT_BASE"
fi

echo "   Writable artifact path: $OUTPUT_BASE/$OUTPUT_DIR/"

mkdir -p "$OUTPUT_BASE/$OUTPUT_DIR"
LOG_FILE="$OUTPUT_BASE/$OUTPUT_DIR/euler_test.log"
RESULTS_FILE="$OUTPUT_BASE/$OUTPUT_DIR/euler_results.json"

echo "   Running $PROBLEMS Euler Level 50 problems..."
echo "   Each problem tests: algorithms, proofs, optimization, mathematical reasoning"
echo ""

# Skip strict live service verification unless caller explicitly overrides
if [ -z "${NIODOO_SKIP_SMOKE+x}" ]; then
    export NIODOO_SKIP_SMOKE=1
fi

# Run the test with timeout protection
EULER_ARGS=(
    --problems "$PROBLEMS"
    --output "$RESULTS_FILE"
    --timeout "$TIMEOUT"
    --verbose
)

if [ "$SMOKE_MODE" -eq 1 ]; then
    EULER_ARGS+=(--smoke)
fi

set +e
timeout --foreground --kill-after "$KILL_AFTER_BUFFER" "$TOTAL_TIMEOUT" "$EULER_BINARY" \
    "${EULER_ARGS[@]}" \
    2>&1 | tee "$LOG_FILE"
PIPE_STATUS=("${PIPESTATUS[@]}")
timeout_exit=${PIPE_STATUS[0]}
tee_exit=${PIPE_STATUS[1]:-0}
set -e

if [ "$tee_exit" -ne 0 ]; then
    echo -e "${RED}💥 Log capture failed (tee exit code $tee_exit)${NC}"
fi

if [ "$timeout_exit" -ne 0 ]; then
    echo -e "${RED}💥 Test execution failed or timed out${NC}"
    echo "   Exit code: $timeout_exit"
    echo "   Check $LOG_FILE for details"
    if [ -f "$RESULTS_FILE" ]; then
        echo "   Partial results may be in $RESULTS_FILE"
    fi
    EXIT_CODE=$timeout_exit
else
    EXIT_CODE=0
fi

cd ..

# Step 5: Analyze Results
echo ""
echo -e "${BLUE}📊 Step 5: Intelligence Analysis${NC}"

if [ -f "$OUTPUT_BASE/$OUTPUT_DIR/euler_results.json" ]; then
    echo -e "${GREEN}✅ Results file generated${NC}"
    
    # Extract key metrics using jq if available
    if command -v jq >/dev/null 2>&1; then
        echo ""
        echo -e "${PURPLE}🎓 INTELLIGENCE SUMMARY:${NC}"
        jq -r '
            .summary | 
            "📊 Problems: \(.completed_problems)/\(.total_problems)",
            "📈 Average Quality: \(.average_quality)/10",
            "🧮 Mathematical Depth: \(.average_math_depth)/10"
        ' "$OUTPUT_BASE/$OUTPUT_DIR/euler_results.json"
        
        echo ""
        echo -e "${PURPLE}🚪 AUTONOMOUS GATING ANALYSIS:${NC}"
        jq -r '
            .gating_analysis |
            "❌ Learning Gate: \(.learning_gate_count) problems (failures → Gemini correction)",
            "😐 Indifferent: \(.indifferent_count) problems (mediocre → discarded)", 
            "✅ Memory Gate: \(.memory_gate_count) problems (high quality → Golden Memory)",
            "🌟 Golden Qualified: \(.golden_memory_qualified) problems (novel/extreme insights)"
        ' "$OUTPUT_BASE/$OUTPUT_DIR/euler_results.json"
        
        echo ""
        echo -e "${PURPLE}🎯 FINAL ASSESSMENT:${NC}"
        jq -r '.intelligence_assessment.mathematical_reasoning_grade' "$OUTPUT_BASE/$OUTPUT_DIR/euler_results.json"
        jq -r '.intelligence_assessment.system_intelligence_level' "$OUTPUT_BASE/$OUTPUT_DIR/euler_results.json"
        
    else
        echo "   Results available in JSON format"
        echo "   Install 'jq' for detailed analysis"
    fi
else
    echo -e "${RED}❌ No results file generated${NC}"
    echo "   Check $OUTPUT_DIR/euler_test.log for errors"
    
    if [ -f "$OUTPUT_DIR/euler_test.log" ]; then
        echo ""
        echo -e "${YELLOW}📋 Last 20 lines of log:${NC}"
        tail -20 "$OUTPUT_DIR/euler_test.log"
    fi
fi

# Step 6: System Synthesis Assessment
echo ""
echo -e "${BLUE}🎉 Step 6: System Synthesis Assessment${NC}"

if [ -f "$OUTPUT_BASE/$OUTPUT_DIR/euler_results.json" ] && command -v jq >/dev/null 2>&1; then
    MEMORY_GATE_COUNT=$(jq -r '.gating_analysis.memory_gate_count' "$OUTPUT_BASE/$OUTPUT_DIR/euler_results.json")
    AVG_QUALITY=$(jq -r '.summary.average_quality' "$OUTPUT_BASE/$OUTPUT_DIR/euler_results.json")
    
    echo "   Memory Gate Success Rate: $MEMORY_GATE_COUNT/$PROBLEMS"
    echo "   Average Quality Score: $AVG_QUALITY/10"
    
    if [ "$MEMORY_GATE_COUNT" -ge 7 ] && [ "$(echo "$AVG_QUALITY > 7.0" | bc -l 2>/dev/null || echo 0)" -eq 1 ]; then
        echo ""
        echo -e "${GREEN}🎉 SYSTEM SYNTHESIS SUCCESS!${NC}"
        echo -e "${GREEN}   Full NIODOO system demonstrates mathematical intelligence${NC}"
        echo -e "${GREEN}   Autonomous gating system working correctly${NC}"
        echo -e "${GREEN}   Ready for production mathematical reasoning tasks${NC}"
    elif [ "$MEMORY_GATE_COUNT" -ge 4 ]; then
        echo ""
        echo -e "${YELLOW}⚠️ PARTIAL SUCCESS - System shows mathematical capability${NC}"
        echo -e "${YELLOW}   Continue tuning and autonomous learning${NC}"
    else
        echo ""
        echo -e "${YELLOW}🔄 SYSTEM NEEDS IMPROVEMENT${NC}"
        echo -e "${YELLOW}   Focus on mathematical reasoning enhancement${NC}"
        echo -e "${YELLOW}   Autonomous gating will help system learn from failures${NC}"
    fi
else
    echo -e "${YELLOW}   Manual analysis required - check results file${NC}"
fi

echo ""
echo -e "${PURPLE}📁 Test Results Location: $OUTPUT_BASE/$OUTPUT_DIR/${NC}"
echo -e "${PURPLE}📊 Complete Results: $OUTPUT_BASE/$OUTPUT_DIR/euler_results.json${NC}"
echo -e "${PURPLE}📝 Test Log: $OUTPUT_BASE/$OUTPUT_DIR/euler_test.log${NC}"
echo ""
echo -e "${GREEN}🎯 Euler Mathematical Intelligence Test Complete!${NC}"

exit $EXIT_CODE
