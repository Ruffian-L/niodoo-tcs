#!/bin/bash
# REAL ABLATION TESTS - Actually executes and captures real results
# No fake data, no expected results - just real execution

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="real_ablation_results_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  RUNNING REAL ABLATION TESTS                                ║"
echo "║  Actual execution - no fake data                             ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Results: $RESULTS_DIR"
echo ""

# Test prompts for actual execution
TEST_PROMPTS=(
    "Explain quantum computing"
    "What is machine learning?"
    "Describe topological data analysis"
)

# Function to run actual pipeline test
run_real_test() {
    local name=$1
    local env_vars=$2
    local output_file="$RESULTS_DIR/${name}_results.json"
    local log_file="$RESULTS_DIR/${name}.log"
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Testing: $name"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Set environment
    eval "$env_vars"
    
    # Track metrics
    local total_time=0
    local success_count=0
    local fail_count=0
    local latencies=()
    
    # Run each prompt
    for i in "${!TEST_PROMPTS[@]}"; do
        local prompt="${TEST_PROMPTS[$i]}"
        echo "  Prompt $((i+1))/${#TEST_PROMPTS[@]}: ${prompt:0:50}..."
        
        local start=$(date +%s.%N)
        
        # Actually run the pipeline
        if cargo run --release --bin niodoo_real_integrated -- \
            --prompt "$prompt" \
            --output json \
            > "$log_file.tmp" 2>&1; then
            local end=$(date +%s.%N)
            local duration=$(echo "$end - $start" | bc 2>/dev/null || echo "0")
            latencies+=("$duration")
            success_count=$((success_count + 1))
            echo "    ✅ Success (${duration}s)"
        else
            fail_count=$((fail_count + 1))
            echo "    ❌ Failed"
        fi
        
        # Small delay between requests
        sleep 0.5
    done
    
    # Calculate statistics
    local avg_latency=0
    local min_latency=999999
    local max_latency=0
    
    if [ ${#latencies[@]} -gt 0 ]; then
        local sum=0
        for lat in "${latencies[@]}"; do
            sum=$(echo "$sum + $lat" | bc 2>/dev/null || echo "$sum")
            if (( $(echo "$lat < $min_latency" | bc -l 2>/dev/null || echo 0) )); then
                min_latency=$lat
            fi
            if (( $(echo "$lat > $max_latency" | bc -l 2>/dev/null || echo 0) )); then
                max_latency=$lat
            fi
        done
        avg_latency=$(echo "scale=3; $sum / ${#latencies[@]}" | bc 2>/dev/null || echo "0")
    fi
    
    # Save real results
    cat > "$output_file" << EOF
{
  "name": "$name",
  "timestamp": "$(date -Iseconds)",
  "success_count": $success_count,
  "fail_count": $fail_count,
  "total_prompts": ${#TEST_PROMPTS[@]},
  "latencies": [$(IFS=','; echo "${latencies[*]}")],
  "avg_latency_sec": $avg_latency,
  "min_latency_sec": $min_latency,
  "max_latency_sec": $max_latency,
  "success_rate": $(echo "scale=2; $success_count * 100 / ${#TEST_PROMPTS[@]}" | bc 2>/dev/null || echo "0")
}
EOF
    
    echo "  Results: $success_count/${#TEST_PROMPTS[@]} successful"
    echo "  Avg latency: ${avg_latency}s"
    echo ""
}

# Baseline: Full system
echo "═══════════════════════════════════════════════════════════════"
echo "BASELINE: Full System (All Components Enabled)"
echo "═══════════════════════════════════════════════════════════════"
run_real_test "baseline_full" ""

# Ablation 1: No Curator
echo "═══════════════════════════════════════════════════════════════"
echo "ABLATION 1: Disable Curator"
echo "═══════════════════════════════════════════════════════════════"
run_real_test "no_curator" "export ENABLE_CURATOR=false"

# Ablation 2: No RCE
echo "═══════════════════════════════════════════════════════════════"
echo "ABLATION 2: Disable RCE"
echo "═══════════════════════════════════════════════════════════════"
run_real_test "no_rce" "export RCE_ENABLED=false"

# Ablation 3: No ERAG
echo "═══════════════════════════════════════════════════════════════"
echo "ABLATION 3: Bypass ERAG"
echo "═══════════════════════════════════════════════════════════════"
run_real_test "no_erag" "export ERAG_BYPASS=true"

# Ablation 4: No nToken
echo "═══════════════════════════════════════════════════════════════"
echo "ABLATION 4: Bypass nToken"
echo "═══════════════════════════════════════════════════════════════"
run_real_test "no_ntoken" "export N_TOKENS_BYPASS=1"

# Generate comparison report
echo "═══════════════════════════════════════════════════════════════"
echo "GENERATING REAL RESULTS COMPARISON"
echo "═══════════════════════════════════════════════════════════════"

# Extract metrics
baseline_success=$(jq -r '.success_rate' "$RESULTS_DIR/baseline_full_results.json" 2>/dev/null || echo "0")
baseline_latency=$(jq -r '.avg_latency_sec' "$RESULTS_DIR/baseline_full_results.json" 2>/dev/null || echo "0")

no_curator_success=$(jq -r '.success_rate' "$RESULTS_DIR/no_curator_results.json" 2>/dev/null || echo "0")
no_curator_latency=$(jq -r '.avg_latency_sec' "$RESULTS_DIR/no_curator_results.json" 2>/dev/null || echo "0")

no_rce_success=$(jq -r '.success_rate' "$RESULTS_DIR/no_rce_results.json" 2>/dev/null || echo "0")
no_rce_latency=$(jq -r '.avg_latency_sec' "$RESULTS_DIR/no_rce_results.json" 2>/dev/null || echo "0")

no_erag_success=$(jq -r '.success_rate' "$RESULTS_DIR/no_erag_results.json" 2>/dev/null || echo "0")
no_erag_latency=$(jq -r '.avg_latency_sec' "$RESULTS_DIR/no_erag_results.json" 2>/dev/null || echo "0")

no_ntoken_success=$(jq -r '.success_rate' "$RESULTS_DIR/no_ntoken_results.json" 2>/dev/null || echo "0")
no_ntoken_latency=$(jq -r '.avg_latency_sec' "$RESULTS_DIR/no_ntoken_results.json" 2>/dev/null || echo "0")

# Generate report
cat > "$RESULTS_DIR/REAL_RESULTS.md" << EOF
# REAL Ablation Test Results
**Generated:** $(date -Iseconds)
**Test Execution:** Actual pipeline runs - no fake data

## Baseline (Full System)
- Success Rate: ${baseline_success}%
- Avg Latency: ${baseline_latency}s

## Ablation Results

### No Curator
- Success Rate: ${no_curator_success}%
- Avg Latency: ${no_curator_latency}s
- Impact: $(echo "scale=1; ${no_curator_success} - ${baseline_success}" | bc 2>/dev/null || echo "N/A")% success rate change
- Latency Change: $(echo "scale=1; ${no_curator_latency} - ${baseline_latency}" | bc 2>/dev/null || echo "N/A")s

### No RCE
- Success Rate: ${no_rce_success}%
- Avg Latency: ${no_rce_latency}s
- Impact: $(echo "scale=1; ${no_rce_success} - ${baseline_success}" | bc 2>/dev/null || echo "N/A")% success rate change
- Latency Change: $(echo "scale=1; ${no_rce_latency} - ${baseline_latency}" | bc 2>/dev/null || echo "N/A")s

### No ERAG
- Success Rate: ${no_erag_success}%
- Avg Latency: ${no_erag_latency}s
- Impact: $(echo "scale=1; ${no_erag_success} - ${baseline_success}" | bc 2>/dev/null || echo "N/A")% success rate change
- Latency Change: $(echo "scale=1; ${no_erag_latency} - ${baseline_latency}" | bc 2>/dev/null || echo "N/A")s

### No nToken
- Success Rate: ${no_ntoken_success}%
- Avg Latency: ${no_ntoken_latency}s
- Impact: $(echo "scale=1; ${no_ntoken_success} - ${baseline_success}" | bc 2>/dev/null || echo "N/A")% success rate change
- Latency Change: $(echo "scale=1; ${no_ntoken_latency} - ${baseline_latency}" | bc 2>/dev/null || echo "N/A")s

## Conclusion

These are REAL test results from actual pipeline execution.
No fake data, no expected values - just what actually happened.

EOF

cat "$RESULTS_DIR/REAL_RESULTS.md"
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  REAL TESTS COMPLETE                                         ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "📊 Results: $RESULTS_DIR/"
echo "📄 Report: $RESULTS_DIR/REAL_RESULTS.md"
echo ""
echo "All results are from ACTUAL execution - no fake data!"

