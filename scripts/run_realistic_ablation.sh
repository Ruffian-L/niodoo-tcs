#!/bin/bash
# REALISTIC Ablation Tests - Works with what we have
# Captures REAL failures and REAL successes - no fake data

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="real_ablation_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  REAL ABLATION TESTS - Capturing ACTUAL Results              ║"
echo "║  Success OR Failure - both are real data                    ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Simple test: Can we even initialize the pipeline?
test_pipeline_init() {
    local name=$1
    local env_vars=$2
    local log_file="$RESULTS_DIR/${name}_init.log"
    
    echo "Testing: $name (Pipeline Init)"
    eval "$env_vars"
    
    local start=$(date +%s.%N)
    if timeout 30 cargo run --release --bin niodoo_real_integrated -- --prompt "test" --output json \
        > "$log_file" 2>&1; then
        local end=$(date +%s.%N)
        local duration=$(echo "$end - $start" | bc 2>/dev/null || echo "0")
        echo "  ✅ INIT SUCCESS (${duration}s)"
        echo "{\"name\": \"$name\", \"init\": \"success\", \"duration\": $duration}" > "$RESULTS_DIR/${name}_init.json"
        return 0
    else
        local end=$(date +%s.%N)
        local duration=$(echo "$end - $start" | bc 2>/dev/null || echo "0")
        echo "  ❌ INIT FAILED (${duration}s)"
        
        # Capture the actual error
        local error=$(tail -5 "$log_file" | head -1)
        echo "{\"name\": \"$name\", \"init\": \"failed\", \"duration\": $duration, \"error\": \"$error\"}" > "$RESULTS_DIR/${name}_init.json"
        return 1
    fi
}

# Test compilation with different configs
test_compilation() {
    local name=$1
    local env_vars=$2
    local log_file="$RESULTS_DIR/${name}_compile.log"
    
    echo "Testing: $name (Compilation Check)"
    eval "$env_vars"
    
    local start=$(date +%s.%N)
    if cargo check --bin niodoo_real_integrated > "$log_file" 2>&1; then
        local end=$(date +%s.%N)
        local duration=$(echo "$end - $start" | bc 2>/dev/null || echo "0")
        echo "  ✅ COMPILES (${duration}s)"
        echo "{\"name\": \"$name\", \"compile\": \"success\", \"duration\": $duration}" > "$RESULTS_DIR/${name}_compile.json"
        return 0
    else
        local end=$(date +%s.%N)
        local duration=$(echo "$end - $start" | bc 2>/dev/null || echo "0")
        echo "  ❌ COMPILE FAILED (${duration}s)"
        local error=$(grep -i "error" "$log_file" | head -1 || echo "unknown")
        echo "{\"name\": \"$name\", \"compile\": \"failed\", \"duration\": $duration, \"error\": \"$error\"}" > "$RESULTS_DIR/${name}_compile.json"
        return 1
    fi
}

echo "═══════════════════════════════════════════════════════════════"
echo "BASELINE: Full System"
echo "═══════════════════════════════════════════════════════════════"
test_compilation "baseline" ""
test_pipeline_init "baseline" ""

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "ABLATION 1: No Curator"
echo "═══════════════════════════════════════════════════════════════"
test_compilation "no_curator" "export ENABLE_CURATOR=false"
test_pipeline_init "no_curator" "export ENABLE_CURATOR=false"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "ABLATION 2: No RCE"
echo "═══════════════════════════════════════════════════════════════"
test_compilation "no_rce" "export RCE_ENABLED=false"
test_pipeline_init "no_rce" "export RCE_ENABLED=false"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "ABLATION 3: No ERAG"
echo "═══════════════════════════════════════════════════════════════"
test_compilation "no_erag" "export ERAG_BYPASS=true"
test_pipeline_init "no_erag" "export ERAG_BYPASS=true"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "REAL RESULTS SUMMARY"
echo "═══════════════════════════════════════════════════════════════"

# Generate report from actual results
cat > "$RESULTS_DIR/REAL_RESULTS.md" << 'EOF'
# REAL Ablation Test Results
**Generated:** $(date -Iseconds)

## What This Shows

These are REAL results from actual test execution. Failures are REAL failures.
Successes are REAL successes. No fake data.

## Compilation Results

EOF

for test in baseline no_curator no_rce no_erag; do
    if [ -f "$RESULTS_DIR/${test}_compile.json" ]; then
        echo "" >> "$RESULTS_DIR/REAL_RESULTS.md"
        echo "### $test" >> "$RESULTS_DIR/REAL_RESULTS.md"
        cat "$RESULTS_DIR/${test}_compile.json" | jq '.' >> "$RESULTS_DIR/REAL_RESULTS.md" 2>/dev/null || cat "$RESULTS_DIR/${test}_compile.json" >> "$RESULTS_DIR/REAL_RESULTS.md"
    fi
done

cat >> "$RESULTS_DIR/REAL_RESULTS.md" << 'EOF'

## Pipeline Init Results

EOF

for test in baseline no_curator no_rce no_erag; do
    if [ -f "$RESULTS_DIR/${test}_init.json" ]; then
        echo "" >> "$RESULTS_DIR/REAL_RESULTS.md"
        echo "### $test" >> "$RESULTS_DIR/REAL_RESULTS.md"
        cat "$RESULTS_DIR/${test}_init.json" | jq '.' >> "$RESULTS_DIR/REAL_RESULTS.md" 2>/dev/null || cat "$RESULTS_DIR/${test}_init.json" >> "$RESULTS_DIR/REAL_RESULTS.md"
    fi
done

cat >> "$RESULTS_DIR/REAL_RESULTS.md" << 'EOF'

## Conclusion

These are REAL test results. If something failed, it REALLY failed.
If something succeeded, it REALLY succeeded.

No fake data. Just real execution results.

EOF

cat "$RESULTS_DIR/REAL_RESULTS.md"
echo ""
echo "📊 Results: $RESULTS_DIR/"
echo "📄 Report: $RESULTS_DIR/REAL_RESULTS.md"
echo ""
echo "✅ REAL tests executed - failures and successes are REAL!"

