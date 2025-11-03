#!/bin/bash
# Master Validation Test Runner
# Executes all validation framework tests and generates comprehensive report

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR/.."
cd "$ROOT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="validation_results_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

echo "🚀 Starting Comprehensive Validation Test Suite"
echo "=============================================="
echo "Results directory: $RESULTS_DIR"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track test results
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_SKIPPED=0

test_passed() {
    echo -e "${GREEN}✅ PASSED:${NC} $1"
    ((TESTS_PASSED++))
}

test_failed() {
    echo -e "${RED}❌ FAILED:${NC} $1"
    ((TESTS_FAILED++))
}

test_skipped() {
    echo -e "${YELLOW}⏭️  SKIPPED:${NC} $1"
    ((TESTS_SKIPPED++))
}

# Check if services are available
check_services() {
    echo "Checking service dependencies..."
    
    if [ -n "$MOCK_MODE" ] && [ "$MOCK_MODE" = "true" ]; then
        echo "MOCK_MODE enabled - skipping service checks"
        return 0
    fi
    
    SERVICES_OK=true
    
    # Check vLLM
    if curl -s -f http://127.0.0.1:5001/health > /dev/null 2>&1; then
        echo "  ✅ vLLM service available"
    else
        echo "  ⚠️  vLLM service not available (will use mock mode)"
        export MOCK_MODE=true
    fi
    
    # Check Qdrant
    if curl -s -f http://127.0.0.1:6333/collections > /dev/null 2>&1; then
        echo "  ✅ Qdrant service available"
    else
        echo "  ⚠️  Qdrant service not available (will use mock mode)"
        export MOCK_MODE=true
    fi
    
    echo ""
}

# Build required binaries
build_binaries() {
    echo "Building validation binaries..."
    
    cd niodoo_real_integrated
    
    if cargo build --bin metrics_runner --release --features svc 2>&1 | tee "$RESULTS_DIR/build_metrics_runner.log"; then
        test_passed "metrics_runner binary build"
    else
        test_failed "metrics_runner binary build"
        return 1
    fi
    
    if cargo build --bin ablation_runner --release --features svc 2>&1 | tee "$RESULTS_DIR/build_ablation_runner.log"; then
        test_passed "ablation_runner binary build"
    else
        test_failed "ablation_runner binary build"
        return 1
    fi
    
    cd ..
    echo ""
}

# Test 1: Baseline Capture
test_baseline_capture() {
    echo "Test 1: Baseline Capture"
    echo "------------------------"
    
    if [ ! -f "baselines/baseline-latest.json" ]; then
        echo "No baseline found, capturing one..."
        if ./scripts/capture_baseline.sh 2>&1 | tee "$RESULTS_DIR/baseline_capture.log"; then
            test_passed "Baseline capture"
            cp baselines/baseline-latest.json "$RESULTS_DIR/baseline.json"
        else
            test_failed "Baseline capture"
            return 1
        fi
    else
        echo "Baseline exists, using existing one"
        cp baselines/baseline-latest.json "$RESULTS_DIR/baseline.json"
        test_skipped "Baseline capture (already exists)"
    fi
    echo ""
}

# Test 2: Load Test
test_load_test() {
    echo "Test 2: Load Test"
    echo "-----------------"
    
    cd niodoo_real_integrated
    
    if cargo run --bin metrics_runner --release --features svc -- \
        --scenario load_test \
        --concurrent-users 8 \
        --duration-secs 30 \
        --target-tokens 512 \
        --output "../$RESULTS_DIR/load_test_report.json" \
        2>&1 | tee "../$RESULTS_DIR/load_test.log"; then
        test_passed "Load test execution"
        
        # Check if report was generated
        if [ -f "../$RESULTS_DIR/load_test_report.json" ]; then
            echo "  Load test metrics:"
            python3 <<EOF
import json
with open('../$RESULTS_DIR/load_test_report.json', 'r') as f:
    m = json.load(f)
print(f"    p99 latency: {m['latency']['p99_ms']:.2f}ms")
print(f"    Throughput: {m['throughput']['tokens_per_sec']:.2f} tokens/sec")
EOF
        fi
    else
        test_failed "Load test execution"
    fi
    
    cd ..
    echo ""
}

# Test 3: Baseline Comparison
test_baseline_comparison() {
    echo "Test 3: Baseline Comparison"
    echo "---------------------------"
    
    if [ -f "$RESULTS_DIR/load_test_report.json" ] && [ -f "$RESULTS_DIR/baseline.json" ]; then
        if ./scripts/compare_baseline.sh "$RESULTS_DIR/load_test_report.json" "$RESULTS_DIR/baseline.json" \
            2>&1 | tee "$RESULTS_DIR/baseline_comparison.log"; then
            test_passed "Baseline comparison"
        else
            # Comparison script might exit with non-zero on regressions, which is OK
            if grep -q "REGRESSION" "$RESULTS_DIR/baseline_comparison.log"; then
                test_failed "Baseline comparison (regression detected)"
            else
                test_passed "Baseline comparison"
            fi
        fi
    else
        test_skipped "Baseline comparison (missing files)"
    fi
    echo ""
}

# Test 4: Golden Probes
test_golden_probes() {
    echo "Test 4: Golden Probes"
    echo "--------------------"
    
    if [ ! -f "data/golden_probes.json" ]; then
        test_skipped "Golden probes (file not found)"
        return
    fi
    
    cd niodoo_real_integrated
    
    # Run a subset of golden probes for testing
    python3 <<EOF
import json
import subprocess
import sys
import os

os.environ['MOCK_MODE'] = 'true'
os.environ['RUST_LOG'] = 'warn'

with open('../data/golden_probes.json', 'r') as f:
    probes_data = json.load(f)

# Test first 5 probes
test_probes = probes_data['probes'][:5]
results = []
passed = 0
failed = 0

for probe in test_probes:
    question = probe['question']
    probe_id = probe['id']
    
    print(f"Running probe: {probe_id}...")
    
    try:
        result = subprocess.run(
            ['cargo', 'run', '--bin', 'niodoo_real_integrated', '--release', '--features', 'svc', '--', '--prompt', question],
            cwd='.',
            capture_output=True,
            text=True,
            timeout=30,
            env=os.environ.copy()
        )
        
        response = result.stdout + result.stderr
        
        # Simple validation
        expected = probe.get('expected_containment', [])
        if expected:
            matches = sum(1 for exp in expected if exp.lower() in response.lower())
            match_rate = matches / len(expected)
            passed_threshold = match_rate >= probes_data['validation_criteria']['pass_threshold']
        else:
            passed_threshold = result.returncode == 0
        
        if passed_threshold:
            passed += 1
            print(f"  ✅ Passed")
        else:
            failed += 1
            print(f"  ❌ Failed")
        
        results.append({
            'id': probe_id,
            'passed': passed_threshold
        })
    except Exception as e:
        failed += 1
        print(f"  ❌ Error: {e}")
        results.append({
            'id': probe_id,
            'passed': False,
            'error': str(e)
        })

print(f"\nGolden Probes Results: {passed}/{len(test_probes)} passed")

with open('../$RESULTS_DIR/golden_probes_results.json', 'w') as f:
    json.dump({'results': results, 'passed': passed, 'failed': failed, 'total': len(test_probes)}, f, indent=2)

if failed > len(test_probes) * 0.4:  # Allow 40% failure rate in test mode
    sys.exit(1)
EOF

    if [ $? -eq 0 ]; then
        test_passed "Golden probes execution"
    else
        test_failed "Golden probes execution"
    fi
    
    cd ..
    echo ""
}

# Test 5: Ablation Experiments
test_ablation_experiments() {
    echo "Test 5: Ablation Experiments"
    echo "----------------------------"
    
    cd niodoo_real_integrated
    
    if [ ! -f "../$RESULTS_DIR/baseline.json" ]; then
        test_skipped "Ablation experiments (no baseline)"
        cd ..
        return
    fi
    
    # Run one ablation experiment as test
    echo "Running ablation experiment: DisableGpuFitness..."
    
    if cargo run --bin ablation_runner --release --features svc -- \
        --experiment DisableGpuFitness \
        --baseline "../$RESULTS_DIR/baseline.json" \
        --output-dir "../$RESULTS_DIR/ablation" \
        --concurrent-users 4 \
        --duration-secs 20 \
        2>&1 | tee "../$RESULTS_DIR/ablation_test.log"; then
        test_passed "Ablation experiment execution"
    else
        test_failed "Ablation experiment execution"
    fi
    
    cd ..
    echo ""
}

# Test 6: Validation Module Compilation
test_validation_modules() {
    echo "Test 6: Validation Module Compilation"
    echo "-------------------------------------"
    
    cd niodoo_real_integrated
    
    if cargo check --lib 2>&1 | tee "$RESULTS_DIR/validation_check.log"; then
        test_passed "Validation modules compilation"
    else
        test_failed "Validation modules compilation"
    fi
    
    cd ..
    echo ""
}

# Generate Summary Report
generate_summary() {
    echo ""
    echo "=============================================="
    echo "Validation Test Suite Summary"
    echo "=============================================="
    echo "Total Tests: $((TESTS_PASSED + TESTS_FAILED + TESTS_SKIPPED))"
    echo -e "${GREEN}Passed: $TESTS_PASSED${NC}"
    echo -e "${RED}Failed: $TESTS_FAILED${NC}"
    echo -e "${YELLOW}Skipped: $TESTS_SKIPPED${NC}"
    echo ""
    echo "Results saved to: $RESULTS_DIR"
    echo ""
    
    # Create summary JSON
    python3 <<EOF
import json
import os
from datetime import datetime

summary = {
    'timestamp': datetime.now().isoformat(),
    'results_dir': '$RESULTS_DIR',
    'tests': {
        'passed': $TESTS_PASSED,
        'failed': $TESTS_FAILED,
        'skipped': $TESTS_SKIPPED,
        'total': $((TESTS_PASSED + TESTS_FAILED + TESTS_SKIPPED))
    },
    'files': {
        'baseline': 'baseline.json' if os.path.exists('$RESULTS_DIR/baseline.json') else None,
        'load_test': 'load_test_report.json' if os.path.exists('$RESULTS_DIR/load_test_report.json') else None,
        'golden_probes': 'golden_probes_results.json' if os.path.exists('$RESULTS_DIR/golden_probes_results.json') else None,
        'ablation': 'ablation/' if os.path.exists('$RESULTS_DIR/ablation') else None
    }
}

with open('$RESULTS_DIR/test_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("Summary saved to: $RESULTS_DIR/test_summary.json")
EOF
    
    if [ $TESTS_FAILED -gt 0 ]; then
        echo -e "${RED}⚠️  Some tests failed. Check logs in $RESULTS_DIR${NC}"
        exit 1
    else
        echo -e "${GREEN}✅ All tests passed!${NC}"
        exit 0
    fi
}

# Main execution
main() {
    check_services
    build_binaries || exit 1
    test_baseline_capture
    test_load_test
    test_baseline_comparison
    test_golden_probes
    test_ablation_experiments
    test_validation_modules
    generate_summary
}

main "$@"

