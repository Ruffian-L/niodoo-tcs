#!/bin/bash
# End-to-End Validation Test Runner - Executes actual tests

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR/.."
cd "$ROOT_DIR"

# Ensure cargo is in PATH
export PATH="$HOME/.cargo/bin:$PATH"
source $HOME/.cargo/env 2>/dev/null || true

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="e2e_validation_results_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║     END-TO-END VALIDATION TEST EXECUTION                      ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Results directory: $RESULTS_DIR"
echo ""

# Check for cargo
if ! command -v cargo >/dev/null 2>&1; then
    echo "❌ Cargo not found. Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
    export PATH="$HOME/.cargo/bin:$PATH"
    source $HOME/.cargo/env
fi

echo "🔧 Cargo version: $(cargo --version)"
echo ""

# Set up environment
export MOCK_MODE=true
export RUST_LOG=warn

# Test 1: Build binaries
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 1: Building Validation Binaries"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

cd niodoo_real_integrated

echo "Building metrics_runner..."
if cargo build --bin metrics_runner --release --features svc 2>&1 | tee "../$RESULTS_DIR/build_metrics_runner.log"; then
    echo "✅ metrics_runner built successfully"
else
    echo "❌ metrics_runner build failed"
    exit 1
fi

echo ""
echo "Building ablation_runner..."
if cargo build --bin ablation_runner --release --features svc 2>&1 | tee "../$RESULTS_DIR/build_ablation_runner.log"; then
    echo "✅ ablation_runner built successfully"
else
    echo "❌ ablation_runner build failed"
    exit 1
fi

echo ""
echo "Building main pipeline binary..."
if cargo build --bin niodoo_real_integrated --release --features svc 2>&1 | tee "../$RESULTS_DIR/build_main.log"; then
    echo "✅ Pipeline binary built successfully"
else
    echo "❌ Pipeline binary build failed"
    exit 1
fi

cd ..
echo ""

# Test 2: Golden Probes Execution
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 2: Golden Probes Execution (E2E)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python3 <<PYTHON
import json
import subprocess
import os
import time
import sys

with open('data/golden_probes.json', 'r') as f:
    probes_data = json.load(f)

print(f"Executing {len(probes_data['probes'])} golden probes through pipeline...")
print()

results = []
passed = 0
failed = 0
errors = 0

# Test first 10 probes
test_probes = probes_data['probes'][:10]

for i, probe in enumerate(test_probes, 1):
    probe_id = probe['id']
    category = probe.get('category', 'unknown')
    question = probe['question']
    expected = probe.get('expected_containment', [])
    
    print(f"[{i:2d}/10] {probe_id:15s} ({category:15s})", end=" ", flush=True)
    
    try:
        # Run actual pipeline
        result = subprocess.run(
            ['cargo', 'run', '--bin', 'niodoo_real_integrated', '--release', '--features', 'svc', '--', '--prompt', question],
            cwd='niodoo_real_integrated',
            capture_output=True,
            text=True,
            timeout=60,
            env={**os.environ, 'MOCK_MODE': 'true', 'RUST_LOG': 'warn'}
        )
        
        response = result.stdout + result.stderr
        
        # Extract response text (simplified - would parse JSON in production)
        response_lower = response.lower()
        
        # Check keyword matching
        if expected:
            matches = sum(1 for exp in expected if exp.lower() in response_lower)
            match_rate = matches / len(expected) if expected else 1.0
            passed_threshold = match_rate >= probes_data['validation_criteria']['pass_threshold']
        else:
            match_rate = 1.0
            passed_threshold = result.returncode == 0
        
        if passed_threshold:
            print("✅ PASS")
            passed += 1
        else:
            print(f"❌ FAIL (match: {match_rate:.2f})")
            failed += 1
        
        results.append({
            'id': probe_id,
            'category': category,
            'question': question,
            'passed': passed_threshold,
            'match_rate': match_rate,
            'returncode': result.returncode,
            'response_length': len(response)
        })
        
    except subprocess.TimeoutExpired:
        print("⏱️  TIMEOUT")
        failed += 1
        errors += 1
        results.append({
            'id': probe_id,
            'category': category,
            'passed': False,
            'error': 'timeout'
        })
    except Exception as e:
        print(f"❌ ERROR: {str(e)[:30]}")
        failed += 1
        errors += 1
        results.append({
            'id': probe_id,
            'category': category,
            'passed': False,
            'error': str(e)
        })

print()
print(f"Golden Probes Results: {passed}/{len(test_probes)} passed ({passed/len(test_probes)*100:.1f}%)")
if errors > 0:
    print(f"  Errors: {errors}")

with open('$RESULTS_DIR/golden_probes_e2e.json', 'w') as f:
    json.dump({
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total': len(test_probes),
        'passed': passed,
        'failed': failed,
        'errors': errors,
        'pass_rate': passed / len(test_probes),
        'results': results
    }, f, indent=2)

if failed > len(test_probes) * 0.5:
    sys.exit(1)
PYTHON

GOLDEN_EXIT=$?
echo ""

# Test 3: Load Test Execution
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 3: Load Test Execution (E2E)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

cd niodoo_real_integrated

echo "Running metrics_runner load test (30 seconds, 4 concurrent users)..."
if timeout 120 cargo run --bin metrics_runner --release --features svc -- \
    --scenario load_test \
    --concurrent-users 4 \
    --duration-secs 30 \
    --target-tokens 256 \
    --output "../$RESULTS_DIR/load_test_e2e.json" \
    --mock-mode \
    2>&1 | tee "../$RESULTS_DIR/load_test.log"; then
    
    if [ -f "../$RESULTS_DIR/load_test_e2e.json" ]; then
        echo ""
        echo "✅ Load Test Completed:"
        cd ..
        python3 <<PYTHON
import json
try:
    with open('$RESULTS_DIR/load_test_e2e.json', 'r') as f:
        m = json.load(f)
    print(f"  p99 latency: {m['latency']['p99_ms']:.2f}ms")
    print(f"  p95 latency: {m['latency']['p95_ms']:.2f}ms")
    print(f"  p50 latency: {m['latency']['p50_ms']:.2f}ms")
    print(f"  Throughput: {m['throughput']['tokens_per_sec']:.2f} tokens/sec")
    print(f"  Requests/sec: {m['throughput']['requests_per_sec']:.2f}")
    if m.get('quality_slis'):
        print(f"  TCS Stability CV: {m['quality_slis'].get('tcs_stability_cv', 'N/A')}")
        print(f"  RCE β_meta Compliance: {m['quality_slis'].get('rce_beta_meta_compliance', 'N/A')}")
except Exception as e:
    print(f"  ⚠️  Error reading results: {e}")
PYTHON
        cd niodoo_real_integrated
    else
        echo "⚠️  Load test completed but no output file found"
    fi
else
    echo "❌ Load test failed (check logs)"
    cd ..
fi

cd ..
echo ""

# Test 4: Baseline Capture
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 4: Baseline Capture (E2E)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

cd niodoo_real_integrated

echo "Running baseline capture (60 seconds, 8 concurrent users)..."
if timeout 180 cargo run --bin metrics_runner --release --features svc -- \
    --scenario baseline \
    --concurrent-users 8 \
    --duration-secs 60 \
    --target-tokens 512 \
    --output "../$RESULTS_DIR/baseline_e2e.json" \
    --mock-mode \
    2>&1 | tee "../$RESULTS_DIR/baseline_capture.log"; then
    
    if [ -f "../$RESULTS_DIR/baseline_e2e.json" ]; then
        echo ""
        echo "✅ Baseline Captured:"
        cd ..
        python3 <<PYTHON
import json
try:
    with open('$RESULTS_DIR/baseline_e2e.json', 'r') as f:
        m = json.load(f)
    print(f"  p99 latency: {m['latency']['p99_ms']:.2f}ms")
    print(f"  Throughput: {m['throughput']['tokens_per_sec']:.2f} tokens/sec")
    print(f"  Duration: {m['duration_secs']:.2f}s")
except Exception as e:
    print(f"  ⚠️  Error reading results: {e}")
PYTHON
        # Copy to baselines directory
        mkdir -p baselines
        cp "$RESULTS_DIR/baseline_e2e.json" "baselines/baseline-e2e-test.json"
        echo "  ✅ Baseline saved to baselines/baseline-e2e-test.json"
        cd niodoo_real_integrated
    fi
else
    echo "⚠️  Baseline capture had issues (check logs)"
    cd ..
fi

cd ..
echo ""

# Test 5: Ablation Experiment
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 5: Ablation Experiment (E2E)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ -f "$RESULTS_DIR/baseline_e2e.json" ]; then
    cd niodoo_real_integrated
    
    echo "Running ablation experiment: DisableGpuFitness..."
    if timeout 180 cargo run --bin ablation_runner --release --features svc -- \
        --experiment DisableGpuFitness \
        --baseline "../$RESULTS_DIR/baseline_e2e.json" \
        --output-dir "../$RESULTS_DIR/ablation" \
        --concurrent-users 4 \
        --duration-secs 30 \
        2>&1 | tee "../$RESULTS_DIR/ablation_test.log"; then
        
        if [ -d "../$RESULTS_DIR/ablation" ]; then
            echo ""
            echo "✅ Ablation Experiment Completed:"
            cd ..
            ls -lh "$RESULTS_DIR/ablation/"
            if [ -f "$RESULTS_DIR/ablation/ablation-DisableGpuFitness.json" ]; then
                python3 <<PYTHON
import json
try:
    with open('$RESULTS_DIR/ablation/ablation-DisableGpuFitness.json', 'r') as f:
        result = json.load(f)
    if result.get('comparison'):
        comp = result['comparison']
        print(f"  Latency change: {comp.get('latency_change_p99_ms', 0):+.2f}ms ({comp.get('latency_change_pct', 0):+.1f}%)")
        print(f"  Regression detected: {comp.get('regression_detected', False)}")
except Exception as e:
    print(f"  ⚠️  Error reading results: {e}")
PYTHON
            fi
            cd niodoo_real_integrated
        fi
    else
        echo "⚠️  Ablation experiment had issues (check logs)"
        cd ..
    fi
else
    echo "⏭️  Skipping ablation experiment (no baseline available)"
fi

cd ..
echo ""

# Final Summary
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "E2E VALIDATION TEST EXECUTION SUMMARY"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python3 <<PYTHON
import json
import os
import glob
from datetime import datetime

print("Test Execution Results:")
print()

summary = {
    'timestamp': datetime.now().isoformat(),
    'results_dir': '$RESULTS_DIR',
    'tests': {}
}

# Golden Probes
if os.path.exists('$RESULTS_DIR/golden_probes_e2e.json'):
    with open('$RESULTS_DIR/golden_probes_e2e.json', 'r') as f:
        gp = json.load(f)
    summary['tests']['golden_probes'] = {
        'passed': gp['passed'],
        'total': gp['total'],
        'pass_rate': gp['pass_rate']
    }
    print(f"  ✅ Golden Probes: {gp['passed']}/{gp['total']} passed ({gp['pass_rate']*100:.1f}%)")

# Load Test
if os.path.exists('$RESULTS_DIR/load_test_e2e.json'):
    with open('$RESULTS_DIR/load_test_e2e.json', 'r') as f:
        lt = json.load(f)
    summary['tests']['load_test'] = {
        'p99_latency_ms': lt['latency']['p99_ms'],
        'throughput_tokens_per_sec': lt['throughput']['tokens_per_sec']
    }
    print(f"  ✅ Load Test: p99={lt['latency']['p99_ms']:.2f}ms, throughput={lt['throughput']['tokens_per_sec']:.2f} tokens/sec")

# Baseline
if os.path.exists('$RESULTS_DIR/baseline_e2e.json'):
    with open('$RESULTS_DIR/baseline_e2e.json', 'r') as f:
        bl = json.load(f)
    summary['tests']['baseline'] = {
        'p99_latency_ms': bl['latency']['p99_ms'],
        'throughput_tokens_per_sec': bl['throughput']['tokens_per_sec']
    }
    print(f"  ✅ Baseline: Captured successfully")

# Ablation
ablation_files = glob.glob('$RESULTS_DIR/ablation/*.json')
if ablation_files:
    summary['tests']['ablation'] = {'experiments': len(ablation_files)}
    print(f"  ✅ Ablation Experiments: {len(ablation_files)} completed")

# Save summary
with open('$RESULTS_DIR/e2e_test_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print()
print(f"Results directory: $RESULTS_DIR/")
print()
print("🎉 End-to-End Validation Tests Complete!")
print()
print("Next steps:")
print("  - Review results in: $RESULTS_DIR/")
print("  - Compare baseline: ./scripts/compare_baseline.sh $RESULTS_DIR/baseline_e2e.json")
print("  - Check logs for details: $RESULTS_DIR/*.log")
PYTHON

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
