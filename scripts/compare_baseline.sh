#!/bin/bash
# Baseline Comparison Script
# Performs statistical analysis comparing current metrics against baseline

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR/.."
cd "$ROOT_DIR"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
CURRENT_FILE="${1:-metrics_report.json}"
BASELINE_FILE="${2:-baselines/baseline-latest.json}"

if [ ! -f "$CURRENT_FILE" ]; then
    echo -e "${RED}❌ Error: Current metrics file not found: $CURRENT_FILE${NC}"
    exit 1
fi

if [ ! -f "$BASELINE_FILE" ]; then
    echo -e "${RED}❌ Error: Baseline file not found: $BASELINE_FILE${NC}"
    echo "Run ./scripts/capture_baseline.sh first to create a baseline"
    exit 1
fi

echo -e "${BLUE}📊 Comparing Metrics Against Baseline${NC}"
echo "=========================================="
echo "Current:  $CURRENT_FILE"
echo "Baseline: $BASELINE_FILE"
echo ""

# Use Python for statistical analysis
python3 <<EOF
import json
import sys
import math
from typing import List, Tuple, Optional

def bootstrap_percentile_ci(values: List[float], percentile: float, n_samples: int = 10000, confidence: float = 0.95) -> Tuple[float, float]:
    """Bootstrap confidence interval for a percentile metric"""
    import random
    if not values:
        return (0.0, 0.0)
    
    n = len(values)
    bootstrap_samples = []
    
    for _ in range(n_samples):
        sample = [values[random.randint(0, n-1)] for _ in range(n)]
        sample.sort()
        index = int(percentile * (len(sample) - 1))
        bootstrap_samples.append(sample[index])
    
    bootstrap_samples.sort()
    alpha = 1.0 - confidence
    lower_idx = int(alpha / 2.0 * n_samples)
    upper_idx = int((1.0 - alpha / 2.0) * n_samples)
    
    return (bootstrap_samples[lower_idx], bootstrap_samples[upper_idx])

def cohens_d(values1: List[float], values2: List[float]) -> float:
    """Compute Cohen's d effect size"""
    if not values1 or not values2:
        return 0.0
    
    mean1 = sum(values1) / len(values1)
    mean2 = sum(values2) / len(values2)
    
    var1 = sum((x - mean1) ** 2 for x in values1) / len(values1)
    var2 = sum((x - mean2) ** 2 for x in values2) / len(values2)
    
    pooled_std = math.sqrt((var1 + var2) / 2.0)
    if pooled_std == 0.0:
        return 0.0
    
    return (mean1 - mean2) / pooled_std

def interpret_effect_size(d: float) -> str:
    """Interpret Cohen's d effect size"""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"

def compare_metric(baseline: float, current: float, name: str, lower_is_better: bool = True) -> dict:
    """Compare a single metric value"""
    diff = current - baseline
    pct_change = (diff / baseline * 100) if baseline != 0 else 0.0
    
    is_regression = False
    if lower_is_better:
        is_regression = diff > 0  # Increase is bad
    else:
        is_regression = diff < 0  # Decrease is bad
    
    status = "✅" if not is_regression else "⚠️"
    if abs(pct_change) < 1.0:
        status = "➡️"  # No significant change
    
    return {
        "name": name,
        "baseline": baseline,
        "current": current,
        "diff": diff,
        "pct_change": pct_change,
        "is_regression": is_regression,
        "status": status
    }

def compare_latency_metrics(baseline: dict, current: dict) -> List[dict]:
    """Compare latency metrics"""
    results = []
    
    for percentile in ["p50", "p95", "p99"]:
        baseline_key = f"{percentile}_ms"
        current_key = f"{percentile}_ms"
        
        if baseline_key in baseline and current_key in current:
            b_val = baseline[baseline_key]
            c_val = current[current_key]
            results.append(compare_metric(b_val, c_val, f"Latency {percentile}", lower_is_better=True))
    
    return results

def compare_quality_slis(baseline: dict, current: dict) -> List[dict]:
    """Compare Quality SLI metrics"""
    results = []
    
    slis = [
        ("tcs_stability_cv", "TCS Stability CV", True, 0.1),  # SLO: < 0.1
        ("rce_beta_meta_compliance", "RCE β_meta Compliance", False, None),  # SLO: in [0.8, 1.2]
    ]
    
    for key, name, lower_is_better, slo_threshold in slis:
        b_val = baseline.get(key)
        c_val = current.get(key)
        
        if b_val is not None and c_val is not None:
            result = compare_metric(b_val, c_val, name, lower_is_better)
            
            # Check SLO compliance
            if slo_threshold is not None:
                if lower_is_better:
                    result["slo_compliant"] = c_val < slo_threshold
                else:
                    # For compliance, check if in range [0.8, 1.2]
                    result["slo_compliant"] = 0.8 <= c_val <= 1.2
            
            results.append(result)
    
    return results

# Load JSON files
try:
    with open("$CURRENT_FILE", 'r') as f:
        current_data = json.load(f)
    
    with open("$BASELINE_FILE", 'r') as f:
        baseline_data = json.load(f)
except Exception as e:
    print(f"❌ Error loading JSON files: {e}", file=sys.stderr)
    sys.exit(1)

print("=" * 60)
print("LATENCY METRICS COMPARISON")
print("=" * 60)

latency_results = compare_latency_metrics(baseline_data.get("latency", {}), current_data.get("latency", {}))
for result in latency_results:
    status_icon = result["status"]
    print(f"{status_icon} {result['name']:20s} | Baseline: {result['baseline']:8.2f} ms | Current: {result['current']:8.2f} ms | Change: {result['pct_change']:+6.2f}%")

print("")
print("=" * 60)
print("THROUGHPUT METRICS COMPARISON")
print("=" * 60)

throughput_b = baseline_data.get("throughput", {})
throughput_c = current_data.get("throughput", {})

if "requests_per_sec" in throughput_b and "requests_per_sec" in throughput_c:
    result = compare_metric(throughput_b["requests_per_sec"], throughput_c["requests_per_sec"], "Requests/sec", lower_is_better=False)
    print(f"{result['status']} {result['name']:20s} | Baseline: {result['baseline']:8.2f} | Current: {result['current']:8.2f} | Change: {result['pct_change']:+6.2f}%")

if "tokens_per_sec" in throughput_b and "tokens_per_sec" in throughput_c:
    result = compare_metric(throughput_b["tokens_per_sec"], throughput_c["tokens_per_sec"], "Tokens/sec", lower_is_better=False)
    print(f"{result['status']} {result['name']:20s} | Baseline: {result['baseline']:8.2f} | Current: {result['current']:8.2f} | Change: {result['pct_change']:+6.2f}%")

print("")
print("=" * 60)
print("QUALITY SLI COMPARISON")
print("=" * 60)

sli_results = compare_quality_slis(baseline_data.get("quality_slis", {}), current_data.get("quality_slis", {}))
for result in sli_results:
    status_icon = result["status"]
    slo_status = "✅ SLO" if result.get("slo_compliant", True) else "❌ SLO BREACH"
    print(f"{status_icon} {result['name']:30s} | Baseline: {result['baseline']:8.4f} | Current: {result['current']:8.4f} | {slo_status}")

print("")
print("=" * 60)
print("SUMMARY")
print("=" * 60)

regressions = [r for r in latency_results + sli_results if r.get("is_regression", False)]
if regressions:
    print(f"⚠️  {len(regressions)} regression(s) detected:")
    for r in regressions:
        print(f"   - {r['name']}: {r['pct_change']:+.2f}%")
else:
    print("✅ No regressions detected")

print("")
print("For detailed statistical analysis (bootstrap CI, Cohen's d),")
print("use the Rust validation/stats module or extend this script.")
EOF

echo ""
echo -e "${GREEN}✅ Comparison complete${NC}"

