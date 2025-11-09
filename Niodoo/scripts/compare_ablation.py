#!/usr/bin/env python3
"""
Ablation Test Comparison Script

Compares baseline (fixed compass) vs topology (computed compass) results
to prove that topology-driven filtering improves system performance.
"""

import json
import sys
from pathlib import Path

def load_baseline(path):
    """Load baseline JSON file."""
    with open(path, 'r') as f:
        return json.load(f)

def compare_results(baseline_path, topology_path):
    """Compare baseline vs topology results."""
    baseline = load_baseline(baseline_path)
    topology = load_baseline(topology_path)
    
    print("=" * 60)
    print("ABLATION TEST RESULTS: Baseline vs Topology")
    print("=" * 60)
    print()
    
    print("BASELINE (Fixed Compass Filter):")
    print(f"  Iterations: {baseline['iterations']}")
    print(f"  Avg Quality: {baseline['quality_score']['avg']:.2f}")
    print(f"  Avg ROUGE-L: {baseline['rouge_l']['avg']:.3f}")
    print(f"  Avg Latency: {baseline['latencies_ms']['avg']:.1f}ms")
    print()
    
    print("TOPOLOGY (Computed Compass Filter):")
    print(f"  Iterations: {topology['iterations']}")
    print(f"  Avg Quality: {topology['quality_score']['avg']:.2f}")
    print(f"  Avg ROUGE-L: {topology['rouge_l']['avg']:.3f}")
    print(f"  Avg Latency: {topology['latencies_ms']['avg']:.1f}ms")
    print()
    
    # Calculate improvements
    quality_delta = topology['quality_score']['avg'] - baseline['quality_score']['avg']
    quality_pct = (quality_delta / baseline['quality_score']['avg']) * 100
    
    rouge_delta = topology['rouge_l']['avg'] - baseline['rouge_l']['avg']
    rouge_pct = (rouge_delta / baseline['rouge_l']['avg']) * 100
    
    latency_delta = topology['latencies_ms']['avg'] - baseline['latencies_ms']['avg']
    latency_pct = (latency_delta / baseline['latencies_ms']['avg']) * 100
    
    print("=" * 60)
    print("IMPROVEMENTS:")
    print("=" * 60)
    print(f"  Quality Score: {quality_delta:+.2f} ({quality_pct:+.1f}%)")
    print(f"  ROUGE-L:       {rouge_delta:+.3f} ({rouge_pct:+.1f}%)")
    print(f"  Latency:       {latency_delta:+.1f}ms ({latency_pct:+.1f}%)")
    print()
    
    # Verdict
    if quality_delta > 0.5:  # 5% improvement threshold
        print("✅ VERDICT: Topology-driven filtering IMPROVES quality")
    elif quality_delta < -0.5:
        print("❌ VERDICT: Topology-driven filtering DEGRADES quality")
    else:
        print("⚠️  VERDICT: No significant difference (within noise)")
    print()
    
    # Statistical significance (simple threshold)
    if abs(quality_delta) < 0.3:
        print("⚠️  Note: Difference may not be statistically significant")
        print("   Consider running more iterations for stronger evidence")
    
    return {
        "quality_improvement": quality_delta,
        "quality_improvement_pct": quality_pct,
        "rouge_improvement": rouge_delta,
        "verdict": "improved" if quality_delta > 0.5 else "degraded" if quality_delta < -0.5 else "neutral"
    }

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: compare_ablation.py <baseline.json> <topology.json>")
        sys.exit(1)
    
    baseline_path = sys.argv[1]
    topology_path = sys.argv[2]
    
    if not Path(baseline_path).exists():
        print(f"Error: Baseline file not found: {baseline_path}")
        sys.exit(1)
    
    if not Path(topology_path).exists():
        print(f"Error: Topology file not found: {topology_path}")
        sys.exit(1)
    
    result = compare_results(baseline_path, topology_path)
    
    # Write summary
    with open("reports/ablation_summary.json", "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"\nSummary written to: reports/ablation_summary.json")

