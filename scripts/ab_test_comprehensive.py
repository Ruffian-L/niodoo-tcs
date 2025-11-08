#!/usr/bin/env python3
"""Comprehensive A/B Test Framework for NIODOO System

Replaces traditional test suites with empirical A/B testing to prove system superiority.
Compares baseline vs treatment configurations with statistical analysis.
"""

import asyncio
import json
import os
import time
import requests
import subprocess
import sys
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class TestMetrics:
    """Metrics collected during A/B test"""
    latency_p50: float
    latency_p95: float
    latency_p99: float
    latency_mean: float
    throughput: float
    error_rate: float
    request_count: int
    duration_secs: float

@dataclass
class StatisticalComparison:
    """Statistical comparison results"""
    latency_difference_ms: float
    latency_difference_pct: float
    throughput_difference_pct: float
    p_value_latency: float
    p_value_throughput: float
    cohens_d_latency: float
    cohens_d_throughput: float
    effect_size_latency: str
    effect_size_throughput: str
    statistical_significance: bool
    winner: str

class ABTestFramework:
    """Comprehensive A/B testing framework"""
    
    def __init__(self, vllm_endpoint: str = "http://localhost:5001",
                 qdrant_url: str = "http://localhost:6333"):
        self.vllm_endpoint = vllm_endpoint
        self.qdrant_url = qdrant_url
        
    def test_vllm_endpoint(self) -> bool:
        """Test if vLLM endpoint is available"""
        try:
            resp = requests.get(f"{self.vllm_endpoint}/v1/models", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False
    
    def test_qdrant_endpoint(self) -> bool:
        """Test if Qdrant endpoint is available"""
        try:
            resp = requests.get(f"{self.qdrant_url}/collections", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False
    
    def run_pipeline_test(self, config_env: Dict[str, str], 
                         prompts: List[str], duration_secs: int = 60) -> TestMetrics:
        """Run pipeline test with given configuration"""
        print(f"\n🧪 Running test with config: {list(config_env.keys())}")
        
        # Set environment variables
        env = os.environ.copy()
        env.update(config_env)
        env["MOCK_MODE"] = "false"
        
        # Collect latencies
        latencies = []
        errors = 0
        requests_count = 0
        start_time = time.time()
        end_time = start_time + duration_secs
        
        # Run pipeline requests
        prompt_idx = 0
        while time.time() < end_time:
            prompt = prompts[prompt_idx % len(prompts)]
            prompt_idx += 1
            
            try:
                cycle_start = time.time()
                result = subprocess.run(
                    ["cargo", "run", "--release", "--bin", "niodoo_real_integrated", "--quiet",
                     "--prompt", prompt, "--output", "json"],
                    cwd="/workspace/Niodoo-Final/niodoo_real_integrated",
                    env=env,
                    capture_output=True,
                    timeout=30,
                    text=True
                )
                
                if result.returncode == 0:
                    latency_ms = (time.time() - cycle_start) * 1000.0
                    latencies.append(latency_ms)
                    requests_count += 1
                else:
                    errors += 1
            except subprocess.TimeoutExpired:
                errors += 1
            except Exception as e:
                print(f"   Error: {e}")
                errors += 1
            
            time.sleep(0.1)  # Small delay between requests
        
        actual_duration = time.time() - start_time
        
        if not latencies:
            return TestMetrics(
                latency_p50=0.0, latency_p95=0.0, latency_p99=0.0,
                latency_mean=0.0, throughput=0.0, error_rate=1.0,
                request_count=0, duration_secs=actual_duration
            )
        
        latencies.sort()
        n = len(latencies)
        
        return TestMetrics(
            latency_p50=latencies[int(n * 0.50)],
            latency_p95=latencies[int(n * 0.95)],
            latency_p99=latencies[int(n * 0.99)] if n > 0 else latencies[-1],
            latency_mean=statistics.mean(latencies),
            throughput=requests_count / actual_duration if actual_duration > 0 else 0.0,
            error_rate=errors / (requests_count + errors) if (requests_count + errors) > 0 else 0.0,
            request_count=requests_count,
            duration_secs=actual_duration
        )
    
    def cohens_d(self, values1: List[float], values2: List[float]) -> float:
        """Calculate Cohen's d effect size"""
        if not values1 or not values2:
            return 0.0
        
        mean1 = statistics.mean(values1)
        mean2 = statistics.mean(values2)
        
        var1 = statistics.variance(values1) if len(values1) > 1 else 0.0
        var2 = statistics.variance(values2) if len(values2) > 1 else 0.0
        
        pooled_std = ((var1 + var2) / 2.0) ** 0.5
        if pooled_std == 0.0:
            return 0.0
        
        return (mean1 - mean2) / pooled_std
    
    def t_test_approximate(self, values1: List[float], values2: List[float]) -> float:
        """Approximate t-test p-value"""
        if not values1 or not values2:
            return 1.0
        
        mean1 = statistics.mean(values1)
        mean2 = statistics.mean(values2)
        var1 = statistics.variance(values1) if len(values1) > 1 else 0.0
        var2 = statistics.variance(values2) if len(values2) > 1 else 0.0
        
        n1, n2 = len(values1), len(values2)
        pooled_se = ((var1 / n1) + (var2 / n2)) ** 0.5
        
        if pooled_se == 0.0:
            return 1.0
        
        t_stat = (mean1 - mean2) / pooled_se
        # Simplified p-value approximation (two-tailed)
        # For production, use scipy.stats.ttest_ind
        p_value = 2.0 * (1.0 - abs(t_stat) / (abs(t_stat) + 2.0))
        return min(max(p_value, 0.0), 1.0)
    
    def compare_configurations(self, baseline: TestMetrics, treatment: TestMetrics,
                             significance_threshold: float = 0.05) -> StatisticalComparison:
        """Compare baseline vs treatment with statistical analysis"""
        # Latency comparison
        latency_diff = treatment.latency_p99 - baseline.latency_p99
        latency_diff_pct = (latency_diff / baseline.latency_p99 * 100.0) if baseline.latency_p99 > 0 else 0.0
        
        # Throughput comparison
        throughput_diff_pct = ((treatment.throughput - baseline.throughput) / baseline.throughput * 100.0) if baseline.throughput > 0 else 0.0
        
        # Statistical tests (using p99 as proxy - would need full distributions)
        baseline_latencies = [baseline.latency_p99]
        treatment_latencies = [treatment.latency_p99]
        baseline_throughputs = [baseline.throughput]
        treatment_throughputs = [treatment.throughput]
        
        p_value_latency = self.t_test_approximate(baseline_latencies, treatment_latencies)
        p_value_throughput = self.t_test_approximate(baseline_throughputs, treatment_throughputs)
        
        cohens_d_latency = self.cohens_d(baseline_latencies, treatment_latencies)
        cohens_d_throughput = self.cohens_d(baseline_throughputs, treatment_throughputs)
        
        # Effect size categories
        def effect_size_category(d: float) -> str:
            abs_d = abs(d)
            if abs_d < 0.2:
                return "Small"
            elif abs_d < 0.5:
                return "Medium"
            elif abs_d < 0.8:
                return "Large"
            else:
                return "Very Large"
        
        effect_latency = effect_size_category(cohens_d_latency)
        effect_throughput = effect_size_category(cohens_d_throughput)
        
        # Determine winner
        treatment_wins = (latency_diff < 0.0 or throughput_diff_pct > 0.0) and (
            p_value_latency < significance_threshold or p_value_throughput < significance_threshold
        )
        
        baseline_wins = (latency_diff > 0.0 or throughput_diff_pct < 0.0) and (
            p_value_latency < significance_threshold or p_value_throughput < significance_threshold
        )
        
        if treatment_wins:
            winner = "Treatment"
        elif baseline_wins:
            winner = "Baseline"
        else:
            winner = "Inconclusive"
        
        is_significant = p_value_latency < significance_threshold or p_value_throughput < significance_threshold
        
        return StatisticalComparison(
            latency_difference_ms=latency_diff,
            latency_difference_pct=latency_diff_pct,
            throughput_difference_pct=throughput_diff_pct,
            p_value_latency=p_value_latency,
            p_value_throughput=p_value_throughput,
            cohens_d_latency=cohens_d_latency,
            cohens_d_throughput=cohens_d_throughput,
            effect_size_latency=effect_latency,
            effect_size_throughput=effect_throughput,
            statistical_significance=is_significant,
            winner=winner
        )
    
    def run_ab_test(self, baseline_config: Dict[str, str], 
                   treatment_config: Dict[str, str],
                   prompts: List[str],
                   duration_secs: int = 60) -> Dict:
        """Run complete A/B test"""
        print("=" * 60)
        print("A/B TEST: Baseline vs Treatment")
        print("=" * 60)
        
        # Check prerequisites
        if not self.test_vllm_endpoint():
            print("❌ vLLM endpoint not available")
            sys.exit(1)
        if not self.test_qdrant_endpoint():
            print("❌ Qdrant endpoint not available")
            sys.exit(1)
        
        # Run baseline
        baseline_metrics = self.run_pipeline_test(baseline_config, prompts, duration_secs)
        
        # Run treatment
        treatment_metrics = self.run_pipeline_test(treatment_config, prompts, duration_secs)
        
        # Compare
        comparison = self.compare_configurations(baseline_metrics, treatment_metrics)
        
        # Generate report
        result = {
            "timestamp": datetime.now().isoformat(),
            "baseline": asdict(baseline_metrics),
            "treatment": asdict(treatment_metrics),
            "comparison": asdict(comparison)
        }
        
        return result
    
    def print_report(self, result: Dict):
        """Print formatted A/B test report"""
        print("\n" + "=" * 60)
        print("A/B TEST RESULTS")
        print("=" * 60)
        
        baseline = result["baseline"]
        treatment = result["treatment"]
        comp = result["comparison"]
        
        print(f"\n📊 Baseline Metrics:")
        print(f"   Latency P99: {baseline['latency_p99']:.2f}ms")
        print(f"   Throughput: {baseline['throughput']:.2f} req/s")
        print(f"   Error Rate: {baseline['error_rate']*100:.2f}%")
        
        print(f"\n📊 Treatment Metrics:")
        print(f"   Latency P99: {treatment['latency_p99']:.2f}ms")
        print(f"   Throughput: {treatment['throughput']:.2f} req/s")
        print(f"   Error Rate: {treatment['error_rate']*100:.2f}%")
        
        print(f"\n📈 Comparison:")
        print(f"   Latency Difference: {comp['latency_difference_ms']:.2f}ms ({comp['latency_difference_pct']:.1f}%)")
        print(f"   Throughput Difference: {comp['throughput_difference_pct']:.1f}%")
        print(f"   P-value (Latency): {comp['p_value_latency']:.4f}")
        print(f"   P-value (Throughput): {comp['p_value_throughput']:.4f}")
        print(f"   Cohen's d (Latency): {comp['cohens_d_latency']:.2f} ({comp['effect_size_latency']})")
        print(f"   Cohen's d (Throughput): {comp['cohens_d_throughput']:.2f} ({comp['effect_size_throughput']})")
        
        print(f"\n🏆 Winner: {comp['winner']}")
        print(f"   Statistically Significant: {comp['statistical_significance']}")
        
        print("\n" + "=" * 60)

def main():
    
    # Test prompts
    prompts = [
        "Explain quantum computing in detail",
        "Describe the theory of relativity",
        "What is machine learning?",
        "How does neural network training work?",
        "Explain the concept of consciousness"
    ]
    
    # Baseline configuration (default)
    baseline_config = {
        "TOPOLOGY_MODE": "hybrid",
        "RCE_ENABLED": "1",
        "N_TOKENS_BYPASS": "0",
        "ENABLE_CURATOR": "1",
    }
    
    # Treatment configuration (example: disable RCE)
    treatment_config = {
        "TOPOLOGY_MODE": "hybrid",
        "RCE_ENABLED": "0",  # Disabled
        "N_TOKENS_BYPASS": "0",
        "ENABLE_CURATOR": "1",
    }
    
    framework = ABTestFramework()
    result = framework.run_ab_test(baseline_config, treatment_config, prompts, duration_secs=60)
    framework.print_report(result)
    
    # Save results
    output_file = Path("ab_test_results.json")
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n✅ Results saved to: {output_file}")

if __name__ == "__main__":
    main()

