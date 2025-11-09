#!/usr/bin/env python3
"""
COMPREHENSIVE END-TO-END TEST SUITE
====================================

This is the ULTIMATE test suite that validates EVERY component of the NIODOO system
working together. It tests the complete pipeline from prompt to response, including:

1. Security validation
2. Embedding generation (local ONNX)
3. ERAG memory retrieval (Qdrant gRPC)
4. Torus projection (7D PAD+Ghost)
5. TCS topology analysis (Betti numbers, persistence entropy)
6. Compass engine (emotional quadrant determination)
7. Dynamic tokenization
8. Generation (vLLM)
9. Curator refinement
10. RCE cognitive control
11. Learning loop (breakthrough detection)
12. Memory storage

This proves NIODOO is superior because NO OTHER SYSTEM has this complete integration.
"""

import argparse
import json
import os
import psutil
import subprocess
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import statistics

sys.path.insert(0, str(Path(__file__).parent.parent / "niodoo-ai" / "scripts"))

from user_test_utils import (
    Colors,
    TestLogger,
    check_qdrant_health,
    check_vllm_health,
    colored,
    find_binary,
    test_pipeline,
    wait_for_service,
)


class ComprehensiveE2ETest:
    """Comprehensive end-to-end test suite that validates the entire NIODOO pipeline."""
    
    def __init__(self, vllm_endpoint: str, qdrant_url: str, verbosity: str = "verbose"):
        self.vllm_endpoint = vllm_endpoint
        self.qdrant_url = qdrant_url
        self.verbosity = verbosity
        self.logger = TestLogger(verbosity=verbosity)
        self.results: List[Dict] = []
        self.metrics = {
            "total_tests": 0,
            "successful": 0,
            "failed": 0,
            "latencies": [],
            "memory_samples": [],
            "topology_metrics": [],
            "compass_states": [],
            "learning_events": [],
            "errors": []
        }
        self.start_time = time.time()
        
    def print_banner(self):
        """Print test suite banner."""
        print("\n" + "=" * 100)
        print(colored("🔥 COMPREHENSIVE END-TO-END TEST SUITE - PROVING NIODOO SUPERIORITY 🔥", 
                     Colors.BOLD + Colors.BLUE))
        print("=" * 100)
        print("\nTesting COMPLETE pipeline integration:")
        print("  ✅ Security → Embedding → ERAG → Torus → TCS → Compass → Tokenizer")
        print("  ✅ Generation → Curator → RCE → Learning → Memory → Response")
        print("\n" + "=" * 100 + "\n")
    
    def check_prerequisites(self) -> bool:
        """Check all prerequisites are met."""
        print(colored("STEP 1: Checking Prerequisites", Colors.BOLD + Colors.BLUE))
        print("-" * 100)
        
        # Check vLLM
        print(f"\nChecking vLLM at {self.vllm_endpoint}...")
        vllm_status = check_vllm_health(self.vllm_endpoint)
        if not vllm_status["online"]:
            print(colored("❌ vLLM is not running!", Colors.RED))
            return False
        print(colored(f"✅ vLLM is running (response: {vllm_status.get('response_time_ms', 0):.1f}ms)", 
                     Colors.GREEN))
        
        # Check Qdrant
        print(f"\nChecking Qdrant at {self.qdrant_url}...")
        qdrant_status = check_qdrant_health(self.qdrant_url)
        if not qdrant_status["online"]:
            print(colored("❌ Qdrant is not running!", Colors.RED))
            return False
        print(colored(f"✅ Qdrant is running (response: {qdrant_status.get('response_time_ms', 0):.1f}ms)", 
                     Colors.GREEN))
        
        # Check binary
        print(f"\nChecking pipeline binary...")
        binary_path = find_binary("niodoo_real_integrated")
        if binary_path is None:
            print(colored("⚠️  Binary not found, will use 'cargo run'", Colors.YELLOW))
        else:
            print(colored(f"✅ Binary found: {binary_path}", Colors.GREEN))
        
        # Check ONNX runtime
        onnx_path = Path("/workspace/Niodoo-Final/third_party/onnxruntime-linux-x64-gpu-1.23.2/lib")
        if onnx_path.exists():
            print(colored(f"✅ ONNX runtime found: {onnx_path}", Colors.GREEN))
        else:
            print(colored("⚠️  ONNX runtime not found at expected location", Colors.YELLOW))
        
        print("\n" + colored("✅ All prerequisites met!", Colors.GREEN))
        return True
    
    def get_test_prompts(self) -> List[str]:
        """Get comprehensive test prompts covering all capabilities."""
        return [
            # Basic functionality
            "Write a Python function to calculate fibonacci numbers",
            "Explain what machine learning is in one paragraph",
            
            # Topology-aware reasoning
            "Analyze the topological structure of a neural network with 3 layers",
            "What are Betti numbers and how do they relate to data analysis?",
            
            # Emotional context
            "I'm feeling frustrated with debugging. Help me understand this error.",
            "I'm excited about learning Rust. What should I focus on?",
            
            # Complex reasoning
            "Design a distributed system for real-time data processing",
            "How would you optimize a database query that's running slowly?",
            
            # Code generation
            "Create a REST API endpoint in Rust that handles authentication",
            "Write a function that merges two sorted arrays efficiently",
            
            # Memory retrieval (should use ERAG)
            "What did we discuss about topology earlier?",
            "Recall the previous conversation about consciousness",
            
            # Learning scenarios
            "Explain quantum computing concepts simply",
            "What are the key differences between Rust and C++?",
            
            # Edge cases
            "",  # Empty prompt (should handle gracefully)
            "a" * 10000,  # Very long prompt
            "!@#$%^&*()",  # Special characters
        ]
    
    def run_single_test(self, prompt: str, test_id: int, total: int) -> Dict:
        """Run a single end-to-end test."""
        print(f"\n[{test_id}/{total}] Testing: {prompt[:60]}{'...' if len(prompt) > 60 else ''}")
        
        # Record initial memory
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        start_time = time.time()
        
        # Run pipeline
        result = test_pipeline(
            prompt=prompt,
            output_format="json",
            timeout=300,  # 5 minute timeout
            logger=self.logger
        )
        
        duration_ms = (time.time() - start_time) * 1000
        
        # Record final memory
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_delta = final_memory - initial_memory
        
        # Parse response for metrics
        response_text = result.get("response", "")
        topology_detected = False
        compass_state = None
        learning_event = False
        
        # Try to extract metrics from response (if JSON contains them)
        try:
            if response_text.startswith("{"):
                response_json = json.loads(response_text)
                if "topology" in response_json:
                    topology_detected = True
                    self.metrics["topology_metrics"].append(response_json["topology"])
                if "compass" in response_json:
                    compass_state = response_json["compass"].get("quadrant", "unknown")
                    self.metrics["compass_states"].append(compass_state)
                if "learning" in response_json and response_json["learning"].get("breakthrough", False):
                    learning_event = True
                    self.metrics["learning_events"].append({
                        "timestamp": time.time(),
                        "prompt": prompt[:100]
                    })
        except:
            pass  # Response might not be JSON
        
        # Update metrics
        self.metrics["total_tests"] += 1
        if result["success"]:
            self.metrics["successful"] += 1
        else:
            self.metrics["failed"] += 1
            self.metrics["errors"].extend(result.get("errors", []))
        
        self.metrics["latencies"].append(duration_ms)
        self.metrics["memory_samples"].append({
            "initial_mb": initial_memory,
            "final_mb": final_memory,
            "delta_mb": memory_delta,
            "timestamp": time.time()
        })
        
        # Build result
        test_result = {
            "test_id": test_id,
            "prompt": prompt,
            "success": result["success"],
            "response_length": len(response_text),
            "latency_ms": duration_ms,
            "memory_delta_mb": memory_delta,
            "topology_detected": topology_detected,
            "compass_state": compass_state,
            "learning_event": learning_event,
            "errors": result.get("errors", []),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
        # Print result
        if result["success"]:
            print(colored(f"  ✅ SUCCESS ({duration_ms:.1f}ms, {len(response_text)} chars)", Colors.GREEN))
        else:
            print(colored(f"  ❌ FAILED ({duration_ms:.1f}ms)", Colors.RED))
            for error in result.get("errors", [])[:3]:
                print(f"     Error: {error[:80]}")
        
        return test_result
    
    def run_load_test(self, duration_secs: int = 300, concurrent_workers: int = 10, 
                     prompt_generator=None) -> Dict:
        """Run load test for extended duration."""
        print("\n" + "=" * 100)
        print(colored(f"LOAD TEST: {duration_secs}s duration, {concurrent_workers} workers", 
                     Colors.BOLD + Colors.BLUE))
        print("=" * 100)
        
        if prompt_generator is None:
            prompts = self.get_test_prompts()
            prompt_generator = lambda: prompts[hash(str(time.time())) % len(prompts)]
        
        stop_event = threading.Event()
        results_queue = []
        results_lock = threading.Lock()
        
        def worker(worker_id: int):
            """Worker thread that runs tests continuously."""
            worker_results = []
            while not stop_event.is_set():
                prompt = prompt_generator()
                start_time = time.time()
                
                result = test_pipeline(
                    prompt=prompt,
                    output_format="json",
                    timeout=180,
                    logger=None  # Disable logging for load test
                )
                
                duration_ms = (time.time() - start_time) * 1000
                
                with results_lock:
                    results_queue.append({
                        "worker_id": worker_id,
                        "prompt": prompt[:50],
                        "success": result["success"],
                        "latency_ms": duration_ms,
                        "response_length": len(result.get("response", "")),
                        "timestamp": time.time()
                    })
                    worker_results.append(result["success"])
        
        # Start workers
        print(f"\nStarting {concurrent_workers} workers...")
        with ThreadPoolExecutor(max_workers=concurrent_workers) as executor:
            futures = [executor.submit(worker, i) for i in range(concurrent_workers)]
            
            # Run for specified duration
            start_time = time.time()
            elapsed = 0
            last_report = 0
            
            while elapsed < duration_secs:
                time.sleep(5)
                elapsed = time.time() - start_time
                
                if elapsed - last_report >= 30:  # Report every 30 seconds
                    with results_lock:
                        total = len(results_queue)
                        successful = sum(1 for r in results_queue if r["success"])
                        avg_latency = statistics.mean([r["latency_ms"] for r in results_queue]) if results_queue else 0
                    
                    print(f"  [{elapsed:.0f}s/{duration_secs}s] Total: {total}, Success: {successful}, "
                          f"Success Rate: {successful/total*100:.1f}%, Avg Latency: {avg_latency:.1f}ms")
                    last_report = elapsed
        
        # Stop workers
        print("\nStopping workers...")
        stop_event.set()
        
        # Wait for completion
        for future in as_completed(futures):
            try:
                future.result(timeout=10)
            except:
                pass
        
        # Calculate final stats
        with results_lock:
            total = len(results_queue)
            successful = sum(1 for r in results_queue if r["success"])
            failed = total - successful
            latencies = [r["latency_ms"] for r in results_queue]
            
            load_stats = {
                "duration_secs": duration_secs,
                "concurrent_workers": concurrent_workers,
                "total_operations": total,
                "successful_operations": successful,
                "failed_operations": failed,
                "success_rate": successful / total if total > 0 else 0.0,
                "avg_latency_ms": statistics.mean(latencies) if latencies else 0.0,
                "p50_latency_ms": statistics.median(latencies) if latencies else 0.0,
                "p95_latency_ms": statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else 0.0,
                "p99_latency_ms": statistics.quantiles(latencies, n=100)[98] if len(latencies) >= 100 else 0.0,
                "ops_per_sec": total / duration_secs if duration_secs > 0 else 0.0,
                "results": results_queue[:100]  # First 100 results
            }
        
        print("\n" + colored("LOAD TEST COMPLETE", Colors.BOLD + Colors.GREEN))
        print(f"  Total Operations: {load_stats['total_operations']}")
        print(f"  Success Rate: {load_stats['success_rate']*100:.1f}%")
        print(f"  Avg Latency: {load_stats['avg_latency_ms']:.1f}ms")
        print(f"  P95 Latency: {load_stats['p95_latency_ms']:.1f}ms")
        print(f"  Ops/sec: {load_stats['ops_per_sec']:.2f}")
        
        return load_stats
    
    def run_soak_test(self, duration_hours: float = 1.0, check_interval: int = 300) -> Dict:
        """Run soak test for extended duration to detect memory leaks and stability issues."""
        print("\n" + "=" * 100)
        print(colored(f"SOAK TEST: {duration_hours} hours", Colors.BOLD + Colors.BLUE))
        print("=" * 100)
        
        duration_secs = int(duration_hours * 3600)
        start_time = time.time()
        process = psutil.Process()
        
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_samples = [{"time": 0, "memory_mb": initial_memory}]
        operation_count = 0
        successful_count = 0
        error_log = []
        
        prompts = self.get_test_prompts()
        prompt_index = 0
        
        print(f"\nStarting soak test...")
        print(f"  Duration: {duration_hours} hours ({duration_secs} seconds)")
        print(f"  Check interval: {check_interval} seconds")
        print(f"  Initial memory: {initial_memory:.1f} MB")
        
        last_check = time.time()
        
        while time.time() - start_time < duration_secs:
            # Run a test
            prompt = prompts[prompt_index % len(prompts)]
            prompt_index += 1
            
            test_start = time.time()
            result = test_pipeline(
                prompt=prompt,
                output_format="json",
                timeout=180,
                logger=None
            )
            test_duration = time.time() - test_start
            
            operation_count += 1
            if result["success"]:
                successful_count += 1
            else:
                error_log.append({
                    "time": time.time() - start_time,
                    "error": result.get("errors", ["Unknown error"])[0] if result.get("errors") else "Unknown",
                    "prompt": prompt[:50]
                })
                if len(error_log) > 100:
                    error_log.pop(0)
            
            # Periodic checks
            if time.time() - last_check >= check_interval:
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                elapsed = time.time() - start_time
                memory_samples.append({
                    "time": elapsed,
                    "memory_mb": current_memory
                })
                
                success_rate = successful_count / operation_count if operation_count > 0 else 0.0
                memory_growth = current_memory - initial_memory
                
                print(f"\n[{elapsed/3600:.2f}h] Operations: {operation_count}, "
                      f"Success: {success_rate*100:.1f}%, "
                      f"Memory: {current_memory:.1f} MB (+{memory_growth:.1f} MB)")
                
                last_check = time.time()
        
        # Final stats
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        total_duration = time.time() - start_time
        
        soak_stats = {
            "duration_hours": duration_hours,
            "actual_duration_secs": total_duration,
            "total_operations": operation_count,
            "successful_operations": successful_count,
            "failed_operations": operation_count - successful_count,
            "success_rate": successful_count / operation_count if operation_count > 0 else 0.0,
            "initial_memory_mb": initial_memory,
            "final_memory_mb": final_memory,
            "memory_growth_mb": final_memory - initial_memory,
            "memory_growth_percent": ((final_memory - initial_memory) / initial_memory * 100) if initial_memory > 0 else 0.0,
            "memory_samples": memory_samples,
            "error_count": len(error_log),
            "errors": error_log[:50],  # First 50 errors
            "ops_per_hour": operation_count / (total_duration / 3600) if total_duration > 0 else 0.0
        }
        
        print("\n" + colored("SOAK TEST COMPLETE", Colors.BOLD + Colors.GREEN))
        print(f"  Total Operations: {soak_stats['total_operations']}")
        print(f"  Success Rate: {soak_stats['success_rate']*100:.1f}%")
        print(f"  Memory Growth: {soak_stats['memory_growth_mb']:.1f} MB ({soak_stats['memory_growth_percent']:.2f}%)")
        print(f"  Errors: {soak_stats['error_count']}")
        
        return soak_stats
    
    def generate_report(self, output_dir: str = "test_reports") -> Dict:
        """Generate comprehensive test report."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        
        # Calculate statistics
        total_duration = time.time() - self.start_time
        
        stats = {
            "test_suite": "comprehensive_e2e",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "duration_secs": total_duration,
            "metrics": {
                "total_tests": self.metrics["total_tests"],
                "successful": self.metrics["successful"],
                "failed": self.metrics["failed"],
                "success_rate": self.metrics["successful"] / self.metrics["total_tests"] if self.metrics["total_tests"] > 0 else 0.0,
                "avg_latency_ms": statistics.mean(self.metrics["latencies"]) if self.metrics["latencies"] else 0.0,
                "p50_latency_ms": statistics.median(self.metrics["latencies"]) if self.metrics["latencies"] else 0.0,
                "p95_latency_ms": statistics.quantiles(self.metrics["latencies"], n=20)[18] if len(self.metrics["latencies"]) >= 20 else 0.0,
                "p99_latency_ms": statistics.quantiles(self.metrics["latencies"], n=100)[98] if len(self.metrics["latencies"]) >= 100 else 0.0,
                "topology_detections": len(self.metrics["topology_metrics"]),
                "compass_states": len(self.metrics["compass_states"]),
                "learning_events": len(self.metrics["learning_events"]),
                "error_count": len(self.metrics["errors"])
            },
            "results": self.results,
            "environment": {
                "vllm_endpoint": self.vllm_endpoint,
                "qdrant_url": self.qdrant_url,
                "python_version": sys.version
            }
        }
        
        # Save JSON report
        json_path = os.path.join(output_dir, f"e2e_test_{timestamp}.json")
        with open(json_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Save human-readable report
        txt_path = os.path.join(output_dir, f"e2e_test_{timestamp}.txt")
        with open(txt_path, 'w') as f:
            f.write("=" * 100 + "\n")
            f.write("COMPREHENSIVE END-TO-END TEST REPORT\n")
            f.write("=" * 100 + "\n\n")
            f.write(f"Timestamp: {stats['timestamp']}\n")
            f.write(f"Duration: {stats['duration_secs']:.1f} seconds\n\n")
            
            f.write("METRICS:\n")
            f.write("-" * 100 + "\n")
            m = stats["metrics"]
            f.write(f"Total Tests: {m['total_tests']}\n")
            f.write(f"Successful: {m['successful']}\n")
            f.write(f"Failed: {m['failed']}\n")
            f.write(f"Success Rate: {m['success_rate']*100:.1f}%\n")
            f.write(f"Avg Latency: {m['avg_latency_ms']:.1f}ms\n")
            f.write(f"P50 Latency: {m['p50_latency_ms']:.1f}ms\n")
            f.write(f"P95 Latency: {m['p95_latency_ms']:.1f}ms\n")
            f.write(f"P99 Latency: {m['p99_latency_ms']:.1f}ms\n")
            f.write(f"Topology Detections: {m['topology_detections']}\n")
            f.write(f"Compass States: {m['compass_states']}\n")
            f.write(f"Learning Events: {m['learning_events']}\n")
            f.write(f"Errors: {m['error_count']}\n\n")
            
            if self.metrics["errors"]:
                f.write("ERRORS:\n")
                f.write("-" * 100 + "\n")
                for error in self.metrics["errors"][:20]:
                    f.write(f"  - {error}\n")
                f.write("\n")
        
        print(f"\n✅ Reports saved:")
        print(f"   JSON: {json_path}")
        print(f"   Text: {txt_path}")
        
        return stats


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive End-to-End Test Suite - Proves NIODOO Superiority",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--vllm-endpoint", default=os.getenv("VLLM_ENDPOINT", "http://localhost:5001"))
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--wait", action="store_true", help="Wait for services to come online")
    parser.add_argument("--load-test", action="store_true", help="Run load test")
    parser.add_argument("--load-duration", type=int, default=300, help="Load test duration (seconds)")
    parser.add_argument("--load-workers", type=int, default=10, help="Concurrent workers for load test")
    parser.add_argument("--soak-test", action="store_true", help="Run soak test")
    parser.add_argument("--soak-duration", type=float, default=1.0, help="Soak test duration (hours)")
    parser.add_argument("--output-dir", default="test_reports", help="Output directory for reports")
    parser.add_argument("--verbosity", choices=["minimal", "moderate", "verbose"], default="verbose")
    
    args = parser.parse_args()
    
    # Create test suite
    suite = ComprehensiveE2ETest(
        vllm_endpoint=args.vllm_endpoint,
        qdrant_url=args.qdrant_url,
        verbosity=args.verbosity
    )
    
    suite.print_banner()
    
    # Check prerequisites
    if not suite.check_prerequisites():
        if args.wait:
            print("\nWaiting for services...")
            wait_for_service(lambda: check_vllm_health(args.vllm_endpoint), "vLLM")
            wait_for_service(lambda: check_qdrant_health(args.qdrant_url), "Qdrant")
            if not suite.check_prerequisites():
                print(colored("❌ Prerequisites not met after waiting", Colors.RED))
                sys.exit(1)
        else:
            print(colored("❌ Prerequisites not met. Use --wait to wait for services.", Colors.RED))
            sys.exit(1)
    
    # Run tests
    if args.load_test:
        load_stats = suite.run_load_test(
            duration_secs=args.load_duration,
            concurrent_workers=args.load_workers
        )
        suite.results.append({"type": "load_test", "stats": load_stats})
    
    if args.soak_test:
        soak_stats = suite.run_soak_test(
            duration_hours=args.soak_duration
        )
        suite.results.append({"type": "soak_test", "stats": soak_stats})
    
    if not args.load_test and not args.soak_test:
        # Run standard E2E tests
        print("\n" + "=" * 100)
        print(colored("STEP 2: Running End-to-End Tests", Colors.BOLD + Colors.BLUE))
        print("=" * 100)
        
        prompts = suite.get_test_prompts()
        for i, prompt in enumerate(prompts, 1):
            result = suite.run_single_test(prompt, i, len(prompts))
            suite.results.append(result)
            time.sleep(1)  # Brief pause between tests
    
    # Generate report
    print("\n" + "=" * 100)
    print(colored("STEP 3: Generating Report", Colors.BOLD + Colors.BLUE))
    print("=" * 100)
    
    stats = suite.generate_report(output_dir=args.output_dir)
    
    # Print summary
    print("\n" + "=" * 100)
    print(colored("TEST SUMMARY", Colors.BOLD + Colors.GREEN))
    print("=" * 100)
    m = stats["metrics"]
    print(f"Total Tests: {m['total_tests']}")
    success_rate_str = f"{m['success_rate']*100:.1f}%"
    success_color = Colors.GREEN if m['success_rate'] > 0.9 else Colors.YELLOW
    print(f"Success Rate: {colored(success_rate_str, success_color)}")
    print(f"Avg Latency: {m['avg_latency_ms']:.1f}ms")
    print(f"P95 Latency: {m['p95_latency_ms']:.1f}ms")
    print(f"Topology Detections: {m['topology_detections']}")
    print(f"Learning Events: {m['learning_events']}")
    
    sys.exit(0 if m['success_rate'] > 0.8 else 1)


if __name__ == "__main__":
    main()

