#!/usr/bin/env python3
"""
Complete Validation Study - Does Topology Actually Help?
Compares baseline vs topology-trained models on real code tasks
"""

import torch
import json
import time
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from rouge_score import rouge_scorer
from datetime import datetime

class ValidationRunner:
    def __init__(self):
        self.base_model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
        self.topology_adapter = "outputs/qwen25-coder-topology-601samples"
        self.results = {
            "baseline": [],
            "topology": [],
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "test_samples": 0
            }
        }
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
    def load_model(self, use_adapter=False):
        """Load base model with optional LoRA adapter"""
        print(f"\nLoading {'topology-trained' if use_adapter else 'baseline'} model...")
        
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        
        if use_adapter:
            model = PeftModel.from_pretrained(model, self.topology_adapter)
            
        return model, tokenizer
    
    def create_test_cases(self):
        """Generate test cases for code understanding"""
        return [
            {
                "prompt": "Find the bug in this code:\ndef get_last_element(arr):\n    return arr[len(arr)]\n",
                "topology": "[TOPOLOGY]\nBetti-0: 1\nBetti-1: 0\nPersistence: 0.34\nKnot complexity: 1.2\n",
                "expected": "off-by-one error",
                "task": "bug_detection"
            },
            {
                "prompt": "Explain what this function does:\ndef mystery(n):\n    if n <= 1: return 1\n    return n * mystery(n-1)\n",
                "topology": "[TOPOLOGY]\nBetti-0: 1\nBetti-1: 1 (recursion detected)\nPersistence: 0.89\nKnot complexity: 3.4\n",
                "expected": "factorial",
                "task": "code_understanding"
            },
            {
                "prompt": "Suggest a refactoring for:\nif x > 0:\n    result = True\nelse:\n    result = False\n",
                "topology": "[TOPOLOGY]\nBetti-0: 1\nBetti-1: 0\nPersistence: 0.12\nKnot complexity: 0.8\n",
                "expected": "result = x > 0",
                "task": "refactoring"
            },
            {
                "prompt": "What's wrong with this loop?\nfor i in range(10):\n    print(i)\n    i = 0\n",
                "topology": "[TOPOLOGY]\nBetti-0: 1\nBetti-1: 1 (infinite loop risk)\nPersistence: 0.67\nKnot complexity: 2.8\n",
                "expected": "reassignment doesn't affect loop",
                "task": "bug_detection"
            },
            {
                "prompt": "Optimize this code:\nsum = 0\nfor item in items:\n    if item > 0:\n        sum += item\n",
                "topology": "[TOPOLOGY]\nBetti-0: 1\nBetti-1: 0\nPersistence: 0.45\nKnot complexity: 1.5\n",
                "expected": "sum(x for x in items if x > 0)",
                "task": "optimization"
            }
        ]
    
    def test_model(self, model, tokenizer, test_cases, use_topology=False):
        """Run test cases through model"""
        results = []
        
        for i, test in enumerate(test_cases, 1):
            print(f"  Test {i}/{len(test_cases)}: {test['task']}...", end=" ")
            
            # Build prompt
            if use_topology:
                prompt = f"[INST] {test['prompt']}\n{test['topology']}[/INST]"
            else:
                prompt = f"[INST] {test['prompt']} [/INST]"
            
            # Generate response
            start_time = time.time()
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.7,
                    do_sample=True
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(prompt):].strip()
            latency = time.time() - start_time
            
            # Calculate ROUGE score
            scores = self.scorer.score(test['expected'].lower(), response.lower())
            
            # Check if expected answer is in response
            contains_answer = test['expected'].lower() in response.lower()
            
            result = {
                "task": test['task'],
                "prompt": test['prompt'][:50] + "...",
                "response": response[:200],
                "rouge1": scores['rouge1'].fmeasure,
                "rouge2": scores['rouge2'].fmeasure,
                "rougeL": scores['rougeL'].fmeasure,
                "contains_expected": contains_answer,
                "latency_ms": latency * 1000
            }
            
            results.append(result)
            print(f"ROUGE-L: {result['rougeL']:.3f}, Correct: {contains_answer}")
            
        return results
    
    def run_full_validation(self):
        """Run complete validation study"""
        print("="*70)
        print("NIODOO VALIDATION STUDY")
        print("Question: Does topology training actually help?")
        print("="*70)
        
        test_cases = self.create_test_cases()
        self.results["metadata"]["test_samples"] = len(test_cases)
        
        # Test 1: Baseline model (no topology training)
        print("\n[1/2] Testing BASELINE model (no topology)...")
        base_model, base_tokenizer = self.load_model(use_adapter=False)
        self.results["baseline"] = self.test_model(base_model, base_tokenizer, test_cases, use_topology=False)
        del base_model
        torch.cuda.empty_cache()
        
        # Test 2: Topology-trained model
        print("\n[2/2] Testing TOPOLOGY-TRAINED model...")
        topo_model, topo_tokenizer = self.load_model(use_adapter=True)
        self.results["topology"] = self.test_model(topo_model, topo_tokenizer, test_cases, use_topology=True)
        del topo_model
        torch.cuda.empty_cache()
        
        # Generate report
        self.generate_report()
        
    def generate_report(self):
        """Generate comprehensive comparison report"""
        print("\n" + "="*70)
        print("VALIDATION RESULTS")
        print("="*70)
        
        baseline_results = self.results["baseline"]
        topology_results = self.results["topology"]
        
        # Calculate averages
        baseline_avg_rouge = sum(r['rougeL'] for r in baseline_results) / len(baseline_results)
        topology_avg_rouge = sum(r['rougeL'] for r in topology_results) / len(topology_results)
        
        baseline_correct = sum(1 for r in baseline_results if r['contains_expected'])
        topology_correct = sum(1 for r in topology_results if r['contains_expected'])
        
        baseline_latency = sum(r['latency_ms'] for r in baseline_results) / len(baseline_results)
        topology_latency = sum(r['latency_ms'] for r in topology_results) / len(topology_results)
        
        print(f"\n📊 ROUGE-L Scores:")
        print(f"  Baseline:  {baseline_avg_rouge:.4f}")
        print(f"  Topology:  {topology_avg_rouge:.4f}")
        rouge_improvement = ((topology_avg_rouge - baseline_avg_rouge) / baseline_avg_rouge) * 100
        print(f"  Change:    {rouge_improvement:+.2f}%")
        
        print(f"\n🎯 Accuracy (contains expected answer):")
        print(f"  Baseline:  {baseline_correct}/{len(baseline_results)} ({baseline_correct/len(baseline_results)*100:.1f}%)")
        print(f"  Topology:  {topology_correct}/{len(topology_results)} ({topology_correct/len(topology_results)*100:.1f}%)")
        
        print(f"\n⚡ Latency:")
        print(f"  Baseline:  {baseline_latency:.1f}ms")
        print(f"  Topology:  {topology_latency:.1f}ms")
        
        # Verdict
        print(f"\n" + "="*70)
        print("VERDICT:")
        print("="*70)
        
        if rouge_improvement > 5:
            print("✅ TOPOLOGY HELPS! Significant improvement in ROUGE scores.")
            print(f"   Topology training provided {rouge_improvement:.1f}% better responses.")
        elif rouge_improvement > 0:
            print("⚠️  MARGINAL BENEFIT. Topology shows slight improvement.")
            print(f"   Only {rouge_improvement:.1f}% better - may not justify complexity.")
        else:
            print("❌ TOPOLOGY IS THEATER. No meaningful improvement.")
            print(f"   Baseline performs {abs(rouge_improvement):.1f}% better!")
            print("   External critic was RIGHT - strip the topology.")
        
        # Save detailed results
        output_file = Path("validation_results.json")
        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n💾 Detailed results saved to: {output_file}")
        print("="*70)

if __name__ == "__main__":
    runner = ValidationRunner()
    runner.run_full_validation()
