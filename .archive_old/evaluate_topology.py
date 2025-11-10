# scripts/evaluate_topology.py
"""
Complete topology evaluation for TCS models.
Measures: Sinkhorn distance, Betti accuracy, geometric reasoning.
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from ripser import ripser
from persim import wasserstein
from geomloss import SamplesLoss
import json
from datetime import datetime

class TopologyEvaluator:
    def __init__(self, model_path, device="cuda"):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.device = device
        
        # Sinkhorn loss for smooth comparisons
        self.sinkhorn = SamplesLoss(
            loss="sinkhorn",
            p=1,
            blur=0.05,
            backend="tensorized"
        )
    
    def get_embeddings(self, text):
        """Extract model embeddings."""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            return outputs.hidden_states[-1][0].cpu().float().numpy()
    
    def compute_topology(self, embeddings, subsample=100):
        """Compute persistent homology."""
        if len(embeddings) > subsample:
            indices = np.random.choice(len(embeddings), subsample, replace=False)
            embeddings = embeddings[indices]
        
        result = ripser(embeddings, maxdim=1)
        return result['dgms']
    
    def eval_betti_accuracy(self):
        """Test 1: Betti number accuracy."""
        test_cases = [
            {"text": "A sphere has no holes or loops", "expected_b1": 0},
            {"text": "A torus has two independent loops", "expected_b1": 2},
            {"text": "A circle is a 1-dimensional loop", "expected_b1": 1},
            {"text": "A Klein bottle is non-orientable with Betti numbers [1,2,0]", "expected_b1": 2},
            {"text": "A double torus has genus 2 with four independent cycles", "expected_b1": 4},
            {"text": "The Möbius strip has one non-contractible loop", "expected_b1": 1},
            {"text": "A trefoil knot is the simplest non-trivial knot", "expected_b1": 1},
            {"text": "The figure-eight knot has crossing number 4", "expected_b1": 1},
        ]
        
        results = []
        correct = 0
        
        for case in test_cases:
            embs = self.get_embeddings(case['text'])
            dgms = self.compute_topology(embs)
            pred_b1 = len(dgms[1])
            
            is_correct = pred_b1 == case['expected_b1']
            if is_correct:
                correct += 1
            
            results.append({
                "text": case['text'],
                "expected": case['expected_b1'],
                "predicted": pred_b1,
                "correct": is_correct
            })
        
        accuracy = correct / len(test_cases)
        return accuracy, results
    
    def eval_paraphrase_stability(self):
        """Test 2: Topology stability under paraphrasing."""
        paraphrase_pairs = [
            ("The torus has genus one", "A torus is a surface of genus 1"),
            ("A sphere is simply connected", "The sphere has trivial fundamental group"),
            ("Klein bottles are non-orientable", "The Klein bottle lacks orientability"),
            ("Betti numbers count holes", "Holes are counted by Betti numbers"),
            ("The trefoil knot is chiral", "Chirality characterizes the trefoil"),
        ]
        
        distances = []
        stable_count = 0
        threshold = 0.3  # Wasserstein threshold for "stable"
        
        for orig, para in paraphrase_pairs:
            embs_orig = self.get_embeddings(orig)
            embs_para = self.get_embeddings(para)
            
            dgm_orig = self.compute_topology(embs_orig)[1]
            dgm_para = self.compute_topology(embs_para)[1]
            
            if len(dgm_orig) > 0 and len(dgm_para) > 0:
                w_dist = wasserstein(dgm_orig, dgm_para)
                distances.append(w_dist)
                
                if w_dist < threshold:
                    stable_count += 1
        
        stability_rate = stable_count / len(paraphrase_pairs)
        mean_distance = np.mean(distances) if distances else float('inf')
        
        return stability_rate, mean_distance, distances
    
    def eval_geometric_reasoning(self):
        """Test 3: Geometric reasoning questions."""
        questions = [
            {
                "prompt": "Explain what Betti numbers measure",
                "keywords": ["holes", "loops", "connected", "components", "cycles"]
            },
            {
                "prompt": "What is the difference between a torus and a sphere topologically?",
                "keywords": ["genus", "hole", "loop", "homeomorphic", "Betti"]
            },
            {
                "prompt": "Why is a Möbius strip non-orientable?",
                "keywords": ["twist", "surface", "one-sided", "boundary", "orientation"]
            },
            {
                "prompt": "What does it mean for two spaces to be homeomorphic?",
                "keywords": ["continuous", "bijection", "deformation", "topology", "preserved"]
            },
        ]
        
        results = []
        
        for q in questions:
            # Generate response
            inputs = self.tokenizer(q["prompt"], return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.7,
                    do_sample=True
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Score based on keyword presence
            keywords_found = sum(1 for kw in q["keywords"] if kw.lower() in response.lower())
            score = keywords_found / len(q["keywords"])
            
            results.append({
                "prompt": q["prompt"],
                "response": response,
                "keywords_found": keywords_found,
                "keywords_total": len(q["keywords"]),
                "score": score
            })
        
        avg_score = np.mean([r["score"] for r in results])
        return avg_score, results
    
    def eval_sinkhorn_alignment(self):
        """Test 4: Sinkhorn distance for topology alignment."""
        # Define target topologies
        targets = {
            "sphere": {
                "text": "A sphere is a 2-dimensional surface with no holes",
                "target_dgm": np.array([[0.1, 0.15]])  # Minimal H1
            },
            "torus": {
                "text": "A torus is a surface with genus 1 and two independent loops",
                "target_dgm": np.array([[0.2, 0.8], [0.25, 0.75]])  # Two loops
            },
            "klein": {
                "text": "A Klein bottle is non-orientable with two non-contractible cycles",
                "target_dgm": np.array([[0.15, 0.7], [0.2, 0.65]])  # Two loops
            }
        }
        
        sinkhorn_distances = []
        
        for name, target in targets.items():
            embs = self.get_embeddings(target["text"])
            pred_dgm = self.compute_topology(embs)[1]
            
            if len(pred_dgm) > 0:
                pred_tensor = torch.tensor(pred_dgm, dtype=torch.float32, device=self.device)
                target_tensor = torch.tensor(target["target_dgm"], dtype=torch.float32, device=self.device)
                
                # Compute Sinkhorn distance
                sink_dist = self.sinkhorn(pred_tensor, target_tensor).item()
                sinkhorn_distances.append(sink_dist)
        
        mean_sinkhorn = np.mean(sinkhorn_distances) if sinkhorn_distances else float('inf')
        return mean_sinkhorn, sinkhorn_distances
    
    def run_full_eval(self):
        """Run complete evaluation suite."""
        print("Running topology evaluation...")
        
        # Test 1: Betti accuracy
        print("\n[1/4] Betti Number Accuracy...")
        betti_acc, betti_results = self.eval_betti_accuracy()
        
        # Test 2: Paraphrase stability
        print("[2/4] Paraphrase Stability...")
        stability, mean_w_dist, w_distances = self.eval_paraphrase_stability()
        
        # Test 3: Geometric reasoning
        print("[3/4] Geometric Reasoning...")
        reasoning_score, reasoning_results = self.eval_geometric_reasoning()
        
        # Test 4: Sinkhorn alignment
        print("[4/4] Sinkhorn Alignment...")
        mean_sinkhorn, sinkhorn_dists = self.eval_sinkhorn_alignment()
        
        # Aggregate results
        results = {
            "timestamp": datetime.now().isoformat(),
            "betti_accuracy": betti_acc,
            "paraphrase_stability": stability,
            "mean_wasserstein": mean_w_dist,
            "geometric_reasoning": reasoning_score,
            "mean_sinkhorn": mean_sinkhorn,
            "detailed_results": {
                "betti_tests": betti_results,
                "reasoning_tests": reasoning_results,
                "wasserstein_distances": w_distances,
                "sinkhorn_distances": sinkhorn_dists
            }
        }
        
        # Print summary
        print("\n" + "="*60)
        print("TOPOLOGY EVALUATION RESULTS")
        print("="*60)
        print(f"Betti Number Accuracy:     {betti_acc:.1%}")
        print(f"Paraphrase Stability:      {stability:.1%}")
        print(f"Mean Wasserstein Distance: {mean_w_dist:.4f}")
        print(f"Geometric Reasoning:       {reasoning_score:.1%}")
        print(f"Mean Sinkhorn Distance:    {mean_sinkhorn:.4f}")
        print("="*60)
        
        return results

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to model")
    parser.add_argument("--output", default="logs/evals/topology_eval_detailed.json")
    args = parser.parse_args()
    
    # Run evaluation
    evaluator = TopologyEvaluator(args.model)
    results = evaluator.run_full_eval()
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {args.output}")

if __name__ == "__main__":
    main()