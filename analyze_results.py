#!/usr/bin/env python3
"""Analyze topology evaluation results"""
import pandas as pd
import sys

def analyze(csv_path):
    df = pd.read_csv(csv_path)
    
    print("=" * 80)
    print("TOPOLOGY EVALUATION RESULTS")
    print("=" * 80)
    
    # Group by mode
    full = df[df['mode'] == 'full']
    erag = df[df['mode'] == 'erag']
    
    print(f"\n{'MODE':<20} {'COUNT':<10} {'ROUGE-L':<15} {'ENTROPY':<15} {'BETTI-1':<15}")
    print("-" * 80)
    
    if len(erag) > 0:
        print(f"{'ERAG (baseline)':<20} {len(erag):<10} {erag['rouge_l'].mean():<15.6f} {erag['persistence_entropy'].mean():<15.6f} {erag['betti_1'].mean():<15.6f}")
    
    if len(full) > 0:
        print(f"{'Full Topology':<20} {len(full):<10} {full['rouge_l'].mean():<15.6f} {full['persistence_entropy'].mean():<15.6f} {full['betti_1'].mean():<15.6f}")
    
    # Detailed stats
    print("\n" + "=" * 80)
    print("DETAILED STATISTICS")
    print("=" * 80)
    
    if len(erag) > 0:
        print("\nERAG (Baseline):")
        print(erag[['rouge_l', 'persistence_entropy', 'betti_1']].describe())
    
    if len(full) > 0:
        print("\nFull Topology:")
        print(full[['rouge_l', 'persistence_entropy', 'betti_1']].describe())
    
    # Improvements
    if len(erag) > 0 and len(full) > 0:
        print("\n" + "=" * 80)
        print("IMPROVEMENT METRICS")
        print("=" * 80)
        
        rouge_improve = ((full['rouge_l'].mean() - erag['rouge_l'].mean()) / erag['rouge_l'].mean()) * 100
        entropy_reduction = ((erag['persistence_entropy'].var() - full['persistence_entropy'].var()) / erag['persistence_entropy'].var()) * 100
        
        print(f"\nROUGE-L Improvement: {rouge_improve:+.2f}%")
        print(f"Entropy Variance Reduction: {entropy_reduction:+.2f}%")
        print(f"Betti-1 Difference: {full['betti_1'].mean() - erag['betti_1'].mean():+.2f}")
        
        if rouge_improve > 0:
            print("\n✓ Topology improves ROUGE-L score")
        else:
            print("\n✗ Topology does not improve ROUGE-L score")
        
        if entropy_reduction > 0:
            print("✓ Topology reduces entropy variance (more stable)")
        else:
            print("✗ Topology does not reduce entropy variance")

if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "/workspace/Niodoo-Final/results/topo_proof_real.csv"
    analyze(csv_path)


