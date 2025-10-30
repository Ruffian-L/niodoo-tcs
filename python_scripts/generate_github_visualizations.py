#!/usr/bin/env python3
"""
Generate visualizations from benchmark CSV data for GitHub README.
Creates plots showing ROUGE improvements, LoRA training, and learning metrics.
"""

import csv
import os
import glob
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14

def load_latest_csv():
    """Load the most recent benchmark CSV file."""
    csv_files = glob.glob('results/benchmarks/topology/topology_benchmark_*.csv')
    if not csv_files:
        print("No CSV files found in results/benchmarks/topology/")
        return None
    latest = max(csv_files, key=os.path.getctime)
    print(f"Loading: {latest}")
    return latest

def plot_rouge_improvement(csv_path, output_dir='docs/images'):
    """Plot ROUGE scores over cycles showing improvement."""
    os.makedirs(output_dir, exist_ok=True)
    
    cycles = []
    rouge_baseline = []
    rouge_hybrid = []
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                cycles.append(int(row['cycle']))
                rouge_baseline.append(float(row['rouge_baseline']))
                rouge_hybrid.append(float(row['rouge_hybrid']))
            except (ValueError, KeyError):
                continue
    
    if not cycles:
        print("No valid data found")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Filter out placeholder values (0.9999999995)
    valid_baseline = [(c, r) for c, r in zip(cycles, rouge_baseline) if r < 0.9]
    valid_hybrid = [(c, r) for c, r in zip(cycles, rouge_hybrid) if r < 0.9]
    
    if valid_baseline:
        cycles_b, rouge_b = zip(*valid_baseline)
        ax.plot(cycles_b, rouge_b, 'o-', label='Baseline', alpha=0.7, color='#ff6b6b', linewidth=2)
    
    if valid_hybrid:
        cycles_h, rouge_h = zip(*valid_hybrid)
        ax.plot(cycles_h, rouge_h, 's-', label='Hybrid (NIODOO)', alpha=0.7, color='#4ecdc4', linewidth=2)
    
    # Calculate and show improvement
    if valid_baseline and valid_hybrid:
        avg_baseline = np.mean([r for _, r in valid_baseline])
        avg_hybrid = np.mean([r for _, r in valid_hybrid])
        improvement = ((avg_hybrid - avg_baseline) / avg_baseline) * 100 if avg_baseline > 0 else 0
        
        ax.axhline(y=avg_baseline, color='#ff6b6b', linestyle='--', alpha=0.5, label=f'Baseline avg: {avg_baseline:.3f}')
        ax.axhline(y=avg_hybrid, color='#4ecdc4', linestyle='--', alpha=0.5, label=f'Hybrid avg: {avg_hybrid:.3f}')
        
        ax.text(0.02, 0.98, f'Average Improvement: {improvement:+.1f}%', 
                transform=ax.transAxes, fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_xlabel('Cycle Number', fontweight='bold')
    ax.set_ylabel('ROUGE Score', fontweight='bold')
    ax.set_title('ROUGE Score Improvements Over Time\n(NIODOO Learning from Conversations)', fontweight='bold', pad=20)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'rouge_improvements.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def plot_entropy_stability(csv_path, output_dir='docs/images'):
    """Plot entropy convergence over cycles."""
    os.makedirs(output_dir, exist_ok=True)
    
    cycles = []
    entropy = []
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                cycles.append(int(row['cycle']))
                entropy.append(float(row['entropy_hybrid']))
            except (ValueError, KeyError):
                continue
    
    if not cycles:
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(cycles, entropy, 'o-', color='#95e1d3', linewidth=2, markersize=4, alpha=0.7)
    ax.axhline(y=2.0, color='r', linestyle='--', linewidth=2, label='Target: 2.0 bits')
    ax.fill_between(cycles, 1.8, 2.2, alpha=0.2, color='green', label='Target range: 1.8-2.2 bits')
    
    avg_entropy = np.mean(entropy)
    std_entropy = np.std(entropy)
    
    ax.text(0.02, 0.98, f'Mean: {avg_entropy:.3f} bits\nStd: {std_entropy:.3f} bits', 
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    ax.set_xlabel('Cycle Number', fontweight='bold')
    ax.set_ylabel('Entropy (bits)', fontweight='bold')
    ax.set_title('Entropy Stability Over Time\n(Consciousness Compass Convergence)', fontweight='bold', pad=20)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'entropy_stability.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def plot_latency_comparison(csv_path, output_dir='docs/images'):
    """Plot latency comparison baseline vs hybrid."""
    os.makedirs(output_dir, exist_ok=True)
    
    cycles = []
    latency_baseline = []
    latency_hybrid = []
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                cycles.append(int(row['cycle']))
                latency_baseline.append(float(row['latency_baseline_ms']))
                latency_hybrid.append(float(row['latency_hybrid_ms']))
            except (ValueError, KeyError):
                continue
    
    if not cycles:
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(cycles, latency_baseline, 'o-', label='Baseline', alpha=0.7, color='#ff6b6b', linewidth=2, markersize=3)
    ax.plot(cycles, latency_hybrid, 's-', label='Hybrid (NIODOO)', alpha=0.7, color='#4ecdc4', linewidth=2, markersize=3)
    
    avg_baseline = np.mean(latency_baseline)
    avg_hybrid = np.mean(latency_hybrid)
    improvement = ((avg_baseline - avg_hybrid) / avg_baseline) * 100 if avg_baseline > 0 else 0
    
    ax.axhline(y=avg_baseline, color='#ff6b6b', linestyle='--', alpha=0.5)
    ax.axhline(y=avg_hybrid, color='#4ecdc4', linestyle='--', alpha=0.5)
    
    ax.text(0.98, 0.98, f'Baseline avg: {avg_baseline:.0f}ms\nHybrid avg: {avg_hybrid:.0f}ms\nSpeedup: {improvement:+.1f}%', 
            transform=ax.transAxes, fontsize=11, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_xlabel('Cycle Number', fontweight='bold')
    ax.set_ylabel('Latency (ms)', fontweight='bold')
    ax.set_title('Latency Comparison: Baseline vs Hybrid\n(NIODOO Performance)', fontweight='bold', pad=20)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'latency_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def plot_learning_summary(csv_path, output_dir='docs/images'):
    """Create a summary plot showing multiple metrics."""
    os.makedirs(output_dir, exist_ok=True)
    
    cycles = []
    rouge_hybrid = []
    entropy = []
    latency = []
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                cycles.append(int(row['cycle']))
                rouge_hybrid.append(float(row['rouge_hybrid']) if float(row['rouge_hybrid']) < 0.9 else np.nan)
                entropy.append(float(row['entropy_hybrid']))
                latency.append(float(row['latency_hybrid_ms']))
            except (ValueError, KeyError):
                continue
    
    if not cycles:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # ROUGE over cycles
    axes[0, 0].plot(cycles, rouge_hybrid, 'o-', color='#4ecdc4', linewidth=2, markersize=3)
    axes[0, 0].set_xlabel('Cycle')
    axes[0, 0].set_ylabel('ROUGE Score')
    axes[0, 0].set_title('ROUGE Score Over Time')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Entropy over cycles
    axes[0, 1].plot(cycles, entropy, 'o-', color='#95e1d3', linewidth=2, markersize=3)
    axes[0, 1].axhline(y=2.0, color='r', linestyle='--', linewidth=1)
    axes[0, 1].set_xlabel('Cycle')
    axes[0, 1].set_ylabel('Entropy (bits)')
    axes[0, 1].set_title('Entropy Stability')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Latency over cycles
    axes[1, 0].plot(cycles, latency, 'o-', color='#f38181', linewidth=2, markersize=3)
    axes[1, 0].set_xlabel('Cycle')
    axes[1, 0].set_ylabel('Latency (ms)')
    axes[1, 0].set_title('Response Latency')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Statistics summary
    axes[1, 1].axis('off')
    stats_text = f"""
    Learning Metrics Summary
    {'='*30}
    
    ROUGE Scores:
    • Mean: {np.nanmean(rouge_hybrid):.3f}
    • Std:  {np.nanstd(rouge_hybrid):.3f}
    
    Entropy:
    • Mean: {np.mean(entropy):.3f} bits
    • Std:  {np.std(entropy):.3f} bits
    • Target: 2.0 ± 0.1 bits
    
    Latency:
    • Mean: {np.mean(latency):.0f} ms
    • P50:  {np.percentile(latency, 50):.0f} ms
    • P95:  {np.percentile(latency, 95):.0f} ms
    
    Total Cycles: {len(cycles)}
    """
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                    verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle('NIODOO Learning Metrics Dashboard\n(Real Data from Production Runs)', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'learning_dashboard.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def main():
    print("🎨 Generating visualizations for GitHub README...\n")
    
    csv_path = load_latest_csv()
    if not csv_path:
        print("❌ No CSV file found. Run benchmarks first.")
        return
    
    output_dir = 'docs/images'
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n📊 Creating plots...")
    plot_rouge_improvement(csv_path, output_dir)
    plot_entropy_stability(csv_path, output_dir)
    plot_latency_comparison(csv_path, output_dir)
    plot_learning_summary(csv_path, output_dir)
    
    print(f"\n✅ All visualizations saved to {output_dir}/")
    print("\n📝 Update README.md to reference these images:")
    print("   - docs/images/rouge_improvements.png")
    print("   - docs/images/entropy_stability.png")
    print("   - docs/images/latency_comparison.png")
    print("   - docs/images/learning_dashboard.png")

if __name__ == '__main__':
    main()

