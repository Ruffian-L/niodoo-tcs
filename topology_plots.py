import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# Create output directory if it doesn't exist
output_dir = '/workspace/Niodoo-Final/results/benchmarks/topology/plots'
os.makedirs(output_dir, exist_ok=True)

# Load the JSON data
json_path = '/workspace/Niodoo-Final/results/benchmarks/topology/topology_benchmark_20251029_134425.json'
with open(json_path, 'r') as f:
    data = json.load(f)

# Extract summary
summary = data['summary']

# Extract records into DataFrame
records = data['records']
df = pd.DataFrame(records)

# Set style for tweet-ready plots (clean, high-contrast)
plt.style.use('default')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['savefig.dpi'] = 300  # High res for tweets

# Plot 1: Bar chart for key summary metrics (Baseline vs Hybrid)
metrics = ['rouge', 'persistence_entropy', 'spectral_gap']
baseline_vals = [summary['avg_rouge_baseline'], summary['avg_persistence_entropy_baseline'], summary['avg_spectral_gap_baseline']]
hybrid_vals = [summary['avg_rouge_hybrid'], summary['avg_persistence_entropy_hybrid'], summary['avg_spectral_gap_hybrid']]
x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots()
ax.bar(x - width/2, baseline_vals, width, label='Baseline', color='skyblue', alpha=0.8)
ax.bar(x + width/2, hybrid_vals, width, label='Hybrid (Topology)', color='orange', alpha=0.8)

ax.set_ylabel('Score')
ax.set_title('Topology Benchmark: Key Metrics Comparison')
ax.set_xticks(x)
ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'summary_metrics_comparison.png'))
plt.close()

# Plot 2: Line plot for ROUGE scores per cycle
fig, ax = plt.subplots()
cycles = df['cycle']
ax.plot(cycles, df['rouge_baseline'], marker='o', label='Baseline', color='skyblue', linewidth=2)
ax.plot(cycles, df['rouge_hybrid'], marker='s', label='Hybrid', color='orange', linewidth=2)
ax.set_xlabel('Cycle')
ax.set_ylabel('ROUGE Score')
ax.set_title('ROUGE Scores Across 20 Cycles')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'rouge_per_cycle.png'))
plt.close()

# Plot 3: Bar chart for average latencies
fig, ax = plt.subplots()
modes = ['Baseline', 'Hybrid']
latencies = [summary['avg_latency_baseline_ms']/1000, summary['avg_latency_hybrid_ms']/1000]  # Convert to seconds
ax.bar(modes, latencies, color=['skyblue', 'orange'], alpha=0.8)
ax.set_ylabel('Average Latency (seconds)')
ax.set_title('Latency Comparison')
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels on bars
for i, v in enumerate(latencies):
    ax.text(i, v + 0.1, f'{v:.1f}s', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'latency_comparison.png'))
plt.close()

# Plot 4: Scatter plot for Spectral Gap vs Persistence Entropy (Hybrid only, colored by label)
fig, ax = plt.subplots()
colors = {'joy': 'gold', 'sadness': 'blue', 'anger': 'red', 'fear': 'purple', 'surprise': 'green', 
          'love': 'pink', 'gratitude': 'brown', 'pride': 'orange', 'neutral': 'gray'}
scatter = ax.scatter(df['persistence_entropy_hybrid'], df['spectral_gap_hybrid'], 
                    c=df['label'].map(colors), alpha=0.7, s=100)
ax.set_xlabel('Persistence Entropy (Hybrid)')
ax.set_ylabel('Spectral Gap (Hybrid)')
ax.set_title('Hybrid Topology: Spectral Gap vs Persistence Entropy by Emotion')
# Add legend (manually, since colors are categorical)
for label, color in colors.items():
    ax.scatter([], [], c=color, label=label, s=100)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'spectral_gap_vs_entropy_hybrid.png'), bbox_inches='tight')
plt.close()

print(f"Plots saved to {output_dir}:")
print("- summary_metrics_comparison.png")
print("- rouge_per_cycle.png")
print("- latency_comparison.png")
print("- spectral_gap_vs_entropy_hybrid.png")

