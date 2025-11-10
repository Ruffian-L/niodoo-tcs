#!/usr/bin/env python3
"""
Generate system architecture diagram from mermaid description.
Creates a visual representation of the NIODOO-TCS pipeline.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from matplotlib.patches import ConnectionPatch
import numpy as np
from pathlib import Path

# Set style
plt.rcParams['figure.facecolor'] = '#0a0a0a'
plt.rcParams['axes.facecolor'] = '#0a0a0a'
plt.rcParams['savefig.facecolor'] = '#0a0a0a'
plt.rcParams['text.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['font.size'] = 10

def create_system_architecture():
    """Create system architecture diagram."""
    
    fig = plt.figure(figsize=(16, 12), facecolor='#0a0a0a')
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Define colors
    colors = {
        'input': '#e1f5ff',
        'output': '#c8e6c9',
        'process': '#4ecdc4',
        'decision': '#fff9c4',
        'memory': '#f3e5f5',
        'arrow': '#96ceb4'
    }
    
    # Define node positions (x, y, width, height)
    nodes = {
        'user_prompt': (2, 10, 2, 0.8),
        'embedding': (2, 8.5, 2, 0.8),
        'erag_retrieval': (2, 7, 2, 0.8),
        'topology': (2, 5.5, 2, 0.8),
        'compass': (2, 4, 2, 0.8),
        'tokenizer': (2, 2.5, 2, 0.8),
        'generation': (2, 1, 2, 0.8),
        'curator': (5, 1, 2, 0.8),
        'response': (8, 1, 2, 0.8),
        'quality': (11, 2.5, 2, 0.8),
        'low_quality': (11, 4, 2, 0.8),
        'high_quality': (11, 5.5, 2, 0.8),
        'buffer_check': (11, 7, 2, 0.8),
        'training': (11, 8.5, 2, 0.8),
        'save_weights': (11, 10, 2, 0.8),
        'load_adapter': (8, 10, 2, 0.8),
        'memory_store': (5, 7, 2, 0.8),
    }
    
    # Draw nodes
    node_labels = {
        'user_prompt': 'User Prompt',
        'embedding': 'Embedding Layer\n768D Vectors',
        'erag_retrieval': 'ERAG Memory\nRetrieval',
        'topology': 'Topology Analysis\nKnot, Betti',
        'compass': 'Consciousness\nCompass',
        'tokenizer': 'Tokenizer\nEnhancement',
        'generation': 'Generation\nEngine (vLLM)',
        'curator': 'Curator\nRefinement',
        'response': 'Response\nOutput',
        'quality': 'Quality\nAssessment',
        'low_quality': 'Create Training\nSample',
        'high_quality': 'Store in\nMemory',
        'buffer_check': 'Buffer Size\n>= 20?',
        'training': 'QLoRA\nTraining',
        'save_weights': 'Save Adapter\nWeights',
        'load_adapter': 'Load Adapter\nNext Cycle',
        'memory_store': 'ERAG Memory\nStore',
    }
    
    node_colors = {
        'user_prompt': colors['input'],
        'embedding': colors['process'],
        'erag_retrieval': colors['process'],
        'topology': colors['process'],
        'compass': colors['process'],
        'tokenizer': colors['process'],
        'generation': colors['process'],
        'curator': colors['process'],
        'response': colors['output'],
        'quality': colors['decision'],
        'low_quality': colors['process'],
        'high_quality': colors['memory'],
        'buffer_check': colors['decision'],
        'training': colors['process'],
        'save_weights': colors['process'],
        'load_adapter': colors['process'],
        'memory_store': colors['memory'],
    }
    
    # Draw all nodes
    drawn_nodes = {}
    for node_id, (x, y, w, h) in nodes.items():
        color = node_colors.get(node_id, colors['process'])
        box = FancyBboxPatch((x-w/2, y-h/2), w, h,
                            boxstyle="round,pad=0.05",
                            facecolor=color,
                            edgecolor='white',
                            linewidth=2,
                            alpha=0.9)
        ax.add_patch(box)
        
        # Add label
        ax.text(x, y, node_labels[node_id], ha='center', va='center',
               fontsize=9, fontweight='bold', color='black' if color == colors['input'] or color == colors['output'] else 'white')
        
        drawn_nodes[node_id] = (x, y)
    
    # Draw arrows (from, to)
    arrows = [
        ('user_prompt', 'embedding'),
        ('embedding', 'erag_retrieval'),
        ('erag_retrieval', 'topology'),
        ('topology', 'compass'),
        ('compass', 'tokenizer'),
        ('tokenizer', 'generation'),
        ('generation', 'curator'),
        ('curator', 'response'),
        ('response', 'quality'),
        ('quality', 'low_quality'),
        ('quality', 'high_quality'),
        ('low_quality', 'buffer_check'),
        ('buffer_check', 'training'),
        ('buffer_check', 'high_quality'),
        ('training', 'save_weights'),
        ('save_weights', 'load_adapter'),
        ('load_adapter', 'user_prompt'),
        ('high_quality', 'memory_store'),
        ('memory_store', 'erag_retrieval'),
    ]
    
    for from_node, to_node in arrows:
        if from_node in drawn_nodes and to_node in drawn_nodes:
            x1, y1 = drawn_nodes[from_node]
            x2, y2 = drawn_nodes[to_node]
            
            # Calculate arrow path
            dx = x2 - x1
            dy = y2 - y1
            
            # Adjust for node size
            node_w, node_h = nodes[from_node][2], nodes[from_node][3]
            if abs(dx) > abs(dy):  # Horizontal
                x1 += np.sign(dx) * node_w/2
                x2 -= np.sign(dx) * node_w/2
            else:  # Vertical
                y1 += np.sign(dy) * node_h/2
                y2 -= np.sign(dy) * node_h/2
            
            arrow = FancyArrowPatch((x1, y1), (x2, y2),
                                   arrowstyle='->', mutation_scale=20,
                                   color=colors['arrow'], linewidth=2,
                                   alpha=0.7, zorder=1)
            ax.add_patch(arrow)
    
    # Add title
    ax.text(8, 11.5, 'NIODOO-TCS System Architecture', 
           ha='center', va='center', fontsize=20, fontweight='bold', 
           color='#4ecdc4')
    
    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor=colors['input'], edgecolor='white', label='Input'),
        mpatches.Patch(facecolor=colors['output'], edgecolor='white', label='Output'),
        mpatches.Patch(facecolor=colors['process'], edgecolor='white', label='Process'),
        mpatches.Patch(facecolor=colors['decision'], edgecolor='white', label='Decision'),
        mpatches.Patch(facecolor=colors['memory'], edgecolor='white', label='Memory'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', 
             facecolor='#1a1a2e', edgecolor='#4ecdc4', fontsize=9)
    
    # Save
    output_dir = Path('figures')
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'system_architecture.png'
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#0a0a0a',
                edgecolor='none', pad_inches=0.2)
    print(f"✅ Created system architecture diagram: {output_path}")
    
    plt.close()

if __name__ == '__main__':
    print("🎨 Generating system architecture diagram...\n")
    create_system_architecture()
    print("\n✨ Done! Diagram ready for README.md")

