#!/bin/bash
set -e

echo "=== Running Topology Evaluation ==="
cd /workspace/Niodoo-Final/niodoo_real_integrated

# Clean and rebuild if needed
echo "Building topology_eval example..."
cargo build --example topology_eval --features examples --release

# Set environment variables
export OLLAMA_ENDPOINT=http://127.0.0.1:11434
export VLLM_ENDPOINT=http://127.0.0.1:5001
export QDRANT_URL=http://127.0.0.1:6333

# Run evaluation with 20 prompts (quick test)
echo "Running evaluation..."
cargo run --example topology_eval --features examples --release -- \
    --num-prompts 20 \
    --seed 42 \
    --out /workspace/Niodoo-Final/results/topo_proof_real.csv \
    --modes erag full

echo "Evaluation complete. Results in results/topo_proof_real.csv"


