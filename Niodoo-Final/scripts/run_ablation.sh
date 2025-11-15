#!/usr/bin/env bash
set -euo pipefail

cd /workspace/Niodoo-Final

CSV_DIR="/workspace/Niodoo-Final/results"
OUT_CSV="$CSV_DIR/topology_eval.csv"
mkdir -p "$CSV_DIR"

cargo run -p niodoo_real_integrated --example topology_eval -- --num-prompts 100 --seed 42 --out "$OUT_CSV" --modes erag erag+lora full

python3 scripts/analyze_topology.py "$OUT_CSV"

echo "Artifacts written to $CSV_DIR"


