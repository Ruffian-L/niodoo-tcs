# Niodoo-TCT

Experimental topological compression toolkit for nTokens research.

## Goals

- Minimal INT8 quantized pipeline for nToken structural encoding
- Persistent homology (H0/H1) baseline running on consumer GPUs
- Early sheaf-structure scaffold for progressive fidelity layers

## Layout

- `ntokens/`: core python package
- `scripts/`: runnable demos and benchmarks
- `tests/`: lightweight correctness checks
- `scripts/extract_features.py`: batch hidden-state → topology feature CLI

## Getting Started

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/demo_encode.py
# Extract topology features from saved hidden states
python scripts/extract_features.py path/to/hidden_states.pt
```

Full roadmap lives in `docs/ROADMAP.md`.
