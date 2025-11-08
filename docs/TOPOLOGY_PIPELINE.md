# Topology-Aware Training Pipeline

This document summarizes the flow between the Niodoo-TCT topological toolkit
and the niodoo-ai fine-tuning stack for Mistral 7B.

## Overview

1. **Hidden-State Extraction** – Transformer activations are gathered during
   inference and saved as tensors (`hidden_states`) or pre-computed topology
   vectors (`topology_features`).
2. **Feature Vectorisation** – `Niodoo-TCT/ntokens/features.py` converts raw
   activations into interpretable descriptors (Betti curves, persistence
   statistics, sheaf energy). The CLI `scripts/extract_features.py` batch
   processes stored tensors.
3. **Prompt Augmentation** – `niodoo_ai/topology.py` consumes feature vectors,
   generates compact summaries, and attaches them to `[INST]` prompts using a
   configurable topology prefix.
4. **QLoRA Training Loop** – `niodoo_ai/training.py` loads Mistral 7B in 4-bit
   precision, applies LoRA adapters, and fine-tunes using the topology-enhanced
   dataset. Training artifacts are written to `runtime.output_dir`.

## Data Requirements

- JSONL records must include `instruction`, `input`, `output`, and either:
  - `topology_features`: list of floats produced by `Niodoo-TCT`, or
  - `hidden_state_path`: path to a `.pt`/`.npy` file with hidden states.
- Optional evaluation data can be provided as a separate JSONL file or created
  automatically via `data.eval_split_ratio`.

## Command Reference

```bash
# Generate topology features (optional pre-processing)
python Niodoo-TCT/scripts/extract_features.py hidden_states.pt > features.json

# Materialise tokenised datasets with embedded topology
python niodoo-ai/scripts/prepare_data.py config/default.yaml --output ./processed

# Fine-tune using QLoRA
python niodoo-ai/scripts/train_topology.py config/default.yaml

# Evaluate the resulting checkpoint
python niodoo-ai/scripts/evaluate_topology.py config/default.yaml --model ./outputs
```

## Key Modules

- `Niodoo-TCT/ntokens/features.py`: Feature extractor producing normalized
  vectors and summaries.
- `niodoo_ai/topology.py`: Augmentor responsible for string formatting and
  feature serialization.
- `niodoo_ai/data.py`: Dataset builder that merges prompts, responses, and
  topology annotations before tokenization.
- `niodoo_ai/training.py`: Mistral QLoRA training orchestration (Trainer setup,
  evaluation, checkpointing).

## Evaluation Enhancements (2025-11)

- `niodoo_ai/training.py` now exposes topology-aware alignment metrics during
  evaluation: vector MSE, Sinkhorn distance, and Betti-sum RMSE are emitted when
  running `scripts/evaluate_topology.py`.
- `scripts/evaluate_topology.py` also reports paraphrase stability by probing a
  small set of symmetry-preserving sentence pairs; stability ≥0.8 indicates the
  topology head is invariant to paraphrase-level perturbations.
- Generated dataset manifests (`prepare_data.py`) include `topology_metadata_summary`
  and teacher-key coverage so downstream jobs can verify that Betti numbers and
  persistence statistics are populated before training.

## Integration Notes

- Both projects assume the repository root is on `PYTHONPATH`. When running
  scripts directly, invoke them with paths relative to `/workspace/Niodoo-Final`.
- LoRA target modules default to the standard Mistral attention/projection
  layers; customize in YAML if you experiment with alternative architectures.
- `TopologyDataset` retains numeric feature vectors for analytics while
  `Trainer` automatically drops them from model inputs via signature pruning.



