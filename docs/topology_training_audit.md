<!-- Audit summarizing current topology-training assets prior to upgrades -->
# Topology Training Audit (November 2025)

This note captures the observed state of the topology-aware training toolchain
before implementing the upgrade plan. It focuses on the `niodoo-ai` fine-tuning
stack and the `Niodoo-TCT` feature extraction scripts.

## Configurations (`niodoo-ai/config/`)

- `default.yaml` only exposes generic QLoRA hyperparameters. There are **no
  knobs for topology-aware training** such as `lambda_topology`, persistence
  subsampling limits, Sinkhorn blur, or teacher-distillation caches.
- The config still points at `./data/train.jsonl`, yet that file does not exist
  in the repository; running the pipeline requires the user to craft it
  manually.
- RTX 5090–specific toggles (e.g. dataset workers, batch sizing, fused ops)
  are not surfaced despite being called out in `config/rtx5090.env`.

## Data & Augmentation (`niodoo_ai/data.py`, `topology.py`)

- `TopologyDataset` correctly expects either `topology_features` or
  `hidden_state_path`, but it **does not validate contents** beyond presence.
  There is no check for dimensional consistency, persistence diagram payloads,
  or Betti metadata.
- `TopologyAugmentor` converts `HiddenStateFeatureAdapter` outputs to text and
  vectors but discards the richer sections (Betti curves, persistence stats)
  that the adapter already computes. The downstream training loop only receives
  a flat vector.

## Feature Extraction (`Niodoo-TCT/scripts/extract_features.py`)

- The CLI prints a JSON blob with the flattened feature vector and its length
  (`{"dim": ..., "values": [...]}`). It **never emits persistence diagrams,
  Betti numbers, sheaf energy, or entropy summaries** that the Implementation
  Guide expects for TopKD and hybrid losses.
- There is no batch mode for multiple tensors or schema indicating how to join
  outputs back into training records.

## Training Loop (`niodoo_ai/training.py`)

- Training still relies on the stock `Trainer` loss = cross-entropy. There is
  **no topological regulariser** (e.g. Sinkhorn Wasserstein on persistence
  diagrams) nor hooks for teacher topology supervision.
- LoRA defaults remain conservative (`r=64`, `alpha=16`). The guide recommends a
  higher rank (≥128) for topology-heavy objectives.
- No logging of topology-specific metrics (Wasserstein distance, Betti
  accuracy) is present; evaluation simply mirrors base LM perplexity.

## Evaluation Tooling (`niodoo-ai/scripts/evaluate_topology.py`)

- The evaluator reuses `Trainer.evaluate()` and returns the standard Hugging
  Face metrics. There is **no computation of topological preservation scores**
  (Betti accuracy, Wasserstein distance, paraphrase stability).

## Dataset Inventory (`data/` and related)

- The repository does not ship curated JSONL corpora with topology annotations.
  The only referenced assets live outside the module (e.g. `Niodoo-AI/data/`),
  so the training workflow begins with a missing file.

## Summary

The current pipeline successfully shells together QLoRA with topology feature
injection but lacks:

1. Configurable topology-aware loss terms and hardware presets.
2. Rich feature exports (persistence diagrams, Betti metadata) from
   `Niodoo-TCT`.
3. Dataset validation and schema guarantees for topology-labelled corpora.
4. Evaluation metrics that quantify structure preservation.

These gaps inform the implementation tasks that follow.


