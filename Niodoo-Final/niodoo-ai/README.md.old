# niodoo-ai

Topology-aware fine-tuning stack for Mistral 7B. This package consumes structural
features from the `Niodoo-TCT` toolkit, augments prompts with topological
signals, and runs a QLoRA training loop tailored for structural reasoning tasks.

## Quick Start

### 1. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Prepare datasets

Create `data/train.jsonl` (and optional `data/eval.jsonl`) with records shaped as:

```json
{"instruction": "Identify root", "input": "tree spec", "output": "answer", "topology_features": [0.12, 0.34, 0.56]}
```

Alternatively provide `hidden_state_path` pointing to tensors saved from model
forward passes—`Niodoo-TCT` will derive features on demand.

### 3. Configure training

Copy and edit `config/default.yaml` to match hardware, dataset paths, and LoRA
preferences. Critical parameters:

- `model.base_model`: Hugging Face identifier or local path for Mistral 7B
- `data.train_file` / `data.eval_file`: JSONL sources
- `optimizer`: learning rate, epochs, accumulation schedule
- `runtime.output_dir`: destination for checkpoints and metrics

### 4. Pre-tokenise (optional but recommended)

```bash
python scripts/prepare_data.py config/default.yaml --output ./processed
```

This stores Arrow datasets with cached topology vectors under `./processed`.

### 5. Train with QLoRA

```bash
python scripts/train_topology.py config/default.yaml
```

### 6. Evaluate

```bash
python scripts/evaluate_topology.py config/default.yaml --model ./outputs
```

Metrics are written to `outputs/eval_metrics.json` and printed to stdout.

## Package Layout

- `niodoo_ai/config.py`: Typed configuration loader for YAML files
- `niodoo_ai/data.py`: JSONL ingestion + tokenizer integration
- `niodoo_ai/topology.py`: Feature formatting powered by `Niodoo-TCT`
- `niodoo_ai/training.py`: QLoRA training/evaluation orchestration
- `scripts/prepare_data.py`: Batch feature extraction + dataset materialisation
- `scripts/train_topology.py`: Fire-and-forget training entry point
- `scripts/evaluate_topology.py`: Validation harness
- `tests/`: Lightweight unit tests for config parsing and prompt augmentation

## Notes

- Requires CUDA-capable hardware for efficient QLoRA; set `model.load_in_4bit`
  to `false` for CPU-only experimentation (slower).
- The training loop uses Hugging Face `Trainer` with LoRA adapters from `peft`.
- Prompt template defaults to `[INST] ... [/INST]` format with a topology block
  prefixed by `[TOPOLOGY]`. Adjust via `data.prompt_template` or
  `data.feature_prefix`.
- `Niodoo-TCT` must be importable (ensure repo root is on `PYTHONPATH`).
