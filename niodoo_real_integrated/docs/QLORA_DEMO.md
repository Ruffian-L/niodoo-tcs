# QLoRA Learning Demo

The learning demo exercises the full NIODOO pipeline, captures per-cycle metrics, and
persists QLoRA adapter updates so you can watch the agent improve across runs.

## Prerequisites

- Ollama running locally with an instruction-tuned model (defaults assume
  `ollama serve` and `ollama pull llama3` or `qwen2`).
- Qdrant and any other services referenced by `RuntimeConfig` should already be
  available (the standard `docker-compose` bundle from the repository works).
- Environment variables commonly used for the pipeline should be set:
  - `OLLAMA_BASE_URL` (default `http://127.0.0.1:11434`)
  - `MODEL_NAME` or `VLLM_MODEL` if you are routing to GPU inference
  - `QDRANT_URL`
- Optional overrides specific to the demo:
  - `LORA_ADAPTER_PATH` – where LoRA weights are loaded from and saved to
    (default `lora_weights.safetensors`).
  - `LEARNING_DEMO_CYCLES` – number of iterations to run (default `20`).
  - `LEARNING_DEMO_TARGET_ROUGE` – ROUGE threshold that triggers persistence
    (default `0.7`).
  - `LEARNING_DEMO_LOG` – destination for the per-cycle TSV log
    (default `learning_demo_log.tsv`).

## Running the Demo

```bash
cargo run --bin learning_demo --release
```

- The binary initialises the standard pipeline and reloads a saved adapter if
  `LORA_ADAPTER_PATH` already exists.
- Each prompt cycle records ROUGE, entropy, threat/healing flags, observed
  QLoRA updates, and latency to `learning_demo_log.tsv` (or your override).
- Whenever the cycle registers a threat, falls below the ROUGE threshold, or
  produces QLoRA updates, the adapter is persisted so improvements accumulate.

## Interpreting Results

At completion the demo prints a summary:

- Average ROUGE for the first and second halves of the run highlights learning
  progress (expect the second half to be higher once the adapter picks up
  curated fixes).
- Average entropy provides a quick read on emotional stability.
- Threat/healing counts let you see how often self-healing pathways were
  engaged.
- `QLoRA updates observed` tallies how many cycles generated LoRA training
  events.
- Adapter and log file paths are echoed so you can feed them into longer runs
  or inspection notebooks.

For longer experiments you can rerun the demo; it will automatically reload the
saved adapter and continue refining it.

