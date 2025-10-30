# Niodoo-Final: Topological Cognitive AI System

Niodoo-Final is an advanced AI framework integrating topological data analysis (TDA), self-learning mechanisms, and hybrid generation for consciousness-aligned intelligence. It features retry-optimized benchmarks, LoRA fine-tuning, and topological metrics like knot complexity for richer latent representations. This is the 'nuclear' drop—battle-tested with 64-cycle benchmarks showing ~20x speedups and superior quality.

## Key Features
- **Topological Cognition**: Computes knot complexity, Betti numbers, persistence entropy for emotional state analysis.
- **Self-Learning**: DQN rewards, LoRA adapters, and meta-updates for continual improvement.
- **Hybrid Generation**: Combines models with ERAG memory retrieval and compass-guided exploration.
- **Benchmarks**: Proven on GoEmotions with high ROUGE (~0.885 hybrid avg) and optimized retries.
- **Scalable**: Parallel swarms, deterministic RNG, and edge-optimized builds.

## Quick Start

### RunPod (fully automated)
```bash
bash /workspace/Niodoo-Final/scripts/runpod_bootstrap.sh --force
```
- Installs apt packages, Rust toolchain, and Python venv (Torch/cu121 + requirements)
- Downloads the vLLM model when `HF_TOKEN` is set and provisions Qdrant/Ollama binaries
- Builds the workspace and boots vLLM, Qdrant, Ollama, and metrics with health checks
- Flags: `--skip-services`, `--skip-build`, `--skip-model-download`, `--skip-qdrant`, `--skip-ollama`

### Manual workspace setup
1. Install Rust 1.87: `rustup install 1.87 && rustup default 1.87`
2. Clone repo: `git clone [your-repo-url]`
3. Enter repo: `cd Niodoo-Final`
4. Export repo root: `export NIODOO_ROOT=$(pwd)`
5. Build once: `cargo build -p niodoo_real_integrated`

### Bring services online
```bash
# Start vLLM, Qdrant, and Ollama via service manager
./unified_service_manager.sh start

# Logs persist under /tmp/niodoo_logs/
ls /tmp/niodoo_logs
```

### Run the QLoRA learning demo (20 cycles)
```bash
CARGO_TARGET_DIR=.cargo-target \
cargo run -p niodoo_real_integrated --bin learning_demo

# Persisted adapters land in ./lora_weights.safetensors
```

### One-off prompt run
```bash
cargo run -p niodoo_real_integrated -- --prompt "Analyze this emotion"
```

Additional benchmarks and scripts live under `run_*.sh` (see docs/ for details).

## Benchmarks
- 64 cycles: Hybrid ROUGE 0.885 vs Baseline 0.585; ~13min total (~20x faster than naive).
- See results/benchmarks/topology/ for JSON/CSV outputs.

## Continuous Integration
- GitHub Actions (`.github/workflows/ci.yml`) runs on every push/PR.
- Checks `cargo fmt`, `cargo clippy`, and full workspace tests in mock mode with `NIODOO_ROOT=$(pwd)`.

## Contributing
Fork, PRs welcome! See CONTRIBUTING.md.

## License
MIT
