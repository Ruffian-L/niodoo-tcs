# Niodoo-Final: Topological Cognitive AI System

Niodoo-Final is an advanced AI framework integrating topological data analysis (TDA), self-learning mechanisms, and hybrid generation for consciousness-aligned intelligence. It features retry-optimized benchmarks, LoRA fine-tuning, and topological metrics like knot complexity for richer latent representations. This is the 'nuclear' drop—battle-tested with 64-cycle benchmarks showing ~20x speedups and superior quality.

## Key Features
- **Topological Cognition**: Computes knot complexity, Betti numbers, persistence entropy for emotional state analysis.
- **Self-Learning**: DQN rewards, LoRA adapters, and meta-updates for continual improvement.
- **Hybrid Generation**: Combines models with ERAG memory retrieval and compass-guided exploration.
- **Benchmarks**: Proven on GoEmotions with high ROUGE (~0.885 hybrid avg) and optimized retries.
- **Scalable**: Parallel swarms, deterministic RNG, and edge-optimized builds.

## Installation
1. Install Rust 1.87: `rustup install 1.87 && rustup default 1.87`
2. Clone repo: `git clone [your-repo-url]`
3. Build: `cargo build --release --all-features`
4. Dependencies: Qdrant, Ollama (Qwen models), optional vLLM/Claude.

Set env vars (see original README for details).

## Usage
Run a prompt:
```bash
cargo run --release -- --prompt \"Analyze this emotion\"
```

Benchmark:
```bash
./run_topology_benchmark.sh
```

For full docs, see docs/ folder.

## Benchmarks
- 64 cycles: Hybrid ROUGE 0.885 vs Baseline 0.585; ~13min total (~20x faster than naive).
- See results/benchmarks/topology/ for JSON/CSV outputs.

## Contributing
Fork, PRs welcome! See CONTRIBUTING.md.

## License
MIT
