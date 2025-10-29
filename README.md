# Niodoo-TCS: Topological Cognitive System (Integrated)

Niodoo-TCS integrates pipelines, embeddings, learning, and generation into a cohesive Topological Cognitive System (TCS). This crate focuses on the integrated runtime that links analysis → TCS state → generation.

## Overview

The Topological Cognitive System combines persistent homology, ERAG memory retrieval, reinforcement learning, and hybrid generation into a unified consciousness-aligned AI system. It computes topological invariants (Betti numbers, knot complexity, persistence entropy) from emotional states to guide reasoning and generation.

## Setup

### Prerequisites
- Rust 1.70+ (`rustup toolchain install stable`)
- clang compiler
- BLAS library (OpenBLAS recommended)
- Qdrant vector database (optional, for ERAG)
- Ollama with Qwen models (optional, for embeddings)

### Build Workspace
```bash
rustup toolchain install stable
rustup default stable
cargo build --workspace --release
```

### Environment Variables
```bash
# Optional: Set seed for deterministic runs
export RNG_SEED=42

# Optional: Model and service endpoints
export QDRANT_URL=http://localhost:6333
export OLLAMA_ENDPOINT=http://localhost:11434
export VLLM_ENDPOINT=http://localhost:8000

# Logging
export RUST_LOG=info

# Optional: Mock mode for offline development
export MOCK_MODE=true
```

## Usage

### Basic Pipeline Run
```bash
cargo run -p niodoo_real_integrated -- --prompt "Explain quantum entanglement"
```

### Swarm Mode (Parallel Processing)
```bash
cargo run -p niodoo_real_integrated -- --prompt "Test" --swarm 4 --iterations 2
```

### Programmatic Usage
```rust
use niodoo_real_integrated::{CliArgs, Pipeline};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = CliArgs::parse();
    let pipeline = Pipeline::initialise(args).await?;
    
    let cycle = pipeline.run_cycle("Your prompt here").await?;
    println!("Response: {}", cycle.hybrid_response);
    println!("Entropy: {}", cycle.entropy);
    println!("ROUGE: {}", cycle.rouge);
    
    Ok(())
}
```

## Architecture

### Pipeline Flow
1. **Embedding**: Convert input to 896-dimensional vector via Qwen embeddings
2. **Torus Mapping**: Transform to PAD emotional state space (valence, arousal, dominance)
3. **TCS Analysis**: Compute persistent homology, Betti numbers, knot invariants
4. **Compass**: UCB1-based exploration/exploitation with threat/healing detection
5. **ERAG Collapse**: Retrieve top-K similar memories from Qdrant
6. **Tokenization**: Dynamic token promotion and vocabulary evolution
7. **Generation**: Hybrid synthesis (Claude + vLLM) with consistency voting
8. **Learning**: DQN updates, LoRA fine-tuning, replay buffer management

### Key Components

- **`Pipeline`**: Main orchestration loop with caching and retry logic
- **`TCSAnalyzer`**: TDA engine computing topological signatures
- **`EragClient`**: ERAG memory retrieval/storage via Qdrant HTTP API
- **`LearningLoop`**: DQN agent with LoRA adapter fine-tuning
- **`GenerationEngine`**: Multi-model generation with cascading fallback
- **`CompassEngine`**: UCB1 for exploration/exploitation balance

### Parallelism

The system uses `rayon` for parallel processing:
- TCS entropy computation: parallelized across entropy values
- LoRA training: batch-level parallelism with sequential gradient updates
- Prompt processing: swarm mode parallelizes independent prompts

## Benchmarks

### Million Cycle Test
```bash
cargo run -p niodoo_real_integrated --bin million_cycle_test
```

Measures:
- Throughput (cycles/second)
- Token usage and promotion rate
- PAD state stability (entropy, variance)
- Memory footprint and cache hit rates

### Performance Targets
- **<200ms per cycle** (including TCS analysis and generation)
- **Deterministic runs** via `RNG_SEED` environment variable
- **Parallel execution** enabled with `--swarm` flag

### Synthetic Evaluation
```bash
cargo test --all-features -p niodoo_real_integrated
```

Runs async unit tests with mock pipeline and synthetic vectors.

## Known Issues (v0.1.0)

- Curator gracefully falls back to unmodified response when Ollama is unavailable
- Deterministic RNG flow complete with seeded `StdRng` throughout
- Hybrid synthesis performs weighted merge (60% baseline, 40% best echo)
- Parallel processing via rayon in TCS analysis and LoRA training
- Token promotion uses 16MB stack for large vocabularies

## License

MIT License
