# Niodoo-TCS: Topological Cognitive System

A cutting-edge AI system that combines topological data analysis, emotional reasoning, and dynamic memory systems to create truly adaptive intelligence.

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Rust 1.75+ (with `cargo`)
- 16GB+ RAM (32GB recommended for full features)
- NVIDIA GPU with CUDA support (optional, for acceleration)

### 1. Start Services with Docker

```bash
# Start vLLM server (for generation)
docker run -d --gpus all -p 5001:8000 \
  -v /path/to/models:/models \
  vllm/vllm-openai:latest \
  --model /models/Qwen2.5-7B-Instruct-AWQ \
  --port 8000

# Start Qdrant (for vector storage)
docker run -d -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant:latest
```

### 2. Set Environment Variables

```bash
export VLLM_ENDPOINT=http://localhost:5001
export QDRANT_URL=http://localhost:6333
export QDRANT_COLLECTION=niodoo_memories
```

### 3. Run Your First Prompt

```bash
# Build the project
cargo build --release

# Run with a simple prompt
cargo run --release --bin niodoo_real_integrated -- \
  --prompt "What is the meaning of consciousness?" \
  --hardware laptop
```

### Example Output

```
Prompt: What is the meaning of consciousness?
Entropy: 2.341
ROUGE: 0.756
Hybrid Response: Consciousness is the subjective experience of awareness...
```

## Architecture

- **Embedding**: Qwen2.5 stateful embeddings (896D)
- **Torus Projection**: Maps embeddings to 7D PAD+ghost emotional space
- **Topology**: Persistent homology, knot invariants, TQFT signatures
- **Memory**: ERAG (Emotional Retrieval-Augmented Generation) with Qdrant
- **Generation**: vLLM with hybrid retrieval-augmented responses
- **Learning**: DQN + QLoRA fine-tuning for adaptive improvement
- **Token Promotion**: Dynamic vocabulary evolution based on patterns

## Configuration

See `config/default.toml` for full configuration options. Key settings:

- `vllm_endpoint`: vLLM server URL
- `qdrant_url`: Qdrant vector database URL
- `token_promotion_interval`: How often to promote new tokens
- `learning_window`: DQN learning window size

## Development

```bash
# Run tests
cargo test

# Run Phase 2 E2E test
cargo test --test phase2_e2e

# Format code
cargo fmt

# Check code
cargo clippy
```

## Documentation

- API docs: `cargo doc --open`
- Pipeline docs: See `Pipeline::process_prompt` for detailed stage documentation

## License

MIT

