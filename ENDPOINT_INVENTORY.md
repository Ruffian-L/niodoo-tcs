# Endpoint Inventory & Service Matrix

## Service Endpoints

### Core Services

| Service | Protocol | Port | Endpoint | Status | Purpose |
|---------|----------|------|----------|--------|---------|
| **Qdrant** | HTTP | 6333 | `http://127.0.0.1:6333` | ✅ Running | Vector database for ERAG memory |
| **Ollama** | HTTP | 11434 | `http://127.0.0.1:11434` | ⚠️ Down | Embeddings & Curator (qwen2:0.5b) |
| **vLLM** | HTTP | 5001 | `http://127.0.0.1:5001` | ✅ Running | Primary generation backend |
| **Metrics** | HTTP | 9092 | `http://127.0.0.1:9092/metrics` | ✅ Running | Prometheus metrics |

### Qdrant Collections

| Collection | Vector Dim | Distance | Status | Purpose |
|------------|------------|----------|--------|---------|
| `experiences` | 896 | Cosine | ✅ Active | Main ERAG memory store |
| `failures` | 896 | Cosine | ✅ Active | Failure pattern storage |

### API Endpoints

#### Qdrant Endpoints
- `GET /collections/{collection_name}` - Collection info
- `POST /collections/{collection_name}/points/search` - Vector search
- `PUT /collections/{collection_name}/points` - Insert vectors
- `DELETE /collections/{collection_name}` - Delete collection

#### Ollama Endpoints
- `POST /api/embeddings` - Generate embeddings (896-dim output)
- `POST /api/generate` - Text generation
- `GET /api/tags` - List available models

#### vLLM Endpoints
- `POST /v1/chat/completions` - Chat completions
- `GET /health` - Health check
- `GET /v1/models` - List loaded models

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `QDRANT_URL` | `http://127.0.0.1:6333` | Qdrant server URL |
| `QDRANT_COLLECTION` | `experiences` | Default collection name |
| `QDRANT_VECTOR_DIM` | `896` | Vector dimension (matches Qwen embedder) |
| `OLLAMA_ENDPOINT` | `http://127.0.0.1:11434` | Ollama server URL |
| `VLLM_ENDPOINT` | `http://127.0.0.1:5001` | vLLM server URL |
| `METRICS_ENDPOINT` | `http://127.0.0.1:9092/metrics` | Prometheus metrics |

### Service Health Commands

```bash
# Check Qdrant
curl http://127.0.0.1:6333/collections/experiences

# Check Ollama (when running)
curl http://127.0.0.1:11434/api/tags

# Check vLLM
curl http://127.0.0.1:5001/health

# Check Metrics
curl http://127.0.0.1:9092/metrics
```

### Start Ollama Service

```bash
# Start Ollama service
ollama serve

# Pull required embedding model
ollama pull qwen2:0.5b

# Verify it's running
curl http://127.0.0.1:11434/api/tags
```

### Dimension Consistency

- **Qwen Embedder Output**: 896 dimensions
- **Qdrant Collection**: 896 dimensions (configured in `erag.rs:105`)
- **LoRA Trainer**: Configured to match embedding dimension from RuntimeConfig
- **ERAG Memory**: Stores 896-dim vectors

## Current Status

✅ **Working**
- Qdrant vector database
- vLLM generation backend
- Prometheus metrics
- Pipeline runs through embedding → topology → generation → retries → ERAG persistence

⚠️ **Needs Attention**
- Ollama service offline (curator and embeddings unavailable)
- Requires: `ollama serve && ollama pull qwen2:0.5b`

✅ **Fixed**
- LoRA trainer dimension mismatch (now configures to 896-dim from RuntimeConfig)
- Empty collapse result handling (skips training when features are zero)
- Feature vector padding to correct dimensions

## Integration Points

1. **Embedding Flow**: Input → Ollama → 896-dim vector → Qdrant storage
2. **Generation Flow**: Prompt → vLLM → Response → Topology analysis
3. **Learning Flow**: Curated samples → LoRA trainer (896-dim) → Weight updates
4. **Memory Flow**: Experiences → ERAG collapse → Qdrant retrieval → Context injection

