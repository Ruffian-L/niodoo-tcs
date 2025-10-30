# RunPod Endpoints Configuration

## 🚀 Bootstrap Everything

```bash
bash /workspace/Niodoo-Final/scripts/runpod_bootstrap.sh
```

- Installs apt packages, Rust toolchain, Python venv (Torch CU121, vLLM, requirements)
- Downloads the vLLM model (requires `HF_TOKEN`) and provisions Qdrant/Ollama binaries
- Builds the full Rust workspace and launches vLLM, Qdrant, Ollama, and metrics with health checks
- Flags: `--force`, `--skip-services`, `--skip-build`, `--skip-model-download`, `--skip-qdrant`, `--skip-ollama`
- Customize via env vars (examples):
  - `VLLM_MODEL=/workspace/models/Qwen2.5-7B-Instruct-AWQ`
  - `QDRANT_VERSION=1.11.3`, `QDRANT_ROOT=/workspace/qdrant`
  - `OLLAMA_ROOT=/workspace/ollama`, `ENABLE_METRICS=0`

```bash
# RunPod startup command example
bash /workspace/Niodoo-Final/scripts/runpod_bootstrap.sh --force
```

## 🔌 All Endpoints

### **vLLM Endpoint**
```bash
export VLLM_ENDPOINT=http://127.0.0.1:8000
```
- **Default:** `http://127.0.0.1:8000`
- **Fallback env vars:** `VLLM_ENDPOINT`, `VLLM_ENDPOINT_TAILSCALE`, `TEST_ENDPOINT_VLLM`
- **Model:** `/workspace/models/hf_cache/hub/models--Qwen--Qwen2.5-7B-Instruct-AWQ`
- **Location:** `tcs_runtime.env:3`

### **Ollama Endpoint**
```bash
export OLLAMA_ENDPOINT=http://127.0.0.1:11434
```
- **Default:** `http://127.0.0.1:11434`
- **Fallback env vars:** `OLLAMA_ENDPOINT`, `OLLAMA_ENDPOINT_TAILSCALE`
- **Models available:**
  - `qwen-consciousness:latest` (30.5B, GGUF Q4_K_M)
  - `qwen3-coder:30b` (30.5B, GGUF Q4_K_M)
  - `qwen2.5-coder:32b` (32.8B, GGUF Q4_K_M)
  - `qwen2.5-coder:1.5b` (986 MB)
- **Location:** `tcs_runtime.env:4`

### **Qdrant Endpoint**
```bash
export QDRANT_URL=http://127.0.0.1:6333
export QDRANT_COLLECTION=experiences
export QDRANT_VECTOR_SIZE=896
```
- **Default:** `http://127.0.0.1:6333`
- **Fallback env vars:** `QDRANT_URL`, `QDRANT_URL_TAILSCALE`, `TEST_ENDPOINT_QDRANT`
- **Collection:** `experiences`
- **Vector Dimension:** `896` (matches Qwen embedder output)
- **Location:** `tcs_runtime.env:5-7`

---

## 📁 Configuration Files

### **1. tcs_runtime.env** (Primary Configuration)
```bash
export VLLM_MODEL=/workspace/models/hf_cache/hub/models--Qwen--Qwen2.5-7B-Instruct-AWQ
export VLLM_ENDPOINT=http://127.0.0.1:8000
export OLLAMA_ENDPOINT=http://127.0.0.1:11434
export QDRANT_URL=http://127.0.0.1:6333
export QDRANT_COLLECTION=experiences
export QDRANT_VECTOR_SIZE=896
export RUST_LOG=info,tcs_core=debug
```

### **2. curator_executor/config.toml**
```toml
[vllm]
endpoint = "http://localhost:8000"
api_key = "runpod-h200-test"
curator_model = "Qwen2.5-0.5B-Instruct"
executor_model = "Qwen2.5-Coder-7B-Instruct"

[memory_core]
qdrant_url = "http://localhost:6333"
collection_name = "experiences"
vector_dim = 896
max_memory_size = 100000
```

### **3. niodoo_integrated/rut_gauntlet_config.toml**
```toml
[qdrant]
url = "http://localhost:6333"
collection_name = "niodoo_embeddings"
vector_size = 896

[vllm]
endpoint = "http://localhost:8000/v1/completions"
model = "Qwen/Qwen2.5-32B-Instruct-AWQ"
max_tokens = 512
temperature = 0.7
```