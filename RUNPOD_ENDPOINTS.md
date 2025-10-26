# RunPod Endpoints Configuration

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
vector_dim = 768
max_memory_size = 100000
```

### **3. niodoo_integrated/rut_gauntlet_config.toml**
```toml
[qdrant]
url = "http://localhost:6333"
collection_name = "niodoo_embeddings"
vector_size = 768

[vllm]
endpoint = "http://localhost:8000/v1/completions"
model = "Qwen/Qwen2.5-32B-Instruct-AWQ"
max_tokens = 512
temperature = 0.7
```

---

## 🔧 Configuration Loading Logic

### **vLLM Endpoint Loading** (`niodoo_real_integrated/src/config.rs:421-437`)
```rust
let mut vllm_keys: Vec<&str> = vec!["VLLM_ENDPOINT"];
if matches!(args.hardware, HardwareProfile::Laptop5080Q) {
    vllm_keys.insert(0, "VLLM_ENDPOINT_TAILSCALE");
} else {
    vllm_keys.push("VLLM_ENDPOINT_TAILSCALE");
}
vllm_keys.push("TEST_ENDPOINT_VLLM");
let vllm_endpoint = env_with_fallback(&vllm_keys)
    .unwrap_or_else(|| "http://127.0.0.1:8000".to_string())
    .trim()
    .trim_end_matches('/')
    .replace("/v1/chat/completions", "")
    .replace("/v1/completions", "")
    .replace("/v1/embeddings", "")
    .trim_end_matches('/')
    .to_string();
```

### **Qdrant Endpoint Loading** (`niodoo_real_integrated/src/config.rs:442-453`)
```rust
let mut qdrant_keys: Vec<&str> = vec!["QDRANT_URL"];
if matches!(args.hardware, HardwareProfile::Laptop5080Q) {
    qdrant_keys.insert(0, "QDRANT_URL_TAILSCALE");
} else {
    qdrant_keys.push("QDRANT_URL_TAILSCALE");
}
qdrant_keys.push("TEST_ENDPOINT_QDRANT");
let qdrant_url = env_with_fallback(&qdrant_keys)
    .unwrap_or_else(|| "http://127.0.0.1:6333".to_string())
    .trim()
    .trim_end_matches('/')
    .to_string();
```

### **Ollama Endpoint Loading** (`niodoo_real_integrated/src/config.rs:462-463`)
```rust
let ollama_endpoint = env_with_fallback(&["OLLAMA_ENDPOINT", "OLLAMA_ENDPOINT_TAILSCALE"])
    .unwrap_or_else(|| "http://127.0.0.1:11434".to_string());
```

---

## 🚀 Usage Examples

### **Loading the Environment**
```bash
source tcs_runtime.env
```

### **Testing Endpoints**
```bash
# Test vLLM
curl $VLLM_ENDPOINT/health

# Test Qdrant
curl $QDRANT_URL/health

# Test Ollama
curl $OLLAMA_ENDPOINT/api/tags
```

### **RunPod Setup**
```bash
# From runpod_setup.sh
bash runpod_setup.sh
```

---

## 📍 Key File Locations

| Component | File | Line(s) |
|-----------|------|---------|
| **Environment Variables** | `tcs_runtime.env` | 1-17 |
| **Config Loading** | `niodoo_real_integrated/src/config.rs` | 421-463 |
| **Curator Config** | `curator_executor/config.toml` | 1-32 |
| **Gauntlet Config** | `niodoo_integrated/rut_gauntlet_config.toml` | 13-29 |
| **Setup Script** | `runpod_setup.sh` | All |

---

## 🔗 Quick Reference

```bash
# All endpoints in one place
VLLM_ENDPOINT=http://127.0.0.1:8000
OLLAMA_ENDPOINT=http://127.0.0.1:11434
QDRANT_URL=http://127.0.0.1:6333
QDRANT_COLLECTION=experiences
QDRANT_VECTOR_SIZE=896
VLLM_MODEL=/workspace/models/hf_cache/hub/models--Qwen--Qwen2.5-7B-Instruct-AWQ
```

