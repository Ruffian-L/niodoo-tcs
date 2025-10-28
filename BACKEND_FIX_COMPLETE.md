# Backend Connection Fixes - Complete

## Issues Identified and Fixed

### 1. vLLM 404 Not Found - Model ID Mismatch

**Root Cause:**
- vLLM exposes model ID as `/workspace/models/Qwen2.5-7B-Instruct-AWQ` (the full path)
- Pipeline was configured with `/workspace/models/hf_cache/models--Qwen--Qwen2.5-7B-Instruct-AWQ` in `tcs_runtime.env`
- GenerationEngine sends whatever model name is configured via `config.vllm_model`

**Fix Applied:**
1. Updated `tcs_runtime.env` to use correct model path:
   ```bash
   export VLLM_MODEL=/workspace/models/Qwen2.5-7B-Instruct-AWQ
   ```

2. Added logging to `niodoo_real_integrated/src/generation.rs` to dump model ID before each request:
   ```rust
   info!(
       endpoint = %self.endpoint,
       model = %self.model,
       "vLLM chat completion request"
   );
   ```

**Verification:**
```bash
curl http://127.0.0.1:5001/v1/models
# Returns: {"id": "/workspace/models/Qwen2.5-7B-Instruct-AWQ", ...}
```

### 2. Curator Ollama Connection - Hardcoded Model Name

**Root Cause:**
- Curator's `refine()` function was hardcoding model name as `"qwen2"` instead of using configured value
- Curator config passes `self.config.model_name` but code ignored it
- Ollama expects `"qwen2:0.5b"` (with tag) but was getting `"qwen2"`

**Fix Applied:**
1. Updated `niodoo_real_integrated/src/curator.rs` line 149-156:
   ```rust
   // BEFORE:
   "model": "qwen2", // Hardcoded!
   
   // AFTER:
   "model": self.config.model_name, // Uses configured value
   ```

2. Added logging to show what model/endpoint curator is calling:
   ```rust
   info!(
       ollama_url = %ollama_url,
       model = %self.config.model_name,
       "Curator refine calling Ollama"
   );
   ```

3. Updated `tcs_runtime.env` to set curator model:
   ```bash
   export CURATOR_MODEL=qwen2:0.5b
   ```

**Verification:**
```bash
curl http://127.0.0.1:11434/api/tags
# Returns: {"models": [{"name": "qwen2:0.5b", ...}]}
```

## Configuration Files Updated

### tcs_runtime.env
```bash
export VLLM_MODEL=/workspace/models/Qwen2.5-7B-Instruct-AWQ
export CURATOR_MODEL=qwen2:0.5b
export VLLM_ENDPOINT=http://127.0.0.1:5001
export OLLAMA_ENDPOINT=http://127.0.0.1:11434
```

### Code Changes
1. `niodoo_real_integrated/src/curator.rs` - Fixed hardcoded model name
2. `niodoo_real_integrated/src/generation.rs` - Added model ID logging

## Next Steps

1. **Source the environment:**
   ```bash
   source tcs_runtime.env
   ```

2. **Run the pipeline:**
   ```bash
   cargo run -p niodoo_real_integrated --bin niodoo_real_integrated -- --prompt "Your prompt here"
   ```

3. **Check logs for model IDs:**
   Look for these log lines to confirm correct model IDs:
   - `"vLLM chat completion request"` - shows model ID being sent to vLLM
   - `"Curator refine calling Ollama"` - shows model ID being sent to Ollama

## Expected Behavior

With these fixes:
- ✅ vLLM should respond with 200 instead of 404
- ✅ Curator should successfully call Ollama
- ✅ Pipeline should produce real outputs instead of fallbacks
- ✅ Logs will show exact model IDs/endpoints being used

## Debugging

If issues persist, check logs for:
1. Model ID mismatches in "vLLM chat completion request" logs
2. Ollama connection errors in "Curator refine calling Ollama" logs
3. Environment variable values with `env | grep -E "VLLM_MODEL|CURATOR_MODEL"`



