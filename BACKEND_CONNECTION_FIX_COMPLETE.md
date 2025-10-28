# Backend Connection Fix - Complete ✅

## Issues Fixed

### 1. vLLM 404 Not Found - Root Cause Found and Fixed

**Problem:** Pipeline was getting 404 errors from vLLM because model ID mismatch.

**Root Cause:** `.env.production` had `VLLM_MODEL_ID=Qwen/Qwen2.5-7B-Instruct-AWQ` but vLLM was actually serving model ID `/workspace/models/Qwen2.5-7B-Instruct-AWQ`.

**Investigation Process:**
1. Added logging to dump model IDs being sent
2. Traced environment variable loading with `env_with_fallback()`
3. Found `.env.production` loads before `tcs_runtime.env` via `prime_environment()`
4. Discovered `.env.production` had wrong model ID

**Fix Applied:**
```bash
# Updated .env.production
VLLM_MODEL_ID=/workspace/models/Qwen2.5-7B-Instruct-AWQ  # Was: Qwen/Qwen2.5-7B-Instruct-AWQ
```

**Verification:**
```bash
# Check vLLM's actual model ID
curl http://127.0.0.1:5001/v1/models | jq -r '.data[0].id'
# Returns: /workspace/models/Qwen2.5-7B-Instruct-AWQ

# Check logs show correct model ID
grep "vLLM chat completion request" | grep "model=/workspace/models/Qwen2.5-7B-Instruct-AWQ"
# ✓ Match!
```

### 2. Curator Ollama Connection - Fixed

**Problem:** Curator was hardcoding model name as `"qwen2"` instead of using configured value.

**Root Cause:** In `niodoo_real_integrated/src/curator.rs` line 149, the model was hardcoded instead of using `self.config.model_name`.

**Fix Applied:**
1. Changed hardcoded `"qwen2"` to `self.config.model_name`
2. Added logging to show model/endpoint being called
3. Updated `tcs_runtime.env` to set `CURATOR_MODEL=qwen2:0.5b`

**Verification:**
```bash
# Check logs show curator using correct model
grep "Curator refine calling Ollama" | grep "model=qwen2:0.5b"
# ✓ Match!
```

## Configuration Files Updated

### `.env.production`
```bash
VLLM_MODEL_ID=/workspace/models/Qwen2.5-7B-Instruct-AWQ  # Fixed
```

### `tcs_runtime.env`
```bash
export VLLM_MODEL=/workspace/models/Qwen2.5-7B-Instruct-AWQ
export VLLM_MODEL_ID=/workspace/models/Qwen2.5-7B-Instruct-AWQ
export CURATOR_MODEL=qwen2:0.5b
```

### Code Changes
1. `niodoo_real_integrated/src/curator.rs` - Fixed hardcoded model name, added logging
2. `niodoo_real_integrated/src/generation.rs` - Added model ID logging
3. `curator_executor/config.toml` - Updated model paths

## Test Results

```bash
# Pipeline now runs successfully
cd /workspace/Niodoo-Final && source tcs_runtime.env && \
cargo run -p niodoo_real_integrated --bin niodoo_real_integrated -- --prompt "hello"

# Output shows:
# - No fallbacks
# - Correct model IDs in logs
# - Successful completion with metrics exported
```

## Environment Variable Priority

The pipeline checks env vars in this order (first match wins):
1. `MAIN_MODEL`
2. `VLLM_MODEL_ID` ← **This was overriding from .env.production**
3. `VLLM_MODEL`
4. `VLLM_MODEL_PATH`

## Key Lessons

1. **`.env.production` loads before `tcs_runtime.env`** - The `prime_environment()` function in config.rs loads `.env.production` and `.env` before other configs
2. **Model ID must match exactly** - vLLM exposes `/workspace/models/Qwen2.5-7B-Instruct-AWQ` not `Qwen/Qwen2.5-7B-Instruct-AWQ`
3. **Hardcoded values bypass config** - The curator was hardcoding the model name instead of using config
4. **Logging is essential** - Without model ID logging, this would have been very hard to debug

## Next Steps

Pipeline is now ready to run with proper backend connections:
```bash
# Source environment
source tcs_runtime.env

# Run pipeline
cargo run -p niodoo_real_integrated --bin niodoo_real_integrated -- --prompt "Your prompt"

# Or run your existing pipeline script
```

Both backends are now properly configured and responding. ✅


