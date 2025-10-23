# LoRA Implementation Guide for Niodoo

This document provides step-by-step instructions for enabling LoRA adapter support in the Niodoo system.

## Current Status

✅ **Code Modifications Complete**: `generation.rs` has been updated to support LoRA
🔴 **Infrastructure Changes Needed**: vLLM server needs restart with `--enable-lora` flag

---

## Part 1: Code Changes (DONE)

### Modified Files
- **`niodoo_real_integrated/src/generation.rs`**

### Changes Made

#### 1. GenerationEngine struct (line 40-52)
Added `lora_name: Option<String>` field to store LoRA adapter name.

#### 2. GenerationEngine::new() (line 63-73)
Initialize `lora_name: None` in constructor.

#### 3. New method: with_lora() (line 88-92)
```rust
pub fn with_lora(mut self, lora_name: Option<String>) -> Self {
    self.lora_name = lora_name;
    self
}
```

Allows builder pattern configuration:
```rust
let engine = GenerationEngine::new("http://localhost:8000", "qwen-model")
    .with_lora(Some("my-adapter".to_string()))
```

#### 4. ChatCompletionRequest struct (line 506-517)
Added `lora_name` field with `#[serde(skip_serializing_if = "Option::is_none")]` to only include it when set.

#### 5. Updated send_chat() (line 354-362)
Now passes `lora_name: self.lora_name.clone()` to ChatCompletionRequest.

#### 6. Updated warmup() (line 393-410)
Now includes `lora_name` in warmup requests (optional - may help pre-load adapters).

---

## Part 2: Infrastructure Setup (REQUIRED)

### Step 1: Stop Current vLLM Server
```bash
# Find and kill vLLM process
pkill -f "vllm.entrypoints.openai.api_server"

# Verify it's stopped
sleep 2
ps aux | grep vllm | grep -v grep
```

### Step 2: Update Startup Script

Locate the vLLM startup script (likely at `/home/beelink/vllm-service/simple-start.sh`):

**Current command**:
```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model /home/beelink/models/Qwen2.5-7B-Instruct-AWQ \
  --host 0.0.0.0 \
  --port 8000 \
  --quantization awq \
  --dtype auto \
  --gpu-memory-utilization 0.9 \
  --max-model-len 4096 \
  --tensor-parallel-size 1 \
  --trust-remote-code
```

**New command with LoRA flags**:
```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model /home/beelink/models/Qwen2.5-7B-Instruct-AWQ \
  --host 0.0.0.0 \
  --port 8000 \
  --quantization awq \
  --dtype auto \
  --gpu-memory-utilization 0.9 \
  --max-model-len 4096 \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --enable-lora \
  --max-loras 4 \
  --max-lora-rank 16 \
  --lora-dtype auto
```

**Key new flags**:
- `--enable-lora`: Activates LoRA support
- `--max-loras 4`: Allow up to 4 LoRA adapters in memory simultaneously
- `--max-lora-rank 16`: Maximum rank for LoRA adapters (affects memory usage)
- `--lora-dtype auto`: Use auto data type for LoRA weights

### Step 3: Verify LoRA Models Exist

Before starting vLLM with LoRA, ensure LoRA adapters exist:

```bash
# Check for existing LoRA models
ls -la /home/beelink/models/
```

You'll need LoRA adapters trained for Qwen2.5-7B-Instruct-AWQ. These should be in a directory structure like:
```
/home/beelink/models/
  ├── Qwen2.5-7B-Instruct-AWQ/        # Base model
  └── qwen2.5-7b-lora-adapter-1/       # LoRA adapter (if available)
      ├── adapter_config.json
      └── adapter_model.bin
```

### Step 4: Start vLLM with LoRA Support

```bash
cd /home/beelink/vllm-service
source venv/bin/activate
nohup python3 -m vllm.entrypoints.openai.api_server \
  --model /home/beelink/models/Qwen2.5-7B-Instruct-AWQ \
  --host 0.0.0.0 \
  --port 8000 \
  --quantization awq \
  --dtype auto \
  --gpu-memory-utilization 0.9 \
  --max-model-len 4096 \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --enable-lora \
  --max-loras 4 \
  --max-lora-rank 16 \
  --lora-dtype auto > vllm.log 2>&1 &
```

### Step 5: Verify LoRA Support is Enabled

```bash
# Check logs
tail -50 /home/beelink/vllm-service/vllm.log | grep -i "lora\|enable"

# Test API endpoint
curl -s http://localhost:8000/version | grep -o '"version"[^}]*'
```

---

## Part 3: Using LoRA in Niodoo Code

### Basic Usage

```rust
use niodoo_real_integrated::generation::GenerationEngine;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Create engine without LoRA (default)
    let engine = GenerationEngine::new(
        "http://localhost:8000",
        "/home/beelink/models/Qwen2.5-7B-Instruct-AWQ"
    )?;

    // Or with LoRA adapter
    let engine = GenerationEngine::new(
        "http://localhost:8000",
        "/home/beelink/models/Qwen2.5-7B-Instruct-AWQ"
    )?
    .with_lora(Some("my-custom-adapter".to_string()));

    // Use as normal
    let result = engine.request_text("Your prompt").await?;
    println!("Response: {}", result);

    Ok(())
}
```

### Configuration Options

```rust
// No LoRA (default behavior)
let engine = GenerationEngine::new("http://localhost:8000", "model")?;

// With specific LoRA adapter
let engine = GenerationEngine::new("http://localhost:8000", "model")?
    .with_lora(Some("adapter-name".to_string()));

// With Claude fallback + LoRA
let engine = GenerationEngine::new("http://localhost:8000", "model")?
    .with_claude(claude_client)
    .with_lora(Some("my-adapter".to_string()));

// With GPT fallback + LoRA
let engine = GenerationEngine::new("http://localhost:8000", "model")?
    .with_gpt(gpt_client)
    .with_lora(Some("my-adapter".to_string()));
```

---

## Part 4: Testing

### Test 1: Basic LoRA Request

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/home/beelink/models/Qwen2.5-7B-Instruct-AWQ",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 50,
    "lora_name": "my-adapter"
  }' | python3 -m json.tool
```

**Expected**:
- Response includes completion
- No warning about ignored fields
- Response quality reflects LoRA adaptation

### Test 2: Check vLLM Logs

```bash
tail -100 /home/beelink/vllm-service/vllm.log | grep -E "(lora|adapter|WARNING)"
```

**Expected**:
- NO warning about "following fields were ignored"
- Possible logs showing adapter loading

### Test 3: Compile Niodoo Code

```bash
cd ~/Niodoo-Final/niodoo_real_integrated
cargo build --release 2>&1 | grep -E "(error|warning)" | head -20
```

**Expected**:
- generation.rs should have NO errors
- Pre-existing errors in other files are NOT your responsibility

---

## Troubleshooting

### Issue: "lora_name field ignored" warning in vLLM logs

**Cause**: Server not started with `--enable-lora` flag

**Fix**: Restart vLLM with LoRA flags (Step 2-4 above)

### Issue: vLLM crashes when loading LoRA adapter

**Possible causes**:
1. LoRA adapter not compatible with Qwen2.5-7B
2. LoRA adapter corrupted or missing files
3. GPU memory insufficient (try `--max-lora-rank 8` instead of 16)

**Fix**:
```bash
# Reduce memory pressure
--max-lora-rank 8
--max-cpu-loras 2

# Check logs for specific error
tail -200 /home/beelink/vllm-service/vllm.log | grep -i "error\|traceback"
```

### Issue: Requests with lora_name timeout

**Possible causes**:
1. LoRA adapter very large or slow to load
2. GPU memory thrashing

**Fix**:
```bash
# Increase timeout in generation.rs:
pub fn new(...) -> Result<Self> {
    let client = Client::builder()
        .timeout(Duration::from_secs(30))  // Was 5, now 30
        .build()?;
```

### Issue: LoRA adapter not found

**Error message**: "LoRA adapter 'xxx' not found"

**Fix**:
1. Verify adapter directory exists: `ls /home/beelink/models/`
2. Verify adapter files exist: `ls adapter-dir/adapter_*.bin`
3. Use correct adapter name (case-sensitive)

---

## Performance Tuning

### Memory-Conservative Setup
```bash
--enable-lora \
--max-loras 2 \
--max-lora-rank 8 \
--max-cpu-loras 1
```

### Performance-Optimized Setup
```bash
--enable-lora \
--max-loras 8 \
--max-lora-rank 32 \
--max-cpu-loras 4
```

### Parameters Explained
- `max-loras`: More = more adapters in GPU memory simultaneously
- `max-lora-rank`: Higher = more adapter parameters, more accurate but more memory
- `max-cpu-loras`: How many adapters to keep on CPU for fast swapping
- `lora-dtype`: `auto` = match base model dtype, `float16` = lower memory, `bfloat16` = better numerics

---

## Monitoring

### Check LoRA Memory Usage
```bash
# While requests are running, monitor GPU memory
nvidia-smi -l 1  # Update every 1 second

# Look for vLLM process and adapter memory
ps aux | grep vllm
```

### Check Active Adapters
```bash
# (If vLLM API provides this endpoint)
curl -s http://localhost:8000/v1/loras | python3 -m json.tool
```

---

## Rollback

If LoRA causes issues, you can quickly revert:

### Stop vLLM
```bash
pkill -f "vllm.entrypoints.openai.api_server"
```

### Restart without LoRA
```bash
cd /home/beelink/vllm-service
source venv/bin/activate
nohup python3 -m vllm.entrypoints.openai.api_server \
  --model /home/beelink/models/Qwen2.5-7B-Instruct-AWQ \
  --host 0.0.0.0 \
  --port 8000 \
  --quantization awq \
  --dtype auto \
  --gpu-memory-utilization 0.9 \
  --max-model-len 4096 \
  --tensor-parallel-size 1 \
  --trust-remote-code > vllm.log 2>&1 &
```

Code will still work - just won't use LoRA (lora_name will be ignored or skipped).

---

## Next Steps

1. ✅ Code modifications are complete (generation.rs updated)
2. 📋 Verify LoRA adapters exist for Qwen2.5-7B
3. 🔄 Update and restart vLLM with `--enable-lora` flag
4. ✅ Update Niodoo initialization to use `.with_lora(Some("adapter"))`
5. 🧪 Test with curl commands above
6. 📊 Monitor performance and adjust parameters as needed

---

## Questions?

Refer to the Agent 3 Report (`agent3-report.md`) for detailed technical findings and vLLM version compatibility information.

---

**Last Updated**: 2025-10-22
**Status**: Ready for implementation
**Confidence**: HIGH (95%)
