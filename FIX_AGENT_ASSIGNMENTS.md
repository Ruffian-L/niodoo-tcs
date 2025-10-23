# FIX AGENT ASSIGNMENTS
**Synthesized from 7 Validator Reports**
**Deploy ALL 10 IN PARALLEL** - Single ssh command with multiple &

---

## PRIORITY 0: COMPILATION BLOCKERS (Must fix first)

### FIX-1: Add Missing Tracing Import
**File**: `niodoo_real_integrated/src/pipeline.rs`
**Line**: Top of file (imports section)
**Fix**:
```rust
use tracing::warn;
```
**Validation**: Check line 113 `warn!(?error, "vLLM warmup failed");` compiles
**Time**: 30 seconds

---

### FIX-2: Fix Candle Version Conflict
**Files**:
- `Cargo.toml` (root workspace)
- `niodoo_real_integrated/Cargo.toml`

**Problem**: Root uses candle 0.9.1 (git), package uses 0.8 (crates.io)

**Fix**:
```toml
# In root Cargo.toml [workspace.dependencies]:
candle-core = "0.8"
candle-nn = "0.8"
# Remove the git = "..." lines

# In niodoo_real_integrated/Cargo.toml [dependencies]:
# Ensure it uses workspace = true for candle
```
**Validation**: `cargo check --workspace` should pass
**Time**: 2 minutes

---

### FIX-3: Add futures Dependency for FutureExt
**File**: `niodoo_real_integrated/Cargo.toml`
**Fix**:
```toml
[dependencies]
futures = { workspace = true }
```

**File**: `niodoo_real_integrated/src/pipeline.rs`
**Add import**:
```rust
use futures::FutureExt;
```
**Validation**: Line 183 `.map()` should now work
**Time**: 1 minute

---

### FIX-4: Fix RuntimeConfig Missing Field
**File**: `niodoo_real_integrated/src/config.rs`
**Line**: 222 (RuntimeConfig::load() method)
**Fix**: Add this field to the struct initialization:
```rust
Ok(Self {
    // ... existing fields ...
    enable_consistency_voting: false,  // ADD THIS LINE
})
```
**Validation**: Config struct should compile
**Time**: 30 seconds

---

### FIX-5: Fix LoRA SafeTensors Serialization
**File**: `niodoo_real_integrated/src/lora_trainer.rs`
**Lines**: 184-202

**Problem**: API type mismatches between candle and safetensors

**Fix** (replace save_adapter method):
```rust
pub fn save_adapter(&self, path: &str) -> Result<()> {
    use safetensors::tensor::{Dtype, TensorView};
    use std::collections::HashMap;

    // Get tensor data as bytes
    let lora_a_data = self.lora_a.to_vec1::<f32>()?;
    let lora_b_data = self.lora_b.to_vec1::<f32>()?;

    // Convert to bytes
    let lora_a_bytes = unsafe {
        std::slice::from_raw_parts(
            lora_a_data.as_ptr() as *const u8,
            lora_a_data.len() * std::mem::size_of::<f32>(),
        )
    };
    let lora_b_bytes = unsafe {
        std::slice::from_raw_parts(
            lora_b_data.as_ptr() as *const u8,
            lora_b_data.len() * std::mem::size_of::<f32>(),
        )
    };

    // Create tensor views with correct types
    let mut tensors = HashMap::new();
    tensors.insert(
        "lora_a".to_string(),
        TensorView::new(
            Dtype::F32,
            self.lora_a.dims().to_vec(),
            lora_a_bytes,
        )?,
    );
    tensors.insert(
        "lora_b".to_string(),
        TensorView::new(
            Dtype::F32,
            self.lora_b.dims().to_vec(),
            lora_b_bytes,
        )?,
    );

    // Serialize to file
    safetensors::serialize_to_file(&tensors, &None, path)?;
    Ok(())
}
```
**Validation**: LoRA trainer should compile
**Time**: 10 minutes

---

## PRIORITY 1: RELIABILITY IMPROVEMENTS

### FIX-6: Add API Client Retry Logic
**File**: `niodoo_real_integrated/src/api_clients.rs`
**Add to ClaudeClient and GptClient**:
```rust
async fn generate_with_retry(&self, prompt: &str, max_tokens: usize) -> Result<String> {
    let mut backoff_ms = 100;
    for attempt in 1..=3 {
        match timeout(Duration::from_secs(5), self.generate(prompt, max_tokens)).await {
            Ok(Ok(response)) => return Ok(response),
            Ok(Err(e)) if attempt == 3 => return Err(e),
            Ok(Err(e)) => {
                warn!("API attempt {} failed: {}", attempt, e);
                tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
                backoff_ms *= 10;
            }
            Err(_) if attempt == 3 => return Err(anyhow!("Timeout after 3 attempts")),
            Err(_) => {
                warn!("API attempt {} timed out", attempt);
                tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
                backoff_ms *= 10;
            }
        }
    }
    unreachable!()
}
```
**Validation**: Test with mock API failures
**Time**: 15 minutes

---

### FIX-7: Add 429 Rate Limit Detection
**File**: `niodoo_real_integrated/src/api_clients.rs`
**In generate() methods, add after .send():**:
```rust
let status = response.status();
if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
    if let Some(retry_after) = response.headers().get("retry-after") {
        let seconds = retry_after.to_str()?.parse::<u64>()?;
        warn!("Rate limited, retrying after {}s", seconds);
        tokio::time::sleep(Duration::from_secs(seconds)).await;
        // Retry logic here
    }
}
```
**Validation**: Mock 429 response
**Time**: 10 minutes

---

## PRIORITY 2: PERFORMANCE OPTIMIZATION

### FIX-8: Parallel API Cascade
**File**: `niodoo_real_integrated/src/generation.rs`
**Replace generate_with_fallback() to try APIs in parallel**:
```rust
pub async fn generate_with_fallback(&self, prompt: &str) -> Result<String> {
    // Launch all APIs in parallel with timeouts
    let claude_future = async {
        if let Some(ref c) = self.claude {
            timeout(Duration::from_secs(5), c.generate(prompt, 512)).await.ok()
        } else {
            None
        }
    };
    let gpt_future = async {
        if let Some(ref g) = self.gpt {
            timeout(Duration::from_secs(5), g.generate(prompt, 512)).await.ok()
        } else {
            None
        }
    };

    // Race them - return first success
    tokio::select! {
        Some(Ok(response)) = claude_future => Ok(response),
        Some(Ok(response)) = gpt_future => Ok(response),
        else => self.vllm.generate(prompt, 512).await  // Fallback
    }
}
```
**Validation**: Measure worst-case latency (should be ~5s not 10.5s)
**Time**: 20 minutes

---

### FIX-9: Add Async Pipeline Metrics
**File**: `niodoo_real_integrated/src/pipeline.rs`
**Add timing instrumentation**:
```rust
use std::time::Instant;

// Before each stage:
let start = Instant::now();
// ... stage execution ...
let elapsed = start.elapsed().as_millis();
info!(stage = "embedding", latency_ms = elapsed);
```
**Add for**: embedding, compass, erag, generation stages
**Validation**: Check logs show per-stage timing
**Time**: 10 minutes

---

### FIX-10: Optimize MCTS Simulation Count
**File**: `niodoo_real_integrated/src/mcts.rs` (if exists)
**Current**: 100 simulations (may be too slow)
**Add adaptive simulation count**:
```rust
pub fn search_adaptive(&self, root_state: &PadGhostState, max_time_ms: u64) -> Result<Vec<MctsAction>> {
    let start = Instant::now();
    let mut simulations = 0;

    while start.elapsed().as_millis() < max_time_ms as u128 && simulations < 100 {
        // Run one simulation
        self.simulate_once(&mut root)?;
        simulations += 1;
    }

    info!(simulations, "MCTS completed");
    Ok(root.best_action_sequence())
}
```
**Validation**: Measure MCTS latency with different budgets (50ms, 100ms, 200ms)
**Time**: 15 minutes

---

## DEPLOYMENT COMMAND (ALL 10 AT ONCE)

```bash
ssh beelink 'cd ~/Niodoo-Final && \
  PATH=~/.npm-global/bin:$PATH claude -p --dangerously-skip-permissions --model claude-haiku-4-5 "FIX-1: Add tracing import to pipeline.rs line 1" > logs/fix1.log 2>&1 & \
  PATH=~/.npm-global/bin:$PATH claude -p --dangerously-skip-permissions --model claude-haiku-4-5 "FIX-2: Fix candle version in root Cargo.toml to 0.8 from crates.io" > logs/fix2.log 2>&1 & \
  PATH=~/.npm-global/bin:$PATH claude -p --dangerously-skip-permissions --model claude-haiku-4-5 "FIX-3: Add futures dependency and import FutureExt in pipeline.rs" > logs/fix3.log 2>&1 & \
  PATH=~/.npm-global/bin:$PATH claude -p --dangerously-skip-permissions --model claude-haiku-4-5 "FIX-4: Add enable_consistency_voting field to RuntimeConfig init at config.rs:222" > logs/fix4.log 2>&1 & \
  PATH=~/.npm-global/bin:$PATH claude -p --dangerously-skip-permissions --model claude-haiku-4-5 "FIX-5: Rewrite save_adapter in lora_trainer.rs with proper safetensors API" > logs/fix5.log 2>&1 & \
  PATH=~/.npm-global/bin:$PATH claude -p --dangerously-skip-permissions --model claude-haiku-4-5 "FIX-6: Add retry logic with exponential backoff to API clients" > logs/fix6.log 2>&1 & \
  PATH=~/.npm-global/bin:$PATH claude -p --dangerously-skip-permissions --model claude-haiku-4-5 "FIX-7: Add 429 rate limit detection and Retry-After header parsing" > logs/fix7.log 2>&1 & \
  PATH=~/.npm-global/bin:$PATH claude -p --dangerously-skip-permissions --model claude-haiku-4-5 "FIX-8: Change generation.rs cascade to parallel with tokio::select!" > logs/fix8.log 2>&1 & \
  PATH=~/.npm-global/bin:$PATH claude -p --dangerously-skip-permissions --model claude-haiku-4-5 "FIX-9: Add per-stage timing metrics to pipeline.rs" > logs/fix9.log 2>&1 & \
  PATH=~/.npm-global/bin:$PATH claude -p --dangerously-skip-permissions --model claude-haiku-4-5 "FIX-10: Add adaptive time-boxed MCTS simulation" > logs/fix10.log 2>&1 & \
  wait && echo "All fixes complete"'
```

---

## VALIDATION AFTER ALL FIXES

```bash
# 1. Compilation test
cargo check --workspace

# 2. Run mini gauntlet (10 prompts)
cargo run -p niodoo_real_integrated --bin niodoo_real_integrated -- --output=csv > logs/post-fix-test.csv

# 3. Compare to baseline
# Expected improvements:
# - Compilation: SUCCESS (was FAILED)
# - Latency: <2000ms (was 2004ms, worst-case reduced from 10.5s to 5s)
# - ROUGE: >0.65 (maintain)
# - No crashes!
```

---

## SUMMARY

**P0 (Compilation blockers)**: Fixes 1-5 (15 minutes total)
**P1 (Reliability)**: Fixes 6-7 (25 minutes total)
**P2 (Performance)**: Fixes 8-10 (45 minutes total)

**Total estimated time**: 85 minutes with all agents in parallel = ~10-15 minutes wall time

**Expected outcome**: System compiles, runs, and is more robust!
