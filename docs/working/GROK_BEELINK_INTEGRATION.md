# Grok: Beelink Integration Instructions

**Mission**: Wire TCS embedder → Niodoo consciousness pipeline and run E2E tests on beelink.

**Hardware**: RTX Quadro 6000 48GB, CUDA 12.8, PyTorch 2.8.0+cu128

---

## STEP 0: TLS Handshake Pre-Check (CRITICAL)

**BEFORE attempting cargo build, verify git TLS works:**

```bash
ssh -i ~/.ssh/temp_beelink_key beelink@100.113.10.90

# On beelink:
cd /home/beelink/Desktop/Niodoo-Final

# Test git TLS handshake
git clone https://github.com/rust-lang/cargo.git /tmp/test-tls
if [ $? -eq 0 ]; then
    echo "✅ Git TLS working"
    rm -rf /tmp/test-tls
else
    echo "⚠️ Git TLS BROKEN - using vendored dependencies"
    # Check if vendor/ directory exists (should be synced from laptop)
    if [ ! -d vendor ]; then
        echo "❌ ERROR: vendor/ not found. Cannot build offline."
        echo "ACTION REQUIRED: Run 'cargo vendor' on laptop and wait for Syncthing sync."
        exit 1
    fi
fi
```

**If TLS is broken:**
- Read `GIT_TLS_TROUBLESHOOTING.md` for solutions
- Use vendored dependencies (cargo build --offline)
- Notify user of TLS issue

---

## STEP 1: Environment Verification

```bash
# Verify Syncthing synced files from laptop
cd /home/beelink/Desktop/Niodoo-Final
git status  # Should show recent changes

# Check if models are present
ls -lh models/qwen2.5-coder/
# Expected: model.onnx, model.onnx.data, tokenizer.json

# If symlinks are broken (models moved from laptop):
find . -xtype l  # Find broken symlinks
# Fix with:
# ln -sf /home/beelink/Desktop/models/qwen2.5-coder ./models/qwen2.5-coder

# Verify ONNX Runtime
ldconfig -p | grep onnxruntime
# Should show: libonnxruntime.so.1.18.1

# Verify CUDA
nvidia-smi
nvcc --version  # Should show 12.8
```

---

## STEP 2: Build Verification

```bash
cd /home/beelink/Desktop/Niodoo-Final

# Try normal build first
cargo build --release --all

# If git TLS fails, use offline mode:
cargo build --release --all --offline

# Expected output:
#   Compiling niodoo-core v0.1.0
#   Compiling tcs-ml v0.1.0
#   Compiling niodoo-consciousness v0.1.0
#   Finished release [optimized] target(s) in XXs
```

**Known State:**
- ✅ 0 compilation errors (verified by 10-Claude storm)
- ✅ Core systems are REAL (no stubs in pipeline)
- ⚠️ 172 warnings (non-blocking)

---

## STEP 3: Create Integration Bridge

**What's Missing:** TCS and Niodoo are in the same workspace but NOT wired together.

**Current State:**
- TCS embedder (tcs-ml): Generates 896D vectors with KV cache
- Niodoo consciousness: Expects embeddings → emotional mapping → compass → ERAG

**Create Bridge Module:**

```bash
# Create new module
mkdir -p niodoo-core/src/tcs_bridge
```

**File: niodoo-core/src/tcs_bridge/mod.rs**

```rust
//! TCS → Niodoo integration bridge
//! Maps TCS embeddings to Niodoo's emotional vectors

use tcs_ml::QwenEmbedder;
use crate::rag_integration::EmotionalVector;
use crate::consciousness_compass::CompassState;
use anyhow::Result;

pub struct TcsNioodooBridge {
    embedder: QwenEmbedder,
}

impl TcsNioodooBridge {
    pub fn new(model_path: &str) -> Result<Self> {
        let embedder = QwenEmbedder::new(model_path)?;
        Ok(Self { embedder })
    }

    /// Convert text → TCS embedding → Niodoo emotional vector
    pub fn text_to_emotional_vector(&self, text: &str) -> Result<EmotionalVector> {
        // Step 1: Get 896D embedding from TCS
        let embedding = self.embedder.embed(text)?;

        // Step 2: Map to 5D emotional space
        // Use PCA-like projection from 896D → 5D
        // (Simplified - real version would use learned projection matrix)
        let joy = self.project_dimension(&embedding, 0);
        let sadness = self.project_dimension(&embedding, 1);
        let anger = self.project_dimension(&embedding, 2);
        let fear = self.project_dimension(&embedding, 3);
        let surprise = self.project_dimension(&embedding, 4);

        Ok(EmotionalVector {
            joy,
            sadness,
            anger,
            fear,
            surprise,
        })
    }

    /// Extract consciousness state from text
    pub fn text_to_compass_state(&self, text: &str) -> Result<CompassState> {
        let emotional_vec = self.text_to_emotional_vector(text)?;
        Ok(CompassState::from_emotional_vector(&emotional_vec))
    }

    fn project_dimension(&self, embedding: &[f32], dim: usize) -> f32 {
        // Simplified projection: average specific ranges
        let range_size = embedding.len() / 5;
        let start = dim * range_size;
        let end = start + range_size;

        let sum: f32 = embedding[start..end].iter().sum();
        let avg = sum / range_size as f32;

        // Normalize to [0, 1]
        (avg.tanh() + 1.0) / 2.0
    }
}
```

**Add to niodoo-core/src/lib.rs:**

```rust
pub mod tcs_bridge;
```

**Update niodoo-core/Cargo.toml dependencies:**

```toml
[dependencies]
tcs-ml = { path = "../tcs-ml", features = ["onnx"] }
# ... existing deps
```

---

## STEP 4: Create E2E Integration Test

**File: niodoo-core/tests/integration_tcs_to_consciousness.rs**

```rust
//! E2E test: Text → TCS embedding → Emotional vector → Compass state → ERAG

use niodoo_core::tcs_bridge::TcsNioodooBridge;
use niodoo_core::rag_integration::RagEngine;
use niodoo_core::consciousness_compass::StrategicAction;
use std::env;

#[test]
fn test_full_consciousness_pipeline() {
    let model_path = env::var("QWEN_MODEL_PATH")
        .unwrap_or_else(|_| "/home/beelink/Desktop/models/qwen2.5-coder".to_string());

    // Step 1: Create bridge
    let bridge = TcsNioodooBridge::new(&model_path)
        .expect("Failed to create TCS bridge");

    // Step 2: Test stuck state detection
    let stuck_text = "I've tried everything and nothing works. I'm completely stuck.";
    let stuck_state = bridge.text_to_compass_state(stuck_text)
        .expect("Failed to analyze stuck text");

    println!("Stuck text analysis:");
    println!("  Stuck: {:?}", stuck_state.stuck);
    println!("  Confidence: {:?}", stuck_state.confidence);
    println!("  Strategy: {:?}", stuck_state.strategic_imperative());

    assert!(matches!(stuck_state.strategic_imperative(),
                     StrategicAction::Panic | StrategicAction::Persist),
            "Should detect stuck state");

    // Step 3: Test unstuck state detection
    let unstuck_text = "I just figured it out! The solution was so simple.";
    let unstuck_state = bridge.text_to_compass_state(unstuck_text)
        .expect("Failed to analyze unstuck text");

    println!("\nUnstuck text analysis:");
    println!("  Stuck: {:?}", unstuck_state.stuck);
    println!("  Confidence: {:?}", unstuck_state.confidence);
    println!("  Strategy: {:?}", unstuck_state.strategic_imperative());

    assert!(matches!(unstuck_state.strategic_imperative(),
                     StrategicAction::Discover | StrategicAction::Master),
            "Should detect unstuck state");

    // Step 4: Test intrinsic reward calculation
    let reward = unstuck_state.intrinsic_reward(&stuck_state);
    println!("\nIntrinsic reward (stuck→unstuck): {}", reward);
    assert!(reward > 5.0, "Should have high intrinsic reward for breakthrough");
}

#[test]
fn test_emotional_vector_mapping() {
    let model_path = env::var("QWEN_MODEL_PATH")
        .unwrap_or_else(|_| "/home/beelink/Desktop/models/qwen2.5-coder".to_string());

    let bridge = TcsNioodooBridge::new(&model_path)
        .expect("Failed to create TCS bridge");

    // Test various emotional states
    let test_cases = vec![
        ("I'm so happy and excited!", "joy"),
        ("This is terrible and frustrating.", "anger/sadness"),
        ("I'm scared of what might happen.", "fear"),
        ("Wow, I didn't expect that!", "surprise"),
    ];

    for (text, expected_emotion) in test_cases {
        let emotional_vec = bridge.text_to_emotional_vector(text)
            .expect("Failed to map emotion");

        println!("\nText: \"{}\"", text);
        println!("Expected: {}", expected_emotion);
        println!("Emotional vector:");
        println!("  Joy: {:.3}", emotional_vec.joy);
        println!("  Sadness: {:.3}", emotional_vec.sadness);
        println!("  Anger: {:.3}", emotional_vec.anger);
        println!("  Fear: {:.3}", emotional_vec.fear);
        println!("  Surprise: {:.3}", emotional_vec.surprise);

        // Verify vector is normalized (values in [0, 1])
        assert!(emotional_vec.joy >= 0.0 && emotional_vec.joy <= 1.0);
        assert!(emotional_vec.sadness >= 0.0 && emotional_vec.sadness <= 1.0);
        assert!(emotional_vec.anger >= 0.0 && emotional_vec.anger <= 1.0);
        assert!(emotional_vec.fear >= 0.0 && emotional_vec.fear <= 1.0);
        assert!(emotional_vec.surprise >= 0.0 && emotional_vec.surprise <= 1.0);
    }
}
```

---

## STEP 5: Run Integration Tests

```bash
cd /home/beelink/Desktop/Niodoo-Final

# Set model path
export QWEN_MODEL_PATH=/home/beelink/Desktop/models/qwen2.5-coder

# Run E2E test
cargo test --test integration_tcs_to_consciousness --features onnx -- --nocapture

# Expected output:
# running 2 tests
# test test_emotional_vector_mapping ... ok
# test test_full_consciousness_pipeline ... ok
#
# test result: ok. 2 passed; 0 failed
```

**If tests fail, check:**
1. Model path is correct (`ls $QWEN_MODEL_PATH`)
2. ONNX Runtime is found (`ldconfig -p | grep onnxruntime`)
3. CUDA is available (`nvidia-smi`)

---

## STEP 6: Verify Full Pipeline

```bash
# Run existing TCS tests
cargo test -p tcs-ml --lib --features onnx

# Run existing Niodoo tests
cargo test -p niodoo-core --lib

# Run workspace-wide tests
cargo test --workspace --lib --features onnx

# Expected: All tests pass
```

---

## Troubleshooting

### Issue: Model symlinks broken
```bash
find . -xtype l
# If models/qwen2.5-coder is broken:
rm models/qwen2.5-coder
ln -s /home/beelink/Desktop/models/qwen2.5-coder models/qwen2.5-coder
```

### Issue: ONNX Runtime not found
```bash
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
ldconfig -p | grep onnxruntime
```

### Issue: CUDA version mismatch
```bash
# Beelink has CUDA 12.8
nvcc --version
nvidia-smi
# If mismatch, update CUDA toolkit
```

### Issue: Git TLS handshake failure
**See GIT_TLS_TROUBLESHOOTING.md for complete solutions.**

Quick fix:
```bash
# Use vendored dependencies
cargo build --release --offline
```

---

## Expected Final State

After completion:

✅ TCS embedder → Niodoo bridge implemented
✅ Integration tests passing
✅ Full pipeline verified:
   - INPUT text
   - → TCS embedding (896D)
   - → Emotional vector (5D)
   - → Compass state (2-bit)
   - → ERAG retrieval
   - → Strategic action

✅ Beelink build working (with or without git TLS)
✅ All workspace tests passing

---

## Report Back

After completing integration, report:

1. **TLS Status**: Did git TLS work? If not, what solution was used?
2. **Build Status**: Any compilation errors? Warnings count?
3. **Test Results**: How many tests passed/failed?
4. **Bridge Quality**: Does the 896D→5D projection make sense?
5. **Performance**: Latency for full pipeline (text → compass state)?

---

## Notes

- **NO HARDCODING**: Use config files or env vars for all paths
- **NO PRINTLN**: Use tracing::info/warn/error
- **NO STUBS**: Real implementations only (already verified)
- **NO UNWRAP**: Proper error handling with Result<T>

The core systems are already REAL and working. You're just connecting the pipes.

**Good luck! 🚀**
