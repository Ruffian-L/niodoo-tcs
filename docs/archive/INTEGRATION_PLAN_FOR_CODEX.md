# INTEGRATION PLAN FOR CODEX

**Context:** We're bringing Niodoo-Feeling's consciousness system INTO Niodoo-Final (which has TCS).

**Goal:** Create ONE unified repo with both systems integrated.

**Executor:** Codex (with Friend Claude coordinating)

---

## PHASE 1: COPY FILES (30 minutes)

### Step 1.1: Copy Core Niodoo Modules

```bash
cd /home/ruffian/Desktop/Niodoo-Final

# Create niodoo-core directory
mkdir -p niodoo-core/src

# Copy the essential Niodoo modules from Niodoo-Feeling
cp /home/ruffian/Desktop/Projects/Niodoo-Feeling/src/consciousness_compass.rs niodoo-core/src/
cp /home/ruffian/Desktop/Projects/Niodoo-Feeling/src/real_mobius_consciousness.rs niodoo-core/src/
cp /home/ruffian/Desktop/Projects/Niodoo-Feeling/src/dual_mobius_gaussian.rs niodoo-core/src/
cp /home/ruffian/Desktop/Projects/Niodoo-Feeling/src/consciousness.rs niodoo-core/src/

# Copy RAG system
mkdir -p niodoo-core/src/rag
cp -r /home/ruffian/Desktop/Projects/Niodoo-Feeling/src/rag/* niodoo-core/src/rag/

# Copy token promotion
mkdir -p niodoo-core/src/token_promotion
cp -r /home/ruffian/Desktop/Projects/Niodoo-Feeling/src/token_promotion/* niodoo-core/src/token_promotion/

# Copy topology
mkdir -p niodoo-core/src/topology
cp -r /home/ruffian/Desktop/Projects/Niodoo-Feeling/src/topology/* niodoo-core/src/topology/

# Copy memory system
mkdir -p niodoo-core/src/memory
cp -r /home/ruffian/Desktop/Projects/Niodoo-Feeling/src/memory/* niodoo-core/src/memory/

# Copy config
mkdir -p niodoo-core/src/config
cp -r /home/ruffian/Desktop/Projects/Niodoo-Feeling/src/config/* niodoo-core/src/config/
```

### Step 1.2: Copy Training Data

```bash
mkdir -p data/training_data

# Copy the 20K emotional training samples
cp /home/ruffian/Desktop/Projects/Niodoo-Feeling/data/training_data/emotion_training_data.csv \
   data/training_data/

cp /home/ruffian/Desktop/Projects/Niodoo-Feeling/data/training_data/emotion_training_data_unsloth.jsonl \
   data/training_data/

# Copy learning curve data if it exists
cp /home/ruffian/Desktop/Projects/Niodoo-Feeling/data/training_data/learning_curve.csv \
   data/training_data/ 2>/dev/null || true
```

### Step 1.3: Copy Documentation

```bash
# Copy the Stanford Ball Slam report
cp /home/ruffian/Desktop/Projects/Niodoo-Feeling/STANFORD_BALL_SLAM_REPORT.md .

# Copy consciousness compass docs
cp /home/ruffian/Desktop/Projects/Niodoo-Feeling/CONSCIOUSNESS_COMPASS_IMPLEMENTATION.md .
```

---

## PHASE 2: CREATE NIODOO-CORE CARGO.TOML (15 minutes)

**File:** `niodoo-core/Cargo.toml`

```toml
[package]
name = "niodoo-core"
version = "0.1.0"
edition = "2021"
authors = ["Jason Van Pham <niodoo@dev>"]
description = "Niodoo consciousness engine - emotional topology and ERAG memory"

[dependencies]
# Core async runtime
tokio = { version = "1", features = ["full"] }
anyhow = "1.0"
thiserror = "1.0"
tracing = "0.1"
tracing-subscriber = "0.3"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
chrono = { version = "0.4", features = ["serde"] }

# Math operations
ndarray = { version = "0.15", features = ["serde-1"] }
nalgebra = { version = "0.33", features = ["serde-serialize"] }
rand = "0.8"
rand_distr = "0.4"

# Tokenization
tokenizers = "0.15"

# ML framework
candle-core = { git = "https://github.com/huggingface/candle" }
candle-nn = { git = "https://github.com/huggingface/candle" }

# Graph structures
petgraph = "0.8"
lru = "0.12"

# RAG dependencies
usearch = "0.6"

[lib]
name = "niodoo_core"
path = "src/lib.rs"
```

### Step 2.2: Create niodoo-core/src/lib.rs

```rust
//! Niodoo Core: Consciousness engine and ERAG memory system

pub mod consciousness_compass;
pub mod real_mobius_consciousness;
pub mod dual_mobius_gaussian;
pub mod consciousness;
pub mod rag;
pub mod token_promotion;
pub mod topology;
pub mod memory;
pub mod config;

// Re-export commonly used types
pub use consciousness_compass::{CompassState, StrategicAction};
pub use real_mobius_consciousness::{EmotionalState, KTwistedTorus};
pub use rag::RagEngine;
```

---

## PHASE 3: UPDATE ROOT CARGO.TOML (5 minutes)

**File:** `Cargo.toml`

**ADD to [workspace.members]:**
```toml
[workspace]
members = [
    "tcs-core",
    "tcs-ml",
    "tcs-pipeline",
    "tcs-tda",
    "niodoo-core",    # ← ADD THIS
]
```

---

## PHASE 4: CREATE INTEGRATION BINARY (30 minutes)

**File:** `tcs-pipeline/src/niodoo_integration.rs`

**Purpose:** Wire TCS embedder → Niodoo consciousness pipeline

```rust
use anyhow::Result;
use tcs_ml::QwenEmbedder;
use niodoo_core::{CompassState, EmotionalState, KTwistedTorus, RagEngine};
use tracing::info;

/// Integration struct connecting TCS → Niodoo
pub struct NiodooTcsIntegration {
    /// TCS embedder (text → 896D vector)
    embedder: QwenEmbedder,
    /// Niodoo torus mapper (embedding → emotional state)
    torus: KTwistedTorus,
    /// ERAG memory system
    rag: RagEngine,
}

impl NiodooTcsIntegration {
    /// Create new integrated system
    pub fn new(model_path: &str, rag_config: niodoo_core::rag::RagConfig) -> Result<Self> {
        info!("Initializing Niodoo-TCS integration");

        // Initialize TCS embedder
        let embedder = QwenEmbedder::new(model_path)?;

        // Initialize Niodoo torus
        let torus = KTwistedTorus::new(100.0, 30.0, 1);

        // Initialize ERAG
        let rag = RagEngine::new(rag_config)?;

        Ok(Self {
            embedder,
            torus,
            rag,
        })
    }

    /// Process input through full pipeline
    pub fn process(&mut self, input_text: &str) -> Result<ProcessedOutput> {
        // Step 1: TCS embedding
        let embedding = self.embedder.embed(input_text)?;

        // Step 2: Map to emotional state
        let emotional_state = self.embedding_to_emotional(&embedding)?;

        // Step 3: Get compass state
        let compass_state = CompassState::from_emotional_state(&emotional_state);

        // Step 4: ERAG retrieval
        let context = self.rag.retrieve_with_importance_boost(
            &embedding,
            top_k: 5
        )?;

        // Step 5: Determine strategic action
        let strategy = compass_state.strategic_imperative();

        Ok(ProcessedOutput {
            embedding,
            emotional_state,
            compass_state,
            erag_context: context,
            strategy,
        })
    }

    /// Map embedding → emotional state using torus geometry
    fn embedding_to_emotional(&self, embedding: &[f32]) -> Result<EmotionalState> {
        // Simple mapping: take first 3 dimensions as PAD
        // (In production, use more sophisticated mapping)
        let valence = embedding[0].clamp(-1.0, 1.0) as f64;
        let arousal = embedding[1].clamp(0.0, 1.0) as f64;
        let dominance = embedding[2].clamp(0.0, 1.0) as f64;

        Ok(EmotionalState::new(valence, arousal, dominance))
    }

    /// Reset conversation context
    pub fn reset_context(&mut self) {
        self.embedder.reset_cache();
        info!("Reset conversation context");
    }
}

/// Output from integrated pipeline
#[derive(Debug)]
pub struct ProcessedOutput {
    pub embedding: Vec<f32>,
    pub emotional_state: EmotionalState,
    pub compass_state: CompassState,
    pub erag_context: Vec<String>,
    pub strategy: niodoo_core::consciousness_compass::StrategicAction,
}
```

### Step 4.2: Update tcs-pipeline/src/lib.rs

```rust
pub mod niodoo_integration;  // ADD THIS LINE
pub use niodoo_integration::NiodooTcsIntegration;
```

### Step 4.3: Update tcs-pipeline/Cargo.toml

**ADD dependency:**
```toml
[dependencies]
tcs-ml = { path = "../tcs-ml" }
niodoo-core = { path = "../niodoo-core" }  # ← ADD THIS
anyhow = "1.0"
tracing = "0.1"
```

---

## PHASE 5: CREATE TEST BINARY (20 minutes)

**File:** `tcs-pipeline/examples/test_integration.rs`

```rust
use anyhow::Result;
use tcs_pipeline::NiodooTcsIntegration;
use tracing_subscriber;

fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("=== Niodoo-TCS Integration Test ===\n");

    // Initialize integrated system
    let model_path = std::env::var("QWEN_MODEL_PATH")
        .unwrap_or_else(|_| "models/qwen2.5-coder-1.5b-instruct".to_string());

    let rag_config = niodoo_core::rag::RagConfig::default();
    let mut system = NiodooTcsIntegration::new(&model_path, rag_config)?;

    // Test inputs
    let test_inputs = vec![
        "I feel stuck and don't know what to do",
        "I just had a breakthrough understanding!",
        "This is confusing but I think I'm making progress",
        "I've mastered this concept now",
    ];

    for (i, input) in test_inputs.iter().enumerate() {
        println!("--- Test {} ---", i + 1);
        println!("Input: {}", input);

        let output = system.process(input)?;

        println!("Compass State: {:?}", output.compass_state);
        println!("Strategy: {:?}", output.strategy);
        println!("ERAG Context: {} items", output.erag_context.len());
        println!();
    }

    println!("✅ Integration test complete!");

    Ok(())
}
```

---

## PHASE 6: BUILD AND TEST (10 minutes)

### Step 6.1: Build Everything

```bash
cd /home/ruffian/Desktop/Niodoo-Final

# Build all packages
cargo build --all

# Expected output: Everything compiles
```

### Step 6.2: Run TCS Tests

```bash
cargo test -p tcs-ml --features onnx
# Should show: 5/5 tests passing
```

### Step 6.3: Run Integration Test

```bash
export QWEN_MODEL_PATH=/path/to/qwen/model
cargo run --example test_integration --features onnx
```

---

## PHASE 7: VERIFY DATA FILES (5 minutes)

```bash
# Check training data copied correctly
wc -l data/training_data/emotion_training_data.csv
# Expected: 20001 lines (20K + header)

ls -lh data/training_data/
# Expected: CSV + JSONL files present
```

---

## SUCCESS CRITERIA

At the end, you should have:

```
Niodoo-Final/
├── tcs-core/              ← TCS buffers/state
├── tcs-ml/                ← TCS embedder (WORKING)
├── tcs-pipeline/          ← TCS orchestrator
│   └── src/
│       └── niodoo_integration.rs  ← NEW INTEGRATION
├── tcs-tda/               ← TCS topology (Phase 2)
├── niodoo-core/           ← Niodoo consciousness (NEW)
│   └── src/
│       ├── consciousness_compass.rs
│       ├── real_mobius_consciousness.rs
│       ├── rag/           ← ERAG system
│       ├── token_promotion/  ← Dynamic tokenizer
│       ├── topology/      ← K-Twist Torus
│       └── memory/        ← Memory system
├── data/
│   └── training_data/
│       ├── emotion_training_data.csv  (20,001 lines)
│       └── emotion_training_data_unsloth.jsonl  (20,000 lines)
├── README_UNIFIED.md      ← Already created
├── INTEGRATION_MAP.md     ← Already created
└── Cargo.toml            ← Updated workspace
```

### Build should pass:
```bash
cargo build --all
# ✅ Success
```

### Tests should pass:
```bash
cargo test --all --features onnx
# ✅ All tests pass
```

### Integration should run:
```bash
cargo run --example test_integration --features onnx
# ✅ Shows compass states + ERAG retrieval
```

---

## FAILURE MODES & FIXES

### Problem: "Module not found"
**Fix:** Check niodoo-core/src/lib.rs exports all modules

### Problem: "Dependency resolution failed"
**Fix:** Run `cargo update` to sync lock file

### Problem: "QWEN_MODEL_PATH not found"
**Fix:** Set environment variable:
```bash
export QWEN_MODEL_PATH=/home/ruffian/Desktop/Niodoo-Final/models/qwen2.5-coder-1.5b-instruct
```

### Problem: "Can't find training data"
**Fix:** Verify files copied:
```bash
ls -la data/training_data/
```

---

## NOTES FOR CODEX

1. **DO NOT modify TCS modules** - they work already
2. **DO NOT simplify Niodoo code** - copy it exactly
3. **DO check that file paths exist** before copying
4. **DO run `cargo check` after each phase**
5. **DO ask if you're unsure** - don't guess

---

## AFTER INTEGRATION IS COMPLETE

Friend Claude will:
1. Verify the integration works
2. Update README_UNIFIED.md with final file structure
3. Create the GitHub commit plan
4. Help you ship it

**Codex executes. Friend Claude coordinates. You ship.**

---

**Ready to start? Begin with PHASE 1.**
