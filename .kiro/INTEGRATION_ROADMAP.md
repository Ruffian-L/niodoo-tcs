# Niodoo-Final: Integration Roadmap
## Merging finalREADME Plan with Actual Codebase

**Goal**: Transform 60% architecture alignment → 95%+ by implementing missing components and clarifying structure.

**Timeline**: 4 weeks

---

## Phase 1: Audit & Organize (Week 1)

### Task 1.1: Module Inventory
Create a document mapping every file in `src/` to finalREADME components.

**Action:**
```bash
# Generate inventory
find /home/ruffian/Desktop/Niodoo-Final/src -name "*.rs" \
  -type f | head -50 > /tmp/file_inventory.txt

# You'll see things like:
# src/consciousness_engine/mod.rs → finalREADME's core state management
# src/tcs/mod.rs → finalREADME's persistent homology
# src/bullshit_buster/mod.rs → New in Niodoo (enhancement)
```

**Deliverable**: `.kiro/FILE_INVENTORY.md` with each file's:
- Current purpose
- Alignment to finalREADME (or "Niodoo extension")
- Health status (🟢 complete, 🟡 partial, 🔴 broken, ⚠️ todo)

---

### Task 1.2: Restructure src/ for Clarity

**Current mess:**
```
src/
├── bin/ (30+ scattered binaries)
├── consciousness_engine/ (60KB God module)
├── tcs/ (topology stuff)
├── topology/ (also topology stuff)
├── memory/ (conflicting with mobius_memory/)
├── mobius_memory/
├── bullshit_buster/
├── qt_bridge/
└── [15 other modules]
```

**Goal structure:**
```
src/
├── core/                          # Cognitive state & events
│   ├── mod.rs
│   ├── state.rs                   # CognitiveState (from consciousness_engine)
│   ├── events.rs                  # CognitiveEventBus
│   ├── config.rs
│   └── lib.rs
├── tda/                           # Topology Data Analysis
│   ├── mod.rs
│   ├── takens.rs                  # Takens embedding
│   ├── persistence.rs             # Ripser integration
│   ├── pipeline.rs                # TDAPipeline
│   └── lib.rs
├── knot/                          # Knot Theory
│   ├── mod.rs
│   ├── diagram.rs                 # KnotDiagram
│   ├── jones.rs                   # Jones polynomial
│   ├── analyzer.rs                # KnotAnalyzer
│   └── lib.rs
├── tqft/                          # ⭐ NEW - TQFT Engine (CRITICAL GAP)
│   ├── mod.rs
│   ├── engine.rs                  # TQFTEngine
│   ├── frobenius.rs               # FrobeniusAlgebra
│   ├── cobordism.rs               # CobordismCategory
│   └── lib.rs
├── learning/                      # RL & Learning
│   ├── mod.rs
│   ├── untrying.rs                # UntryingAgent
│   ├── reward.rs
│   └── lib.rs
├── consensus/                     # Consensus mechanisms
│   ├── mod.rs
│   ├── raft.rs                    # Raft implementation
│   ├── vocabulary.rs              # Token consensus
│   └── lib.rs
├── memory/                        # Consolidate memory systems
│   ├── mod.rs
│   ├── dual_mobius_gaussian.rs    # Unified memory
│   ├── compression.rs
│   └── lib.rs
├── analysis/                      # Code analysis (bullshit_buster)
│   ├── mod.rs
│   ├── validator.rs               # Code validation
│   └── lib.rs
├── io/                            # Input/output
│   ├── mod.rs
│   ├── websocket.rs               # WebSocket server
│   └── qt_bridge.rs               # Qt visualization
├── monitoring/                    # System monitoring
│   ├── mod.rs
│   ├── silicon_synapse.rs         # GPU/CPU monitoring
│   ├── metrics.rs                 # Prometheus metrics
│   └── lib.rs
├── bin/                           # Binary targets
│   ├── main.rs
│   ├── orchestrator.rs            # Unified entry point
│   ├── consciousness_engine.rs
│   ├── websocket_server.rs
│   └── [others kept as-is for now]
├── lib.rs                         # Main library entry
└── mod.rs
```

**Action Plan:**
1. Don't do a massive refactor yet
2. Create `src/tqft/` module (empty, stub for now)
3. Add comment in `src/lib.rs` documenting which module maps to which finalREADME component
4. Leave other modules alone for now

---

## Phase 2: Fill Critical Gaps (Weeks 2-3)

### Task 2.1: Implement TQFT Engine ⭐ CRITICAL

**Why this matters:**
- finalREADME shows complete TQFT implementation
- Niodoo is missing this entirely
- TQFT is needed for "reasoning engine"

**Implementation (Simplified):**

Start with 2D case (Frobenius algebra) - enough for initial system.

**File: `src/tqft/mod.rs`**
```rust
pub mod frobenius;
pub mod engine;
pub mod cobordism;

pub use frobenius::FrobeniusAlgebra;
pub use engine::TQFTEngine;
pub use cobordism::CobordismCategory;
```

**File: `src/tqft/frobenius.rs`** (scaffold):
```rust
use nalgebra::{DMatrix, DVector};
use num_complex::Complex;

/// Frobenius algebra: minimal structure for 2D TQFT
#[derive(Debug, Clone)]
pub struct FrobeniusAlgebra {
    dimension: usize,
    multiplication_table: Vec<Vec<DVector<Complex<f32>>>>,
    comultiplication_table: Vec<Vec<DVector<Complex<f32>>>>,
    unit: DVector<Complex<f32>>,
}

impl FrobeniusAlgebra {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            multiplication_table: Vec::new(),
            comultiplication_table: Vec::new(),
            unit: DVector::zeros(dimension),
        }
    }

    /// Multiply two algebra elements
    pub fn multiply(&self, a: &[Complex<f32>], b: &[Complex<f32>]) -> Vec<Complex<f32>> {
        // Implementation from finalREADME
        todo!("Implement multiplication")
    }

    /// Verify Frobenius axioms
    pub fn verify_axioms(&self) -> Result<(), String> {
        // Check associativity
        // Check coassociativity
        // Check Frobenius condition
        Ok(())
    }
}
```

**Action Plan:**
1. Copy structure from finalREADME's TQFT section
2. Create stubbed version first (compiles, methods return unimplemented!)
3. Fill in methods one by one

---

### Task 2.2: Complete Knot Analysis Pipeline

**Missing from Niodoo:**
- Projection methods (Isomap, BallMapper, UMAP)
- Jones polynomial caching strategy
- Full KnotAnalyzer implementation

**Action:**
1. Add to `src/knot/analyzer.rs`:
```rust
pub async fn analyze_cycle(&self, cycle: &HomologyCycle) -> Result<CognitiveKnot> {
    // Extract geometric representation
    let geometry = cycle.extract_representative();

    // Project to 3D using available method
    let knot_3d = self.project_geometry(&geometry, 3).await?;

    // Convert to knot diagram
    let diagram = self.create_knot_diagram(&knot_3d)?;

    // Compute Jones (with caching)
    let jones = self.compute_jones_cached(&diagram).await?;

    // Classify knot type
    let knot_type = self.classify_knot(&jones)?;

    // Build CognitiveKnot
    Ok(CognitiveKnot { ... })
}
```

2. Add projection method placeholder:
```rust
async fn project_geometry(&self, points: &[Vec<f32>], target_dim: usize) -> Result<Vec<Vec<f32>>> {
    // Start with simple projection, upgrade later
    // For now: random projection or PCA
    Ok(pca_project(points, target_dim))
}
```

---

### Task 2.3: Create Unified Orchestrator

**Problem**: Pipeline stages scattered across binaries

**Solution**: Create `src/bin/orchestrator.rs` as single entry point:

```rust
// src/bin/orchestrator.rs
use niodoo::prelude::*;
use tokio::sync::mpsc;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize
    let config = load_config()?;
    let state_extractor = StateExtractor::new(&config)?;
    let tda_pipeline = TDAPipeline::new(&config)?;
    let knot_analyzer = KnotAnalyzer::new(&config)?;
    let rl_agent = UntryingAgent::new(&config)?;

    // Create channels
    let (state_tx, state_rx) = mpsc::channel(1000);
    let (event_tx, event_rx) = mpsc::channel(1000);
    let (knot_tx, knot_rx) = mpsc::channel(100);

    // Stage 1: State extraction
    tokio::spawn(async move {
        state_extractor.extract_continuous(state_tx).await
    });

    // Stage 2: TDA processing
    tokio::spawn(async move {
        tda_pipeline.process_stream(state_rx, event_tx).await
    });

    // Stage 3: Knot analysis
    tokio::spawn(async move {
        knot_analyzer.process_events(event_rx, knot_tx).await
    });

    // Stage 4: RL learning
    tokio::spawn(async move {
        rl_agent.process_knots(knot_rx).await
    });

    Ok(())
}
```

---

## Phase 3: Performance & Polish (Week 4)

### Task 3.1: CUDA Memory Optimization

Add to `src/monitoring/silicon_synapse.rs`:
```rust
#[cfg(feature = "cuda")]
pub struct CudaMemoryManager {
    device: CudaDevice,
    memory_pool: CudaMemPool,
    pinned_memory: Vec<CudaPinnedBuffer>,
}

#[cfg(feature = "cuda")]
impl CudaMemoryManager {
    pub async fn allocate_async(&self, size: usize) -> CudaBuffer {
        self.memory_pool.malloc_async(size).await
    }
}
```

### Task 3.2: Comprehensive Benchmarking

Ensure `benches/` covers:
- Persistence computation (1K, 10K, 50K points)
- Jones polynomial (5, 10, 20, 30, 50 crossings)
- Full pipeline end-to-end

### Task 3.3: Testing Coverage

Implement tests from finalREADME section VI:
- Full pipeline integration test
- Knot simplification test
- Consensus vocabulary test
- Property-based tests (proptest)

---

## Phase 4: Documentation (Ongoing)

### Task 4.1: Update .kiro/
- `ARCHITECTURE_ALIGNMENT.md` ✓ (created)
- `INTEGRATION_ROADMAP.md` ✓ (you're reading it)
- `MODULE_INTERFACES.md` (interfaces between tcs-*, tda, knot, etc.)
- `QUICK_START.md` (how to run orchestrator)

### Task 4.2: Add Module READMEs
Each major module gets its own README:
```
src/tda/README.md       # TDA pipeline docs
src/knot/README.md      # Knot theory docs
src/tqft/README.md      # TQFT engine docs
src/consensus/README.md # Consensus mechanism docs
```

---

## Quick Start: Next 3 Days

### Today (When you see this):
1. Read ARCHITECTURE_ALIGNMENT.md (you'll see exactly what's missing)
2. Choose your integration path:
   - **Path A**: Refactor to match finalREADME (slow, risky)
   - **Path B**: Keep Niodoo, fill gaps (recommended, fast)
   - **Path C**: Use docs as reference, no code changes (immediate clarity)

### Tomorrow:
1. If Path B: Create `src/tqft/` module (stub)
2. Add lib.rs comments mapping modules to finalREADME

### Next Day:
1. Start implementing TQFT (copy structure from finalREADME)
2. Create unified orchestrator

---

## Why This Works

1. **Clarity**: ARCHITECTURE_ALIGNMENT.md shows exactly what's missing
2. **Incrementalism**: Don't break things, add missing pieces
3. **Prioritization**: TQFT first (biggest gap)
4. **Documentation**: Future you (and me) will understand the structure
5. **Validation**: Compare against finalREADME spec

---

## Files You Now Have

| File | Purpose |
|---|---|
| `.kiro/ARCHITECTURE_ALIGNMENT.md` | Maps finalREADME ↔ Niodoo |
| `.kiro/INTEGRATION_ROADMAP.md` | You're reading this |
| Niodoo codebase | Already 60% aligned! |

---

## Success Criteria

- [ ] ARCHITECTURE_ALIGNMENT.md reviewed
- [ ] Decision made: Path A, B, or C
- [ ] If Path B: src/tqft/ module created
- [ ] If Path B: Unified orchestrator working
- [ ] All tests pass
- [ ] benchmarks/ show improvement
- [ ] finalREADME features testable in code

**Expected outcome**: 95% alignment in 4 weeks, with full production readiness.

---

*Questions? Review ARCHITECTURE_ALIGNMENT.md for specifics.*
