# 🗺️ Niodoo-Final Code Location Map

**Quick reference for WHERE everything lives**

---

## 🎯 Core Components

### Consciousness Engine
**Location**: `niodoo-core/src/consciousness/`
```
consciousness_compass.rs          (17KB)  - 2-bit Panic/Persist/Discover/Master
consciousness_constants.rs                - Constants for consciousness states
consciousness_state_inversion.rs          - State inversion logic
real_mobius_consciousness.rs              - Real Möbius topology implementation
dual_mobius_gaussian.rs                   - Dual Möbius + Gaussian combination
```

### Memory Systems (ERAG)
**Location**: `niodoo-core/src/`
```
memory/                                   - Core memory module
kv_cache.rs                               - Key-value cache for Qwen
advanced_memory_retrieval.rs              - Advanced retrieval algorithms
dream_state_processor.rs                  - Dream/sleep consolidation
memory_sync_master.rs                     - Multi-layer memory sync
true_nonorientable_memory.rs              - Non-orientable memory topology
optimized_memory_management.rs            - Performance-optimized manager
personal_memory.rs                        - Personal memory engine
guessing_memory_system.rs                 - Guessing sphere memory
```

### RAG System
**Location**: `niodoo-core/src/`
```
rag.rs                                    - Core RAG implementation
rag_integration.rs                  (24KB) - ERAG wave-collapse integration
qwen_integration.rs                       - Qwen embedder integration
qwen_curator.rs                     (50KB) - Learning event curation
```

### Topology Engine
**Location**: `niodoo-core/src/topology/`
```
mobius_gaussian_framework.rs              - Möbius + Gaussian framework
mobius_torus_k_twist.rs                   - K-twist Möbius torus
topology_engine.rs                        - Main topology processor
```

### Dynamic Tokenizer
**Location**: `niodoo-core/src/token_promotion/`
```
(Files in this directory implement 0% OOV tokenizer)
```

### Configuration
**Location**: `niodoo-core/src/config/`
```
mod.rs                                    - Main config module
phase6_config.rs                          - Phase 6 integration config
```

### Integration Phases
**Location**: `niodoo-core/src/`
```
phase6_integration.rs                     - Phase 6 full integration
phase7_consciousness_psychology.rs        - Phase 7 psychology integration
engines.rs                                - Engine orchestration
```

---

## 🧬 TCS Framework

### ML/Embeddings
**Location**: `tcs-ml/src/`
```
qwen_embedder.rs              - Stateful Qwen ONNX embedder (4/5 tests ✅)
qwen_config.rs                - Configuration for Qwen
lib.rs                        - Public exports
bin/test_qwen_stateful.rs     - Smoke tests
```

### Core Topology
**Location**: `tcs-core/src/`
```
(Topology primitives and core data structures)
```

### TDA (Topological Data Analysis)
**Location**: `tcs-tda/src/`
```
(Persistent homology, sheaf theory implementations)
```

### Knot Theory
**Location**: `tcs-knot/src/`
```
(Knot classification algorithms)
```

### Pipeline
**Location**: `tcs-pipeline/src/`
```
(Integration pipeline orchestration)
```

---

## 💾 Training Data (THE GOLD)

**Location**: `data/training_data/`
```
emotion_training_data.csv       (20,001 lines)  - Core emotion training
emotion_training_data.json      (8.1MB)         - JSON format
learning_curve.csv              (1,001 lines)   - Convergence proof
learning_events.csv             (51 lines)      - Learning event log
continual_training_data.csv     (51 lines)      - Continual learning data

TOTAL: 21,104 lines of REAL training data
```

**Key Exports**:
- `niodoo-core/src/bin/training_data_export.rs` (36KB) - THE WORKING TRAINER

---

## 🔧 Build Configuration

### Root Workspace
**Location**: `Cargo.toml` (root)
```toml
[workspace]
members = [
    "tcs-ml",
    "tcs-core",
    "tcs-tda",
    "tcs-knot",
    "tcs-tqft",
    "tcs-consensus",
    "tcs-pipeline",
    "niodoo-core",       # Production consciousness engine
    "constants_core",    # Shared constants
    "src",               # Experimental (372 files)
]

[workspace.dependencies]
# All unified here
```

### Package Manifests
```
niodoo-core/Cargo.toml     - Consciousness engine dependencies
tcs-ml/Cargo.toml          - Qwen embedder dependencies
constants_core/Cargo.toml  - Constants package
```

---

## 🧪 Tests & Benchmarks

### Working Tests
**Location**: `tcs-ml/src/qwen_embedder.rs`
```rust
#[cfg(test)]
mod tests {
    // 4/5 PASSING:
    merge_errors_when_present_shrinks_context ✅
    merge_appends_when_incremental_present ✅
    merge_falls_back_when_present_expands_beyond_sum ✅
    merge_returns_present_when_full_sequence ✅
}
```

### Benchmarks
**Location**: `niodoo-core/benches/`
```
(Performance benchmarks for consciousness operations)
```

---

## 📂 File Counts

| Directory | Files | Purpose |
|-----------|-------|---------|
| `niodoo-core/` | 30+ | Production consciousness engine |
| `tcs-ml/` | 5 | Qwen embedder (WORKING) |
| `tcs-core/` | ~10 | Topology primitives |
| `constants_core/` | 3 | Shared constants |
| `src/` | 372 | Experimental/legacy code |
| `data/training_data/` | 5 | 21K+ lines training data |

**Total Lines of Code**: ~149,000 (production only)

---

## 🚨 Error Locations

**Current Issues**: 19 unique errors (38 total)

### Error Hotspots
1. `niodoo-core/src/qwen_integration.rs` - Missing `run_validation_comparison` method
2. `niodoo-core/src/lib.rs` - Missing `parking_lot` import
3. Various files - Type annotation needed (7 locations)
4. `niodoo-core/src/rag.rs` - `LogitsProcessor` missing Debug trait

---

## 🔍 Quick Navigation

### "Where is the Möbius topology?"
→ `niodoo-core/src/topology/mobius_torus_k_twist.rs`
→ `niodoo-core/src/real_mobius_consciousness.rs`
→ `niodoo-core/src/dual_mobius_gaussian.rs`

### "Where is ERAG?"
→ `niodoo-core/src/rag_integration.rs` (24KB - wave-collapse)
→ `niodoo-core/src/rag.rs` (core implementation)

### "Where is the 2-bit compass?"
→ `niodoo-core/src/consciousness_compass.rs` (17KB)

### "Where is the Qwen integration?"
→ `tcs-ml/src/qwen_embedder.rs` (stateful KV cache - TESTS PASSING)
→ `niodoo-core/src/qwen_integration.rs` (consciousness integration)
→ `niodoo-core/src/qwen_curator.rs` (50KB - learning curation)

### "Where is the dynamic tokenizer?"
→ `niodoo-core/src/token_promotion/` (0% OOV achievement)

### "Where is the training data?"
→ `data/training_data/emotion_training_data.csv` (20,001 lines - THE GOLD)

### "Where is the working trainer?"
→ `niodoo-core/src/bin/training_data_export.rs` (36KB)

### "Where are the memory systems?"
→ `niodoo-core/src/memory/` (guessing spheres)
→ `niodoo-core/src/optimized_memory_management.rs` (pooling)
→ `niodoo-core/src/personal_memory.rs` (personal engine)
→ `niodoo-core/src/memory_sync_master.rs` (multi-layer sync)

---

## 🎯 Critical Files for GitHub Release

**Must be working**:
1. ✅ `tcs-ml/src/qwen_embedder.rs` (4/5 tests passing)
2. ⚠️ `niodoo-core/src/lib.rs` (19 errors)
3. ✅ `data/training_data/emotion_training_data.csv` (proof)
4. ⚠️ `niodoo-core/src/qwen_integration.rs` (needs method)
5. ✅ `constants_core/src/consciousness.rs` (public constants)

**Must be documented**:
1. ⏳ `README.md` (architecture overview)
2. ⏳ `ARCHITECTURE.md` (technical deep-dive)
3. ✅ `INTEGRATION_STATUS_REPORT.md` (current status)
4. ✅ `NIODOO_MASTER_CHECKLIST.md` (progress tracking)

---

**Last Updated**: 2025-10-18T16:35:00Z
**Purpose**: Quick reference for navigating 149K lines of code
