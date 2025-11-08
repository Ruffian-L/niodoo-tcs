# NIODOO Component Inventory

## Status Legend
- ✅ **ACTIVE**: Currently used in main pipeline
- ⚠️ **CONDITIONAL**: Used only under specific conditions
- ❌ **DEAD**: Not used, commented out, or archived
- 🔄 **SEPARATE**: Separate system/executable, not integrated into main pipeline
- 📦 **BACKUP**: Backup file (*.full), not actively used

## Active Components (lib.rs)

### Core Pipeline Components ✅

| Module | File | Status | Purpose | Dependencies |
|--------|------|--------|---------|--------------|
| `pipeline` | `pipeline/core.rs` | ✅ ACTIVE | Main processing pipeline | All other components |
| `pipeline::cache` | `pipeline/cache.rs` | ✅ ACTIVE | Pipeline caching system | None |
| `pipeline::metrics` | `pipeline/metrics.rs` | ✅ ACTIVE | Stage timing metrics | None |
| `pipeline::state` | `pipeline/state.rs` | ✅ ACTIVE | Pipeline state types | None |
| `pipeline::stages` | `pipeline/stages.rs` | ✅ ACTIVE | Pipeline stage implementations | curator, tcs_analysis |

### Core Systems ✅

| Module | File | Status | Purpose | Dependencies |
|--------|------|--------|---------|--------------|
| `config` | `config.rs` | ✅ ACTIVE | Runtime configuration | None |
| `embedding` | `embedding.rs` | ✅ ACTIVE | Local ONNX embedding | tcs_ml |
| `erag` | `erag.rs` | ✅ ACTIVE | Memory retrieval (gRPC) | qdrant_client |
| `generation` | `generation.rs` | ✅ ACTIVE | vLLM generation | reqwest |
| `compass` | `compass.rs` | ✅ ACTIVE | Consciousness compass | None |
| `learning` | `learning.rs` | ✅ ACTIVE | QLoRA learning loop | erag, token_manager |
| `curator` | `curator.rs` | ✅ ACTIVE | Quality assessment | reqwest, vLLM or Ollama |
| `tcs_analysis` | `tcs_analysis.rs` | ⚠️ CONDITIONAL | Topological analysis | Only in Hybrid mode |
| `rce` | `rce/*.rs` | ✅ ACTIVE (metrics), ⚠️ ACTIONS GATED | β_meta, consensus, optional control | None |
| `security` | `security.rs` | ✅ ACTIVE | Prompt security | None |
| `token_manager` | `token_manager.rs` | ✅ ACTIVE | Dynamic tokenization | None |
| `tokenizer` | `tokenizer.rs` | ✅ ACTIVE | Tokenizer utilities | None |

### Memory & Topology ✅

| Module | File | Status | Purpose | Dependencies |
|--------|------|--------|---------|--------------|
| `weighted_episodic_mem` | `weighted_episodic_mem.rs` | ✅ ACTIVE | Weighted memory system | erag |
| `weight_evolution` | `weight_evolution.rs` | ✅ ACTIVE | Weight evolution | weighted_episodic_mem |
| `gpu_fitness` | `gpu_fitness.rs` | ✅ ACTIVE | GPU fitness calculation | None |
| `topology_memory` | `topology_memory.rs` | ✅ ACTIVE | Topology memory analyzer | None |
| `memory_consolidation` | `memory_consolidation.rs` | ✅ ACTIVE | Memory consolidation | weighted_episodic_mem |
| `mcts` | `mcts.rs` | ✅ ACTIVE | Monte Carlo tree search | None |
| `torus` | `torus.rs` | ✅ ACTIVE | Torus projection mapping | None |

### Supporting Systems ✅

| Module | File | Status | Purpose | Dependencies |
|--------|------|--------|---------|--------------|
| `data` | `data.rs` | ✅ ACTIVE | Data structures | None |
| `util` | `util.rs` | ✅ ACTIVE | Utilities (ROUGE, seeding) | None |
| `metrics` | `metrics.rs` | ✅ ACTIVE | Prometheus metrics | None |
| `health` | `health.rs` | ✅ ACTIVE | Health checks | None |
| `signals` | `signals.rs` | ✅ ACTIVE | Failure signals | None |
| `circuit_breaker` | `circuit_breaker.rs` | ✅ ACTIVE | Circuit breaker pattern | None |
| `consonance` | `consonance.rs` | ✅ ACTIVE | Consonance computation | compass, curator |
| `hyperfocus` | `hyperfocus.rs` | ✅ ACTIVE | Hyperfocus detection | signals |
| `resource_budget` | `resource_budget.rs` | ✅ ACTIVE | Resource budget tracking | None |
| `degradation_tiers` | `degradation_tiers.rs` | ✅ ACTIVE | Graceful degradation | resource_budget |
| `temporal_tda` | `temporal_tda.rs` | ✅ ACTIVE | Temporal TDA analysis | None |

### API & Integration ✅

| Module | File | Status | Purpose | Dependencies |
|--------|------|--------|---------|--------------|
| `api_clients` | `api_clients.rs` | ✅ ACTIVE | API client utilities | reqwest |
| `curator_parser` | `curator_parser.rs` | ✅ ACTIVE | Curator response parsing | None |
| `embedded_qdrant` | `embedded_qdrant.rs` | ⚠️ CONDITIONAL | Embedded Qdrant spawn | Only if feature enabled |
| `vector_store` | `vector_store.rs` | ✅ ACTIVE | Vector store utilities | None |

### Advanced Features ✅

| Module | File | Status | Purpose | Dependencies |
|--------|------|--------|---------|--------------|
| `tcs_lora` | `tcs_lora.rs` | ✅ ACTIVE | TCS-aware LoRA training | learning |
| `tcs_predictor` | `tcs_predictor.rs` | ✅ ACTIVE | TCS-based prediction | tcs_analysis |
| `topology_crawler` | `topology_crawler.rs` | ✅ ACTIVE | Topology crawling | None |
| `lora_trainer` | `lora_trainer.rs` | ✅ ACTIVE | LoRA training utilities | None |
| `conversation_log` | `conversation_log.rs` | ✅ ACTIVE | Conversation logging | None |
| `memory_architect` | `memory_architect.rs` | ✅ ACTIVE | Memory architecture | None |
| `graph_exporter` | `graph_exporter.rs` | ✅ ACTIVE | Graph export utilities | None |
| `emotional_graph` | `emotional_graph.rs` | ✅ ACTIVE | Emotional graph processing | None |
| `benchmark` | `benchmark.rs` | ✅ ACTIVE | Benchmarking utilities | None |

### Evaluation & Testing ✅

| Module | File | Status | Purpose | Dependencies |
|--------|------|--------|---------|--------------|
| `eval` | `eval/mod.rs` | ✅ ACTIVE | Evaluation framework | None |
| `eval::metrics` | `eval/metrics.rs` | ✅ ACTIVE | Evaluation metrics | None |
| `eval::synthetic` | `eval/synthetic.rs` | ✅ ACTIVE | Synthetic data generation | None |

### Mock Systems ✅

| Module | File | Status | Purpose | Dependencies |
|--------|------|--------|---------|--------------|
| `mock_qdrant` | `mock_qdrant.rs` | ✅ ACTIVE | Mock Qdrant for testing | None |
| `mock_vllm` | `mock_vllm.rs` | ✅ ACTIVE | Mock vLLM for testing | None |
| `test_support` | `test_support.rs` | ✅ ACTIVE | Test utilities | None |

## Dead/Unused Code

### Backup Files 📦

| File | Status | Notes |
|------|--------|-------|
| `pipeline.rs.full` | 📦 BACKUP | Backup of old pipeline implementation |
| `config.rs.full` | 📦 BACKUP | Backup of old config |
| `learning.rs.full` | 📦 BACKUP | Backup of old learning code |

### Alternative Implementations ❌

| Path | Status | Notes |
|------|--------|-------|
| `pipeline_v2/` | ❌ DEAD | Alternative pipeline implementation (not used) |
| `pipeline_legacy.rs` | ❌ DEAD | Commented out in lib.rs (line 24-26) |
| `config_v2/` | ❌ DEAD | Alternative config system (not used) |

### Separate Systems 🔄

| Path | Status | Notes |
|------|--------|-------|
| `curator_executor/` | 🔄 SEPARATE | Standalone system with knowledge distillation, memory curation, executor |
| `consciousness_engine/` | ❓ UNKNOWN | Need to check if used |
| `cpp-qt-brain-integration/` | ❓ UNKNOWN | C++ integration, need to check if used |

## Component Initialization Order

Based on `Pipeline::initialise()`:

1. **Config Loading**: `RuntimeConfig::load()`
2. **Dataset Loading**: `load_emotional_dataset()`
3. **Stats Computation**: `compute_dataset_stats()`
4. **Thresholds**: `Thresholds` struct creation
5. **Embedder**: `QwenStatefulEmbedder::new()` (LOCAL)
6. **Torus Strategy**: `TorusSeedStrategy` initialization
7. **Compass**: `CompassEngine::new()`
8. **Qdrant Process**: Optional embedded Qdrant spawn
9. **ERAG**: `EragClient::new()` (gRPC connection)
10. **Tokenizer**: `DynamicTokenizerManager::initialise()`
11. **Generator**: `GenerationEngine::new_with_config()` (vLLM)
12. **Config Arc**: `Arc<RwLock<RuntimeConfig>>`
13. **Security**: `PromptSecurityManager::new()`
14. **Learning**: `LearningLoop::new()`
15. **TCS Analyzer**: Conditional (only if Hybrid mode)
16. **Curator**: Conditional (if `enable_curator` - but should always be true!)
17. **Caches**: `PipelineCache` for embeddings and collapse
18. **Weighted Memory**: `SmoothWeightEvolution`, `GPUMemoryFitnessCalculator`, etc.
19. **MCTS**: `MctsDaydreamer::new()`
20. **Discovery Queue**: Background discovery processing
21. **Cascade Tracker**: `CascadeTracker::new()`
22. **Hyperfocus Detector**: `HyperfocusDetector::new()`

## Dependencies Summary

### External Services Required
- **vLLM**: GenerationEngine, Curator (if using vLLM backend)
- **Qdrant**: EragClient (via gRPC)

### External Services Optional
- **Ollama**: Curator (only if backend set to Ollama)

### Local Dependencies
- **tcs_ml**: QwenStatefulEmbedder (ONNX model)
- **qdrant_client**: EragClient
- **reqwest**: GenerationEngine, Curator
- **tcs-rce**: RCE primitives (β_meta, Laplacians wrapper)
- **tcs-knot**: Experimental (optional via `knot` feature)

## Critical Notes

1. **Curator is PIVOTAL**: Should always be enabled, not optional
2. **Embeddings are LOCAL**: No external service needed
3. **Qdrant uses gRPC**: Automatic conversion from HTTP URLs
4. **TCS Analyzer is Conditional**: Only in Hybrid topology mode
5. **Two Curator Systems**: Integrated (`curator.rs`) and separate (`curator_executor/`)

