## RCE Dependencies

- `niodoo_real_integrated` → `tcs-rce` (β_meta, Laplacian wrappers)
- `niodoo_real_integrated` → Prometheus metrics (RCE gauges/counters)
- Optional: `tcs-knot` via feature `knot` (default off)

## Flow Hooks

- TCS → RCE (TopologicalSignature)
- RCE → Metrics (Prometheus)
- RCE → Optional actions (Retry gating, Hyperfocus, ERAG ordering, Curriculum)
# NIODOO Dependency Map

## Component Dependency Graph

```
Pipeline (main.rs)
│
├── Config (config.rs)
│   └── RuntimeConfig::load()
│       ├── Environment variables
│       ├── CLI args
│       └── Optional YAML file
│
├── Embedding (embedding.rs)
│   └── QwenStatefulEmbedder
│       └── tcs_ml::QwenEmbedder (LOCAL ONNX - NO EXTERNAL SERVICE!)
│
├── ERAG Memory (erag.rs)
│   └── EragClient
│       └── QdrantClient (gRPC on port 6334)
│           └── [EXTERNAL SERVICE: Qdrant]
│
├── Generation (generation.rs)
│   └── GenerationEngine
│       └── reqwest::Client
│           └── [EXTERNAL SERVICE: vLLM on port 5001]
│
├── Compass (compass.rs)
│   └── CompassEngine
│       └── (no external dependencies)
│
├── Curator (curator.rs) ⚠️ CRITICAL
│   └── Curator
│       ├── CuratorBackend::Vllm
│       │   └── [EXTERNAL SERVICE: vLLM on port 5001]
│       └── CuratorBackend::Ollama (optional)
│           └── [EXTERNAL SERVICE: Ollama on port 11434]
│
├── Learning (learning.rs)
│   └── LearningLoop
│       ├── EragClient (for memory storage)
│       ├── DynamicTokenizerManager
│       └── Config (for adapter paths)
│
├── TCS Analysis (tcs_analysis.rs) ⚠️ CONDITIONAL
│   └── TCSAnalyzer
│       └── (only if TopologyMode::Hybrid)
│
├── Token Manager (token_manager.rs)
│   └── DynamicTokenizerManager
│       └── (no external dependencies)
│
├── Security (security.rs)
│   └── PromptSecurityManager
│       └── (no external dependencies)
│
├── Weighted Memory System
│   ├── SmoothWeightEvolution
│   ├── GPUMemoryFitnessCalculator
│   ├── TopologyMemoryAnalyzer
│   └── MemoryConsolidationManager
│       └── (all use EragClient internally)
│
└── MCTS (mcts.rs)
    └── MctsDaydreamer
        └── (no external dependencies)
```

## Service Dependencies

### External Services

```
┌─────────────────────────────────────────────────────────┐
│                    EXTERNAL SERVICES                    │
└─────────────────────────────────────────────────────────┘

┌─────────────────┐
│   vLLM (5001)   │
│                 │
│  REQUIRED for:  │
│  - Generation   │
│  - Curator      │
│    (if Vllm     │
│     backend)    │
└─────────────────┘
         ▲
         │
    ┌────┴────┐
    │         │
┌───┴───┐ ┌──┴────────┐
│ Gen   │ │ Curator   │
│Engine │ │ (Vllm)    │
└───────┘ └───────────┘


┌─────────────────┐
│  Qdrant (6334)  │
│     (gRPC)      │
│                 │
│  REQUIRED for:  │
│  - ERAG Memory  │
└─────────────────┘
         ▲
         │
    ┌────┴────┐
    │         │
┌───┴───┐ ┌──┴──────────────┐
│ ERAG  │ │ Weighted Memory │
│Client │ │ (uses ERAG)     │
└───────┘ └─────────────────┘


┌─────────────────┐
│ Ollama (11434)  │
│                 │
│  OPTIONAL for:  │
│  - Curator      │
│    (only if     │
│     backend =   │
│     Ollama)     │
└─────────────────┘
         ▲
         │
    ┌────┴────┐
    │         │
┌───┴───┐   (not used if Vllm backend)
│Curator│
│(Ollama│
│backend│
└───────┘
```

## Data Flow Dependencies

### Prompt Processing Flow

```
User Prompt
    │
    ▼
┌─────────────────┐
│ Security Manager│ ← No dependencies
└─────────────────┘
    │
    ▼
┌─────────────────┐
│    Embedder     │ ← LOCAL ONNX (no external service!)
└─────────────────┘
    │
    ▼
┌─────────────────┐
│   ERAG Client   │ ← Qdrant (gRPC)
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  TCS Analyzer   │ ← Conditional (only Hybrid mode)
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  Compass Engine │ ← No dependencies
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ Token Manager   │ ← No dependencies
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  Generation     │ ← vLLM
└─────────────────┘
    │
    ▼
┌─────────────────┐
│    Curator      │ ← vLLM OR Ollama
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  Learning Loop  │ ← ERAG, Tokenizer
└─────────────────┘
    │
    ▼
┌─────────────────┐
│   ERAG Store    │ ← Qdrant (gRPC)
└─────────────────┘
    │
    ▼
Response Output
```

## Circular Dependencies

### None Identified
The system is designed with a clear linear flow. No circular dependencies detected.

## Initialization Order Dependencies

Components must be initialized in this order due to dependencies:

1. **Config** (no dependencies)
2. **Dataset & Stats** (no dependencies)
3. **Thresholds** (depends on stats)
4. **Embedder** (no external dependencies)
5. **Compass** (depends on thresholds)
6. **ERAG** (depends on Qdrant service)
7. **Tokenizer** (no dependencies)
8. **Generator** (depends on vLLM service)
9. **Config Arc** (wrapper for config)
10. **Security** (depends on config)
11. **Learning** (depends on ERAG, Tokenizer, Config)
12. **TCS Analyzer** (conditional, no dependencies)
13. **Curator** (depends on vLLM or Ollama service)
14. **Caches** (no dependencies)
15. **Weighted Memory** (depends on ERAG)
16. **MCTS** (no dependencies)
17. **Supporting Systems** (CascadeTracker, HyperfocusDetector - no dependencies)

## What Can Run Standalone

### Fully Standalone (No External Services)
- Security Manager
- Compass Engine
- Token Manager
- TCS Analyzer
- MCTS Daydreamer
- Weight Evolution (computational only)
- Topology Memory Analyzer

### Requires External Services
- **ERAG**: Requires Qdrant
- **Generation**: Requires vLLM
- **Curator**: Requires vLLM OR Ollama
- **Learning**: Requires ERAG (which requires Qdrant)

## Critical Dependency Notes

1. **Embeddings are LOCAL**: QwenStatefulEmbedder uses local ONNX models, no external service needed
2. **Qdrant uses gRPC**: Code automatically converts HTTP URLs to gRPC (port 6334)
3. **Curator is CRITICAL**: Should always be enabled, backend can be vLLM or Ollama
4. **vLLM is Multi-Purpose**: Used by both Generation and Curator (if using vLLM backend)
5. **Ollama is Optional**: Only needed if curator backend is explicitly set to Ollama

## Service Startup Requirements

### Minimum Required Services
1. **vLLM** (port 5001) - For generation
2. **Qdrant** (port 6334 gRPC) - For memory

### Optional Services
3. **Ollama** (port 11434) - Only if curator backend = Ollama

### No External Service Needed
- Embeddings (local ONNX)

