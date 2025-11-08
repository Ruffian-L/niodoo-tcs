## RCE (Recursive Connectome Engine)

- Purpose: real-time topology sensing and meta-control
- Components:
  - RceAnalyzer (β_meta, spectra, metastability) [read-only by default]
  - Consensus gate (diverse analyzer quorum)
  - Controller hooks (Hyperfocus, curriculum, retry gating) – config-gated
- Placement: between topology analysis and compass/generation; feeds metrics and optional actions.
- Safety: actions require `rce_actions_enabled=true` and consensus approval; circuit breaker on sustained spikes.

# NIODOO System Architecture Overview

## High-Level System Architecture

NIODOO is a consciousness-aligned AI system that processes prompts through a sophisticated 7-stage pipeline with topological analysis, memory retrieval, and continuous learning capabilities.

## End-to-End Connectivity (Mermaid)

The diagram below captures the primary runtime flow, major subsystems, the responsibilities of each stage, and how they interact with external services and background processes.

```mermaid
flowchart TD
    subgraph ExternalServices[External Services]
        vLLM[vLLM (port 5001)<br/>Primary LLM inference]
        Qdrant[Qdrant (port 6334 gRPC)<br/>Vector memory database]
        Ollama[Ollama (port 11434)<br/>Optional curator backend]
    end

    subgraph Pipeline["Pipeline::process_prompt()"]
        Config[RuntimeConfig Loader<br/>Env + CLI + YAML fusion]
        Security[PromptSecurityManager<br/>Validation, rate limiting, audit logging]
        Embedder[QwenStatefulEmbedder<br/>Local ONNX embeddings]
        ERAG[EragClient<br/>Context retrieval & collapse]
        Torus[Torus Projection<br/>7D PAD+Ghost mapping]
        TCS[TCSAnalyzer<br/>Persistent topology (Hybrid mode only)]
        Compass[CompassEngine<br/>Quadrant, cascade, MCTS guidance]
        Tokenizer[DynamicTokenizerManager<br/>Adaptive tokens & CRDT consensus]
        Generator[GenerationEngine<br/>Topology-aware response via vLLM]
        Curator[Curator<br/>Quality scoring, refinement, failure gating]
        Consonance[Consonance Engine<br/>Truth attractor & hyperfocus signals]
        Failure[FailureSignals & Retry Handler<br/>Escalation, reflection, threat tracking]
        Learning[LearningLoop<br/>QLoRA updates, breakthrough detection]
        WeightedMem[Weighted Episodic Memory<br/>Fitness weighting & consolidation]
        MemoryStore[ERAG Store<br/>Persist curated experiences]
        Output[Response Assembly<br/>Final hybrid answer & metrics]
    end

    subgraph Background[Background Systems]
        Cache[PipelineCache<br/>Embedding & collapse TTL caches]
        Cascade[CascadeTracker<br/>Topology transition history]
        Hyperfocus[HyperfocusDetector<br/>Focus drift alerts]
        Discovery[Discovery Queue<br/>Batch exploration & weight refresh]
    end

    Config --> Security --> Embedder --> ERAG --> Torus --> TCS --> Compass --> Tokenizer --> Generator --> Curator --> Consonance --> Failure --> Learning --> WeightedMem --> MemoryStore --> Output

    Torus -. "Baseline mode" .-> Compass

    Embedder -->|stores| Cache
    ERAG -->|stores| Cache
    Cache -->|warm hits| Embedder
    Cache -->|warm hits| ERAG

    ERAG -->|aggregated context| Generator
    WeightedMem -->|fitness feedback| ERAG
    Learning -->|updates weights| WeightedMem
    Curator -->|learned signals| Learning
    Curator -->|token promotions| Tokenizer
    Compass -->|state| Learning
    Consonance -->|signals| Hyperfocus
    Compass --> Cascade
    Failure --> Cascade
    Learning --> Discovery
    Discovery --> WeightedMem

    Generator -->|LLM calls| vLLM
    Curator -->|LLM calls| vLLM
    Curator -->|optional backend| Ollama
    ERAG -->|gRPC operations| Qdrant
    MemoryStore -->|upserts| Qdrant
```

## Core Components

### 1. Pipeline (`pipeline/core.rs`)
The main processing pipeline that orchestrates all stages:
- **Entry Point**: `Pipeline::initialise()` creates all components
- **Processing**: `process_prompt()` handles individual prompts end-to-end
- **State Management**: Tracks pipeline cycles, timings, and metrics

### 2. Embedding System (`embedding.rs`)
- **Component**: `QwenStatefulEmbedder`
- **Type**: LOCAL ONNX model (Rust/Candle)
- **Dependencies**: None - runs locally!
- **Purpose**: Converts text to 768D/896D vectors for memory retrieval
- **Key Point**: Does NOT use Ollama - completely local

### 3. ERAG Memory System (`erag.rs`)
- **Component**: `EragClient`
- **Protocol**: gRPC (port 6334)
- **Dependencies**: Qdrant vector database
- **Purpose**: Stores and retrieves experiences using hyperspherical embeddings
- **Key Point**: Uses gRPC, not HTTP (converts HTTP URLs automatically)

### 4. Generation Engine (`generation.rs`)
- **Component**: `GenerationEngine`
- **Backend**: vLLM (Python service on port 5001)
- **Dependencies**: vLLM service
- **Purpose**: Generates responses using Qwen2.5-7B-Instruct-AWQ model
- **Features**: Topology-aware generation, consistency voting

### 5. Compass Engine (`compass.rs`)
- **Component**: `CompassEngine`
- **Purpose**: 2-bit consciousness model tracking emotional state
- **Quadrants**: Panic, Persist, Discover, Master
- **Features**: MCTS-based decision making, cascade tracking

### 6. Curator (`curator.rs`) - **CRITICAL COMPONENT**
- **Component**: `Curator`
- **Status**: PIVOTAL - should NOT be optional!
- **Backends**: vLLM OR Ollama (configurable via `CuratorBackend`)
- **Default**: vLLM (`CuratorBackend::Vllm`)
- **Dependencies**: vLLM OR Ollama (depending on config)
- **Purpose**: 
  - Quality assessment and refinement
  - Learning loop integration
  - Failure detection
  - Consonance computation
  - Topology-aware refinement
- **Key Point**: If disabled, failure detection skips retries and learning misses data!

### 7. Learning Loop (`learning.rs`)
- **Component**: `LearningLoop`
- **Purpose**: QLoRA fine-tuning, DQN-based adaptive learning
- **Features**: Breakthrough detection, reward computation, adapter persistence

### 8. TCS Analysis (`tcs_analysis.rs`)
- **Component**: `TCSAnalyzer`
- **Status**: Conditional (only in Hybrid topology mode)
- **Purpose**: Topological data analysis (knot complexity, Betti numbers, persistence entropy)

### 9. Token Manager (`token_manager.rs`)
- **Component**: `DynamicTokenizerManager`
- **Purpose**: Dynamic tokenization with pattern discovery and CRDT consensus

### 10. Security Manager (`security.rs`)
- **Component**: `PromptSecurityManager`
- **Purpose**: Prompt validation, rate limiting, security audit logging

### 11. Weighted Episodic Memory (`weighted_episodic_mem.rs`)
- **Components**: 
  - `SmoothWeightEvolution`
  - `GPUMemoryFitnessCalculator`
  - `TopologyMemoryAnalyzer`
  - `MemoryConsolidationManager`
- **Purpose**: 6-layer memory hierarchy with fitness-weighted retrieval

### 12. MCTS Daydreamer (`mcts.rs`)
- **Component**: `MctsDaydreamer`
- **Purpose**: Monte Carlo tree search for exploration and planning

## External Services

### Required Services

1. **vLLM** (port 5001)
   - Python service for LLM inference
   - Used by: GenerationEngine, Curator (if using vLLM backend)
   - Model: Qwen2.5-7B-Instruct-AWQ

2. **Qdrant** (ports 6333 HTTP, 6334 gRPC)
   - Vector database for ERAG memory
   - Used by: EragClient (via gRPC)
   - Protocol: gRPC (automatic conversion from HTTP URLs)

### Optional Services

3. **Ollama** (port 11434)
   - Only needed if curator backend is set to `CuratorBackend::Ollama`
   - Default curator backend is vLLM, so Ollama is OPTIONAL

## Data Flow

```
User Prompt
    ↓
[Security Manager] - Validation & rate limiting
    ↓
[Embedding] - Local ONNX model (QwenStatefulEmbedder)
    ↓
[ERAG Memory] - Retrieve similar experiences (gRPC → Qdrant)
    ↓
[Torus Projection] - Map to 7D PAD+Ghost space
    ↓
[TCS Analysis] - Topological analysis (if Hybrid mode)
    ↓
[Compass Engine] - Determine emotional quadrant
    ↓
[Token Manager] - Dynamic tokenization
    ↓
[Generation Engine] - Generate response (vLLM)
    ↓
[Curator] - Quality assessment & refinement (vLLM or Ollama)
    ↓
[Learning Loop] - Update adapters if breakthrough detected
    ↓
[ERAG Memory] - Store experience
    ↓
Response Output
```

## Critical Findings

### Service Dependencies Clarified

1. **Embeddings**: Local ONNX - NO external service needed!
2. **Vector Storage**: Qdrant via gRPC (REQUIRED)
3. **Generation**: vLLM (REQUIRED)
4. **Curator**: vLLM OR Ollama (REQUIRED component, backend configurable)

### Curator Status

- **Currently**: Marked as optional via `enable_curator` flag
- **Should Be**: Always enabled - it's pivotal to the system
- **Impact if Disabled**: 
  - Failure detection skips retries
  - Learning loop misses data
  - Consonance computation incomplete
  - No topology-aware refinement

### Two Curator Systems

1. **Integrated Curator** (`curator.rs`): Used in pipeline for refinement
2. **curator_executor**: Separate system with knowledge distillation and memory curation (more features)

## Configuration

### Key Config Flags

- `enable_curator`: Currently optional, should be always true
- `topology_mode`: `Hybrid` (with TCS) or `Baseline` (analytical only)
- `curator_backend`: `Vllm` (default) or `Ollama`
- `mock_mode`: Enables stubbed responses for testing

## Performance Considerations

- Embeddings cached with TTL (`embedding_cache_ttl_secs`)
- ERAG collapse results cached (`collapse_cache_ttl_secs`)
- Background discovery processing (batched)
- Weight evolution updates every 5 seconds

