# NIODOO System Flowchart - Refined to Actual Implementation

```mermaid
flowchart TD
    A["CLI Args / Config<br/>Hardware Profile, Prompts, Swarm Size"] --> B["Pipeline Orchestrator<br/>niodoo_process() + Tokio Async<br/>Beelink/5080Q Tuning"]
    
    B --> C["Emotional Embedding<br/>embed_text_emotional<br/>Base 384D → PAD 768D + 7D VAE Ghosts<br/>Möbius Twist + Valence Injection"]
    
    C --> D{"Compass Decision<br/>MCTS + Shannon/KL Entropy<br/>PAD Quadrants: Panic/Persist/Discover/Master<br/>Dynamic Thresholds from Window"}
    
    D -->|"Parallel: explore/retrieve/generate"| E["ERAG Retrieval + EchoMemoria<br/>Qdrant Top-K Docs<br/>Cosine Sim on Emotional Torus"]
    D -->|"threat/healing flags"| F["Cascading Generation<br/>Claude → GPT → vLLM Local<br/>Retry/Backoff + Curator Quality Gate<br/>BullshitDetector: Coherence/ROUGE-L"]
    
    E --> G["Context Fusion + Learning Events<br/>Compass Branches + Retrieved Memories<br/>Breakthroughs: Novelty + Reward Calc"]
    F --> G
    
    G --> H["Hybrid Response + Metrics<br/>Latency/Entropy/ROUGE Percentiles<br/>Quadrant Threats/Healings"]
    
    H --> I["Memory Storage + Learning<br/>ERAG Upsert<br/>Experience Replay Buffer<br/>Learning Event Tracking"]
    
    %% Embeddings + TCS Subsystem
    subgraph "Embeddings + TCS Bridge"
        J["File Watcher + Dual-Write<br/>notify Debounce + Sidecar/ChromaDB<br/>Patterns: *.rs/*.py + Hash Validation"]
        J --> K["Embedding Engine + niodoo-tcs-bridge<br/>fastembed + Persistent Homology<br/>Takens' Embedding Theorem<br/>Knot Classification + Betti Numbers"]
        K --> L["Topo Analysis<br/>Non-Euclidean Geometry + TQFT Reasoning<br/>Causal Cobordisms + Cognitive Knots<br/>GPU Opts: RTX 6000 / 10k pts in 2.3s"]
        L --> M["MCP Notifier + Swarm Sync<br/>JSON-RPC + Topo-Swarm AI<br/>Constants_Core: Shared Thresholds"]
    end
    
    B -.-> K
    C -.-> L
    
    %% Core Consciousness Engine
    subgraph "niodoo-core: Consciousness Engine"
        N["PadGhostState + Sigma Variance<br/>Pleasure/Arousal/Dominance + Noise<br/>Intrinsic Reward + UCB Branches"]
        N --> O["Topo Quantum Field Theory<br/>Emergent Loops + Slime Mold Flux<br/>Epigenetic Bits + Oxytocin Pumps"]
        O --> P["Curator Executor<br/>Assessment Prompts + Voting<br/>Quality Threshold + Timeout"]
    end
    
    D -.-> N
    I -.-> O
    
    %% Testing/Validation
    subgraph "Rut Gauntlet + Legacy"
        Q["100 Prompts / Labyrinth Mode<br/>Frustration Ruts + Dijkstra 7x7x7<br/>Raw/Blueprint Validation"]
        Q --> R["Dynamic Thresholds + First 20 Cycles<br/>Entropy High + MCTS C + Mirage Sigma"]
        R --> S["Cycle Processing + Exports<br/>CSV/JSON/Plot/Prometheus<br/>Threat/Healing Rates by Quadrant"]
        S --> T["Verdict: Torque Achieved?<br/>Stability, Activation, Breakthroughs >60%"]
    end
    
    Q -.-> B
    S -.-> U["Legacy Monolith src/<br/>Fallback Impl + Smoke Tests"]
    
    %% Self-Adaptive Loop: Partially Implemented
    subgraph "Self-Adaptive Learning Loop<br/>(Partially Implemented - Retry Logic Working,<br/>Full ES/MAML/QLoRA Coming Soon)"
        V["Failure Detection<br/>ROUGE <0.5 OR Entropy_delta >0.1<br/>Circuit Breaker + Retry Budget<br/>✅ IMPLEMENTED"]
        V --> W["Inner Loop: Reflexion + Self-Consistency<br/>Retrieve Qdrant Failures + Reflection Prompts<br/>τ Adaptive Entropy (AEnt/EPO)<br/>⚠️ PARTIAL"]
        W --> X["Outer Loop: ES Perturbation + Reptile Meta-Init<br/>Gaussian Mutation + Fitness (ROUGE + Entropy_delta)<br/>QLoRA Trigger on Breakthrough Cluster<br/>🔜 COMING SOON"]
        X --> Y["Convergence Check<br/>Entropy Stabilize 0.6-0.8 + ROUGE Plateau<br/>McNemar Test + Chaos Injection<br/>🔜 COMING SOON"]
        Y --> Z["Event Bus: Param Update + EWC Reg<br/>Experience Replay Buffer (Prioritized Sum-Tree)<br/>Back to Pipeline<br/>⚠️ PARTIAL"]
    end
    
    %% Response to Failure Detect
    H -.-> V
    %% Loop Back to Orchestrator
    Z -.-> B
    
    %% Styles
    style A fill:#e1f5fe
    style H fill:#c8e6c9
    style T fill:#fff3e0
    style M fill:#f3e5f5
    style O fill:#fce4ec
    style Z fill:#ffecb3
    style V fill:#fff9c4,stroke:#f9a825,stroke-width:2px
    style W fill:#fff9c4,stroke:#f9a825,stroke-width:2px
    style X fill:#fff9c4,stroke:#f9a825,stroke-width:2px
    style Y fill:#fff9c4,stroke:#f9a825,stroke-width:2px
```

## Key Changes Made:

### Removed Aspirational Elements:
- **Removed**: "Gaussian Spheres + MTKGP Embeddings" (not actually implemented)
- **Removed**: "KL Entropy" specific (kept Shannon entropy which is implemented)
- **Removed**: "Hamming Sim" (only Cosine similarity is actually implemented)
- **Removed**: "ChromaDB" from ERAG retrieval (only Qdrant is actually used)
- **Simplified**: Learning pipeline (removed claims about QLoRA training - exists as stubs/docs only)
- **Simplified**: "Distributed Vocab Evolution" → more accurate description of what's actually implemented

### Kept Actual Implementations:
- ✅ Emotional embedding with Möbius twist
- ✅ Compass with MCTS and PAD quadrants  
- ✅ ERAG with Qdrant retrieval
- ✅ Cascading generation (Claude → GPT → vLLM)
- ✅ Bullshit detector with coherence
- ✅ TCS topology analysis
- ✅ Rut gauntlet testing
- ✅ File watcher + ChromaDB sync subsystem
- ✅ Retry loop with failure detection (actually working)

### Self-Learning Loop Status:
- ✅ **Failure Detection**: Fully implemented with ROUGE/entropy checks
- ⚠️ **Reflexion Loop**: Partially implemented (reflection prompts work, but not full self-consistency)
- 🔜 **ES Perturbation**: Coming soon (blueprint exists, not implemented)
- 🔜 **QLoRA Training**: Coming soon (LoRA infrastructure exists as stubs)
- ⚠️ **Experience Replay**: Partially implemented (some buffering exists)

This flowchart accurately separates what's working now from what's planned/partial.

