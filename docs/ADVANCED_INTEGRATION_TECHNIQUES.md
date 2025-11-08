# Advanced Integration Techniques & Literature Survey

## Purpose

This document provides a literature-backed survey of advanced integration techniques and optimization strategies applicable to NIODOO-TCS's back-half pipeline optimization (Phase 1-6). These techniques inform our implementation roadmap and provide a foundation for post-MVP enhancements.

## Phase 1: ERAG & Qdrant Optimization

### Candle→Qdrant Integration Patterns

**NUMA-Aware Batching**: Modern multi-socket systems benefit from NUMA-aware memory allocation. When batching Qdrant upserts, consider:
- Aligning batch boundaries with NUMA node boundaries
- Using memory pools that respect NUMA topology
- Batching reduces cache line contention and improves memory locality

**Pooled gRPC via Tonic**: The `tonic` crate (Rust gRPC implementation) supports connection pooling:
- Reuse gRPC channels across requests
- Implement connection pooling with `tonic::transport::Channel::from_static()` with lazy connection
- Reduce connection overhead for batch operations

**Rationale**: Our Phase 1.2 batched upsert implementation leverages these patterns:
- Batch size (128) tuned for NUMA locality
- Background flush task reduces connection overhead
- Circuit breaker protection aligns with reliability patterns

### Semantic Caching Extensions

**123× Exact-Hit Speedups**: Semantic caching can provide massive speedups for exact query matches:
- Cache collapse results by embedding hash
- Use TTL caches (300s) as mentioned in proposal
- Future enhancement: integrate semantic cache layer post-batching optimization

**Integration Point**: Once ERAG retrieval latency is under control (Phase 1.2 complete), semantic caching becomes a high-leverage addition.

## Phase 2: TCSAnalyzer Acceleration

### Persistent Laplacian vs. Homology

**Current Approach**: Full persistent homology computation (Vietoris-Rips filtrations)
- O(n³) complexity for n=896D points
- CPU-bound, 150-300ms per cycle

**Literature Discussion**: Persistent Laplacian offers alternative approach:
- Computes eigenvalues of persistent Laplacian matrices
- Can be faster than full homology computation
- Maintains topological signal detection

**Future Enhancements** (Post-giotto/Gudhi baseline):
- **Ripser**: Optimized C++ implementation of persistent homology
- **OpenPH**: Open-source persistent homology library
- **CubeCL**: GPU-accelerated homology computation

**Rationale**: Our Phase 2.1 giotto-tda integration provides the baseline (60% speedup). Once stable, we can evaluate Ripser/OpenPH for further gains while maintaining β₁ fidelity ≥95%.

**Refined Fallback Triggers**: Our Phase 2.3 adaptive fallback design uses:
- Differential metrics (KS/Wasserstein distance, Δβ counts) instead of global entropy spikes
- Learned classifier for risk assessment
- Aligns with reliability/orchestration best practices

## Phase 3: LearningLoop Optimization

### QLoRA/Candle Footprint Numbers

**VRAM Reduction**: 
- fp32 adapters: 6-8GB on 7B models
- fp16 adapters: 3-4GB (50% reduction) - **Phase 3.1 target**
- 4-bit NF4: Further reduction possible (future enhancement)

**DQN Variants**:
- **Prioritized Replay**: Sample transitions based on TD-error magnitude
- **Dueling Heads**: Separate value and advantage streams
- **Phase 3.2 target**: Async policy updates + batched replay buffers

**Future Iterations**:
- **ES-DQN**: Evolution Strategies + DQN hybrid
- **iDDQN**: Improved DQN with better exploration
- Keep in "next iteration" list after fp16 baseline is stable

**Rationale**: Phase 3.1 (fp16 adapters) and Phase 3.2 (async training) align with these DQN best practices. Prioritized replay can be added post-MVP.

## Phase 4: Curator & Weighted Memory

### Six-Layer Memory Hierarchy

**Current Implementation**: Weighted Episodic Memory with:
- SmoothWeightEvolution
- GPUMemoryFitnessCalculator
- TopologyMemoryAnalyzer
- MemoryConsolidationManager

**Research Alignment**: Our fitness-weighted retrieval aligns with neuroscience-inspired episodic memory research:
- Multi-factor fitness: temporal decay, PAD salience, β₁ connectivity, retrieval count, consonance
- Three-phase temporal decay dynamics
- PAD emotional salience calculation

**RL-Informed LSM-Tree Work**:
- **RusKey**: Research on LSM-tree optimization with RL
- **CAMAL**: Combined memory and learning storage systems
- **Future Enhancement**: Long-term storage tuning once core batching is done (Phase 1 complete)

**Rationale**: Phase 4.3 (GPU fitness) and Phase 4.4 (CRDT consolidation) establish the foundation. RL-informed storage tuning can follow.

## Alignment & Orchestration

### Reliability/Orchestration Patterns

**Circuit Breakers**: Already implemented in `EragClient`:
- Protects Qdrant requests from cascading failures
- Exponential backoff via `CircuitBreakerConfig`

**Curator Feedback Loops**: Phase 4.2 implements:
- Links curator outcomes to curiosity coefficients
- RLAIF for curator-derived rewards
- Aligns with reliability best practices

**Rationale**: Our roadmap already incorporates these patterns. Phase 5 (Telemetry) adds comprehensive monitoring.

## Exploration Controller

### Topological Learning Extensions

**TopoLoss**: Topology-aware loss functions for neural networks
**Sheaf Neural Networks**: Sheaf theory applied to neural architecture
**TREPH**: Topological Representation Enhancement for Predictive Hierarchy

**Future Phases**: These represent rich topological-aware learning extensions:
- Note as inspirations rather than immediate scope
- Consider for Phase 7+ when expanding into richer topo-aware learning
- Current focus: maintaining topological fidelity (β₁ ≥95%) while optimizing performance

**Rationale**: Phase 2 (TCSAnalyzer) establishes the topological foundation. TopoLoss/sheaf/TREPH can build on this foundation post-MVP.

## Integration Roadmap

### Immediate (Phase 1-6)
- ✅ NUMA-aware batching patterns (Phase 1.2)
- ✅ Pooled gRPC via tonic (Phase 1.2)
- ✅ Scalar quantization (Phase 1.3)
- ✅ Adaptive fallbacks (Phase 2.3)
- ✅ QLoRA fp16 (Phase 3.1)
- ✅ Async DQN (Phase 3.2)
- ✅ GPU fitness (Phase 4.3)

### Post-MVP Enhancements
- Semantic caching (123× speedups)
- Ripser/OpenPH/CubeCL evaluation
- Prioritized replay + dueling heads
- ES-DQN/iDDQN variants
- RL-informed LSM-tree (RusKey/CAMAL)
- TopoLoss/sheaf/TREPH integration

### Long-Term Vision
- Topology surrogate models
- Multi-agent CRDT studies
- PAD+Ghost ablation studies
- Multi-node federation

## References

- Qdrant Quantization: https://qdrant.tech/documentation/guides/quantization/
- Qdrant Optimization: https://qdrant.tech/documentation/guides/optimize/
- HNSW Index Management: https://qdrant.tech/documentation/concepts/indexing/
- Candle Rust ML Framework: https://github.com/huggingface/candle
- Tonic gRPC: https://github.com/hyperium/tonic
- giotto-tda: https://arxiv.org/pdf/2004.02551
- QLoRA: https://arxiv.org/abs/2305.14314

## Notes

This survey strengthens the justification for our optimization roadmap and provides a shopping list for post-MVP enhancements. The immediate focus remains on Phase 1-6 optimizations, with these techniques informing design decisions and future expansion paths.



