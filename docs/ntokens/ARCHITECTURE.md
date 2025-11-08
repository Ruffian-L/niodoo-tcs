# Topological Connection Tokens (nTokens) Architecture

## 1. Purpose and Scope

Topological Connection Tokens (nTokens) are the structural currency of NIODOO's Recursive Connectome Engine (RCE). Each nToken is a compositional object that encodes:

- **Relational topology** – how sentence entities, predicates, and contextual cues connect as higher-order simplices.
- **Inference flow** – functorial mapping from grammatical derivations to semantic operators.
- **Value alignment** – ethical and affective constraints shaping admissible reasoning paths.
- **Memory linkage** – hyperbolic embeddings and graph indices for associative recall.
- **Temporal flux** – cobordism-informed evolution across processing cycles.

This document specifies the mathematical underpinnings, software architecture, data representations, GPU runtime strategy, and validation requirements for introducing nTokens into the NIODOO codebase.

## 2. Mathematical Foundations

### 2.1 Functorial Compositionality

- Adopt the **DisCoCat** formalism where grammatical categories form a rigid monoidal category `Preg` and semantic realizations live in a symmetric monoidal category `Sem`.
- Implement the strong monoidal functor `F: Preg → Sem` inside `ntokens::semantics`, ensuring `F(f ∘ g) = F(f) ∘ F(g)` and `F(f ⊗ g) = F(f) ⊗ F(g)`.
- Represent derivations as string diagrams, stored as typed adjacency lists that preserve wire topology.

### 2.2 Cellular Sheaf Structures

- Use **cellular sheaves** over dependency complexes. Stalks hold local semantic tensors or density matrices; restriction maps enforce relational constraints.
- Sheaf Laplacian `L_F = δᵀ δ` drives Neural Sheaf Diffusion layers; cohomology groups detect inconsistencies (e.g., unresolved anaphora).
- Compute cohomology via sparse linear algebra backed by CubeCL kernels for `d` and `δ` operators.

### 2.3 Persistent and Zigzag Homology

- Construct multiparameter filtrations on semantic simplicial complexes using scale (`ε_sem`), value alignment (`ε_val`), and temporal depth (`ε_time`).
- Employ **Multipers** with differentiable signed barcode measures (Hilbert, Euler, rank) for gradient-based optimization.
- Extend to zigzag persistence to capture addition/removal of relations during streaming inference; store histories as cobordism morphisms.

### 2.4 Hyperbolic Memory Geometry

- Model long-term memory in a `d`-dimensional Poincaré ball. Radial norm indicates importance/recency; angular separation encodes similarity.
- Use gyrovector arithmetic for updates. Value embeddings share the same manifold to permit geodesic constraint projection.

### 2.5 Value Alignment via Topological Constraints

- Map hard constraints to boundary points `∂B`; soft constraints become geodesic penalties.
- Incorporate constraint satisfaction into filtration weights and sheaf restriction maps, ensuring violations manifest as infinite or large costs during optimization.

## 3. nToken Data Model

```
struct NToken {
    id: Uuid,
    compositional: CompositionalSignature,
    topology: TopologySignature,
    sheaf: SheafBundle,
    memory: MemoryLinkage,
    values: ValueProfile,
    temporal: TemporalFlux,
    attention: CrossModalAttention,
    metrics: NTokenMetrics,
}
```

### 3.1 CompositionalSignature

- `string_diagram`: compressed representation of the DisCoCat derivation.
- `semantic_tensor`: tensor network or quantum circuit parameters derived from `F(string_diagram)`.
- `morphism_chain`: ordered morphisms enabling inference tracing.

### 3.2 TopologySignature

- `simplicial_complex`: reference to GPU-managed complex storage.
- `persistence`: map of homology degrees `{H0, H1, H2, ...}` with diagrams and signed measures.
- `cohomology`: sheaf Betti numbers and obstruction certificates.

### 3.3 SheafBundle

- `stalks`: heterogeneous typed tensors (Torch-like `DType` support).
- `restriction_maps`: neural modules compiled via CubeCL.
- `laplacian_cache`: reusable operators for diffusion steps.

### 3.4 MemoryLinkage

- `hyperbolic_vector`: Poincaré coordinates (FP16/FP8 mixed precision).
- `graph_edges`: adjacency to prior nTokens (stored in Qdrant via Weaviate bridge).
- `importance_score`: derived from norm and curator feedback.

### 3.5 ValueProfile

- `ethical_embedding`: hyperbolic vector within the value manifold.
- `constraints`: enumerated hard/soft rules with geodesic distances.
- `satisfaction_score`: scalar computed during value gate checks.

### 3.6 TemporalFlux

- `cobordism_chain`: sequence of state transitions, each referencing prior nTokens.
- `zigzag_diagram`: persistence summary over time.
- `decay_parameters`: learned rates for forgetting and consolidation.

### 3.7 CrossModalAttention & Metrics

- Aligns nToken with audio/vision modalities through shared attention keys.
- Metrics include entropy, bottleneck distance deltas, Laplacian spectra, and compute costs.

## 4. Pipeline Integration Blueprint

### 4.1 Processing Stage Insertion

1. **After ERAG retrieval** – instantiate `ntokens::Builder` with dependency graph, retrieved memories, and current runtime config.
2. **Before Compass/TCS** – emit nToken for downstream modules; Compass consumes hyperbolic and value insights, TCS consumes topological invariants.
3. **Curator Feedback Loop** – nToken metrics feed into `integrate_curator()` for quality modulation.

### 4.2 Key Touchpoints

- `niodoo_real_integrated/src/pipeline/core.rs` – add `build_ntoken()` inside prompt processing.
- `niodoo_real_integrated/src/pipeline/stages.rs` – define `PipelineStage::NTokenSynthesis` with timing metrics.
- `niodoo_real_integrated/src/config.rs` – extend runtime configuration for filtration parameters, kernel tuning, and memory/value flags.
- `niodoo_real_integrated/src/tcs_analysis.rs` – share spectrum computations and persistence utilities.

## 5. Module Layout

```
src/ntokens/
  mod.rs
  builder.rs           // orchestrates pipeline stage
  parsing/
    disoccat.rs        // string diagram construction via lambeq output
    functor.rs         // strong monoidal functor implementation
  topology/
    complex.rs         // GPU-backed simplicial/cubical complexes
    filtration.rs      // multiparameter & zigzag filtration logic
    persistence.rs     // Multipers bindings, signed measures
  sheaf/
    bundle.rs          // stalk/restriction data structures
    laplacian.rs       // CubeCL kernels for δ and diffusion
  memory/
    hyperbolic.rs      // gyrovector operations, Weaviate linkage
  value/
    constraints.rs     // geodesic penalty models
  temporal/
    cobordism.rs       // Bord_n functor, history management
  attention.rs         // cross-modal hooks
  metrics.rs           // instrumentation helpers
```

Each submodule exposes trait-based interfaces consumed by the builder and downstream pipeline.

## 6. External Dependencies and Bindings

- **lambeq**: invoked via Python FFI or precomputed metadata to obtain CCG parses and string diagrams.
- **Multipers**: integrate through Rust bindings (existing C++ backend, PyTorch interoperability). Plan dedicated crate `ntokens_multipers` wrapping C ABI.
- **Ripser++ / Gudhi GPU**: optional acceleration for large complexes; evaluate via FFI bridging if CubeCL implementation underperforms.
- **Weaviate**: extend existing REST client to store vectorized persistence signatures.

## 7. GPU Runtime Strategy (H200 NVL)

- Memory budgeting: reserve 40 GB for simplicial complexes, 35 GB for boundary matrices, 30 GB for intermediate persistence calculations, and 20 GB for sheaf diffusion tensors, leaving ~16 GB headroom.
- Mixed precision: leverage FP8 Tensor Cores for large matrix operations; fall back to FP16/BF16 where stability requires.
- Kernel batching: group 64 sentences per batch for 10k-point complexes; adopt pipeline parallelism across NVLink-connected GPUs.
- CubeCL autotuning: enable runtime kernel specialization using `comptime` features for restriction map inference.
- CPU coordination: dedicate host threads to the 0.1% submatrix reductions not suited to GPU parallelism (Ripser++ style split).

## 8. Memory and Value Alignment Integration

- Extend `weighted_episodic_mem.rs` to accept nToken payloads, storing persistence signatures and hyperbolic vectors alongside traditional embeddings.
- Modify fitness scoring to exploit nToken metrics (e.g., persistence entropy, cohomology obstructions) when ranking memories.
- Update value alignment components to operate on nToken `ValueProfile`, ensuring constraint violations trigger curator escalation and compass adjustments.
- Synchronize with Weaviate ingestion pipeline, capturing nToken metadata for associative retrieval.

## 9. Temporal Evolution & Cobordism Handling

- Maintain per-session cobordism chains capturing state transitions; each cobordism corresponds to a natural transformation between nTokens.
- Implement `TemporalFlux::update_from_cycle()` that:
  1. Records boundary manifolds (pre/post nToken states).
  2. Runs zigzag persistence to quantify structural drift.
  3. Adjusts decay parameters for memory consolidation.
- Provide APIs for downstream analytics (e.g., Compass drift detection).

## 10. Observability and Instrumentation

- Emit Prometheus metrics: persistence pairs/sec, sheaf diffusion latency, hyperbolic norm distributions, constraint satisfaction ratios.
- Add structured logs summarizing key invariants per prompt (bottleneck distance shift, cohomology rank changes).
- Integrate tracing spans into `pipeline/core.rs` for `NTokenSynthesis` stage, capturing GPU occupancy.

## 11. Testing Strategy

- **Unit tests**: verify functorial mappings, sheaf restriction consistency, gyrovector arithmetic, constraint evaluations.
- **Property-based tests**: ensure persistence diagrams respect Bottleneck Stability, sheaf Laplacian remains positive semidefinite.
- **Integration tests**: simulate prompt processing with mocked lambeq outputs; validate nToken storage in memory subsystem.
- **Performance harness**: benchmark 10k and 100k point configurations on H200, confirming sub-second and sub-10-second targets respectively.
- **Regression tests**: compare nToken-informed curator outcomes versus baseline to detect degradations.

## 12. Execution Roadmap

1. **Spec & Scaffolding** – finalize data model, create `ntokens` crate structure, stub key traits (no stubs in runtime path).
2. **Topology Core** – implement complexes, filtrations, persistence bindings with full unit coverage.
3. **Sheaf & Memory** – integrate diffusion kernels, hyperbolic embeddings, value constraints.
4. **Pipeline Integration** – wire builder into `pipeline/core.rs`, update configuration, add stage metrics.
5. **GPU Optimization** – profile kernels, tune CubeCL autotuning, validate NVLink scaling.
6. **Validation & Hardening** – execute performance harness, expand observability, document operational procedures.

## 13. Documentation & Change Management

- Update `SYSTEM_ARCHITECTURE.md` and `RUNTIME_FLOW.md` once nToken stage is implemented.
- Maintain living design notes in `docs/ntokens/` for module-specific decisions.
- Record code changes and operational impacts in `CHANGELOG.md` for every commit touching nToken functionality.

## 14. Open Questions

- Preferred integration path for lambeq outputs (runtime call vs. offline preprocessing).
- Strategy for quantum circuit execution (simulate vs. hardware) in `CompositionalSignature`.
- Policy for curator escalation when value constraints conflict with high-utility inference paths.

Resolving these will inform subsequent iterations of the design.



