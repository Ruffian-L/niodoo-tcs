# nToken Module Design

This document expands the `src/ntokens/` module plan, specifying responsibilities, dependency boundaries, and integration notes for differentiable topological computation.

## 1. Crate Topology

- Target: add an optional workspace crate `ntokens` compiled into `niodoo_real_integrated` via feature flag `ntokens`.
- Public API exposes `Builder`, `NToken`, and specialized helper traits; internal modules remain sealed where possible.

```
ntokens
├── builder.rs
├── compositional/
│   ├── disoccat.rs
│   └── functor.rs
├── topology/
│   ├── complex.rs
│   ├── filtration.rs
│   └── persistence.rs
├── sheaf/
│   ├── bundle.rs
│   └── laplacian.rs
├── memory/
│   └── hyperbolic.rs
├── value/
│   └── constraints.rs
├── temporal/
│   ├── cobordism.rs
│   └── zigzag.rs
├── attention.rs
├── metrics.rs
└── errors.rs
```

## 2. Builder Workflow

1. `Builder::from_context(ctx: &PromptContext)` orchestrates the following phases:
   - Parse DisCoCat diagram via `compositional::disoccat` (lambeq output ingestion).
   - Apply `compositional::functor::apply` to obtain semantic tensors.
   - Construct simplicial complex in `topology::complex`.
   - Generate multiparameter filtration with `topology::filtration`.
   - Run persistence via `topology::persistence` (Multipers bindings).
   - Assemble sheaf bundle (`sheaf::bundle`) and compute Laplacian/diffusion.
   - Map memory embeddings through `memory::hyperbolic`.
   - Enforce value constraints via `value::constraints`.
   - Record temporal drift with `temporal::{cobordism, zigzag}`.
2. Returns `NToken` populated with metrics from `metrics` module.

## 3. Topology Module Specifications

### 3.1 `topology::complex`

- Stores cells using coordinate lists (COO) optimized for GPU transfer.
- Supports both simplicial (clique expansion) and cubical complexes.
- Provides builders for `k`-simplices up to configurable maximum (default `k=3`).
- Interfaces with CubeCL kernels for boundary matrix generation; exports `GpuComplexHandle` referencing device buffers.

### 3.2 `topology::filtration`

- Implements multiparameter filtration struct `FiltrationGrid` supporting:
  - Semantic distance (`ε_sem`), value weight (`ε_val`), temporal recency (`ε_time`).
  - Grid discretization with user-configurable resolution.
- Provides deterministic seeding and caching to reuse grids across prompts.
- Exposes smooth (`C¹`) interpolation functions for differentiability.

### 3.3 `topology::persistence`

- Wraps Multipers C++ backend via FFI:
  - Safe Rust API returning `PersistenceDiagram` and `SignedMeasure` structures.
  - Gradient callback handling bridging PyTorch autograd or custom CubeCL autograd stubs.
- Adds fallback CPU implementation (Ripser++) for environments lacking GPU support.
- Ensures Bottleneck-Stability tests in debug asserts.

## 4. Sheaf Module Specifications

- `sheaf::bundle` stores typed stalks using enum-backed trait objects (`SheafStalk` trait for operations).
- Restriction maps implemented as `CubeKernelHandle` produced by small MLP compilers; includes shape validation.
- `sheaf::laplacian` exposes routines to compute `δ`, `δᵀ`, and `L = δᵀδ` with caching of spectral decompositions (for diffusion heat kernels).

## 5. Memory & Value Modules

### 5.1 `memory::hyperbolic`

- Uses `gyrovector` math for addition and scaling.
- Provides conversions to/from Weaviate vector payloads.
- Maintains GPU/CPU parity with tests verifying curvature invariants.

### 5.2 `value::constraints`

- Supports hard constraints (mapped to boundary) and soft constraints (geodesic penalties).
- Interfaces with sheaf restriction maps to attenuate disallowed flows.
- Emits diagnostic metrics for curator escalation.

## 6. Temporal Module

- `temporal::cobordism` models state transitions as morphisms in `Bord_n`; provides combinators to compose transitions.
- `temporal::zigzag` runs zigzag persistence using `topology::persistence` dynamic grids; caches diagrams for sliding windows.

## 7. Error Handling & Diagnostics

- Centralized error enum in `errors.rs` with variants for parsing, GPU kernels, persistence, and constraint violations.
- All public APIs return `Result<T, NTokenError>`; metrics module logs counters upon errors.

## 8. Testing Strategy

- Unit tests in each module with synthetic data (e.g., simple grammars, triangle complexes).
- Integration tests for builder flow to ensure modules compose correctly.
- Benchmark harness gating merges to prevent regressions in GPU kernel latency.

This design formalizes the internal module responsibilities required to deliver differentiable topology within the nToken pipeline stage.



