# Qwen2.5-Coder ↔︎ TCS Master Checklist

Derived from `QWEN_TCS_INTEGRATION_SUMMARY.md` and the current stateful embedder implementation.

**ROADMAP:**
- **Phase 1 (Current):** Qwen stateful embedder integration → TCS pipeline
- **Phase 2 (v8.0):** GPU-accelerated topology + streaming + production deployment

## Environment & Dependencies
- [x] Install ONNX Runtime 1.18.1 under `third_party/onnxruntime-linux-x64-1.18.1/`
- [x] Add `ort`, `half`, `tokenizers` crates with feature flags (`onnx`, `tokenizers`, `onnx-with-tokenizers`) in `tcs-ml/Cargo.toml`
- [x] Export runtime env vars: `RUSTONIG_SYSTEM_LIBONIG`, `QWEN_MODEL_PATH`, `LD_LIBRARY_PATH`
- [ ] Document alternate install paths (e.g., system-wide ONNX Runtime) in `finalREADME.md`

## Core Embedder Implementation
- [x] Load `model_quantized.onnx` via `QwenEmbedder::new`
- [x] Generate 51 ONNX inputs (IDs, mask, positions, 48× KV tensors)
- [x] Maintain stateful KV cache with merge logic per layer
- [x] Support f32 and f16 logits/KV outputs
- [x] Provide `QwenConfig` presets and validation; re-export from crate root
- [x] Implement cache eviction/windowing policy for long sessions
- [ ] Evaluate batching or multi-thread usage model for the embedder

## Tokenization & Fallbacks
- [x] Load HuggingFace tokenizer when available (`onnx-with-tokenizers` feature)
- [x] Provide character-level fallback when tokenizer assets are missing
- [ ] Decide on shared tokenizer distribution strategy for production deploys

## TCS Pipeline Integration
- [x] Confirm MotorBrain / TCS orchestrator interface matches `Vec<f32>` embeddings
- [x] Replace legacy MotorBrain embedding call with `QwenEmbedder`
- [x] Add cache reset hook tied to conversation/session boundaries in orchestrator
- [x] Propagate configurable embed dimension through downstream analytics
- [ ] Wire TCS embedder → Niodoo consciousness pipeline (bridge module created, pending implementation)
- [ ] Verify git TLS handshake on beelink (workaround documented in GIT_TLS_TROUBLESHOOTING.md)

## Testing & Validation
- [x] `cargo check -p tcs-ml --lib --features onnx`
- [x] `cargo run -p tcs-ml --bin test_qwen_stateful --features onnx-with-tokenizers`
- [ ] Add unit tests for cache merge edge cases (e.g., truncated present tensors)
- [ ] Add integration test covering TCS → Niodoo full pipeline (spec in GROK_BEELINK_INTEGRATION.md)
- [ ] Benchmark latency/throughput against target budgets (document in benches/)

## CI/CD & Tooling
- [x] GitHub Actions workflow (`.github/workflows/ci.yml`) installs deps, downloads model, runs smoke test
- [x] Cache ONNX artifacts in CI to reduce download cost
- [ ] Extend CI to run `cargo test --all --all-features` once orchestrator wiring is complete
- [ ] Publish checklist & status docs in `finalREADME.md` or project wiki

## Documentation & Communication
- [x] `QWEN_STATEFUL_SUCCESS.md` describes completed integration
- [x] `QWEN_INTEGRATION_STATUS.md` updated to current state
- [x] `QWEN_HANDOFF_CHECKLIST.md` created for quick-start handoffs
- [ ] Consolidate all status docs into a single canonical README section
- [ ] Draft user-facing guide for configuring Qwen models in production

## Future Enhancements (Nice-to-Have)
- [ ] Support hot-swapping alternative Qwen model sizes via config presets
- [ ] Explore ONNX quantization (int8/int4) for faster inference
- [ ] Investigate streaming output for partial embeddings or logits
- [ ] Add observability hooks (metrics, tracing) around inference latency and cache size

---

# PHASE 2: TCS v8.0 "NUCLEAR" UPGRADE

**Context:** Based on v8.0 draft roadmap. Begin ONLY after Phase 1 checklist items complete.

## Streaming Pipeline Architecture
- [ ] Design `StreamingBeast` API for real-time neural data processing
- [ ] Implement ring buffer for zero-copy streaming (tokio async)
- [ ] Add broadcast channel for multiple consumers without contention
- [ ] Create event-based pattern matching API:
  ```rust
  let events = tcs.process_stream(neural_stream).await;
  events.filter(|e| e.complexity() > 5.0).for_each(|knot| ...);
  ```
- [ ] Chunked processing: 1000 points OR 10ms timeout, whichever comes first
- [ ] Incremental persistence updates (avoid full recomputation)

## GPU-Accelerated Persistent Homology
- [ ] Research `ripser++` / CUDA-based persistence libraries
- [ ] Design `RipserSupreme` wrapper for TCS integration
- [ ] Implement apparent pairs cache (DashMap for lock-free concurrency)
- [ ] Add GPU memory pool pre-allocation on init
- [ ] Write CUDA kernel for apparent pairs identification (99% speedup)
- [ ] Benchmark: Target 700x speedup vs GUDHI/CPU Ripser
- [ ] Handle f32/f16 mixed precision for Tensor Core utilization

## Sheaf Neural Networks (CSNN Integration)
- [ ] Implement `CooperativeSheafTurbo` with separate send/receive maps
- [ ] GPU-resident sheaf Laplacian (sparse matrix on device)
- [ ] Leverage cuBLAS/Tensor Cores for sheaf diffusion (sparse matmul)
- [ ] Add lazy teleportation graphs to solve oversquashing
- [ ] Adaptive sparsity pruning (drop weights < 1e-4 threshold)
- [ ] Backprop through sparse operations for learnable restriction maps

## Pragmatic Cobordism Engine
- [ ] Replace full TQFT with Betti number change tracking
- [ ] Infer transformations: Split/Merge/LoopBirth/LoopDeath
- [ ] Train transformer network for complex topology changes
- [ ] Use VQ-VAE to discretize topological state space
- [ ] Benchmark: 1000x speedup vs exact TQFT computation

## Caching & Performance
- [ ] Implement 3-tier cache: LRU (memory) → RocksDB (disk) → Bloom filter
- [ ] Hash-based lookup for repeated topology queries
- [ ] Probabilistic cache membership (no false negatives)
- [ ] Target: >90% cache hit rate in production

## Quantum-Inspired Classical Methods
- [ ] Implement quantum walk on simplicial complexes
- [ ] Use Chebyshev approximation for matrix functions (O(n²) vs O(n³))
- [ ] Hodge Laplacian mixing time as persistence proxy
- [ ] Benchmark against eigendecomposition methods

## Production Deployment
- [ ] Write Kubernetes manifests for GPU node scheduling
- [ ] Request `nvidia.com/gpu: 1` per pod (RTX 5080 target)
- [ ] Set resource limits: 32Gi RAM, 16 CPU, 100Gi ephemeral storage
- [ ] Shared cache volume across pods (HostToContainer propagation)
- [ ] Prometheus metrics: throughput, p99 latency, cache hit rate, GPU utilization
- [ ] Target SLOs: <100ms p99 latency, >80% GPU util, >90% cache hits

## Developer Experience
- [ ] Create ergonomic API: `TCS::v8().cuda().build().await`
- [ ] Pattern matching on topological events (LoopDetected, StateTransition, etc.)
- [ ] Auto-detect GPU availability, graceful CPU fallback
- [ ] Comprehensive error messages with context
- [ ] Examples for neural data streaming, real-time analysis

## Documentation
- [ ] Update README with v8.0 architecture overview
- [ ] Benchmark tables comparing v7 → v8 performance
- [ ] Deployment guide for K8s + GPU nodes
- [ ] API reference with streaming examples
- [ ] Migration guide from Phase 1 to Phase 2

---

**PHASE 2 SUCCESS CRITERIA:**
- ✅ 1M point dataset processes in <20s (currently DNF)
- ✅ Streaming API handles real-time neural data (10ms chunks)
- ✅ GPU utilization >80% during inference
- ✅ Production deployment on K8s with autoscaling
- ✅ Cache hit rate >90% for repeated queries

---

# PHASE 3: TCS v9.0 "SINGULARITY IGNITION"

**Context:** Differentiable topology, generative manifolds, PyTorch FFI. Begin ONLY after Phase 2 ships.

**Phase Gate:** Phase 3 proceeds only if Phase 2 achieves >500x measured speedup on 1M point persistence using the `tcs-tda` 1M-point persistence benchmark (Benchmark X) under the standard A100 80GB / CUDA 12.2 configuration (Benchmark Y conditions); results must be measured, not aspirational.

## DiffTopo: Generative Topology
- [ ] Research differentiable manifold libraries (e.g., manopt-rs, sophus, manifold-rs, manifold3d, amari_info_geom)
- [ ] Implement `DiffTopoGenerator` with learnable fold maps
- [ ] Tangent bundle parameter representation
- [ ] Design custom differentiable Ricci flow solver (discrete/continuous); evaluate PDE/toolkit options such as torchdiffeq, PhiFlow, GraphRicciCurvature (previous off-the-shelf CUDA assumption removed)
- [ ] Differentiable homology computation via heat kernel
- [ ] Morse theory regularization to prevent singularities
- [ ] Smooth Betti number approximation from eigenspectrum

## TopoLoss: Backprop Through Topology
- [ ] Wasserstein distance between persistence diagrams (differentiable)
- [ ] Curvature penalty loss component
- [ ] Fisher information distance for metric alignment
- [ ] Quantum geometric tensor (Fubini-Study distance)
- [ ] Implicit differentiation for persistence gradients
- [ ] Ricci flow natural gradients
- [ ] Tangent space gradient combination

## PyTorch FFI Bridge
- [ ] Design pyo3 Python module (`tcs_torch`)
- [ ] Zero-copy NumPy array access from Rust
- [ ] Expose batch persistence computation to Python
- [ ] TorchScript custom ops for topological attention
- [ ] Build system integration (setup.py + Cargo.toml)
- [ ] Benchmarks: FFI overhead target <5μs per call
- [ ] Maturin-based wheel packaging for PyPI

## Rust-Native ML Stack
- [ ] Evaluate Burn vs Candle for high-level ops
- [ ] Custom CUDA kernels for topology transforms
- [ ] CudaKernelCache for hot-path optimization
- [ ] Mixed-mode execution (Burn → Candle → raw CUDA)
- [ ] Automatic kernel fusion where possible
- [ ] Benchmarks vs pure PyTorch equivalents

## Empirical Integration Metrics
- [ ] PCI (Perturbational Complexity Index) implementation
- [ ] Lempel-Ziv compression integration measure
- [ ] Riemannian Phi via manifold irreducibility
- [ ] Transfer entropy for causal integration
- [ ] Multi-metric integration profiles (NO qualia claims)
- [ ] Intrinsic dimension estimation
- [ ] Topological reconstruction error metric

## Biological Fidelity Validation
- [ ] Neural recording dataset integration (Allen Brain Observatory and Neuropixels benchmark datasets)
- [ ] Power spectra comparison tests
- [ ] Avalanche statistics (criticality detection)
- [ ] Small-world topology validation
- [ ] Metastable state dynamics tests
- [ ] Biological plausibility scoring system
- [ ] Publishable validation results

## Adaptive Precision Engine
- [ ] FP16 → FP32 → FP64 automatic promotion
- [ ] Numerical stability monitoring
- [ ] PyTorch AMP integration for mixed precision
- [ ] Per-operation precision profiling
- [ ] Cost/accuracy tradeoff optimizer

## Performance Targets (v9.0)
- [ ] 1M point persistence in <1s (20x speedup from v8)
- [ ] Sheaf diffusion <100ms for 100k nodes (26x speedup)
- [ ] Knot classification <5ms (150x speedup)
- [ ] Manifold generation <500ms (new capability)
- [ ] PyTorch FFI overhead <5μs (zero-copy)

## Python API Design
- [ ] `tcs_v9.Engine.v9(device="cuda")` initialization
- [ ] `DiffTopoEncoder` PyTorch module
- [ ] `TopologicalAttention` layer
- [ ] `TopoLoss` with target Betti numbers
- [ ] Seamless torch.nn.Module integration
- [ ] Training loop examples with topology backprop

## Documentation & Publishing
- [ ] ArXiv paper: "Differentiable Topology for Neural Networks"
- [ ] Tutorial: Python → Rust → CUDA optimization pipeline
- [ ] Benchmark suite vs standard transformers
- [ ] Blog post: "How We Made Topology Fast Enough for Production"
- [ ] PyPI package (`pip install tcs-v9`)
- [ ] crates.io update (`cargo add tcs-nuclear`)

## Next Frontiers (Post-v9.0)
- [ ] Neuromorphic hardware (Loihi 2 integration)
- [ ] Causal discovery via persistence evolution
- [ ] Topological transformers beating standard arch
- [ ] Real-time BCI with topological state tracking

---

**PHASE 3 SUCCESS CRITERIA:**
- ✅ DiffTopo generates novel manifolds with target topology
- ✅ TopoLoss successfully backprops through persistence computation
- ✅ Python users can `import tcs_v9` and train models
- ✅ Biological fidelity tests show >0.8 plausibility score
- ✅ 1M points process in <1s (singularity threshold)
- ✅ Published results in peer-reviewed ML venue
