# NIODOO System Setup Guide for AI Assistants

## Quick Start: Understanding This Codebase

This document provides essential context for AI assistants working with the NIODOO codebase. Read these files in order for best understanding.

## Required Reading (In Order)

1. **HOW_TO_START.md** - **START HERE!** Complete step-by-step guide to start all services and endpoints end-to-end
2. **SYSTEM_ARCHITECTURE.md** - High-level overview of all components and how they connect
3. **COMPONENT_INVENTORY.md** - Complete list of all modules with status (ACTIVE/DEAD/CONDITIONAL)
4. **DEPENDENCY_MAP.md** - Visual dependency graph showing what depends on what
5. **RUNTIME_FLOW.md** - Detailed trace of what happens when processing a prompt

## Git Submodules (NEW!)

The repository now includes two git submodules that must be initialized:

- **Niodoo-TCT** (`Niodoo-TCT/`): Topological toolkit for feature extraction, Betti curves, sheaf metrics
  - Repository: https://github.com/Ruffian-L/Niodoo-TCT.git
  - Used for: Hidden-state extraction, topology feature vectorisation
  - See: `docs/TOPOLOGY_PIPELINE.md` for integration details

- **niodoo-ai** (`niodoo-ai/`): Python package for topology-aware training
  - Repository: https://github.com/Ruffian-L/niodoo-ai.git
  - Used for: YAML-driven config, dataset builders, QLoRA training/evaluation
  - See: `docs/TOPOLOGY_PIPELINE.md` for usage

**Initializing submodules:**
```bash
git submodule update --init --recursive
```

**Updating submodules:**
```bash
git submodule update --remote --recursive
```

## Critical System Facts

### Service Dependencies (CRITICAL TO UNDERSTAND!)

- **Embeddings**: `QwenStatefulEmbedder` uses **LOCAL ONNX models** (Rust/Candle). NO external service needed! The `embedding_model_name` config is just a model path.
- **Qdrant**: Uses **both HTTP (port 6333) and gRPC (port 6334)**. Code automatically converts HTTP URLs to gRPC when needed: `http://127.0.0.1:6333` → `grpc://127.0.0.1:6334`
- **Qwen 3 Coder**: **REQUIRED** vLLM service (port 5001). Main generation model for code generation.
- **Qwen 2.5 Topology**: **REQUIRED** vLLM service for curator (port 5001 same instance, or port 5002 if separate). Used for quality assessment and refinement.
- **Ollama**: **OPTIONAL** service (port 11434). Only needed if curator backend is explicitly set to `CuratorBackend::Ollama`. Default is vLLM, so Ollama is usually NOT needed!

**See HOW_TO_START.md for exact startup commands and verification steps.**

### Curator is PIVOTAL - NOT Optional!

**Current State**: Curator is marked as optional via `enable_curator` flag in config.

**Should Be**: Always enabled! It's critical to the system because:
- Called after generation in `integrate_curator()` (line 763 in `pipeline/core.rs`)
- Feeds learning loop via `apply_curator_learned()` (line 810)
- Used for failure detection (line 850-867) - if unavailable, SKIPS RETRIES!
- Used for consonance computation (line 777-797)
- Used for topology-aware refinement (line 838-1010)

**Impact if Disabled**: Failure detection skips retries, learning loop misses data, consonance incomplete.

### Two Curator Systems

1. **Integrated Curator** (`niodoo_real_integrated/src/curator.rs`): Used in pipeline for refinement
2. **curator_executor** (`curator_executor/`): Separate system with knowledge distillation, memory curation, executor (MORE features than integrated curator)

### RCE (Recursive Connectome Engine) - NEW!

**RCE** is a topology-aware cognitive control system that monitors breakthrough detection and guides system behavior:

- **Location**: `tcs-rce/` crate (new workspace crate)
- **Integration**: After curator integration in pipeline (`pipeline/stages.rs` line ~449)
- **Default State**: Enabled in shadow mode (metrics-only, no actions)
- **Purpose**: 
  - Computes β_meta composite metric (Betti derivatives, metastability, persistence entropy)
  - Consensus gate for retry approval
  - ERAG topology-aware reranking
  - Hyperfocus detection and circuit breaker
  - Curriculum scheduling for learning loop

**β_meta Formula** (`tcs-rce/src/beta_meta.rs`):
```
β_meta(t) = α₁·||dβ/dt|| + α₂·σ_R(t)·H_topo(t) + α₃·Σ w_m·|d/dt[n_m]| + α₄·ΔS_sheaf
```
Where:
- `||dβ/dt||`: Norm of Betti number derivatives across dimensions (H₀, H₁, H₂)
- `σ_R`: Metastability proxy (std-dev of recent entropy values)
- `H_topo`: Persistence entropy from topology
- `motif_flux`: Higher-order motif flux (Phase 3)
- `sheaf_divergence`: Sheaf geometry divergence (Phase 3)

**RCE Analyzer** (`niodoo_real_integrated/src/rce/analyzer.rs`):
- Maintains sliding window of entropy values for metastability computation
- Tracks Betti number changes over time for derivative computation
- Records prompt-to-spike latency for observability
- Exports Prometheus metrics: `niodoo_rce_beta_meta_current`, `niodoo_rce_beta_meta_peak`, `niodoo_rce_beta_meta_spikes_total`

**Config Flags** (in `config.rs`):
- `rce_enabled`: Default `true` (enabled)
- `rce_shadow_mode`: Default `true` (metrics-only, safe)
- `rce_actions_enabled`: Default `false` (no behavior changes)
- `rce_consensus.enabled`: Default `false` (no consensus gating)
- `rce_erag_lambda`: Default `0.0` (no ERAG reranking)
- `rce_beta_meta_weights`: Default all 1.0 (alpha_betti, alpha_meta, alpha_motif, alpha_sheaf)

**See**: `docs/RCE_Roadmap.md` for staged enablement guide

**Impact**: When actions enabled, RCE can gate retries, adjust generation temperature, and bias memory retrieval based on topology signals.

### nToken Implementation - NEW!

**nToken** is a topology feature extraction service that provides real-time topological analysis:

- **Client**: `niodoo_real_integrated/src/ntoken_client.rs` - HTTP client for nToken service
- **Service Endpoint**: Set via `NTOKEN_ENDPOINT` environment variable (e.g., `http://127.0.0.1:8000/ntoken`)
- **Bypass Flag**: `n_tokens_bypass` config flag (default `false`) or `N_TOKENS_BYPASS` env var

**Features Extracted**:
- `h1_count`: Number of H₁ persistence features
- `h1_total_persistence`: Total persistence of H₁ features
- `entropy_norm`: Normalized entropy measure
- `sheaf_energy`: Sheaf energy (0.0-1.0, low = consistent story)
- `sheaf_mean_section_norm`: Mean section norm of sheaf
- `raw_features`: Full feature map from service
- `topological_properties`: Hierarchical topological properties

**Integration Points**:

1. **Compass Integration** (`compass.rs` line ~139-158):
   - Fetched early with prompt-only (before context available)
   - Updates PAD state automatically:
     - High H₁ persistence (>2.0) → reduces pleasure/dominance (frustrated)
     - Low sheaf energy (<0.3) → increases pleasure/dominance (relieved)
     - High persistence → increases arousal (tension building)
   - Used in `evaluate_with_rng()` and `evaluate_with_ntoken()` methods

2. **Tokenizer Integration** (`pipeline/stages.rs` line ~207-232):
   - Refetched with full context (better than prompt-only)
   - Falls back to compass features if context fetch fails
   - Used for tokenizer refinement cues

**Pipeline Flow**:
```rust
// Early fetch (prompt only) for compass
let ntoken_features_for_compass = ntoken_client::fetch_features(&endpoint, prompt, None).await?;

// Compass evaluation with nToken features
compass.evaluate_with_rng(&pad_state, topology, &mut rng, ntoken_features_for_compass.as_ref())

// Later refetch (with context) for tokenizer
let ntoken_features = ntoken_client::fetch_features(&endpoint, prompt, Some(&context)).await?;
```

**Failure Handling**: Graceful degradation - if nToken service unavailable, pipeline continues without nToken features.

## Key File Locations

### Main Pipeline
- **Entry Point**: `niodoo_real_integrated/src/main.rs`
- **Pipeline Core**: `niodoo_real_integrated/src/pipeline/core.rs`
- **Pipeline Stages**: `niodoo_real_integrated/src/pipeline/stages.rs`
- **Config**: `niodoo_real_integrated/src/config.rs` (1675 lines - complex!)

### Core Components
- **Embedding**: `niodoo_real_integrated/src/embedding.rs` - LOCAL ONNX, no external service!
- **ERAG Memory**: `niodoo_real_integrated/src/erag.rs` - gRPC to Qdrant
- **Generation**: `niodoo_real_integrated/src/generation.rs` - vLLM API calls
- **Curator**: `niodoo_real_integrated/src/curator.rs` - Quality assessment (CRITICAL!)
- **Learning**: `niodoo_real_integrated/src/learning.rs` - QLoRA fine-tuning
- **Compass**: `niodoo_real_integrated/src/compass.rs` - Consciousness compass (nToken-integrated!)
- **RCE Analyzer**: `tcs-rce/` crate + `niodoo_real_integrated/src/rce/` - Recursive Connectome Engine (NEW!)
- **nToken Client**: `niodoo_real_integrated/src/ntoken_client.rs` - Topology feature extraction service client (NEW!)
- **Security**: `niodoo_real_integrated/src/security.rs` - Rate limiting, audit logging (enhanced)
- **Health**: `niodoo_real_integrated/src/health.rs` - Health check endpoints (requires `svc` feature)
- **Tracing**: `niodoo_real_integrated/src/tracing_integration.rs` - OpenTelemetry integration (requires `otel` feature)
- **Circuit Breaker**: `niodoo_real_integrated/src/circuit_breaker.rs` - Circuit breaker pattern
- **gRPC Inference**: `niodoo_real_integrated/src/grpc_inference/` - gRPC inference server (requires `svc` feature)

### Dead Code (For Reference Only)
- Backup files: `archive/*.full` (moved from `src/`)
- Alternative implementations: `pipeline_v2/`, `pipeline_legacy.rs` (commented out)
- See `archive/README.md` for details

### Validation & Testing Tools (NEW!)
- **Metrics Runner** (`src/bin/metrics_runner.rs`): Validation framework CLI tool
  - Load testing with concurrent users
  - Baseline capture (golden metrics)
  - Cognitive benchmarks
  - Quality SLI tracking
- **Ablation Runner** (`src/bin/ablation_runner.rs`): Systematic component testing
  - 12 ablation experiments (single and multi-component)
  - Statistical significance testing (p-values, Mann-Whitney U test)
  - Bootstrap confidence intervals (95% CI)
  - Component contribution scoring
  - Automated superiority proof generation
  - See `docs/ABLATION_TESTING.md` for complete guide
- **A/B Test Runner** (`src/bin/ab_test_runner.rs`): Configuration comparison framework
  - Compare baseline vs treatment configurations
  - Statistical comparison (t-tests, effect sizes)
  - Automated winner determination
  - Performance and quality metrics comparison
  - See `docs/AB_TESTING.md` for complete guide
- **Python A/B Test Framework** (`scripts/ab_test_comprehensive.py`):
  - Enhanced Python-based A/B testing
  - Statistical analysis with Cohen's d and p-values
  - Automated reporting
- **Superiority Proof Generator** (`scripts/run_superiority_proof.sh`):
  - Aggregates ablation and A/B test results
  - Generates comprehensive superiority reports
  - Identifies critical components
- **Baseline Infrastructure** (`baselines/` directory):
  - Timestamped baseline captures
  - `scripts/capture_baseline.sh`: Automated baseline capture
  - `scripts/compare_baseline.sh`: Statistical comparison tool
  - Bootstrap confidence intervals for percentile metrics

**Note**: Traditional test suites (unit tests, integration tests) have been removed. System superiority is proven through ablation tests and A/B tests that provide empirical evidence of component value.

### Runtime Config Files (NEW!)
- **RTX 5090**: `config/rtx5090.env` - 32GB GDDR7 optimizations
- **H200**: `config/h200.env` - H200 HBM3e optimizations  
- **TCS Runtime**: `tcs_runtime.env` - General TCS runtime defaults
- **Bootstrap Scripts**:
  - `scripts/start_h200_bootstrap.sh`: H200 environment bootstrap
  - `scripts/bootstrap_h200.sh`: Alternative H200 bootstrap
  - `scripts/runpod_bootstrap.sh`: RunPod environment setup

### Deployment & Infrastructure (NEW!)
- **Docker**: `niodoo_real_integrated/Dockerfile` - Container build
- **Docker Compose**: 
  - `docker-compose.yml` - Main services (mcp-app, qdrant, prometheus)
  - `docker-compose.monitoring.yml` - Monitoring stack (Prometheus, Grafana)
- **Kubernetes**: `deployment/k8s/deployment.yaml` - Full K8s manifests
  - Deployment with 3 replicas, HPA (3-10 replicas)
  - Service, ConfigMap, PersistentVolumeClaim
  - Liveness/readiness probes
- **Helm Charts**: `deployment/helm/niodoo/` - Production-ready Helm charts
- **Operations Guide**: `deployment/OPERATIONS_GUIDE.md` - Complete deployment guide

### Observability & Monitoring (NEW!)
- **Prometheus**: `prometheus.yml` - Full scrape config
  - vLLM metrics (port 5001)
  - Qdrant metrics (port 6333)
  - NVIDIA GPU metrics (port 9400)
  - Pipeline metrics (port 9093)
- **Prometheus Alerts**: `prometheus-alerts.yml` - Alert rules
  - HighErrorRate, HighLatency, CircuitBreakerOpen
  - LowCacheHitRate, HighMemoryUsage, ServiceDown
- **Grafana**: `grafana-provisioning/` - Dashboards and datasources
  - Pipeline latency, request rate, cache hit rate
  - Token promotion events, memory usage, circuit breaker status
- **OpenTelemetry**: `tracing_integration.rs` (requires `otel` feature)
  - Distributed tracing with OTLP exporter
  - OTLP endpoint: `OTEL_EXPORTER_OTLP_ENDPOINT` (default: `http://localhost:4317`)
  - Service name: `OTEL_SERVICE_NAME` (default: `niodoo-pipeline`)
  - Automatic trace context propagation
- **Health Checks**: `health.rs` (requires `svc` feature)
  - `/health` endpoint - Liveness probe (200 = healthy, 503 = unhealthy)
  - `/ready` endpoint - Readiness probe (200 = ready)
  - `/metrics` endpoint - Prometheus scraping
  - Component health registry (Healthy/Degraded/Unhealthy)

## Component Initialization Order

When `Pipeline::initialise()` runs, components are created in this order:

1. Config loading
2. Dataset & stats computation
3. Thresholds creation
4. Embedder (LOCAL - no external service)
5. Compass engine
6. ERAG client (connects to Qdrant via gRPC)
7. Tokenizer
8. Generator (connects to vLLM)
9. Security manager (enhanced with audit logging)
10. Learning loop
11. TCS analyzer (conditional - only if Hybrid mode)
12. Curator (conditional - but should always be enabled!)
13. RCE analyzer (conditional - enabled by default, shadow mode)
14. Caches
15. Weighted memory components (GPU fitness calculator)
16. MCTS daydreamer
17. Supporting systems (cascade tracker, hyperfocus detector)

## Runtime Flow Summary

1. Security validation (rate limiting, pattern detection, audit logging)
2. Embedding (LOCAL ONNX - fast!)
3. ERAG retrieval (gRPC → Qdrant)
4. Torus projection (7D PAD+Ghost space)
5. TCS analysis (if Hybrid mode)
6. nToken feature fetch (prompt-only, early for compass)
7. Compass processing (quadrant determination, topology-aware MCTS, nToken PAD updates)
8. ERAG retrieval (gRPC → Qdrant)
9. nToken feature refetch (with full context, for tokenizer refinement)
10. Token manager (dynamic tokenization, uses nToken cues)
11. Generation (vLLM API call - slowest step, topology-aware if enabled)
12. Curator integration (CRITICAL - quality assessment)
13. RCE analyzer (β_meta computation, consensus gate, shadow mode by default)
14. Consonance computation
15. Failure detection (skips retries if curator unavailable OR RCE consensus rejects!)
16. Retry logic (gated by RCE consensus if enabled)
17. Learning loop update (if breakthrough detected, RCE curriculum scheduling)
18. Memory storage (gRPC → Qdrant, topology-aware reranking if `rce_erag_lambda > 0`)
19. Response output

## Common Tasks & Where to Look

### Adding a New Pipeline Stage
- Look at `pipeline/stages.rs` for examples
- Add to `integrate_curator()` or create new method in `Pipeline`
- Update `StageTimings` in `pipeline/metrics.rs`

### Understanding Service Dependencies
- Check `DEPENDENCY_MAP.md` first
- Verify actual usage in `pipeline/core.rs` initialization
- Check config in `config.rs` for service URLs

### Finding Where Something is Used
- Search in `pipeline/core.rs` for initialization
- Check `pipeline/stages.rs` for processing logic
- Look at `COMPONENT_INVENTORY.md` for dependencies

### Debugging Service Issues
- Embeddings: Check `embedding.rs` - it's LOCAL, no network calls!
- Qdrant: Check `erag.rs` - uses gRPC (port 6334), not HTTP!
- vLLM: Check `generation.rs` and `curator.rs`
- Ollama: Only used if curator backend = Ollama (check config)
- RCE: Check `tcs-rce/` crate and `pipeline/stages.rs` for RCE analyzer integration
- GPU: Check `config/h200.env` for H200 settings, or verify CUDA availability

### Working with Submodules
- **Niodoo-TCT**: Feature extraction CLI (`scripts/extract_features.py`), topology metrics
- **niodoo-ai**: Training scripts (`scripts/train_topology.py`, `scripts/prepare_data.py`)
- See `docs/TOPOLOGY_PIPELINE.md` for end-to-end workflow

### H200 GPU Setup
- Run `scripts/start_h200_bootstrap.sh` to bootstrap H200 environment
- Source `config/h200.env` for H200-specific optimizations
- See `docs/H200_PRIMING_GUIDE.md` for complete walkthrough
- vLLM config: FP8 KV cache, FlashInfer attention, DeepGEMM kernels

### RTX 5090 GPU Setup
- Source `config/rtx5090.env` for RTX 5090 optimizations
- **32GB GDDR7** allows aggressive memory utilization (0.95 GPU util)
- **128k context window** with FP8 KV cache
- **16k batched tokens** for high throughput
- **128 concurrent sequences** for maximum parallelism
- ERAG batch size: 512 (large batches for 32GB VRAM)
- Cache prefetch parallelism: 16 (high parallelism)

### nToken Service Setup
- Set `NTOKEN_ENDPOINT` environment variable (e.g., `http://127.0.0.1:8000/ntoken`)
- Service should accept POST requests with `SentenceRequest` JSON
- Returns `SentenceResponse` with topological features
- If service unavailable, pipeline continues gracefully (degraded mode)
- To bypass: Set `N_TOKENS_BYPASS=1` or `n_tokens_bypass=true` in config

### Working with Validation Tools
- **Metrics Runner**: Run load tests and capture baselines
  ```bash
  cargo run --bin metrics_runner -- --scenario load_test --concurrent-users 16 --duration-secs 60
  cargo run --bin metrics_runner -- --scenario baseline --output baselines/baseline-latest.json
  ```
- **Ablation Runner**: Test component contributions (proves system superiority)
  ```bash
  # Run single ablation experiment
  cargo run --bin ablation_runner -- --experiment DisableRce --baseline baselines/baseline-latest.json
  
  # Run all experiments
  for exp in DisableRce BypassNTokens DisableTcsGpu DisableGpuFitness DisableCurator BypassErag DisableCompass DisableLearning DisableTcs DisableTokenizer; do
    cargo run --bin ablation_runner -- --experiment $exp --baseline baselines/baseline-latest.json
  done
  ```
- **A/B Test Runner**: Compare configurations
  ```bash
  cargo run --bin ab_test_runner -- \
    --baseline-name baseline \
    --treatment-name treatment \
    --baseline-config configs/baseline.json \
    --treatment-config configs/treatment.json
  ```
- **Python A/B Test**: `python3 scripts/ab_test_comprehensive.py`
- **Superiority Proof**: Generate comprehensive reports
  ```bash
  ./scripts/run_superiority_proof.sh
  ```
- **Baseline Capture**: `./scripts/capture_baseline.sh`
- **Baseline Comparison**: `./scripts/compare_baseline.sh metrics_report.json [baseline.json]`
- **Soak Test**: `cargo run --bin soak_test` - Extended load testing

### Deployment & Operations
- **Docker Build**: `docker build -t niodoo-pipeline -f niodoo_real_integrated/Dockerfile .`
- **Docker Compose**: `docker-compose up -d` (main services) or `docker-compose -f docker-compose.monitoring.yml up -d` (monitoring)
- **Kubernetes**: `kubectl apply -f deployment/k8s/deployment.yaml`
- **Helm**: `helm install niodoo deployment/helm/niodoo/`
- **Health Checks**: `curl http://localhost:8080/health` (liveness), `curl http://localhost:8080/ready` (readiness)
- **Metrics**: `curl http://localhost:8080/metrics` (Prometheus scraping)
- **OpenTelemetry**: Set `OTEL_EXPORTER_OTLP_ENDPOINT` and `OTEL_SERVICE_NAME` env vars, enable `otel` feature
- **Service Feature**: Build with `--features svc` to enable HTTP/gRPC endpoints

## Critical Code Sections

### Curator Integration (MUST UNDERSTAND!)
```rust
// File: pipeline/core.rs, line ~763
let curated_experience = self.integrate_curator(...).await?;

// File: pipeline/core.rs, line ~850
let curator_unavailable = self.curator.is_none() || ...

// If curator unavailable: SKIPS RETRIES!
if (curator_unavailable || curator_passive) && failure != "none" {
    failure = "none".to_string();
    // Retries skipped!
}
```

### Service Initialization
```rust
// File: pipeline/core.rs, line ~198
let erag = EragClient::new(
    &config.qdrant_url,  // HTTP URL
    &config.qdrant_collection,
    config.qdrant_vector_dim,
    config.similarity_threshold,
).await?;
// Internally converts HTTP → gRPC (port 6334)
```

### Embedding (LOCAL!)
```rust
// File: pipeline/core.rs, line ~141
let embedder = QwenStatefulEmbedder::new(
    &config.embedding_model_name,  // Model path, not Ollama model!
    config.qdrant_vector_dim,
)?;
// Uses LOCAL ONNX model (tcs_ml::QwenEmbedder)
```

### RCE Integration (NEW!)
```rust
// File: pipeline/stages.rs, line ~449
if let Some(analyzer) = self.rce_analyzer.as_mut() {
    let beta = analyzer.update_with_prompt_timestamp(&pad_state, &topology, Some(overall_start));
    // Consensus gate (read-only): combine diverse simple votes
    let mut approved = true;
    if self.config.rce_consensus.enabled {
        let gate = crate::rce::safety::ensemble::ConsensusGate::new(self.config.rce_consensus.clone());
        let vote_beta = beta >= self.config.rce_breakthrough_threshold;
        let vote_meta = analyzer.current_metastability() * topology.persistence_entropy > 0.0;
        let vote_spec = topology.spectral_gap > 0.0;
        approved = gate.approve(&[vote_beta, vote_meta, vote_spec]);
    }
    rce_retry_approved = approved;
    // Actions (if enabled): temperature adjustment, circuit breaker
}
```

### nToken Integration (NEW!)
```rust
// File: pipeline/stages.rs, line ~110-133
// Early fetch (prompt only) for compass
let ntoken_features_for_compass = if !self.config.n_tokens_bypass {
    ntoken_client::fetch_features(&endpoint, prompt, None).await?
} else { None };

// Compass evaluation with nToken features
compass.evaluate_with_rng(
    &pad_state_for_compass,
    Some(&topology_for_compass),
    &mut rng,
    ntoken_features_for_compass.as_ref(), // PAD state updates here!
);

// File: compass.rs, line ~139-158
// nToken PAD state adjustments
if let Some(ntoken) = ntoken_features {
    let h1_persistence_norm = (ntoken.h1_total_persistence / 2.5).tanh();
    let h1_penalty = h1_persistence_norm * 0.3;
    let sheaf_boost = if ntoken.sheaf_energy < 0.3 {
        (0.3 - ntoken.sheaf_energy) * 0.5
    } else { 0.0 };
    // High H₁ → reduces PAD (frustrated)
    // Low sheaf energy → increases PAD (relieved)
    pleasure = (pleasure - h1_penalty + sheaf_boost).clamp(-1.0, 1.0);
}
```

### RCE β_meta Computation (NEW!)
```rust
// File: tcs-rce/src/beta_meta.rs
pub fn compute_beta_meta(weights: BetaMetaWeights, inputs: BetaMetaInputs) -> f64 {
    let term_betti = weights.alpha_betti * inputs.d_betti_norm;
    let term_meta = weights.alpha_meta * (inputs.metastability_sigma_r * inputs.persistence_entropy);
    let term_motif = weights.alpha_motif * inputs.motif_flux;
    let term_sheaf = weights.alpha_sheaf * inputs.sheaf_divergence;
    term_betti + term_meta + term_motif + term_sheaf
}

// File: niodoo_real_integrated/src/rce/analyzer.rs
// Betti derivative computation
let d_betti_norm = (diff0 + diff1 + diff2) / dt_secs.max(1e-6);
// Metastability: entropy std-dev
let sigma_r = self.compute_entropy_std();
// Persistence entropy from topology
let h_topo = topo.persistence_entropy;
```

## Configuration Flags to Know

### Core Flags
- `enable_curator`: Currently optional, but SHOULD always be true!
- `topology_mode`: `Hybrid` (with TCS) or `Baseline` (analytical only)
- `curator_backend`: `Vllm` (default) or `Ollama` (optional)
- `mock_mode`: Enables stubbed responses for testing
- `qdrant_embedded`: Spawns Qdrant as child process (optional)

### RCE Flags (NEW!)
- `rce_enabled`: Default `true` - Enable RCE analyzer
- `rce_shadow_mode`: Default `true` - Safe mode (metrics-only, no actions)
- `rce_actions_enabled`: Default `false` - Enable behavior changes (retry gating, temperature adjustment)
- `rce_consensus.enabled`: Default `false` - Enable consensus gate for retry approval
- `rce_erag_lambda`: Default `0.0` - Topology-aware ERAG reranking weight (0.0 = disabled)
- `rce_breakthrough_threshold`: Default `0.5` - β_meta spike threshold

### GPU/Hardware Flags (NEW!)
- `USE_GPU_FITNESS=1`: Enable GPU-backed episodic fitness scoring
- `TCS_ENABLE_GPU=1`: Enable GPU acceleration for TCS analysis
- `OPTIMIZED_ERAG=1`: Enable batched Qdrant upserts for large VRAM
- `ERAG_BATCH_SIZE`: Batch size for ERAG operations (default varies by hardware)
- `CACHE_PREFETCH_ENABLED=1`: Enable cache prefetching

**RTX 5090 Configuration** (`config/rtx5090.env`):
- **VRAM**: 32GB GDDR7 (Blackwell architecture)
- **vLLM Settings**:
  - `VLLM_GPU_MEMORY_UTILIZATION=0.95` (32GB allows aggressive usage)
  - `VLLM_MAX_MODEL_LEN=128000` (128k context window)
  - `VLLM_MAX_NUM_BATCHED_TOKENS=16384` (16k batched tokens)
  - `VLLM_MAX_NUM_SEQS=128` (high concurrency)
  - `VLLM_KV_CACHE_DTYPE=fp8` (FP8 KV cache for memory efficiency)
  - `VLLM_USE_DEEP_GEMM=1` (DeepGEMM kernels)
  - `VLLM_ATTENTION_BACKEND=FLASH_ATTN` (FlashAttention)
- **ERAG Settings**:
  - `ERAG_BATCH_SIZE=512` (large batches for 32GB VRAM)
  - `CACHE_PREFETCH_PARALLELISM=16` (high parallelism)
- **Generation**:
  - `GENERATION_MAX_TOKENS=8192` (deep sampling)
  - `DYNAMIC_TOKEN_MAX=2048` (large token vocab)
  - `TOKEN_PROMOTION_INTERVAL=20` (frequent promotion)

**H200 Configuration** (`config/h200.env`):
- See `docs/H200_PRIMING_GUIDE.md` for H200 setup
- Similar to RTX 5090 but optimized for H200 HBM3e memory

### Security Flags (NEW!)
- Enhanced audit logging in `security.rs`
- Rate limiting and pattern detection

### Validation Flags (NEW!)
- **Quality SLIs**:
  - `niodoo_quality_sli_tcs_stability_cv`: TCS stability (coefficient of variation of persistence_entropy)
  - `niodoo_quality_sli_rce_beta_meta_compliance`: RCE β_meta range compliance ([0.8, 1.2])
- **Baseline Comparison**: Bootstrap confidence intervals, Cohen's d effect size
- **Ablation Experiments**: Environment variables for component disabling
  - `RCE_ENABLED=0`: Disable RCE
  - `N_TOKENS_BYPASS=1`: Bypass nToken layer
  - `TCS_ENABLE_GPU=0`: Disable GPU acceleration for TCS
  - `USE_GPU_FITNESS=0`: Disable GPU fitness calculation
  - `ENABLE_CURATOR=0`: Disable Curator
  - `ERAG_BYPASS=1`: Bypass ERAG (zero-shot mode)

### Feature Flags (NEW!)
- **`svc` feature**: Enables HTTP/gRPC service endpoints
  - Axum HTTP server for health checks
  - Tower middleware support
  - gRPC inference server (tonic)
  - Required for `/health`, `/ready`, `/metrics` endpoints
- **`otel` feature**: Enables OpenTelemetry distributed tracing
  - OTLP exporter support
  - Distributed trace context propagation
  - Requires `opentelemetry`, `opentelemetry-otlp`, `tracing-opentelemetry` dependencies
- **`gpu` feature**: Enables GPU acceleration
  - CUDA support via Candle
  - GPU fitness calculator
  - TCS GPU acceleration
- **`embedded-qdrant` feature**: Enables embedded Qdrant spawning
- **`knot` feature**: Enables knot theory computations (requires `tcs-knot` crate)

## When Making Changes

### Before Making Changes
1. Read relevant section in `SYSTEM_ARCHITECTURE.md`
2. Check `COMPONENT_INVENTORY.md` for module status
3. Verify dependencies in `DEPENDENCY_MAP.md`
4. Understand flow in `RUNTIME_FLOW.md`

### After Making Changes
1. Update `CHANGELOG.md` with your changes
2. Update relevant documentation files if architecture changed
3. Note any new dependencies or service requirements

## Common Mistakes to Avoid

1. **Assuming Ollama is needed for embeddings** - It's LOCAL ONNX!
2. **Using HTTP for Qdrant** - It uses gRPC (automatic conversion)
3. **Making curator optional** - It's pivotal, should always be enabled
4. **Ignoring curator_unavailable checks** - This causes retries to be skipped
5. **Assuming vLLM is only for generation** - Curator also uses it (by default)
6. **Forgetting to initialize submodules** - Run `git submodule update --init --recursive` after clone
7. **Enabling RCE actions without understanding** - Default shadow mode is safe; enable actions only after testing
8. **Ignoring RCE consensus gate** - Can block retries if consensus rejects (check logs)
9. **Mixing H200 config on non-H200 hardware** - Use hardware-specific configs (`config/h200.env` only for H200, `config/rtx5090.env` only for RTX 5090)
10. **Assuming nToken service is required** - nToken is optional, pipeline degrades gracefully if unavailable
11. **Not setting NTOKEN_ENDPOINT** - nToken features won't be fetched (check logs for warnings)
12. **Ignoring nToken PAD updates** - Compass automatically adjusts PAD state based on nToken features (high H₁ → frustrated, low sheaf → relieved)
13. **Confusing nToken with TCS** - nToken is external HTTP service, TCS is internal topology analysis

## Getting Help

- **How to start everything**: See `HOW_TO_START.md` - **START HERE!**
- **Architecture questions**: See `SYSTEM_ARCHITECTURE.md`
- **Component status**: See `COMPONENT_INVENTORY.md`
- **Dependencies**: See `DEPENDENCY_MAP.md`
- **Runtime behavior**: See `RUNTIME_FLOW.md`
- **Code examples**: See `pipeline/core.rs` and `pipeline/stages.rs`
- **RCE details**: See `docs/RCE_Roadmap.md` for staged enablement
- **H200 setup**: See `docs/H200_PRIMING_GUIDE.md` for GPU optimization
- **RTX 5090 setup**: See `config/rtx5090.env` for RTX 5090 optimizations
- **nToken implementation**: See `niodoo_real_integrated/src/ntoken_client.rs` and `compass.rs` for integration details
- **Topology training**: See `docs/TOPOLOGY_PIPELINE.md` for Niodoo-TCT + niodoo-ai workflow
- **Validation**: See `docs/validation/PROMETHEUS_METRICS.md` for observability
- **Ablation Testing**: See `src/bin/ablation_runner.rs` for component testing
- **Baseline Infrastructure**: See `baselines/README.md` for baseline capture and comparison
- **Deployment**: See `deployment/OPERATIONS_GUIDE.md` for production deployment
- **Monitoring**: See `prometheus.yml` and `docker-compose.monitoring.yml` for observability setup
- **OpenTelemetry**: See `niodoo_real_integrated/src/tracing_integration.rs` for distributed tracing
- **Health Checks**: See `niodoo_real_integrated/src/health.rs` for health endpoint implementation

## Quick Reference

| Component | Service | Port | Protocol | Required? |
|-----------|---------|------|----------|-----------|
| Embeddings | Local ONNX | N/A | N/A | Always |
| ERAG/Qdrant | Qdrant | 6333/6334 | HTTP/gRPC | Yes |
| Qwen 3 Coder | vLLM | 5001 | HTTP | Yes |
| Qwen 2.5 Topology Curator | vLLM | 5001 or 5002 | HTTP | Yes |
| Curator (Ollama) | Ollama | 11434 | HTTP | Optional |
| RCE Analyzer | Local (tcs-rce) | N/A | N/A | Enabled by default (shadow) |
| nToken Service | HTTP Service | Variable | HTTP | Optional (graceful degradation) |
| Main Pipeline Health | HTTP Server | 9090 | HTTP | Optional (requires `svc` feature) |
| Prometheus | Prometheus | 9090 | HTTP | Optional (for metrics) |
| Grafana | Grafana | 3000 | HTTP | Optional (for dashboards) |
| OpenTelemetry | OTLP Collector | 4317 | gRPC | Optional (requires `otel` feature) |
| gRPC Inference | gRPC Server | Variable | gRPC | Optional (requires `svc` feature) |

**See HOW_TO_START.md for complete startup instructions.**

## Testing & Validation Tools

| Tool | Purpose | Location |
|------|---------|----------|
| Metrics Runner | Load testing, baseline capture, cognitive benchmarks | `src/bin/metrics_runner.rs` |
| Ablation Runner | Systematic component testing (6 experiments) | `src/bin/ablation_runner.rs` |
| Baseline Capture | Automated baseline capture script | `scripts/capture_baseline.sh` |
| Baseline Compare | Statistical comparison tool | `scripts/compare_baseline.sh` |
| Validation Tests | End-to-end validation suite | `scripts/run_all_validation_tests.sh` |

## Recent Additions (2025)

### Git Submodules
- **Niodoo-TCT**: Topology toolkit (`Niodoo-TCT/`)
- **niodoo-ai**: Training infrastructure (`niodoo-ai/`)

### RCE (Recursive Connectome Engine)
- New `tcs-rce` crate for topology-aware cognitive control
- β_meta composite metric computation
- Consensus gate for retry approval
- ERAG topology-aware reranking
- Default: Shadow mode (safe, metrics-only)

### H200 GPU Support
- H200-specific optimizations (`config/h200.env`)
- FP8 KV cache, FlashInfer attention, DeepGEMM kernels
- GPU fitness calculator for episodic memory
- See `docs/H200_PRIMING_GUIDE.md`

### Validation Framework
- Prometheus metrics integration
- Quality SLIs (TCS stability, RCE β_meta compliance)
- Metrics runner (`src/bin/metrics_runner.rs`)
- See `docs/validation/PROMETHEUS_METRICS.md`

### Topology-Aware Improvements
- Topology-aware MCTS branch generation in compass
- **nToken Implementation**: Full HTTP client service integration
  - Early fetch (prompt-only) for compass PAD state updates
  - Context-aware refetch for tokenizer refinement
  - Automatic PAD state adjustments based on H₁ persistence and sheaf energy
  - Graceful degradation if service unavailable
- Topology-aware generation prompts (internal only)
- Mistral topology finetuning infrastructure

### RTX 5090 Support
- RTX 5090 specific configuration (`config/rtx5090.env`)
- 32GB GDDR7 optimizations (0.95 GPU utilization)
- 128k context window with FP8 KV cache
- 16k batched tokens, 128 concurrent sequences
- ERAG batch size 512, cache prefetch parallelism 16

### Security Enhancements
- Enhanced audit logging
- Rate limiting improvements
- Pattern detection

### Validation & Testing Infrastructure
- **Metrics Runner**: Comprehensive validation framework
  - Load testing with concurrent users
  - Baseline capture (golden metrics JSON)
  - Cognitive benchmark execution
  - Quality SLI tracking (TCS stability CV, RCE β_meta compliance)
- **Ablation Runner**: Systematic component testing
  - 6 predefined ablation experiments
  - Statistical comparison with baseline
  - Cohen's d effect size calculation
  - Regression detection
- **Baseline Infrastructure**: 
  - Timestamped baseline captures
  - Automated capture scripts
  - Statistical comparison tools
  - Bootstrap confidence intervals
- **Soak Test**: Extended load testing (`soak_test` binary)
  - Memory leak detection
  - Concurrent load testing
  - Stability validation

### Deployment & Infrastructure
- **Docker & Docker Compose**: Full containerization
  - Main services: mcp-app, qdrant, prometheus
  - Monitoring stack: Prometheus, Grafana
- **Kubernetes**: Production-ready manifests
  - Deployment with 3 replicas, HPA (3-10 replicas)
  - Service, ConfigMap, PersistentVolumeClaim
  - Liveness/readiness probes
- **Helm Charts**: Production-ready Helm charts
- **Operations Guide**: Complete deployment documentation

### Observability & Monitoring
- **Prometheus**: Full scrape configuration
  - vLLM metrics (port 5001)
  - Qdrant metrics (port 6333)
  - NVIDIA GPU metrics (port 9400)
  - Pipeline metrics (port 9093)
- **Prometheus Alerts**: Comprehensive alert rules
  - HighErrorRate, HighLatency, CircuitBreakerOpen
  - LowCacheHitRate, HighMemoryUsage, ServiceDown
- **Grafana**: Dashboards and provisioning
  - Pipeline latency, request rate, cache hit rate
  - Token promotion events, memory usage, circuit breaker status
- **OpenTelemetry**: Distributed tracing (requires `otel` feature)
  - OTLP exporter support
  - Automatic trace context propagation
- **Health Checks**: HTTP endpoints (requires `svc` feature)
  - `/health` (liveness), `/ready` (readiness), `/metrics` (Prometheus)

## Remember

- **START HERE = HOW_TO_START.md** (complete end-to-end startup guide)
- **Embeddings = LOCAL ONNX** (no external service needed)
- **Qdrant = HTTP + gRPC** (ports 6333 HTTP, 6334 gRPC)
- **Qwen 3 Coder = vLLM PORT 5001** (main generation model)
- **Qwen 2.5 Topology = vLLM PORT 5001/5002** (curator model)
- **Curator = CRITICAL** (not optional!)
- **Ollama = OPTIONAL** (only if curator backend = Ollama)
- **RCE = SHADOW MODE BY DEFAULT** (safe, metrics-only)
- **Submodules = MUST INITIALIZE** (`git submodule update --init --recursive`)
- **H200 = SPECIAL CONFIG** (use `config/h200.env`, see `docs/H200_PRIMING_GUIDE.md`)
- **RTX 5090 = SPECIAL CONFIG** (use `config/rtx5090.env` for 32GB GDDR7 optimizations)
- **nToken = OPTIONAL SERVICE** (set `NTOKEN_ENDPOINT` env var, graceful degradation if unavailable)
- **nToken PAD Updates = AUTOMATIC** (compass adjusts PAD state based on H₁ persistence and sheaf energy)
- **Validation = BUILT-IN** (metrics_runner and ablation_runner binaries for testing)
- **Baselines = TIMESTAMPED** (capture with `scripts/capture_baseline.sh`, compare with `scripts/compare_baseline.sh`)
- **Deployment = DOCKER/K8S** (Dockerfile, docker-compose, Kubernetes manifests, Helm charts)
- **Monitoring = PROMETHEUS+GRAFANA** (Full observability stack with alerts)
- **Tracing = OPENTELEMETRY** (Distributed tracing with `otel` feature, OTLP exporter)
- **Health = HTTP ENDPOINTS PORT 9090** (Requires `svc` feature: `/health`, `/ready`, `/metrics`)

