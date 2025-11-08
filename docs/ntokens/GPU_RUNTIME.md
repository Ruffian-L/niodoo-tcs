# nToken GPU Runtime & Deployment Strategy

This document details the GPU execution model for nToken synthesis on NVIDIA H200 NVL systems, covering kernel architecture, memory planning, and deployment updates.

## 1. Hardware Baseline

- **Configuration**: 4× NVIDIA H200 NVL (141 GB HBM3e each) connected via NVLink (900 GB/s per link, 3.6 TB/s aggregate bandwidth).
- **Precision**: FP8 Tensor Cores (E4M3/E5M2) for bulk matrix ops; FP16/BF16 for numerically sensitive steps; FP32 fallback for topology reductions.
- **Software stack**: CUDA 13.0, cuDNN 9.x, TensorRT 10.x, CubeCL alpha, Multipers 1.3 w/ KeOps acceleration.

## 2. Kernel Responsibilities

| Kernel | Module | Precision | Notes |
|--------|--------|-----------|-------|
| `complex_build` | `topology::complex` | FP32/FP16 | Constructs boundary matrices from adjacency; relies on coalesced 128-byte loads. |
| `filtration_assign` | `topology::filtration` | FP16 | Evaluates multiparameter filter values per simplex; uses shared-memory tiling for interpolation coefficients. |
| `persistence_reduce` | `topology::persistence` | Mixed (FP16 compute, FP32 accum) | Performs boundary matrix reduction; 99.9% GPU (apparent pairs), 0.1% CPU fallback for pivot resolution. |
| `laplacian_apply` | `sheaf::laplacian` | FP8 (Tensor Core) | Applies sheaf Laplacian during diffusion; uses CubeCL autotuning for block sizes. |
| `gyro_update` | `memory::hyperbolic` | FP16 | Updates hyperbolic embeddings using gyrovector math; leverages Tensor Core fused multiply-add. |
| `constraint_eval` | `value::constraints` | FP16 | Computes geodesic distances and penalty gradients; batched per constraint set. |
| `zigzag_update` | `temporal::zigzag` | FP16 | Maintains sliding window persistence with incremental updates. |

## 3. Memory Budgeting

| Allocation | Size (per GPU) | Description |
|------------|----------------|-------------|
| Boundary matrices | 35 GB | CSR/COO matrices up to 100k simplices; stored in unified memory with preferred location on device. |
| Filtration grids | 8 GB | Multiparameter grid (≤ 64³) plus interpolation coefficients. |
| Persistence buffers | 30 GB | Workspaces for reduction, including pivot stacks. |
| Sheaf tensors | 28 GB | Stalk features, restriction map params, Laplacian caches. |
| Hyperbolic embeddings | 10 GB | Batched gyrovector states for memories in scope. |
| Temporal history | 10 GB | Zigzag snapshots and cobordism metadata. |
| Overhead | ~20 GB | CUDA driver, fragmentation buffer, cubeCL caches, fragmentation headroom. |

Total per GPU ≈ 141 GB (max); orchestrator enforces headroom by adjusting batch size when allocation exceeds 120 GB.

## 4. Execution Pipeline

1. **Batch Scheduling**: Sentence batches grouped by similar complexity; orchestrator ensures total simplex count per batch fits memory budget.
2. **Stream Management**: Use CUDA streams per stage (`complex`, `persistence`, `sheaf`, `memory`). Events synchronize dependencies without blocking.
3. **NVLink Coordination**: For cross-GPU workloads, broadcast shared embeddings and restrict per-GPU reduction results. Use NCCL for reductions when merging persistence stats.
4. **CPU Assist**: Dedicated thread pool handles residual boundary reductions and lambeq parsing (Python FFI). GPU kernels triggered via CubeCL runtime.

## 5. Deployment Updates

### 5.1 `install_runpod_deps.sh`

- Add detection for CubeCL runtime prerequisites (Rust nightly component, SPIR-V tools).
- Ensure Multipers build selects CUDA backend with KeOps; install `pykeops` wheels.
- Export environment variables:
  - `NTOKEN_CUDA_STREAMS=4`
  - `NTOKEN_FP8_ENABLED=1`
  - `NTOKEN_MAX_BATCH=64`
- Validate presence of `libcublasLt.so` (required for FP8 Tensor Core ops).

### 5.2 `tcs_runtime.env`

- Introduce ntoken tuning keys: `NTOKEN_FILTRATION_RES=64`, `NTOKEN_MAX_SIMPLEX_K=3`, `NTOKEN_ZIGZAG_WINDOW=32`.
- Expose `NTOKEN_MIXED_PRECISION=fp8` with fallback to `bf16` when stability issues detected.

### 5.3 Build Flags

- Add workspace feature `ntokens-gpu` enabling CubeCL kernels and linking Multipers CUDA.
- Ensure CI builds with `cargo build --features ntokens,ntokens-gpu,cuda`.

## 6. Performance Targets

- **10k simplex batch**: < 1 s end-to-end (including CPU parsing).
- **100k simplex batch**: < 10 s.
- **Throughput**: 100–1000 sentences/minute across 4 GPUs.
- Monitor via Prometheus metrics `ntoken_gpu_latency`, `ntoken_vram_usage`, `ntoken_fp8_fallback_total`.

## 7. Failure & Recovery

- Kernel watchdog: if `persistence_reduce` exceeds 5 s, abort batch, switch to CPU fallback, and raise alert.
- Memory pressure: when VRAM > 90%, orchestrator halves batch size and retries once before escalating.
- Log GPU context resets and trigger curator downgrade (skip nToken features) if three consecutive failures occur.

## 8. Validation Steps

- Execute synthetic benchmarks post-deployment (scripts TBD) verifying latency and memory budgets.
- Validate gradient flow through Multipers by running autograd tests comparing analytical vs numerical gradients.
- Confirm environment variables load via `start_all_services.sh --hardware h200 --enable-ntokens` dry run.

This strategy ensures nToken synthesis fully exploits H200 capabilities while maintaining operational resilience.



