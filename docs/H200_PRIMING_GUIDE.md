# NIODOO H200 Priming Guide

This walkthrough captures everything you need to wring maximum value out of an NVIDIA H200 that you have on loan for a day.

## 1. Bootstrap the Environment

```bash
cd /workspace/Niodoo-Final
./scripts/start_h200_bootstrap.sh
```

- Validates CUDA availability (`nvidia-smi`) and warns if the driver is < R580
- Installs or wires CUDA 13.0 (Aug 2025) into `PATH`/`LD_LIBRARY_PATH` for FP8/FP4 kernels
- Locates CUDA-enabled ONNX Runtime libs (v1.24.0) and wires them into `LD_LIBRARY_PATH`
- Writes aggressive runtime overrides to `config/h200.env`
- Builds the workspace with GPU features (`cargo build --release --features gpu`)

> **Note:** `config/h200.env` is idempotent – re-run the script any time you refresh the machine.

## 2. Source the Runtime Overrides

```bash
source config/h200.env
```

The key overrides do the following:

- `USE_GPU_FITNESS=1`: forces GPU-backed episodic fitness scoring
- `OPTIMIZED_ERAG=1`: enables batched Qdrant upserts tuned for large VRAM
- `ERAG_BATCH_SIZE=256` / `CACHE_PREFETCH_*`: amps up memory prefetch parallelism
- `GENERATION_MAX_TOKENS=4096`, `DYNAMIC_TOKEN_MAX=1024`: unlock deep sampling while the H200 eats the KV cache overhead
- `VLLM_ATTENTION_BACKEND=flashinfer`, `VLLM_KV_CACHE_DTYPE=fp8`, `VLLM_USE_DEEP_GEMM=1`: turn on Hopper-specific attention + DeepGEMM kernels
- `VLLM_GPU_MEMORY_UTILIZATION=0.92`, `VLLM_MAX_MODEL_LEN=128000`, `VLLM_MAX_NUM_BATCHED_TOKENS=8192`: pre-tune vLLM for 141 GB HBM3e
- `TOKEN_PROMOTION_INTERVAL=30`: faster promotion cadence keeps the tokenizer sharp during heavy throughput

## 3. Start External Services with the H200 Profile

All service scripts respect the exported variables:

```bash
./start_all_services.sh --hardware h200
```

### Manual vLLM launch (if you want to run it yourself)

```bash
source venv/bin/activate
export VLLM_HOST=127.0.0.1
export VLLM_PORT=5001
export VLLM_ALL2ALL_BACKEND=pplx
export VLLM_USE_DEEP_GEMM=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_KV_CACHE_DTYPE=fp8
vllm serve /workspace/models/hf_cache/models--Qwen--Qwen2.5-7B-Instruct-AWQ \
  --host "$VLLM_HOST" --port "$VLLM_PORT" \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.92 \
  --attention-backend "$VLLM_ATTENTION_BACKEND" \
  --kv-cache-dtype "$VLLM_KV_CACHE_DTYPE" \
  --tensor-parallel-size 1 \
  --enable-chunked-prefill \
  --use-deep-gemm \
  --trust-remote-code
```

If you are running services manually:

- vLLM: `VLLM_GPU_MEMORY_UTILIZATION=0.85` keeps enough headroom for engine startup
- Qdrant: keep it on gRPC (`http://127.0.0.1:6333` automatically maps to `grpc://127.0.0.1:6334`)
- Ollama: optional unless curator backend is switched to Ollama

> **MIG tip:** If you want to partition the H200, configure MIG before launching services, e.g. `sudo nvidia-smi mig -cgi 0,1,2,3,4,5,6 -C` for seven 20 GB slices.

## 4. Run the Pipeline

### Interactive prompt

```bash
cargo run --release --features gpu -- \
  --hardware h200 \
  --prompt "sketch the multi-stage recovery plan"
```

### H200 soak smoke test (1k cycles)

```bash
./scripts/smoke_test_h200.sh
```

### Million-cycle battering ram

```bash
TEST_COUNT=250000 WORKERS=256 ./scripts/run_million_cycle_h200.sh
```

## 5. GPU Fitness Verification

After a run, check that GPU fitness metrics are ticking:

```bash
curl -s localhost:9898/metrics | grep gpu_fitness
```

You should see `gpu_fitness_calculations_total` increasing and `gpu_fitness_gpu_available` at `1`.

## 6. Tear-down Checklist

- Stop GPU monitoring spawned by the scripts (`pkill -f nvidia-smi` if needed)
- Archive `logs/million_cycle_*` directories – they contain utilization summaries
- Revert to the default profile with `unset $(cut -d= -f1 config/h200.env)` if you return to the Beelink edge boxes

Enjoy the throughput while the H200 is on the bench.

