# vLLM Startup Guide (November 2025)

This guide captures the current (November 2025) playbooks for bringing up vLLM with the models that already live in `/workspace/models`. No Hugging Face credentials or downloads are required — every flow below points vLLM at the local safetensor trees.

---

## Shared Conventions

- **Model root:** `export VLLM_MODEL_PATH=/workspace/models/Qwen2.5-7B-Instruct-AWQ`
  - Adjust the path if you want to serve a different directory under `/workspace/models`.
  - Ensure the directory contains the standard Hugging Face layout (`config.json`, `model-*.safetensors`, tokenizer files). vLLM happily consumes the tree without calling the Hub.
- **Service port:** The NIODOO stack expects vLLM on `127.0.0.1:5001`. Override with `VLLM_HOST`/`VLLM_PORT` if you must.
- **Readiness probe:** `curl -s http://$VLLM_HOST:$VLLM_PORT/v1/models | jq '.'` is the canonical health check before wiring NIODOO back in.

---

## NVIDIA CUDA (CUDA 12.8+, Hopper/Lovelace/Ampere)

### Install / Upgrade

```bash
cd /workspace/Niodoo-Final
uv venv --python 3.12 --seed
source .venv/bin/activate
pip install --upgrade pip
pip install 'vllm[flashinfer]'  # ships the CUDA 12.8 wheels with FlashAttention + DeepGEMM

# verify toolchain
nvcc --version
```

> **Tip:** If the host already has `venv/` with the Hopper stack, you can reuse it. The wheel targets CUDA 12.8; make sure `/usr/local/cuda` points at 12.8+ (H200 bootstrap scripts do this automatically).

### Serve the local model

```bash
export VLLM_HOST=127.0.0.1
export VLLM_PORT=5001
export VLLM_MODEL_PATH=/workspace/models/Qwen2.5-7B-Instruct-AWQ
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_KV_CACHE_DTYPE=fp8
export VLLM_USE_DEEP_GEMM=1

vllm serve "$VLLM_MODEL_PATH" \
  --host "$VLLM_HOST" --port "$VLLM_PORT" \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.85 \
  --attention-backend "$VLLM_ATTENTION_BACKEND" \
  --kv-cache-dtype "$VLLM_KV_CACHE_DTYPE" \
  --enable-chunked-prefill \
  --use-deep-gemm \
  --tensor-parallel-size 1 \
  --trust-remote-code

curl http://$VLLM_HOST:$VLLM_PORT/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen2.5-7B-Instruct-AWQ","prompt":"ping","max_tokens":8}' | jq '.'
```

Wire NIODOO back in with `./start_all_services.sh --hardware h200` once the curl probe passes.

---

## AMD ROCm (ROCm 6.3+, MI2xx/MI3xx/RDNA3)

### Container-first approach (recommended)

```bash
docker pull vllm/vllm:rocm
docker run --rm -it \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v /workspace/models:/workspace/models:ro \
  -p 5001:5001 \
  vllm/vllm:rocm /bin/bash
```

Inside the container:

```bash
export VLLM_MODEL_PATH=/workspace/models/Qwen2.5-7B-Instruct-AWQ
vllm serve "$VLLM_MODEL_PATH" --host 0.0.0.0 --port 5001 --tensor-parallel-size 1 --max-model-len 16384
```

When the server is up, hit it from the host:

```bash
curl http://127.0.0.1:5001/v1/models | jq '.'
```

### Native install (if you must)

```bash
cd /workspace/Niodoo-Final
uv venv --python 3.12 --seed
source .venv/bin/activate
export PYTORCH_ROCM_ARCH="gfx90a;gfx942;gfx1100"
pip install --upgrade pip
pip install -r third_party/vllm/requirements-rocm.txt
pip install -e third_party/vllm  # assumes repo checkout under third_party/
```

Then reuse the `vllm serve "$VLLM_MODEL_PATH" ...` command from above. Keep `--tensor-parallel-size` at `1` unless you have multiple MI cards.

---

## Intel OpenVINO / XPU (OneAPI 2025.x)

### Build with OpenVINO backend

```bash
cd /workspace/Niodoo-Final
python3.12 -m venv openvino-venv
source openvino-venv/bin/activate
sudo apt-get update && sudo apt-get install -y gcc-12 g++-12 libnuma-dev python3-dev
source /opt/intel/oneapi/setvars.sh
export CC=gcc-12 CXX=g++-12
pip install --upgrade pip
PIP_PRE=1 VLLM_TARGET_DEVICE=openvino \
  PIP_EXTRA_INDEX_URL="https://storage.openvinotoolkit.org/simple/wheels/nightly/" \
  python -m pip install -v vllm
```

### Serve the local model on CPU/XPU

```bash
export VLLM_MODEL_PATH=/workspace/models/Qwen2.5-7B-Instruct-AWQ
vllm serve "$VLLM_MODEL_PATH" \
  --device openvino \
  --host 127.0.0.1 --port 5001 \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --max-num-batched-tokens 2048

curl http://127.0.0.1:5001/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen2.5-7B-Instruct-AWQ","prompt":"ping","max_tokens":8}' | jq '.'
```

---

## Google Cloud TPU v5e / v6e

### TPU VM setup

```bash
gcloud compute tpus tpu-vm ssh my-tpu --zone=us-central2-b --worker=all --project $PROJECT

python3.12 -m venv vllm-tpu
source vllm-tpu/bin/activate
pip install --upgrade pip
pip install vllm-tpu
```

Stage the model directory onto the TPU VM (rsync or pre-mounted persistent disk) so that `/workspace/models/Qwen2.5-7B-Instruct-AWQ` exists.

### Serve from TPU

```bash
export VLLM_MODEL_PATH=/workspace/models/Qwen2.5-7B-Instruct-AWQ
vllm serve "$VLLM_MODEL_PATH" \
  --device tpu \
  --host 0.0.0.0 --port 5001 \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --max-num-batched-tokens 2048

curl http://localhost:5001/v1/models | jq '.'
```

> **Note:** TPU support is still marked “preview” in vLLM 0.11+. Expect first-launch compilations to take a few minutes.

---

## Integrating with NIODOO

- `start_all_services.sh` now defaults to `VLLM_MODEL_ID=/workspace/models/Qwen2.5-7B-Instruct-AWQ`. Override with `export VLLM_MODEL_ID=/path/to/another/model` before running the script for alternate weights.
- `START_VLLM_COMMANDS.txt` and `FIX_VLLM_NOW.txt` mirror the Hopper commands above. Use them for manual overrides when automation fails.
- After any manual launch, re-run `cargo test --lib vllm_bridge::tests -- --nocapture` to verify the bridge before resuming prompts.

Keep this document in sync with upstream release notes (`docs.vllm.ai`) and update the flag sets when vLLM bumps the serve CLI.


