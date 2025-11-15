#!/usr/bin/env bash
# Small soak test: 4 parallel jobs, 20 cycles each

set -euo pipefail

cd /workspace/Niodoo-Final

# Source runtime environment first
if [[ -f "tcs_runtime.env" ]]; then
    set -a
    source tcs_runtime.env
    set +a
    echo "✅ Loaded environment from tcs_runtime.env"
fi

# Override with gRPC + vLLM curator settings
export QDRANT_USE_GRPC=true
export CURATOR_BACKEND=vllm
export CURATOR_VLLM_ENDPOINT=${VLLM_ENDPOINT:-http://127.0.0.1:5001}
export ENABLE_CURATOR=true
export EMBEDDING_DEVICE=${EMBEDDING_DEVICE:-auto}

echo "🧪 Small Soak Test (quick mode)"
echo "==============================="
echo "Config: gRPC Qdrant + vLLM Curator"
echo "QDRANT_USE_GRPC=$QDRANT_USE_GRPC"
echo "CURATOR_BACKEND=$CURATOR_BACKEND"
echo "ENABLE_CURATOR=$ENABLE_CURATOR"
echo "VLLM_ENDPOINT=${VLLM_ENDPOINT:-http://127.0.0.1:5001}"
echo "QDRANT_URL=${QDRANT_URL:-http://127.0.0.1:6333}"
echo ""

mkdir -p logs/soak/small

# Check services
echo "Checking services..."
curl -sS http://127.0.0.1:6333/healthz > /dev/null || { echo "❌ Qdrant not running"; exit 1; }
curl -sS http://127.0.0.1:5001/v1/models > /dev/null || { echo "❌ vLLM not running"; exit 1; }
echo "✅ Services ready"

export SOAK_EMBEDDING_MAX_LATENCY_MS=${SOAK_EMBEDDING_MAX_LATENCY_MS:-1000}
echo "🎯 Embedding latency target: ${SOAK_EMBEDDING_MAX_LATENCY_MS} ms"
echo "Verifying GPU-backed embeddings..."

# Ensure ONNX runtime libraries are discoverable before spawning soak_test
export LD_LIBRARY_PATH="/tmp/cudnn8_extract/cudnn-linux-x86_64-8.9.7.29_cuda11-archive/lib:/workspace/Niodoo-Final/third_party/onnxruntime-linux-x64-gpu-1.18.1/lib:/workspace/Niodoo-Final/third_party/onnxruntime-linux-x64-gpu-1.18.1/lib/cuda_compat:/usr/local/cuda-11.8/lib64:/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}"

python3 <<'PY'
import json
import os
import subprocess
import sys
import time

script_path = "/workspace/Niodoo-Final/src/scripts/real_ai_inference.py"
target_ms = int(os.environ["SOAK_EMBEDDING_MAX_LATENCY_MS"])

start = time.perf_counter()
proc = subprocess.run(
    ["python3", script_path, "embed", "GPU verification probe"],
    capture_output=True,
    text=True,
    check=True,
)
elapsed_ms = (time.perf_counter() - start) * 1000

try:
    payload = json.loads(proc.stdout)
except json.JSONDecodeError as exc:
    sys.stderr.write(f"❌ Failed to parse embedding response: {exc}\n{proc.stdout}\n")
    sys.exit(1)

if payload.get("status") != "success":
    sys.stderr.write(f"❌ Embedding probe failed: {payload!r}\n")
    sys.exit(1)

device = payload.get("device")
if device is None:
    sys.stderr.write("❌ Embedding response missing device telemetry; cannot confirm GPU usage\n")
    sys.exit(1)

if not device.startswith("cuda"):
    sys.stderr.write(f"❌ Embedding probe ran on {device}, expected CUDA device\n")
    sys.exit(1)

latency_ms = float(payload.get("warmup_ms", elapsed_ms))
total_ms = elapsed_ms

if latency_ms > target_ms:
    sys.stderr.write(
        f"❌ Embedding warm-up {latency_ms:.2f} ms exceeded target {target_ms} ms\n"
    )
    sys.exit(1)

if total_ms > target_ms:
    sys.stderr.write(
        f"⚠️ Cold start cost {total_ms:.2f} ms exceeds target {target_ms} ms; warm-up {latency_ms:.2f} ms. Continuing.\n"
    )

print(
    f"✅ GPU embeddings ready on {device} (warm-up {latency_ms:.2f} ms, cold start {total_ms:.2f} ms, target {target_ms} ms)"
)
PY

echo "GPU verification passed. Ensuring soak_test binary is available..."

if [ ! -x "./target/release/soak_test" ]; then
  echo "Building soak_test (release)..."
  mkdir -p /workspace/Niodoo-Final/.tmp
  export TMPDIR=/workspace/Niodoo-Final/.tmp
  cargo build --release --bin soak_test >/tmp/small_soak_build.log 2>&1 || {
    echo "❌ Failed to build soak_test. See /tmp/small_soak_build.log";
    exit 1;
  }
fi

export RUST_LOG=${RUST_LOG:-info}

LOG_FILE="logs/soak/small/soak_small.log"
echo "Running soak_test --quick (logs -> $LOG_FILE)"
./target/release/soak_test --quick > "$LOG_FILE" 2>&1
status=$?

echo ""
if [ $status -eq 0 ]; then
  echo "✅ Small soak test complete!"
else
  echo "⚠️ Soak test exited with status $status"
fi

echo ""
echo "📊 Summary:"
python3 <<'PY'
import json, pathlib
results_path = pathlib.Path('soak_test_results.json')
if not results_path.exists():
    print('No soak_test_results.json produced. Check logs/soak/small/soak_small.log')
    raise SystemExit(0)
data = json.loads(results_path.read_text())
def flag(ok):
    return '✅' if ok else '❌'
success_rate = data.get('success_rate', 0)
avg_latency = data.get('avg_latency_ms', 0)
ops_sec = data.get('ops_per_sec', 0)
failed = data.get('failed_operations', 0)
duration = data.get('duration_secs', 0)
print(f"Duration: {duration:.1f}s | Ops: {data.get('total_operations', 0)} | Ops/sec: {ops_sec:.1f}")
print(f"Success rate: {flag(success_rate >= 0.99)} {success_rate*100:.2f}% (failures={failed})")
print(f"Avg latency: {flag(avg_latency < 1000)} {avg_latency:.1f} ms")
print(f"Peak memory: {data.get('peak_memory_mb', 0):.1f} MB | Growth: {data.get('memory_growth_mb', 0):.1f} MB")
print(f"Threats: {data.get('threat_count', 0)} | Healings: {data.get('healing_count', 0)} | Breakthroughs: {data.get('breakthroughs', 0)}")
PY

echo ""
echo "🗒️ Detailed logs: $LOG_FILE"

