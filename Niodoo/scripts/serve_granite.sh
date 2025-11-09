#!/usr/bin/env bash
# Launch the Granite 3B model with vLLM in OpenAI-compatible mode.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
LOG_DIR="$ROOT_DIR/logs"
LOG_FILE="$LOG_DIR/vllm_granite.log"
MODEL_ID="${VLLM_MODEL_ID:-ibm-granite/granite-3b-code-instruct}"
PORT="${VLLM_PORT:-8000}"
HOST="${VLLM_HOST:-0.0.0.0}"
DTYPE="${VLLM_DTYPE:-auto}"
MAX_LEN="${VLLM_MAX_LEN:-2048}"
EXTRA_ARGS=(${VLLM_EXTRA_ARGS:-})

mkdir -p "$LOG_DIR"

echo "[serve_granite] Starting vLLM with model=$MODEL_ID port=$PORT" | tee "$LOG_FILE"

# Avoid multiple instances if the port is already serving.
if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
  echo "[serve_granite] vLLM already running on port ${PORT}." | tee -a "$LOG_FILE"
  exit 0
fi

# Activate the workspace venv if present.
if [[ -f "$ROOT_DIR/../venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT_DIR/../venv/bin/activate"
fi

# Ensure ONNX runtime libraries are available for optional local embeddings.
export LD_LIBRARY_PATH="$ROOT_DIR/../onnxruntime-linux-x64-1.16.3/lib:${LD_LIBRARY_PATH:-}"

nohup vllm serve "$MODEL_ID" \
  --host "$HOST" \
  --port "$PORT" \
  --dtype "$DTYPE" \
  --max-model-len "$MAX_LEN" \
  --trust-remote-code \
  "${EXTRA_ARGS[@]}" \
  >> "$LOG_FILE" 2>&1 &

PID=$!
echo "[serve_granite] vLLM PID $PID" | tee -a "$LOG_FILE"

echo -n "[serve_granite] Waiting for /v1/models" | tee -a "$LOG_FILE"
for _ in {1..60}; do
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
    echo " - ready" | tee -a "$LOG_FILE"
    exit 0
  fi
  echo -n '.' | tee -a "$LOG_FILE"
  sleep 2
  if ! kill -0 "$PID" >/dev/null 2>&1; then
    echo "\n[serve_granite] vLLM process died unexpectedly" | tee -a "$LOG_FILE"
    exit 1
  fi

done

echo "\n[serve_granite] Timeout waiting for Granite service" | tee -a "$LOG_FILE"
exit 1
