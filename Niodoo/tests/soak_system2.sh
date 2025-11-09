#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
LOG_DIR="$ROOT_DIR/logs"
BASELINE_DIR="$ROOT_DIR/baselines"
LOG_FILE="$LOG_DIR/soak_system2.log"
RESULT_FILE="$BASELINE_DIR/system2.json"

mkdir -p "$LOG_DIR" "$BASELINE_DIR"

if ! curl -sf "http://127.0.0.1:8000/v1/models" >/dev/null; then
  echo "[soak_system2] Granite not responding on 8000" >&2
  exit 1
fi

if ! curl -sf "http://127.0.0.1:6333/healthz" >/dev/null; then
  echo "[soak_system2] Qdrant not responding on 6333" >&2
  exit 1
fi

: >"$LOG_FILE"

source "$ROOT_DIR/../venv/bin/activate" 2>/dev/null || true

cargo run --quiet --bin system2_loop -- \
  --iterations 5 \
  --erag-config config/erag.toml \
  --memory-config config/system2_memory.toml \
  --log-file "$LOG_FILE" \
  --baseline-file "$RESULT_FILE"

echo "[soak_system2] baseline stored at $RESULT_FILE"
