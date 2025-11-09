#!/usr/bin/env bash
# Health check script for the Granite vLLM server.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
LOG_DIR="$ROOT_DIR/logs"
LOG_FILE="$LOG_DIR/check_granite.log"
PORT="${VLLM_PORT:-8000}"
HOST="${VLLM_HOST:-127.0.0.1}"
TIMEOUT="${HEALTH_TIMEOUT:-5}"

mkdir -p "$LOG_DIR"

URL="http://${HOST}:${PORT}/v1/models"
START_TS=$(date --iso-8601=seconds)

{
  echo "[check_granite] ${START_TS} - Checking ${URL}"
  if RESPONSE=$(curl -sf --max-time "$TIMEOUT" "$URL"); then
    MODEL_COUNT=$(echo "$RESPONSE" | python3 -c 'import json,sys; data=json.load(sys.stdin); print(len(data.get("data", []))) if isinstance(data, dict) else print(0)' 2>/dev/null || echo "unknown")
    echo "[check_granite] OK - ${MODEL_COUNT} models reported"
    exit 0
  else
    echo "[check_granite] FAILED - Granite endpoint not reachable"
    exit 1
  fi
} | tee "$LOG_FILE"
