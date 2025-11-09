#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
LOG_DIR="$ROOT_DIR/logs"
BASELINE_DIR="$ROOT_DIR/baselines"
LOG_FILE="$LOG_DIR/soak_system1.log"
RESULT_FILE="$BASELINE_DIR/system1.json"
export RESULT_FILE
CLI_MANIFEST="$ROOT_DIR/Cargo.toml"
PROMPTS=(
  "Explain hyperfocus detection."
  "How do we avoid cooldown drift in ERAG?"
  "What payloads should we log after a breakthrough?"
  "Summarize the memory cadence during a Discover quadrant."
  "When should beta_meta snapshots be persisted?"
)

mkdir -p "$LOG_DIR" "$BASELINE_DIR"

# Ensure Granite service is running
if ! curl -sf "http://127.0.0.1:8000/v1/models" >/dev/null; then
  echo "[soak_system1] Granite not responding on 8000" >&2
  exit 1
fi

# Ensure Qdrant responding
if ! curl -sf "http://127.0.0.1:6333/healthz" >/dev/null; then
  echo "[soak_system1] Qdrant not responding on 6333" >&2
  exit 1
fi

: > "$LOG_FILE"

LATENCIES=()
for prompt in "${PROMPTS[@]}"; do
  start_ms=$(date +%s%3N)
  if output=$(cargo run --quiet --bin niodoo-cli -- --with-memory --compass Discover --prompt "$prompt" 2>>"$LOG_FILE"); then
    status="ok"
  else
    status="error"
  fi
  end_ms=$(date +%s%3N)
  latency=$((end_ms - start_ms))
  LATENCIES+=("$latency")

  {
    echo "[soak_system1] prompt: $prompt"
    echo "[soak_system1] status: $status latency_ms: $latency"
    echo "$output"
    echo "---"
  } >> "$LOG_FILE"

  sleep 0.5
done

export LATENCIES_STR="${LATENCIES[*]}"
python3 - <<'PY'
import json, statistics, os
from pathlib import Path
latencies = [int(x) for x in os.environ['LATENCIES_STR'].split()]
result = {
    "total_requests": len(latencies),
    "latencies_ms": {
        "p50": statistics.median(latencies),
        "avg": statistics.fmean(latencies),
        "max": max(latencies),
    },
    "requests": latencies,
}
Path(os.environ["RESULT_FILE"]).write_text(json.dumps(result, indent=2))
PY

echo "[soak_system1] results stored at $RESULT_FILE"
