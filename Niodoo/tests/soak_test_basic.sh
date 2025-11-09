#!/usr/bin/env bash
# Run a small soak test (12 prompts) against the Granite CLI baseline.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
LOG_DIR="$ROOT_DIR/logs"
BASELINE_DIR="$ROOT_DIR/baselines"
LOG_FILE="$LOG_DIR/soak_system0.log"
RESULT_FILE="$BASELINE_DIR/system0.json"
CLI_MANIFEST="$ROOT_DIR/Cargo.toml"
CLI_ARGS=("--prompt")
PROMPTS=(
  "Explain the difference between Rust's &T and &mut T references."
  "Write a Python function that reverses a string."
  "Summarize the NIODOO 7-stage pipeline in one sentence."
  "Generate a unit test in Rust for a function that sums a slice of i32."
  "Describe how to start Qdrant for NIODOO."
  "Refactor a nested if/else into a match statement in Rust."
  "Provide a bash command to check if vLLM is running on port 8000."
  "Explain the role of the Consciousness Compass."
  "Suggest logging best practices for soak tests."
  "What environment variable controls QLoRA adapter rank?"
  "Give a JSON example of a Granite completion request."
  "List three Prometheus metrics we should monitor for vLLM."
)

mkdir -p "$LOG_DIR" "$BASELINE_DIR"

if [[ ! -x "$ROOT_DIR/scripts/serve_granite.sh" ]]; then
  echo "[soak_test_basic] Missing scripts/serve_granite.sh" >&2
  exit 1
fi

# Ensure the Granite service is running.
"$ROOT_DIR/scripts/serve_granite.sh"

if [[ ! -f "$CLI_MANIFEST" ]]; then
  echo "[soak_test_basic] Expected $CLI_MANIFEST. Build the CLI before running the soak." >&2
  exit 1
fi

DATA_FILE=$(mktemp)
trap 'rm -f "$DATA_FILE"' EXIT

: > "$LOG_FILE"

echo "[soak_test_basic] running ${#PROMPTS[@]} prompts" | tee -a "$LOG_FILE"

for prompt in "${PROMPTS[@]}"; do
  start_ms=$(date +%s%3N)
  if output=$(cargo run --quiet --manifest-path "$CLI_MANIFEST" --bin niodoo-cli -- "${CLI_ARGS[@]}" "$prompt" 2>&1); then
    status="ok"
  else
    status="error"
  fi
  end_ms=$(date +%s%3N)
  latency=$((end_ms - start_ms))

  printf '%s	%s	%s
' "$prompt" "$status" "$latency" >> "$DATA_FILE"

  {
    echo "[soak_test_basic] prompt: $prompt"
    echo "[soak_test_basic] status: $status latency_ms: $latency"
    echo "$output"
    echo "---"
  } >> "$LOG_FILE"

  sleep 0.25
done

python3 - "$DATA_FILE" "$RESULT_FILE" <<'PY'
import json
import statistics
import sys
import time

if len(sys.argv) != 3:
    raise SystemExit("Usage: script <data_file> <output_file>")

data_file, out_file = sys.argv[1:]

prompts = []
statuses = []
latencies = []
with open(data_file, encoding="utf-8") as fh:
    for line in fh:
        prompt, status, latency = line.rstrip('\n').split('\t')
        prompts.append(prompt)
        statuses.append(status)
        latencies.append(int(latency))

successes = sum(1 for s in statuses if s == "ok")
failures = len(statuses) - successes

if latencies:
    latencies_sorted = sorted(latencies)
    p50 = statistics.median(latencies_sorted)
    p95 = latencies_sorted[int(0.95 * (len(latencies_sorted) - 1))]
    p99 = latencies_sorted[int(0.99 * (len(latencies_sorted) - 1))]
    avg = statistics.fmean(latencies_sorted)
else:
    p50 = p95 = p99 = avg = None

data = {
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "total_requests": len(prompts),
    "successes": successes,
    "failures": failures,
    "latencies_ms": {
        "p50": p50,
        "p95": p95,
        "p99": p99,
        "avg": avg,
    },
    "requests": [
        {
            "prompt": prompt,
            "status": status,
            "latency_ms": latency,
        }
        for prompt, status, latency in zip(prompts, statuses, latencies)
    ],
}

with open(out_file, "w", encoding="utf-8") as fh:
    json.dump(data, fh, indent=2)

print(f"[soak_test_basic] wrote baseline to {out_file}")
PY
