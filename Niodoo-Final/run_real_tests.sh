#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ -n "${NIODOO_SKIP_REAL_TESTS:-}" ]]; then
  echo "NIODOO_SKIP_REAL_TESTS set; skipping real stack tests." >&2
  exit 0
fi

source ./tcs_runtime.env

echo "Checking service health before tests..."
curl -fsS "$VLLM_ENDPOINT/v1/models" >/dev/null
curl -fsS "$OLLAMA_ENDPOINT/api/tags" >/dev/null
curl -fsS "$QDRANT_URL/healthz" >/dev/null

RUN_ENV=(
  REAL_TEST=1
  MOCK_MODE=0
  VLLM_ENDPOINT="$VLLM_ENDPOINT"
  OLLAMA_ENDPOINT="$OLLAMA_ENDPOINT"
  QDRANT_URL="$QDRANT_URL"
  TOKENIZER_JSON="$TOKENIZER_JSON"
)

echo "Running topology smoke test (real stack)..."
env "${RUN_ENV[@]}" cargo run --release --bin topology_bench -- --cycles 2 >/tmp/topology_bench_real.log

echo "Running emotion benchmark smoke (real stack)..."
env "${RUN_ENV[@]}" cargo run --release --bin emotion_bench >/tmp/emotion_bench_real.log

echo "Running integration tests with REAL_TEST=1..."
env "${RUN_ENV[@]}" cargo test -p niodoo_real_integrated -- --ignored --test-threads=1 >/tmp/real_tests.log

echo "All real tests completed." 

