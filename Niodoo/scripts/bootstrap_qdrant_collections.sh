#!/usr/bin/env bash
# Create NIODOO ERAG collections and payload schema in Qdrant.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
CONFIG_FILE="$ROOT_DIR/config/erag.toml"
LOG_DIR="$ROOT_DIR/logs"
LOG_FILE="$LOG_DIR/qdrant_bootstrap.log"

mkdir -p "$LOG_DIR"

http_url=${QDRANT_HTTP_URL:-$(grep -E '^http_url' "$CONFIG_FILE" | awk -F'"' '{print $2}' | head -n1)}
collection=${QDRANT_COLLECTION:-$(grep -E '^collection' "$CONFIG_FILE" | awk -F'"' '{print $2}' | head -n1)}
vector_size=${QDRANT_VECTOR_SIZE:-$(grep -E '^vector_size' "$CONFIG_FILE" | awk '{print $3}' | head -n1)}
distance=${QDRANT_DISTANCE:-$(grep -E '^distance' "$CONFIG_FILE" | awk -F'"' '{print $2}' | head -n1)}
api_key=${QDRANT_API_KEY:-$(grep -E '^api_key' "$CONFIG_FILE" | awk -F'"' '{print $2}' | head -n1)}

if [[ -z "$http_url" || -z "$collection" || -z "$vector_size" ]]; then
  echo "[bootstrap_qdrant] Missing configuration values" | tee "$LOG_FILE"
  exit 1
fi

echo "[bootstrap_qdrant] Target collection=$collection vector_size=$vector_size distance=$distance" | tee "$LOG_FILE"

headers=(-H "Content-Type: application/json")
if [[ -n "$api_key" ]]; then
  headers+=(-H "api-key: $api_key")
fi

payload=$(cat <<JSON
{
  "vectors": {
    "size": $vector_size,
    "distance": "${distance:-Cosine}"
  },
  "quantization_config": {
    "scalar": {
      "type": "int8",
      "always_ram": true
    }
  }
}
JSON
)

echo "[bootstrap_qdrant] Creating collection via PUT ${http_url}/collections/${collection}" | tee -a "$LOG_FILE"
if ! curl -sf -X PUT "${http_url}/collections/${collection}" "${headers[@]}" -d "$payload" >>"$LOG_FILE" 2>&1; then
  echo "[bootstrap_qdrant] Collection creation failed" | tee -a "$LOG_FILE"
  exit 1
fi

echo "[bootstrap_qdrant] NOTE: skipping payload schema patch (not exposed via REST in Qdrant >=1.15)." | tee -a "$LOG_FILE"

echo "[bootstrap_qdrant] Bootstrap complete" | tee -a "$LOG_FILE"
