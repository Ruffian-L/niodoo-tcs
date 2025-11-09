#!/usr/bin/env bash
# Start Qdrant with durable storage for the reverse ablation lab.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
DATA_DIR="$ROOT_DIR/qdrant_storage"
LOG_DIR="$ROOT_DIR/logs"
LOG_FILE="$LOG_DIR/qdrant.log"
CONTAINER_NAME="niodoo-qdrant"
IMAGE_TAG="qdrant/qdrant:latest"
HTTP_PORT="${QDRANT_HTTP_PORT:-6333}"
GRPC_PORT="${QDRANT_GRPC_PORT:-6334}"

mkdir -p "$DATA_DIR" "$LOG_DIR"

echo "[start_qdrant] Ensuring Docker is available" | tee "$LOG_FILE"
if ! command -v docker >/dev/null 2>&1; then
  echo "[start_qdrant] ERROR: docker not found in PATH" | tee -a "$LOG_FILE"
  exit 1
fi

if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  echo "[start_qdrant] Container already running" | tee -a "$LOG_FILE"
else
  if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "[start_qdrant] Removing stale container" | tee -a "$LOG_FILE"
    docker rm -f "$CONTAINER_NAME" >>"$LOG_FILE" 2>&1 || true
  fi

  echo "[start_qdrant] Launching ${IMAGE_TAG} on ports ${HTTP_PORT}/${GRPC_PORT}" | tee -a "$LOG_FILE"
  docker run -d \
    --name "$CONTAINER_NAME" \
    --restart unless-stopped \
    -p "${HTTP_PORT}:6333" \
    -p "${GRPC_PORT}:6334" \
    -v "$DATA_DIR:/qdrant/storage" \
    "$IMAGE_TAG" >>"$LOG_FILE" 2>&1
fi

echo -n "[start_qdrant] Waiting for health endpoint" | tee -a "$LOG_FILE"
for _ in {1..30}; do
  if curl -sf "http://127.0.0.1:${HTTP_PORT}/collections" >/dev/null; then
    echo " - ready" | tee -a "$LOG_FILE"
    exit 0
  fi
  echo -n '.' | tee -a "$LOG_FILE"
  sleep 2
 done

echo "\n[start_qdrant] Timeout waiting for Qdrant" | tee -a "$LOG_FILE"
exit 1
