#!/bin/bash
# Start ALL services: vLLM, Qdrant, Ollama

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR"

ENV_FILE="$ROOT_DIR/tcs_runtime.env"
if [ -f "$ENV_FILE" ]; then
    # shellcheck disable=SC1090
    source "$ENV_FILE"
fi

extract_host_port() {
    local url="$1"
    local default_port="$2"
    local host port
    if [[ "$url" =~ ^https?://([^/:]+)(:([0-9]+))? ]]; then
        host="${BASH_REMATCH[1]}"
        port="${BASH_REMATCH[3]}"
        if [ -z "$port" ]; then
            port="$default_port"
        fi
    else
        host="127.0.0.1"
        port="$default_port"
    fi
    printf "%s %s" "$host" "$port"
}

VLLM_ENDPOINT=${VLLM_ENDPOINT:-http://127.0.0.1:5001}
OLLAMA_ENDPOINT=${OLLAMA_ENDPOINT:-http://127.0.0.1:11434}
QDRANT_URL=${QDRANT_URL:-http://127.0.0.1:6333}
VLLM_MODEL=${VLLM_MODEL:-Qwen/Qwen2.5-7B-Instruct-AWQ}

VLLM_BASE=${VLLM_ENDPOINT%/}
OLLAMA_BASE=${OLLAMA_ENDPOINT%/}
QDRANT_BASE=${QDRANT_URL%/}

if [[ "$VLLM_BASE" == */v1 ]]; then
    VLLM_BASE=${VLLM_BASE%/v1}
fi

if [[ "$OLLAMA_BASE" == */api ]]; then
    OLLAMA_BASE=${OLLAMA_BASE%/api}
fi

read -r VLLM_HOST VLLM_PORT < <(extract_host_port "$VLLM_BASE" "5001")
read -r OLLAMA_HOST OLLAMA_PORT < <(extract_host_port "$OLLAMA_BASE" "11434")

echo "🚀 STARTING ALL SERVICES..."
echo ""

# vLLM
echo "1️⃣ Starting vLLM (${VLLM_HOST}:${VLLM_PORT})..."
if curl -s "${VLLM_BASE}/v1/models" > /dev/null 2>&1; then
    echo "   ✅ vLLM already running"
else
    echo "   Starting BIG Qwen 7B model..."
    pkill -9 -f vllm || true
    sleep 2
    cd "$ROOT_DIR"
    if [ -f venv/bin/activate ]; then
        # shellcheck disable=SC1091
        source venv/bin/activate
    fi
    export HF_HUB_ENABLE_HF_TRANSFER=0
    nohup vllm serve /workspace/models/hf_cache/models--Qwen--Qwen2.5-7B-Instruct-AWQ \
        --host "$VLLM_HOST" --port "$VLLM_PORT" \
        --gpu-memory-utilization 0.85 \
        --trust-remote-code \
        > /tmp/vllm_service.log 2>&1 &
    echo "   ⏳ Loading... (2-5 minutes)"
fi

# Qdrant
echo ""
echo "2️⃣ Starting Qdrant (${QDRANT_BASE})..."
if curl -s "${QDRANT_BASE}/collections/experiences" > /dev/null 2>&1; then
    echo "   ✅ Qdrant already running"
else
    echo "   Starting Qdrant..."
    docker restart qdrant || echo "   ⚠️  Qdrant not in docker, check supervisor"
fi

# Ollama
echo ""
echo "3️⃣ Starting Ollama (${OLLAMA_HOST}:${OLLAMA_PORT})..."
if curl -s "${OLLAMA_BASE}/api/embeddings" \
    -H "Content-Type: application/json" \
    -d '{"model":"qwen2.5:0.5b","prompt":"test"}' > /dev/null 2>&1; then
    echo "   ✅ Ollama already running"
else
    echo "   Ensuring Ollama model..."
    OLLAMA_HOST="${OLLAMA_HOST}:${OLLAMA_PORT}" ollama pull qwen2.5:0.5b
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ ALL SERVICES STARTED"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Waiting for vLLM to be ready..."
echo "Watch: tail -f /tmp/vllm_service.log"

for _ in {1..60}; do
    sleep 5
    if curl -s "${VLLM_BASE}/v1/models" > /dev/null 2>&1; then
        echo ""
        echo "✅✅✅ vLLM IS READY! ✅✅✅"
        break
    fi
    printf "."
done

echo ""
echo "🧪 RUNNING TESTS NOW..."
cd "$ROOT_DIR"
if [ -f "$ENV_FILE" ]; then
    # shellcheck disable=SC1090
    source "$ENV_FILE"
fi
cargo test --lib vllm_bridge::tests -- --nocapture



