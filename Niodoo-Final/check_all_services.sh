#!/bin/bash
# Check all three services: vLLM, Qdrant, Ollama

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
VLLM_MODEL=${VLLM_MODEL:-qwen2.5-coder:7b}

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

echo "🔍 CHECKING ALL SERVICES..."
echo ""

# Check vLLM
echo "1️⃣ vLLM Status (${VLLM_HOST}:${VLLM_PORT}):"
if curl -s "${VLLM_BASE}/v1/models" > /dev/null 2>&1; then
    echo "   ✅ vLLM is READY"
    curl -s "${VLLM_BASE}/v1/models" | jq '.data[0].id' || echo "   ⚠️  Unable to parse model list"

    echo ""
    echo "   Testing chat completions:"
    curl -s "${VLLM_BASE}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"${VLLM_MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}" |
        jq '.choices[0].message.content' || echo "   ⚠️  Chat endpoint issue"
else
    echo "   ❌ vLLM NOT READY - Still loading or failed"
    echo "   Check: tail -f /tmp/vllm_service.log"
fi

echo ""
echo "2️⃣ Qdrant Status (${QDRANT_BASE}):"
if curl -s "${QDRANT_BASE}/collections/experiences" > /dev/null 2>&1; then
    echo "   ✅ Qdrant is READY"
    curl -s "${QDRANT_BASE}/collections/experiences" | jq '.config.params.vectors.size' || echo "   ⚠️  Unable to parse collection info"
else
    echo "   ❌ Qdrant NOT READY"
    echo "   Run: docker restart qdrant"
fi

echo ""
echo "3️⃣ Ollama Status (${OLLAMA_HOST}:${OLLAMA_PORT}):"
if curl -s "${OLLAMA_BASE}/api/embeddings" \
    -H "Content-Type: application/json" \
    -d '{"model":"qwen2.5:0.5b","prompt":"test"}' > /dev/null 2>&1; then
    echo "   ✅ Ollama is READY"
else
    echo "   ❌ Ollama NOT READY"
    echo "   Run: OLLAMA_HOST=${OLLAMA_HOST}:${OLLAMA_PORT} ollama pull qwen2.5:0.5b"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ ALL CHECKS COMPLETE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

