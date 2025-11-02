#!/bin/bash
# Start ALL services: vLLM, Qdrant, Ollama

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR"

version_ge() {
    # Returns success if $1 >= $2 (semantic version compare)
    local v1="$1"
    local v2="$2"
    if [ -z "$v1" ] || [ -z "$v2" ]; then
        return 1
    fi
    [ "$(printf '%s\n' "$v2" "$v1" | sort -V | head -n1)" = "$v2" ]
}

ENV_FILE="$ROOT_DIR/tcs_runtime.env"
if [ -f "$ENV_FILE" ]; then
    # shellcheck disable=SC1090
    source "$ENV_FILE"
fi

# Parse CLI overrides (currently only --hardware is supported)
HARDWARE_PROFILE=${HARDWARE:-}
while [[ $# -gt 0 ]]; do
    case "$1" in
        --hardware)
            if [[ -n "${2:-}" ]]; then
                HARDWARE_PROFILE="$2"
                shift 2
                continue
            else
                echo "ERROR: --hardware flag requires a value" >&2
                exit 1
            fi
            ;;
        *)
            echo "⚠️  Ignoring unknown argument: $1" >&2
            shift
            ;;
    esac
done

if [ -n "$HARDWARE_PROFILE" ]; then
    export HARDWARE="$HARDWARE_PROFILE"
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
    printf "%s %s\n" "$host" "$port"
}

VLLM_ENDPOINT=${VLLM_ENDPOINT:-http://127.0.0.1:5001}
OLLAMA_ENDPOINT=${OLLAMA_ENDPOINT:-http://127.0.0.1:11434}
QDRANT_URL=${QDRANT_URL:-http://127.0.0.1:6333}

VLLM_BASE=${VLLM_ENDPOINT%/}
OLLAMA_BASE=${OLLAMA_ENDPOINT%/}
QDRANT_BASE=${QDRANT_URL%/}

if [[ "$VLLM_BASE" == */v1 ]]; then
    VLLM_BASE=${VLLM_BASE%/v1}
fi

if [[ "$OLLAMA_BASE" == */api ]]; then
    OLLAMA_BASE=${OLLAMA_BASE%/api}
fi

read -r DEFAULT_VLLM_HOST DEFAULT_VLLM_PORT < <(extract_host_port "$VLLM_BASE" "5001")
read -r DEFAULT_OLLAMA_HOST DEFAULT_OLLAMA_PORT < <(extract_host_port "$OLLAMA_BASE" "11434")

VLLM_HOST=${VLLM_HOST:-$DEFAULT_VLLM_HOST}
VLLM_PORT=${VLLM_PORT:-$DEFAULT_VLLM_PORT}
OLLAMA_HOST=${OLLAMA_HOST:-$DEFAULT_OLLAMA_HOST}
OLLAMA_PORT=${OLLAMA_PORT:-$DEFAULT_OLLAMA_PORT}

VLLM_MODEL_ID=${VLLM_MODEL_ID:-${VLLM_MODEL:-/workspace/models/hf_cache/models--Qwen--Qwen2.5-7B-Instruct-AWQ}}
VLLM_MODEL_PATH=${VLLM_MODEL_PATH:-$VLLM_MODEL_ID}
VLLM_DTYPE=${VLLM_DTYPE:-bfloat16}
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.85}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-4096}
VLLM_MAX_NUM_BATCHED_TOKENS=${VLLM_MAX_NUM_BATCHED_TOKENS:-4096}
VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-32}
VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-flash-attn}
VLLM_KV_CACHE_DTYPE=${VLLM_KV_CACHE_DTYPE:-fp16}
VLLM_TENSOR_PARALLEL_SIZE=${VLLM_TENSOR_PARALLEL_SIZE:-1}
VLLM_DATA_PARALLEL_SIZE=${VLLM_DATA_PARALLEL_SIZE:-1}
VLLM_ENABLE_CHUNKED_PREFILL=${VLLM_ENABLE_CHUNKED_PREFILL:-0}
VLLM_USE_DEEP_GEMM=${VLLM_USE_DEEP_GEMM:-0}
VLLM_ALL2ALL_BACKEND=${VLLM_ALL2ALL_BACKEND:-}

VLLM_VERSION=$(vllm --version 2>/dev/null | tail -n1 | tr -d '\r')
ATTENTION_FLAG_SUPPORTED=1
DEEP_GEMM_FLAG_SUPPORTED=1
if version_ge "$VLLM_VERSION" "0.11.0"; then
    ATTENTION_FLAG_SUPPORTED=0
    DEEP_GEMM_FLAG_SUPPORTED=0
fi

LOWER_HARDWARE=$(echo "${HARDWARE:-}" | tr '[:upper:]' '[:lower:]')
if [[ "$LOWER_HARDWARE" == "h200" ]]; then
    VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.85}
    VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-32768}
    VLLM_MAX_NUM_BATCHED_TOKENS=${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}
    VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-64}
    VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}
    VLLM_KV_CACHE_DTYPE=${VLLM_KV_CACHE_DTYPE:-fp8}
    VLLM_ENABLE_CHUNKED_PREFILL=${VLLM_ENABLE_CHUNKED_PREFILL:-1}
    VLLM_USE_DEEP_GEMM=${VLLM_USE_DEEP_GEMM:-1}
    VLLM_ALL2ALL_BACKEND=${VLLM_ALL2ALL_BACKEND:-pplx}
fi

echo "🚀 STARTING ALL SERVICES..."
[ -n "$HARDWARE" ] && echo "   Hardware profile: $HARDWARE"
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
    export VLLM_ATTENTION_BACKEND
    export VLLM_KV_CACHE_DTYPE
    export VLLM_ALL2ALL_BACKEND
    export VLLM_USE_DEEP_GEMM

    VLLM_SERVE_ARGS=(
        serve
        "$VLLM_MODEL_ID"
        --host "$VLLM_HOST"
        --port "$VLLM_PORT"
        --dtype "$VLLM_DTYPE"
        --max-model-len "$VLLM_MAX_MODEL_LEN"
        --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION"
        --tensor-parallel-size "$VLLM_TENSOR_PARALLEL_SIZE"
        --trust-remote-code
    )

    if [ -n "$VLLM_MAX_NUM_BATCHED_TOKENS" ]; then
        VLLM_SERVE_ARGS+=(--max-num-batched-tokens "$VLLM_MAX_NUM_BATCHED_TOKENS")
    fi
    if [ -n "$VLLM_MAX_NUM_SEQS" ]; then
        VLLM_SERVE_ARGS+=(--max-num-seqs "$VLLM_MAX_NUM_SEQS")
    fi
    if [ "$ATTENTION_FLAG_SUPPORTED" = "1" ] && [ -n "$VLLM_ATTENTION_BACKEND" ]; then
        VLLM_SERVE_ARGS+=(--attention-backend "$VLLM_ATTENTION_BACKEND")
    fi
    if [ -n "$VLLM_KV_CACHE_DTYPE" ]; then
        VLLM_SERVE_ARGS+=(--kv-cache-dtype "$VLLM_KV_CACHE_DTYPE")
    fi
    if [ "$VLLM_ENABLE_CHUNKED_PREFILL" = "1" ]; then
        VLLM_SERVE_ARGS+=(--enable-chunked-prefill)
    fi
    if [ "$DEEP_GEMM_FLAG_SUPPORTED" = "1" ] && [ "$VLLM_USE_DEEP_GEMM" = "1" ]; then
        VLLM_SERVE_ARGS+=(--use-deep-gemm)
    fi
    if [ -n "$VLLM_EXTRA_ARGS" ]; then
        # shellcheck disable=SC2206
        EXTRA_SPLIT=($VLLM_EXTRA_ARGS)
        VLLM_SERVE_ARGS+=("${EXTRA_SPLIT[@]}")
    fi

    if [ "$ATTENTION_FLAG_SUPPORTED" = "0" ] && [ -n "$VLLM_ATTENTION_BACKEND" ]; then
        echo "   ℹ️  vLLM $VLLM_VERSION auto-selects attention backend; skipping --attention-backend flag"
    fi
    if [ "$DEEP_GEMM_FLAG_SUPPORTED" = "0" ] && [ "$VLLM_USE_DEEP_GEMM" = "1" ]; then
        echo "   ℹ️  vLLM $VLLM_VERSION handles DeepGEMM automatically; skipping --use-deep-gemm flag"
    fi

    echo "   Launching vLLM with dtype=$VLLM_DTYPE, kv-cache=$VLLM_KV_CACHE_DTYPE"
    nohup vllm "${VLLM_SERVE_ARGS[@]}" > /tmp/vllm_service.log 2>&1 &
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
if ! command -v ollama >/dev/null 2>&1; then
    echo "   ⚠️  Ollama CLI not found; skipping (set OLLAMA_ENDPOINT to a remote host if needed)"
elif curl -s "${OLLAMA_BASE}/api/embeddings" \
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



