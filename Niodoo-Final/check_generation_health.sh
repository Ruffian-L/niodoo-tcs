#!/usr/bin/env bash
set -euo pipefail

BACKEND=${GENERATION_BACKEND:-${1:-vllm_gpu}}
VLLM_ENDPOINT=${VLLM_ENDPOINT:-http://127.0.0.1:5001}
OLLAMA_ENDPOINT=${OLLAMA_ENDPOINT:-http://127.0.0.1:11434}

log() {
    printf '[check] %s\n' "$1"
}

fail() {
    printf '[check] ERROR: %s\n' "$1" >&2
    exit 1
}

probe_url() {
    local url=$1
    log "GET ${url}"
    if ! curl -fsSL --max-time 5 "$url" >/dev/null; then
        fail "Request failed: ${url}"
    fi
}

probe_url_optional() {
    local url=$1
    log "GET ${url}"
    if ! curl -fsSL --max-time 5 "$url" >/dev/null; then
        log "WARN: Request failed (continuing): ${url}"
        return 0
    fi
}

case "$BACKEND" in
    ollama_cpu)
        BASE=${OLLAMA_ENDPOINT%/}
        probe_url_optional "${BASE}/api/health"
        probe_url "${BASE}/v1/models"
        log "Ollama OpenAI shim looks healthy at ${BASE}"
        ;;
    *)
        BASE=${VLLM_ENDPOINT%/}
        if ! curl -fsSL --max-time 5 "${BASE}/health" >/dev/null; then
            probe_url "${BASE}/v1/models"
        else
            log "vLLM health endpoint responded"
        fi
        log "vLLM backend looks healthy at ${BASE}"
        ;;
 esac
