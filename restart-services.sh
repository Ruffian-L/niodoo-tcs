#!/bin/bash
# Quick script to restart all services (vllm, qdrant, ollama)
# Saves to /workspace network drive

ROOT=${NIODOO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
"$ROOT"/supervisor.sh restart

