#!/usr/bin/env bash

set -euo pipefail
set -o errtrace

ROOT="/workspace/Niodoo-Final"
LOG_DIR="$ROOT/logs"
STATE_DIR="$ROOT/.bootstrap_state"

mkdir -p "$LOG_DIR" "$STATE_DIR"

LOG_FILE="$LOG_DIR/runpod_bootstrap.log"
touch "$LOG_FILE"

exec > >(tee -a "$LOG_FILE") 2>&1
umask 022

cd "$ROOT"

# colours
BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

FORCE_REFRESH=0
SKIP_BUILD=0
SKIP_SERVICES=0
SKIP_PACKAGES=0
SKIP_MODEL_DOWNLOAD=0
SKIP_QDRANT=0
SKIP_OLLAMA=0

usage() {
    cat <<'EOF'
Usage: scripts/runpod_bootstrap.sh [options]

Options:
  --force                Re-run all idempotent steps (ignores cached markers)
  --skip-packages        Skip apt-get system package installation
  --skip-build           Skip cargo build stage
  --skip-services        Do not start or health-check services
  --skip-model-download  Do not attempt to download missing model artifacts
  --skip-qdrant          Skip Qdrant binary/config provisioning
  --skip-ollama          Skip Ollama binary provisioning
  -h, --help             Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --force)
            FORCE_REFRESH=1
            ;;
        --skip-packages)
            SKIP_PACKAGES=1
            ;;
        --skip-build)
            SKIP_BUILD=1
            ;;
        --skip-services)
            SKIP_SERVICES=1
            ;;
        --skip-model-download)
            SKIP_MODEL_DOWNLOAD=1
            ;;
        --skip-qdrant)
            SKIP_QDRANT=1
            ;;
        --skip-ollama)
            SKIP_OLLAMA=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
    esac
    shift
done

if [[ $FORCE_REFRESH -eq 1 ]]; then
    find "$STATE_DIR" -maxdepth 1 -name '*.done' -type f -delete
fi

timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

log_info() {
    echo -e "$(timestamp) ${GREEN}➜${NC} $1"
}

log_warn() {
    echo -e "$(timestamp) ${YELLOW}⚠${NC} $1"
}

log_error() {
    echo -e "$(timestamp) ${RED}✖${NC} $1"
}

log_section() {
    echo -e "\n${BOLD}${BLUE}==> $1${NC}\n"
}

on_error() {
    local exit_code=$?
    local line=$1
    log_error "Bootstrap failed at line ${line} (exit ${exit_code}). See ${LOG_FILE}."
    exit $exit_code
}

trap 'on_error ${LINENO}' ERR

should_run_step() {
    local step=$1
    if [[ $FORCE_REFRESH -eq 1 ]]; then
        return 0
    fi
    [[ ! -f "$STATE_DIR/${step}.done" ]]
}

mark_step() {
    local step=$1
    touch "$STATE_DIR/${step}.done"
}

run_step() {
    local step=$1
    local description=$2
    shift 2
    if should_run_step "$step"; then
        log_section "$description"
        "$@"
        mark_step "$step"
    else
        log_info "Skipping $description (already completed)."
    fi
}

ensure_directory() {
    local dir=$1
    if [[ ! -d "$dir" ]]; then
        mkdir -p "$dir"
    fi
}

ensure_cargo_env() {
    if [[ -f "$HOME/.cargo/env" ]]; then
        # shellcheck disable=SC1091
        source "$HOME/.cargo/env"
        export PATH="$HOME/.cargo/bin:$PATH"
    else
        log_error "Rust toolchain not initialised (missing $HOME/.cargo/env)."
        return 1
    fi
}

activate_venv() {
    if [[ ! -f "$ROOT/venv/bin/activate" ]]; then
        log_error "Python virtualenv missing at $ROOT/venv."
        return 1
    fi
    # shellcheck disable=SC1091
    source "$ROOT/venv/bin/activate"
}

step_system_packages() {
    if ! command -v apt-get >/dev/null 2>&1; then
        log_warn "apt-get not available; skipping system package installation."
        return 0
    fi

    export DEBIAN_FRONTEND=noninteractive

    log_info "Running apt-get update"
    apt-get update

    local packages=(
        build-essential
        cmake
        pkg-config
        libssl-dev
        curl
        wget
        git
        git-lfs
        python3.11
        python3.11-dev
        python3.11-venv
        python3-pip
        libopenblas-dev
        liblapack-dev
        libatlas-base-dev
        gfortran
        unzip
        tar
        ca-certificates
        jq
    )

    log_info "Installing system packages"
    apt-get install -y --no-install-recommends "${packages[@]}"
}

step_rust_toolchain() {
    if ! command -v rustup >/dev/null 2>&1; then
        log_info "Installing rustup toolchain manager"
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    fi

    ensure_cargo_env

    log_info "Updating Rust toolchain"
    rustup toolchain install stable --profile minimal
    rustup default stable
    rustup component add rustfmt clippy
}

step_python_env() {
    if ! command -v python3.11 >/dev/null 2>&1; then
        log_error "python3.11 is required but not found."
        return 1
    fi

    if [[ ! -d "$ROOT/venv" ]]; then
        log_info "Creating Python virtualenv"
        python3.11 -m venv "$ROOT/venv"
    fi

    activate_venv

    pip install --upgrade pip setuptools wheel
    pip install --upgrade "huggingface_hub[cli]>=0.24.5"
    pip install --upgrade --index-url https://download.pytorch.org/whl/cu121 \
        torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1
    pip install --upgrade --extra-index-url https://download.pytorch.org/whl/cu121 -r "$ROOT/requirements.txt"
}

step_prefetch_models() {
    if [[ $SKIP_MODEL_DOWNLOAD -eq 1 ]]; then
        log_warn "Skipping model download per flag."
        return 0
    fi

    if [[ -z "${VLLM_MODEL:-}" ]]; then
        log_warn "VLLM_MODEL not set; skipping model provisioning."
        return 0
    fi

    if [[ -d "$VLLM_MODEL" && -n "$(ls -A "$VLLM_MODEL" 2>/dev/null)" ]]; then
        log_info "Model already present at $VLLM_MODEL"
        return 0
    fi

    if [[ -z "${HF_TOKEN:-}" ]]; then
        log_warn "HF_TOKEN not set; cannot download $VLLM_MODEL automatically."
        return 0
    fi

    ensure_directory "$VLLM_MODEL"

    activate_venv

    local repo="${VLLM_MODEL_REPO:-Qwen/Qwen2.5-7B-Instruct-AWQ}"
    log_info "Downloading model $repo to $VLLM_MODEL"

    python <<'PY'
import os
import sys
from huggingface_hub import snapshot_download

target = os.environ["VLLM_MODEL"]
repo = os.environ.get("VLLM_MODEL_REPO", "Qwen/Qwen2.5-7B-Instruct-AWQ")
token = os.environ.get("HF_TOKEN")

if not token:
    print("HF_TOKEN missing; aborting download", file=sys.stderr)
    sys.exit(0)

snapshot_download(
    repo_id=repo,
    local_dir=target,
    local_dir_use_symlinks=False,
    token=token,
)
PY
}

step_qdrant() {
    local qdrant_root="${QDRANT_ROOT:-/workspace/qdrant}"
    local qdrant_bin="$qdrant_root/qdrant"
    local desired_version="${QDRANT_VERSION:-1.11.3}"
    local download_url="https://github.com/qdrant/qdrant/releases/download/v${desired_version}/qdrant-v${desired_version}-linux-x86_64.tar.gz"

    if [[ -x "$qdrant_bin" ]]; then
        local current_version
        current_version=$("$qdrant_bin" --version | awk '{print $2}' | tr -d 'v')
        if [[ "$current_version" == "$desired_version" ]]; then
            log_info "Qdrant ${desired_version} already installed at $qdrant_root"
        else
            log_warn "Qdrant version $current_version found; upgrading to $desired_version"
        fi
    fi

    if [[ ! -x "$qdrant_bin" || $("$qdrant_bin" --version | awk '{print $2}' | tr -d 'v') != "$desired_version" ]]; then
        log_info "Fetching Qdrant ${desired_version}"
        local tmpdir
        tmpdir=$(mktemp -d)
        curl -fL "$download_url" -o "$tmpdir/qdrant.tar.gz"
        if ! tar -xzf "$tmpdir/qdrant.tar.gz" -C "$tmpdir"; then
            log_error "Failed to extract Qdrant archive from $download_url"
            return 1
        fi
        local extracted
        extracted=$(find "$tmpdir" -maxdepth 1 -type d -name 'qdrant*' | head -n 1)
        if [[ -z "$extracted" ]]; then
            log_error "Failed to locate extracted Qdrant directory"
            return 1
        fi
        rm -rf "$qdrant_root"
        mkdir -p "$qdrant_root"
        cp -a "$extracted/." "$qdrant_root/"
        chmod +x "$qdrant_bin"
        rm -rf "$tmpdir"
    fi

    ensure_directory "/workspace/qdrant_storage"
    ensure_directory "/workspace/qdrant_snapshots"
    ensure_directory "/workspace/qdrant_storage/wal"

    local config_dir="${QDRANT_CONFIG_DIR:-/workspace/qdrant_config}"
    local config_file="$config_dir/config.yaml"

    if [[ ! -d "$config_dir" ]]; then
        mkdir -p "$config_dir"
    fi

    if [[ ! -f "$config_file" ]]; then
        cat > "$config_file" <<'YAML'
service:
  api_key: null
  http_port: 6333
  grpc_port: 6334
  master_key: null
  enable_tls: false
  enable_cors: true

storage:
  storage_path: "/workspace/qdrant_storage"
  snapshots_path: "/workspace/qdrant_snapshots"
  wal_path: "/workspace/qdrant_storage/wal"
  on_disk_payload: true
  create_default_collection: false

cluster:
  enabled: false

log:
  level: INFO
  file: "/workspace/qdrant_storage/qdrant.log"
YAML
    fi
}

step_ollama() {
    if command -v ollama >/dev/null 2>&1; then
        log_info "Ollama already available at $(command -v ollama)"
        return 0
    fi

    local ollama_root="${OLLAMA_ROOT:-/workspace/ollama}"
    local ollama_bin="$ollama_root/bin/ollama"

    if [[ -x "$ollama_bin" ]]; then
        log_info "Ollama binary detected at $ollama_bin"
        export PATH="$ollama_root/bin:$PATH"
        return 0
    fi

    log_info "Installing Ollama binary"
    local tmpdir
    tmpdir=$(mktemp -d)
    curl -fL https://ollama.com/download/ollama-linux-amd64 -o "$tmpdir/ollama.tar.gz"
    if tar -xzf "$tmpdir/ollama.tar.gz" -C "$tmpdir" 2>/dev/null; then
        if [[ -f "$tmpdir/ollama" ]]; then
            mkdir -p "$ollama_root/bin"
            mv "$tmpdir/ollama" "$ollama_bin"
        else
            local extracted
            extracted=$(find "$tmpdir" -maxdepth 1 -type f -name 'ollama*' | head -n 1)
            if [[ -n "$extracted" ]]; then
                mkdir -p "$ollama_root/bin"
                mv "$extracted" "$ollama_bin"
            else
                log_error "Failed to locate Ollama binary in archive"
                return 1
            fi
        fi
    else
        mkdir -p "$ollama_root/bin"
        mv "$tmpdir/ollama.tar.gz" "$ollama_bin"
    fi

    chmod +x "$ollama_bin"
    export PATH="$ollama_root/bin:$PATH"
    rm -rf "$tmpdir"

    log_info "Ollama installed to $ollama_bin"
}

step_rust_deps() {
    ensure_cargo_env
    log_info "Fetching Rust dependencies"
    cargo fetch
}

step_rust_build() {
    ensure_cargo_env
    log_info "Building workspace (release)"
    cargo build --workspace --release --features onnx
}

check_http_health() {
    local name=$1
    local url=$2
    local attempts=${3:-30}
    local delay=${4:-5}
    local max_timeout=${5:-0}  # Maximum total timeout in seconds (0 = use attempts*delay)

    if ! command -v curl >/dev/null 2>&1; then
        log_warn "curl not available; skipping $name health probe ($url)"
        return 0
    fi

    # Special handling for vLLM - needs longer timeout for model loading
    if [[ "$name" == "vLLM" ]] && [[ $max_timeout -eq 0 ]]; then
        attempts=120  # 120 attempts × 5 seconds = 600 seconds (10 minutes)
        delay=5
        log_info "Using extended timeout for vLLM model loading (up to 10 minutes)"
    fi

    local start_time=$(date +%s)
    for ((i=1; i<=attempts; i++)); do
        if curl -fsS "$url" >/dev/null 2>&1; then
            local elapsed=$(( $(date +%s) - start_time ))
            log_info "$name healthy at $url (took ${elapsed}s)"
            
            # For vLLM, verify model is actually loaded
            if [[ "$name" == "vLLM" ]]; then
                local models_response
                models_response=$(curl -fsS "$url" 2>/dev/null || echo "")
                if [[ -z "$models_response" ]] || [[ "$models_response" == "null" ]] || [[ "$models_response" == "[]" ]]; then
                    log_warn "vLLM endpoint responded but no models loaded yet, continuing to wait..."
                    sleep "$delay"
                    continue
                fi
            fi
            
            return 0
        fi
        
        # Check if max timeout exceeded
        if [[ $max_timeout -gt 0 ]]; then
            local elapsed=$(( $(date +%s) - start_time ))
            if [[ $elapsed -ge $max_timeout ]]; then
                log_error "$name failed health checks at $url (timeout after ${elapsed}s)"
                return 1
            fi
        fi
        
        if [ $((i % 10)) -eq 0 ]; then
            local elapsed=$(( $(date +%s) - start_time ))
            log_warn "$name not ready yet (attempt $i/$attempts, elapsed: ${elapsed}s)"
        fi
        sleep "$delay"
    done

    local elapsed=$(( $(date +%s) - start_time ))
    log_error "$name failed health checks at $url after ${elapsed}s ($attempts attempts)"
    return 1
}

start_services() {
    log_section "Starting services"

    local manager="$ROOT/unified_service_manager.sh"
    local supervisor="$ROOT/supervisor.sh"

    if [[ -x "$manager" ]]; then
        "$manager" stop >/dev/null 2>&1 || true
        "$manager" start
    elif [[ -x "$supervisor" ]]; then
        "$supervisor" stop >/dev/null 2>&1 || true
        "$supervisor" start
    else
        log_error "No service manager found (expected unified_service_manager.sh or supervisor.sh)."
        return 1
    fi
}

health_check_services() {
    local vllm_url="${VLLM_ENDPOINT:-http://127.0.0.1:5001}"
    local qdrant_url="${QDRANT_URL:-http://127.0.0.1:6333}"
    local ollama_url="${OLLAMA_ENDPOINT:-http://127.0.0.1:11434}"
    local metrics_url="${METRICS_ENDPOINT:-http://127.0.0.1:9093/metrics}"

    log_section "Health Checking Services"
    
    # Qdrant health check (quick - should be ready in 10-30 seconds)
    check_http_health "Qdrant" "${qdrant_url%/}/health" 30 2 || {
        log_error "Qdrant health check failed - service may not be running"
        return 1
    }
    
    # vLLM health check (slow - model loading takes 2-5 minutes)
    check_http_health "vLLM" "${vllm_url%/}/v1/models" 120 5 || {
        log_error "vLLM health check failed - model may still be loading or service failed"
        log_info "Check vLLM logs for details: tail -f /tmp/vllm_coder.log"
        return 1
    }
    
    # Ollama health check (optional - only if using Ollama backend)
    if [[ "${CURATOR_BACKEND:-vllm}" == "ollama" ]]; then
        check_http_health "Ollama" "${ollama_url%/}/api/tags" 30 2 || {
            log_warn "Ollama health check failed (optional service)"
        }
    fi
    
    # Metrics endpoint (optional)
    if [[ -n "${METRICS_ENDPOINT:-}" ]]; then
        check_http_health "Metrics" "${metrics_url}" 10 2 || {
            log_warn "Metrics endpoint health check failed (optional)"
        }
    fi
    
    log_info "All critical services are healthy"
}

log_section "Niodoo RunPod Bootstrap"
log_info "Logging to $LOG_FILE"

if [[ $(id -u) -ne 0 ]]; then
    log_warn "Consider running as root for package installation."
fi

if [[ -f "$ROOT/tcs_runtime.env" ]]; then
    # shellcheck disable=SC1091
    set -a
    source "$ROOT/tcs_runtime.env"
    set +a
    log_info "Loaded runtime environment from tcs_runtime.env"
else
    log_warn "tcs_runtime.env not found; relying on default environment variables"
fi

if [[ $SKIP_PACKAGES -eq 0 ]]; then
    run_step "system-packages" "Install system packages" step_system_packages
else
    log_warn "System package installation skipped by flag"
fi

run_step "rust-toolchain" "Provision Rust toolchain" step_rust_toolchain
run_step "python-env" "Provision Python environment" step_python_env

if [[ $SKIP_MODEL_DOWNLOAD -eq 0 ]]; then
    run_step "model-assets" "Fetch model artifacts" step_prefetch_models
else
    log_warn "Model download skipped by flag"
fi

if [[ $SKIP_QDRANT -eq 0 ]]; then
    run_step "qdrant" "Provision Qdrant" step_qdrant
else
    log_warn "Qdrant provisioning skipped by flag"
fi

if [[ $SKIP_OLLAMA -eq 0 ]]; then
    run_step "ollama" "Provision Ollama" step_ollama
else
    log_warn "Ollama provisioning skipped by flag"
fi

run_step "rust-deps" "Prefetch Rust crates" step_rust_deps

if [[ $SKIP_BUILD -eq 0 ]]; then
    run_step "rust-build" "Build Niodoo (release)" step_rust_build
else
    log_warn "Rust build skipped by flag"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    log_info "GPU detected:"
    nvidia-smi
else
    log_warn "nvidia-smi not available; GPU metrics unavailable"
fi

if [[ $SKIP_SERVICES -eq 0 ]]; then
    start_services
    health_check_services
else
    log_warn "Service startup skipped by flag"
fi

log_section "Bootstrap complete"
log_info "Everything ready. You can now start workloads or connect to services."

