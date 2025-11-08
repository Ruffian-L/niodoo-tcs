#!/bin/bash
# Comprehensive RunPod setup script - A100 environment
# Installs dependencies, builds workspace, starts services

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

echo "🔧 NIODOO RunPod Full Setup"
echo "============================"
echo ""

# 1. Source environment
echo "📦 Loading environment..."
if [ -f "${ROOT_DIR}/.runpod_env.sh" ]; then
    source "${ROOT_DIR}/.runpod_env.sh"
fi
if [ -f "${ROOT_DIR}/config/a100.env" ]; then
    source "${ROOT_DIR}/config/a100.env"
    echo "✓ A100 environment loaded"
fi

# 2. Install Rust if needed
if ! command -v cargo &> /dev/null; then
    echo ""
    echo "🔨 Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
    source "$HOME/.cargo/env"
    echo "✓ Rust installed"
else
    echo "✓ Rust already installed: $(rustc --version)"
fi

# 3. Activate Python venv
echo ""
echo "🐍 Setting up Python environment..."
if [ -d "${ROOT_DIR}/venv" ]; then
    source "${ROOT_DIR}/venv/bin/activate"
    echo "✓ Virtual environment activated"
else
    echo "⚠️  No venv found at ${ROOT_DIR}/venv"
    echo "   Creating new venv..."
    python3 -m venv "${ROOT_DIR}/venv"
    source "${ROOT_DIR}/venv/bin/activate"
    echo "✓ Virtual environment created"
fi

# 4. Install Python dependencies
echo ""
echo "📦 Installing Python dependencies..."
pip install --upgrade pip setuptools wheel

# Core ML dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 || \
    pip install torch torchvision torchaudio

pip install transformers>=4.44.0 accelerate>=0.30.0 peft>=0.10.0 trl>=0.10.0 datasets>=3.2.0

# vLLM for serving
pip install vllm>=0.4.0

# Topology dependencies
if [ -f "${ROOT_DIR}/Niodoo-TCT/requirements.txt" ]; then
    echo "  Installing Niodoo-TCT dependencies..."
    pip install -r "${ROOT_DIR}/Niodoo-TCT/requirements.txt"
fi

if [ -f "${ROOT_DIR}/Niodoo-AI/requirements.txt" ]; then
    echo "  Installing Niodoo-AI dependencies..."
    pip install -r "${ROOT_DIR}/Niodoo-AI/requirements.txt"
fi

# Additional topology tools
pip install ripser sentence-transformers scikit-learn

# Hugging Face tools
pip install huggingface-hub hf-transfer

echo "✓ Python dependencies installed"

# 5. Install Cargo dependencies and build
echo ""
echo "🔨 Building Rust workspace..."
if command -v cargo &> /dev/null; then
    cd "${ROOT_DIR}"
    
    # Set cargo target directory
    export CARGO_TARGET_DIR="${ROOT_DIR}/target"
    mkdir -p "${CARGO_TARGET_DIR}"
    
    # Build with GPU features
    echo "  Building with GPU features..."
    cargo build --release --features gpu || {
        echo "⚠️  GPU build failed, trying without GPU features..."
        cargo build --release
    }
    
    echo "✓ Rust workspace built"
else
    echo "⚠️  Cargo not available, skipping Rust build"
fi

# 6. Verify services
echo ""
echo "🔍 Verifying services..."
if command -v vllm &> /dev/null; then
    echo "✓ vLLM: $(vllm --version 2>/dev/null | tail -n1 || echo 'installed')"
else
    echo "⚠️  vLLM not found"
fi

if command -v qdrant &> /dev/null; then
    echo "✓ Qdrant: $(qdrant --version 2>/dev/null || echo 'installed')"
else
    echo "⚠️  Qdrant not found"
fi

# 7. Start services
echo ""
echo "🚀 Starting services..."
cd "${ROOT_DIR}"
export HARDWARE=a100
./start_all_services.sh --hardware a100

echo ""
echo "✅ Setup complete!"
echo ""
echo "Services status:"
echo "  - vLLM: http://${VLLM_HOST:-127.0.0.1}:${VLLM_PORT:-5001}"
echo "  - Qdrant: ${QDRANT_URL:-http://127.0.0.1:6333}"
echo ""
echo "Monitor GPU: watch -n 1 nvidia-smi"
echo "Check vLLM: curl http://${VLLM_HOST:-127.0.0.1}:${VLLM_PORT:-5001}/v1/models"
