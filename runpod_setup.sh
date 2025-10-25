#!/bin/bash
# ============================================================================
# Niodoo-TCS: H200 RunPod Setup Script (2025)
# ============================================================================
# One-shot setup for H200 GPU acceleration
# Install Rust, Python deps, and compile the full stack

set -e

echo "🚀 Niodoo-TCS H200 RunPod Setup"
echo "================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: System packages
echo -e "${YELLOW}[1/7] Installing system dependencies...${NC}"
apt-get update
apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    libssl-dev \
    curl \
    wget \
    git \
    python3.11 \
    python3.11-dev \
    python3-pip \
    libopenblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    gfortran

echo -e "${GREEN}✓ System packages installed${NC}"

# Step 2: Rust toolchain
echo -e "${YELLOW}[2/7] Installing Rust toolchain...${NC}"
if ! command -v cargo &> /dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source $HOME/.cargo/env
fi
rustup update
rustup component add rustfmt clippy

echo -e "${GREEN}✓ Rust installed${NC}"

# Step 3: CUDA & cuDNN setup
echo -e "${YELLOW}[3/7] Configuring CUDA environment...${NC}"
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH

# Verify CUDA
if command -v nvcc &> /dev/null; then
    nvcc --version
    echo -e "${GREEN}✓ CUDA detected${NC}"
else
    echo -e "${RED}⚠ CUDA not found - GPU features may not work${NC}"
fi

# Step 4: Python environment
echo -e "${YELLOW}[4/7] Setting up Python environment...${NC}"
python3.11 -m venv /workspace/Niodoo-Final/venv
source /workspace/Niodoo-Final/venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.1 support
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121

echo -e "${GREEN}✓ Python environment ready${NC}"

# Step 5: Python dependencies
echo -e "${YELLOW}[5/7] Installing Python dependencies...${NC}"
pip install -r /workspace/Niodoo-Final/requirements.txt

echo -e "${GREEN}✓ Python packages installed${NC}"

# Step 6: Rust dependencies (check + update)
echo -e "${YELLOW}[6/7] Checking Rust dependencies...${NC}"
cd /workspace/Niodoo-Final
cargo fetch

echo -e "${GREEN}✓ Rust dependencies resolved${NC}"

# Step 7: Build (release mode for H200)
echo -e "${YELLOW}[7/7] Building Niodoo for H200...${NC}"
cargo build --release --all --features "onnx" 2>&1 | tee build.log

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Build successful!${NC}"
else
    echo -e "${RED}✗ Build failed - check build.log${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}🎉 Setup Complete!${NC}"
echo -e "${GREEN}================================${NC}"
echo ""
echo "Next steps:"
echo "1. Activate venv: source venv/bin/activate"
echo "2. Run tests: cargo test --all"
echo "3. Start server: cargo run --release --bin niodoo-consciousness"
echo ""
echo "Environment:"
echo "  - Rust: $(rustc --version)"
echo "  - Python: $(python --version)"
echo "  - CUDA: $CUDA_HOME"
echo "  - GPU: $(nvidia-smi -L 2>/dev/null || echo 'Not detected')"


