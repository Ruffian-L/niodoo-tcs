#!/bin/bash
# SSH Cheatsheet for Niodoo-TCS Cluster
# Source this file or copy-paste commands as needed

# ==============================================================================
# SSH CONNECTIONS
# ==============================================================================

# Connect to Beelink (Architect - RTX A6000 48GB)
alias ssh-beelink='ssh -i ~/.ssh/temp_beelink_key beelink@100.113.10.90'

# Connect to Oldlaptop (Worker - Intel Ultra 5)
alias ssh-worker='ssh -i ~/.ssh/id_oldlaptop oldlaptop@100.119.255.24'

# Test Beelink connection
alias test-beelink='ssh -i ~/.ssh/temp_beelink_key beelink@100.113.10.90 "whoami"'

# ==============================================================================
# TAILSCALE IPS
# ==============================================================================

# Beelink:    100.113.10.90
# Laptop:     100.126.84.41
# Oldlaptop:  100.119.255.24

# ==============================================================================
# GITEA (PRIVATE GIT)
# ==============================================================================

# Web UI: http://100.113.10.90:3000
# SSH Port: 222
# SSH Key: ~/.ssh/gitea_beelink

# Clone via Gitea SSH
# git clone ssh://git@100.113.10.90:222/username/repo.git

# ==============================================================================
# REMOTE CLAUDE EXECUTION (CLAUDEBALLS)
# ==============================================================================

# Run Claude task on Beelink (Haiku 4.5 - fast & cheap)
run-remote-claude() {
    ssh beelink "PATH=~/.npm-global/bin:\$PATH claude -p --dangerously-skip-permissions --model claude-haiku-4-5 '$1'"
}

# Usage:
# run-remote-claude "Analyze this code and find bugs"

# ==============================================================================
# PROJECT SETUP ON REMOTE
# ==============================================================================

# Full setup on Beelink
setup-beelink() {
    ssh-beelink << 'EOF'
cd ~/Desktop/Niodoo-Final || cd ~/Niodoo-Final || { echo "Project not found"; exit 1; }
export QWEN_MODEL_PATH="$(pwd)/models/qwen2.5-coder/model_quantized.onnx"
export RUSTONIG_SYSTEM_LIBONIG=1
export LD_LIBRARY_PATH="$(pwd)/third_party/onnxruntime-linux-x64-1.18.1/lib:$LD_LIBRARY_PATH"
echo "Environment set. Ready to build."
cargo check --all
EOF
}

# ==============================================================================
# QUICK COMMANDS
# ==============================================================================

# Run smoke test on Beelink
test-remote() {
    ssh-beelink "cd ~/Desktop/Niodoo-Final && cargo run -p tcs-ml --bin test_qwen_stateful --features onnx-with-tokenizers"
}

# Build on Beelink (background, show progress)
build-remote() {
    ssh-beelink "cd ~/Desktop/Niodoo-Final && cargo build --release --all"
}

# Check GPU status on Beelink
check-gpu() {
    ssh-beelink "nvidia-smi"
}

# ==============================================================================
# COPY FILES TO/FROM BEELINK
# ==============================================================================

# Copy file TO Beelink
copy-to-beelink() {
    scp -i ~/.ssh/temp_beelink_key "$1" beelink@100.113.10.90:"$2"
}

# Copy file FROM Beelink
copy-from-beelink() {
    scp -i ~/.ssh/temp_beelink_key beelink@100.113.10.90:"$1" "$2"
}

# Usage:
# copy-to-beelink local_file.rs ~/Niodoo-Final/src/
# copy-from-beelink ~/Niodoo-Final/target/release/binary ./

# ==============================================================================
# SYNCTHING STATUS
# ==============================================================================

# Check Syncthing status (usually http://localhost:8384)
# All nodes sync automatically via Tailscale mesh

# ==============================================================================
# QUICK DIAGNOSTICS
# ==============================================================================

# Full cluster status
cluster-status() {
    echo "=== BEELINK (Architect) ==="
    ssh -i ~/.ssh/temp_beelink_key beelink@100.113.10.90 "uname -a && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader" 2>/dev/null || echo "UNREACHABLE"
    
    echo ""
    echo "=== OLDLAPTOP (Worker) ==="
    ssh -i ~/.ssh/id_oldlaptop oldlaptop@100.119.255.24 "uname -a" 2>/dev/null || echo "UNREACHABLE"
    
    echo ""
    echo "=== LAPTOP (Developer) ==="
    uname -a
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "No NVIDIA GPU"
}

# ==============================================================================
# ENVIRONMENT EXPORT (for local use)
# ==============================================================================

export-niodoo-env() {
    export QWEN_MODEL_PATH="/home/ruffian/Desktop/Niodoo-Final/models/qwen2.5-coder/model_quantized.onnx"
    export RUSTONIG_SYSTEM_LIBONIG=1
    export LD_LIBRARY_PATH="/home/ruffian/Desktop/Niodoo-Final/third_party/onnxruntime-linux-x64-1.18.1/lib:$LD_LIBRARY_PATH"
    echo "✅ Niodoo environment exported"
}

# ==============================================================================
# HELP
# ==============================================================================

niodoo-help() {
    echo "Niodoo-TCS Cluster Commands:"
    echo ""
    echo "Connections:"
    echo "  ssh-beelink        - Connect to Beelink (A6000 GPU)"
    echo "  ssh-worker         - Connect to Oldlaptop"
    echo "  test-beelink       - Test Beelink connection"
    echo ""
    echo "Remote Execution:"
    echo "  run-remote-claude  - Run Claude task on Beelink"
    echo "  test-remote        - Run smoke tests on Beelink"
    echo "  build-remote       - Build project on Beelink"
    echo "  check-gpu          - Check Beelink GPU status"
    echo ""
    echo "Setup:"
    echo "  setup-beelink      - Configure environment on Beelink"
    echo "  export-niodoo-env  - Export env vars locally"
    echo ""
    echo "Diagnostics:"
    echo "  cluster-status     - Check all nodes"
    echo ""
    echo "File Transfer:"
    echo "  copy-to-beelink <local> <remote>"
    echo "  copy-from-beelink <remote> <local>"
}

# ==============================================================================
# AUTO-HELP ON SOURCE
# ==============================================================================

echo "✅ Niodoo-TCS cluster commands loaded!"
echo "Run 'niodoo-help' for command list"
echo ""
echo "Quick start:"
echo "  ssh-beelink          # Connect to GPU box"
echo "  test-beelink         # Test connection"
echo "  export-niodoo-env    # Set local env vars"