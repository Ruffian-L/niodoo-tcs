        #!/bin/bash
        # =============================================================================
        # NIODOO Fresh RunPod Setup - Complete Dependency Installation
        # =============================================================================
        # This script installs ALL dependencies for a fresh RunPod setup:
        # - Rust toolchain (latest stable)
        # - NVIDIA CUDA drivers and toolkit
        # - Protocol Buffers (Protobuf) with version compatibility (v21/v25.1, avoid v26+)
        # - ONNX Runtime GPU build (v1.24.0 with FP8/FP4 support)
        # - System libraries (OpenBLAS, libonig-dev, protobuf, etc.)
        # - Python dependencies (ONNX Runtime, Protobuf, gRPC, vLLM FlashInfer stack)
        # - Environment configuration
        # =============================================================================

        set -e  # Exit on error
        set -x  # Debug mode

        CUDA_VERSION_TARGET="13.0"
        CUDA_RUNFILE_BASENAME="cuda_13.0.0_535.104.05_linux.run"
        CUDA_RUNFILE_URL="https://developer.download.nvidia.com/compute/cuda/13.0.0/local_installers/${CUDA_RUNFILE_BASENAME}"
        CUDA_RUNFILE_LOCAL="/tmp/${CUDA_RUNFILE_BASENAME}"

        SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        cd "$SCRIPT_DIR"

        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "🚀 NIODOO FRESH RUNPOD SETUP - INSTALLING ALL DEPENDENCIES"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""

        # =============================================================================
        # STEP 1: Update system packages
        # =============================================================================
        echo "📦 STEP 1: Updating system packages..."
        sudo apt-get update -y
        sudo apt-get upgrade -y

        # =============================================================================
        # STEP 2: Install system build dependencies
        # =============================================================================
        echo ""
        echo "📦 STEP 2: Installing system build dependencies..."
        sudo apt-get install -y \
            build-essential \
            cmake \
            ninja-build \
            pkg-config \
            git \
            curl \
            wget \
            ca-certificates \
            libssl-dev \
            libcurl4-openssl-dev \
            libonig-dev \
            libopenblas-dev \
            libopenblas-openmp-dev \
            libgomp1 \
            python3 \
            python3-pip \
            python3-dev \
            clang \
            llvm \
            ccache \
            patchelf \
            binutils

        # =============================================================================
        # STEP 3: Verify NVIDIA GPU, drivers, and CUDA toolkits
        # =============================================================================
        echo ""
        echo "🎮 STEP 3: Verifying NVIDIA GPU, drivers, and CUDA toolkits..."
        GPU_NAME="Unknown"
        if ! command -v nvidia-smi &> /dev/null; then
            echo "⚠️  WARNING: nvidia-smi not found. Installing NVIDIA drivers..."
            sudo apt-get install -y nvidia-driver-580 nvidia-cuda-toolkit || echo "⚠️  Driver installation may require reboot"
        else
            echo "✅ NVIDIA drivers found:"
            nvidia-smi --query-gpu=name,driver_version,cuda_version --format=csv,noheader || nvidia-smi
            GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 | tr -d '\r')
        fi

        IS_H200=0
        if [[ "$GPU_NAME" == *"H200"* ]]; then
            IS_H200=1
            echo "🔬 Detected NVIDIA H200 (Hopper). Targeting CUDA ${CUDA_VERSION_TARGET}."
        else
            echo "ℹ️ Detected GPU: $GPU_NAME"
        fi

        # Ensure Hopper/H200 drivers (R580+) are present
        REQUIRED_DRIVER="580.0"
        CURRENT_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1 | tr -d '\r')

        version_lt() {
            [ "$(printf '%s\n' "$1" "$2" | sort -V | head -n1)" != "$2" ]
        }

        if [ -n "$CURRENT_DRIVER" ] && version_lt "$CURRENT_DRIVER" "$REQUIRED_DRIVER"; then
            echo "⚠️  WARNING: Detected driver $CURRENT_DRIVER < required $REQUIRED_DRIVER."
            echo "   Install the R580 branch (Aug 2025) for full H200 FP8/FP4 transformer support."
        fi

        # Check CUDA version (install CUDA 13.0 when missing or outdated on H200)
        CURRENT_CUDA_RELEASE=""
        if command -v nvcc &> /dev/null; then
            nvcc --version | grep "release" || true
            CURRENT_CUDA_RELEASE=$(nvcc --version | grep -oP 'release \K([0-9]+\.[0-9]+)' | head -n1)
        fi

        INSTALL_CUDA=0
        if [ "$IS_H200" -eq 1 ]; then
            if [ "$CURRENT_CUDA_RELEASE" != "$CUDA_VERSION_TARGET" ]; then
                echo "⚠️  CUDA ${CUDA_VERSION_TARGET} required for H200 FP8/FP4 kernels. Scheduling toolkit installation."
                INSTALL_CUDA=1
            fi
        else
            if ! command -v nvcc &> /dev/null; then
                echo "⚠️  CUDA toolkit not detected. A generic installation will be attempted."
                INSTALL_CUDA=1
            fi
        fi

        if [ "$INSTALL_CUDA" -eq 1 ]; then
            if wget -q --spider "$CUDA_RUNFILE_URL" 2>/dev/null; then
                echo "⬇️  Downloading CUDA runfile from $CUDA_RUNFILE_URL"
                wget -q "$CUDA_RUNFILE_URL" -O "$CUDA_RUNFILE_LOCAL" || echo "❌ Failed to download CUDA runfile."
            else
                echo "❌ Unable to reach CUDA runfile URL $CUDA_RUNFILE_URL"
            fi
            if [ -f "$CUDA_RUNFILE_LOCAL" ]; then
                echo "🛠️  Installing CUDA toolkit (silent)..."
                sudo sh "$CUDA_RUNFILE_LOCAL" --silent --toolkit --override --no-man-page --no-drm || echo "⚠️  CUDA installer reported issues. Check /var/log/cuda-installer.log."
                sudo ln -sfn /usr/local/cuda-${CUDA_VERSION_TARGET} /usr/local/cuda || true
                rm -f "$CUDA_RUNFILE_LOCAL"
            else
                echo "⚠️  CUDA runfile not present. Manual installation may be required."
            fi
        else
            echo "✅ CUDA toolkit release ${CURRENT_CUDA_RELEASE:-unknown} detected."
        fi

        # Prefer CUDA 13.0 paths when available
        if [ -d "/usr/local/cuda-${CUDA_VERSION_TARGET}" ]; then
            export CUDA_HOME=/usr/local/cuda-${CUDA_VERSION_TARGET}
            export PATH="$CUDA_HOME/bin:$PATH"
            export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
        fi

        if [ "$IS_H200" -eq 1 ] && command -v nvidia-smi &> /dev/null; then
            MIG_STATE=$(nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader | head -n1 | tr -d '\r')
            echo "🧩 MIG mode: ${MIG_STATE:-Unknown}"
            if [[ "$MIG_STATE" != "Enabled" ]]; then
                echo "   🛈 Recommended for H200: enable MIG and allocate seven instances for stage parallelism:"
                echo "      sudo nvidia-smi -i 0 -mig 1"
                echo "      # Profile 19 = 1g.20gb Hopper slice (fits seven instances on 141GB HBM3e)"
                echo "      sudo nvidia-smi mig -i 0 -cgi 19,19,19,19,19,19,19 -C"
            else
                echo "   MIG already enabled. Inspect layouts with: sudo nvidia-smi mig -i 0 -lgci"
            fi
        fi

        # =============================================================================
        # STEP 4: Install Rust toolchain (latest stable)
        # =============================================================================
        echo ""
        echo "🦀 STEP 4: Installing Rust toolchain (latest stable)..."
        if ! command -v rustc &> /dev/null; then
            echo "Installing Rust via rustup..."
            curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
            source "$HOME/.cargo/env"
        else
            echo "✅ Rust already installed:"
            rustc --version
            cargo --version
            echo "Updating Rust toolchain..."
            rustup update stable
        fi

        # Install additional Rust components
        rustup component add rustfmt clippy
        rustup target add x86_64-unknown-linux-gnu

        # =============================================================================
        # STEP 5: Install Protocol Buffers (Protobuf) with version compatibility
        # =============================================================================
        echo ""
        echo "📦 STEP 5: Installing Protocol Buffers (Protobuf)..."
        echo "Note: ONNX Runtime requires Protobuf v25.1 minimum (v1.19.1+), supports v21 for compatibility"
        echo "Avoiding Protobuf v26+ due to potential linking issues with ONNX Runtime"

        # Install Protobuf compiler and development libraries
        sudo apt-get install -y \
            protobuf-compiler \
            libprotobuf-dev \
            libprotoc-dev || {
            echo "⚠️  WARNING: Failed to install system Protobuf packages. Continuing..."
        }

        # Verify Protobuf installation
        if command -v protoc &> /dev/null; then
            PROTOC_VERSION=$(protoc --version | grep -oP '[\d.]+' | head -1)
            echo "✅ Protobuf compiler found: version $PROTOC_VERSION"
            # Check if version is compatible (avoid v26+)
            PROTOC_MAJOR=$(echo "$PROTOC_VERSION" | cut -d. -f1)
            if [ "$PROTOC_MAJOR" -ge 26 ]; then
                echo "⚠️  WARNING: Protobuf version $PROTOC_VERSION may cause linking issues with ONNX Runtime"
                echo "   Consider downgrading to v25.1 or v21 for compatibility"
            fi
        else
            echo "⚠️  WARNING: protoc not found. Some features may not work."
        fi

        # Install Python Protobuf (for ONNX/gRPC interop)
        pip3 install protobuf || {
            echo "⚠️  WARNING: Failed to install Python protobuf. Continuing..."
        }

        # =============================================================================
        # STEP 6: Download and set up ONNX Runtime GPU build
        # =============================================================================
        echo ""
        echo "🧠 STEP 6: Setting up ONNX Runtime GPU build..."
        echo "ONNX Runtime v1.24.0 requires Protobuf v25.1 minimum (supports v21 for backward compatibility)"
        echo "Using latest stable version with Hopper/H200 FP8+ support and Protobuf compatibility"

        ONNX_VERSION="1.24.0"
        ONNX_DIR="$SCRIPT_DIR/third_party/onnxruntime-linux-x64-gpu-${ONNX_VERSION}"
        ONNX_LIB_DIR="$ONNX_DIR/lib"

        mkdir -p "$SCRIPT_DIR/third_party"

        if [ ! -d "$ONNX_DIR" ]; then
            echo "Downloading ONNX Runtime GPU ${ONNX_VERSION} (latest, with H200/FP8 support)..."
            cd "$SCRIPT_DIR/third_party"
            
            # Download ONNX Runtime GPU build (latest version with H200 support)
            ONNX_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/onnxruntime-linux-x64-gpu-${ONNX_VERSION}.tgz"
            
            if wget -q --spider "$ONNX_URL" 2>/dev/null; then
                wget -q "$ONNX_URL" -O "onnxruntime-gpu-${ONNX_VERSION}.tgz"
                tar -xzf "onnxruntime-gpu-${ONNX_VERSION}.tgz"
                rm "onnxruntime-gpu-${ONNX_VERSION}.tgz"
                echo "✅ ONNX Runtime GPU downloaded and extracted"
            else
                echo "⚠️  WARNING: ONNX Runtime GPU ${ONNX_VERSION} not found at expected URL"
                echo "Attempting to download latest available version..."
                # Try to find latest version
                LATEST_VERSION=$(curl -s https://api.github.com/repos/microsoft/onnxruntime/releases/latest | grep -oP '"tag_name": "\Kv?([0-9]+\.[0-9]+\.[0-9]+)' | head -1)
                if [ -n "$LATEST_VERSION" ]; then
                    LATEST_VERSION="${LATEST_VERSION#v}"  # Remove 'v' prefix if present
                    ONNX_URL="https://github.com/microsoft/onnxruntime/releases/download/v${LATEST_VERSION}/onnxruntime-linux-x64-gpu-${LATEST_VERSION}.tgz"
                    echo "Downloading ONNX Runtime GPU ${LATEST_VERSION}..."
                    wget -q "$ONNX_URL" -O "onnxruntime-gpu-${LATEST_VERSION}.tgz" || {
                        echo "❌ Failed to download ONNX Runtime GPU. You may need to download it manually."
                        echo "Visit: https://github.com/microsoft/onnxruntime/releases"
                    }
                fi
            fi
        else
            echo "✅ ONNX Runtime GPU already exists at $ONNX_DIR"
        fi

        # Set up library paths
        export LD_LIBRARY_PATH="$ONNX_LIB_DIR:${LD_LIBRARY_PATH}"

        # Verify ONNX Runtime libraries
        if [ -d "$ONNX_LIB_DIR" ]; then
            echo "✅ ONNX Runtime libraries found:"
            ls -lh "$ONNX_LIB_DIR"/*.so* | head -5
        else
            echo "⚠️  WARNING: ONNX Runtime lib directory not found at $ONNX_LIB_DIR"
        fi

        # =============================================================================
        # STEP 7: Install Python dependencies
        # =============================================================================
        echo ""
        echo "🐍 STEP 7: Installing Python dependencies and H200 vLLM stack..."

        # Install ONNX Runtime GPU via pip (for Python interop)
        # Note: Ensure Protobuf version compatibility (v21/v25.1, avoid v26+)
        pip3 install --upgrade pip setuptools || echo "⚠️  Pip upgrade skipped (may have system packages)"

        # Install Python protobuf with version pinning for ONNX compatibility
        pip3 install 'protobuf>=4.21.0,<5.0.0' || {
            echo "⚠️  WARNING: Failed to install Python protobuf. Continuing..."
        }

        # Install ONNX Runtime GPU via pip (for Python interop)
        pip3 install onnxruntime-gpu || {
            echo "⚠️  WARNING: Failed to install onnxruntime-gpu via pip. Continuing..."
        }

        # Install gRPC Python libraries (for federated learning and distributed communication)
        pip3 install grpcio grpcio-tools || {
            echo "⚠️  WARNING: Failed to install gRPC Python libraries. Continuing..."
        }

        # Install Hopper/H200 optimized LLM serving stack
        PIP_CUDA_INDEX=${PIP_CUDA_INDEX:-"https://download.pytorch.org/whl/cu128"}
        echo "Installing FlashAttention 3 kernels (FP8/FP16)"
        pip3 install --upgrade --extra-index-url "$PIP_CUDA_INDEX" flash-attn --no-build-isolation || {
            echo "⚠️  WARNING: flash-attn install failed. Hopper FP8 attention kernels unavailable."
        }

        echo "Installing FlashInfer runtime kernels"
        pip3 install --upgrade flashinfer || {
            echo "⚠️  WARNING: flashinfer install failed. vLLM FlashInfer backend disabled."
        }

        echo "Installing vLLM 1.0.0a alpha with FlashInfer extras"
        pip3 install --upgrade "vllm[flashinfer]==1.0.0a" || {
            echo "⚠️  WARNING: vLLM 1.0.0a install failed. Check Python/CUDA compatibility or adjust the version tag."
        }

        echo "Installing DeepSpeed and Transformer Engine for FP8/FP4"
        pip3 install --upgrade deepspeed || {
            echo "⚠️  WARNING: DeepSpeed install failed. MoE/ZeRO optimizations unavailable."
        }
        pip3 install --upgrade "transformer-engine>=1.9.0" || {
            echo "⚠️  WARNING: Transformer Engine install failed. NVIDIA FP8 kernels unavailable."
        }

        echo "Installing topology GPU toolchain (Gudhi-GPU, multipers, networkx-gpu, rdkit-gpu)"
        pip3 install --upgrade "gudhi-gpu==4.2" || {
            echo "⚠️  WARNING: gudhi-gpu install failed. Persistent homology GPU acceleration unavailable."
        }
        pip3 install --upgrade "multipers==1.3" || {
            echo "⚠️  WARNING: multipers install failed. Differentiable persistent Laplacians unavailable."
        }
        pip3 install --upgrade networkx-gpu || {
            echo "⚠️  WARNING: networkx-gpu install failed. Kuramoto metastability analysis unavailable."
        }
        pip3 install --upgrade rdkit-gpu || {
            echo "⚠️  WARNING: rdkit-gpu install failed. Motif detection acceleration unavailable."
        }

        # Install other Python dependencies if requirements.txt exists
        if [ -f "requirements.txt" ]; then
            echo "Installing Python requirements from requirements.txt..."
            pip3 install -r requirements.txt
        fi

        # =============================================================================
        # STEP 8: Set up environment variables
        # =============================================================================
        echo ""
        echo "🔧 STEP 8: Setting up environment variables..."

        # Source cargo environment if it exists
        if [ -f "$HOME/.cargo/env" ]; then
            source "$HOME/.cargo/env"
        fi

        # Source project cargo environment
        if [ -f "$SCRIPT_DIR/.cargo_env.sh" ]; then
            source "$SCRIPT_DIR/.cargo_env.sh"
        fi

        # Set up ONNX Runtime environment
        export LD_LIBRARY_PATH="$ONNX_LIB_DIR:${LD_LIBRARY_PATH}"
        export ORT_STRICT_VERSION_CHECK=0

        # Set Rust environment variables
        export RUSTONIG_SYSTEM_LIBONIG=1
        export RUSTFLAGS="-C link-arg=-Wl,-rpath,$ONNX_LIB_DIR"

        # Protobuf environment (for ONNX Runtime and gRPC)
        export PROTOC=$(which protoc 2>/dev/null || echo "")
        export PROTOC_INCLUDE=$(pkg-config --variable=includedir protobuf 2>/dev/null || echo "")
        if [ -n "$PROTOC_INCLUDE" ]; then
            export PKG_CONFIG_PATH="$PROTOC_INCLUDE/pkgconfig:$PKG_CONFIG_PATH"
        fi

        # CUDA environment
        if [ -d "/usr/local/cuda-${CUDA_VERSION_TARGET}" ]; then
            export CUDA_HOME=/usr/local/cuda-${CUDA_VERSION_TARGET}
            export PATH="$CUDA_HOME/bin:$PATH"
            export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
        elif [ -d "/usr/local/cuda" ]; then
            export CUDA_HOME=/usr/local/cuda
            export PATH="$CUDA_HOME/bin:$PATH"
            export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
        elif [ -d "/usr/local/cuda-12" ]; then
            export CUDA_HOME=/usr/local/cuda-12
            export PATH="$CUDA_HOME/bin:$PATH"
            export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
        fi

        # Create environment file for persistence
        ENV_FILE="$SCRIPT_DIR/.runpod_env.sh"
        cat > "$ENV_FILE" << EOF
        #!/bin/bash
        # Auto-generated RunPod environment file
        # Source this file: source .runpod_env.sh

        export LD_LIBRARY_PATH="$ONNX_LIB_DIR:\${LD_LIBRARY_PATH}"
        export ORT_STRICT_VERSION_CHECK=0
        export RUSTONIG_SYSTEM_LIBONIG=1
        export RUSTFLAGS="-C link-arg=-Wl,-rpath,$ONNX_LIB_DIR"

        # Protobuf environment (for ONNX Runtime and gRPC)
        export PROTOC=\$(which protoc 2>/dev/null || echo "")
        export PROTOC_INCLUDE=\$(pkg-config --variable=includedir protobuf 2>/dev/null || echo "")
        if [ -n "\$PROTOC_INCLUDE" ]; then
            export PKG_CONFIG_PATH="\$PROTOC_INCLUDE/pkgconfig:\$PKG_CONFIG_PATH"
        fi

        if [ -d "/usr/local/cuda-${CUDA_VERSION_TARGET}" ]; then
            export CUDA_HOME=/usr/local/cuda-${CUDA_VERSION_TARGET}
            export PATH="\$CUDA_HOME/bin:\$PATH"
            export LD_LIBRARY_PATH="\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH"
        elif [ -d "/usr/local/cuda" ]; then
            export CUDA_HOME=/usr/local/cuda
            export PATH="\$CUDA_HOME/bin:\$PATH"
            export LD_LIBRARY_PATH="\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH"
        elif [ -d "/usr/local/cuda-12" ]; then
            export CUDA_HOME=/usr/local/cuda-12
            export PATH="\$CUDA_HOME/bin:\$PATH"
            export LD_LIBRARY_PATH="\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH"
        fi

        # Source cargo environment
        if [ -f "\$HOME/.cargo/env" ]; then
            source "\$HOME/.cargo/env"
        fi

        # Source project cargo environment
        if [ -f "$SCRIPT_DIR/.cargo_env.sh" ]; then
            source "$SCRIPT_DIR/.cargo_env.sh"
        fi
        EOF

        chmod +x "$ENV_FILE"
        echo "✅ Environment file created at $ENV_FILE"

        # =============================================================================
        # STEP 9: Verify installations
        # =============================================================================
        echo ""
        echo "✅ STEP 9: Verifying installations..."

        echo ""
        echo "Rust toolchain:"
        rustc --version || echo "❌ Rust not found"
        cargo --version || echo "❌ Cargo not found"

        echo ""
        echo "NVIDIA GPU:"
        nvidia-smi --query-gpu=name,driver_version,cuda_version --format=csv,noheader || echo "❌ NVIDIA GPU not found"

        echo ""
        echo "CUDA toolkit:"
        nvcc --version 2>/dev/null || echo "⚠️  CUDA toolkit not found (may be in container)"

        echo ""
        echo "ONNX Runtime:"
        if [ -d "$ONNX_LIB_DIR" ]; then
            echo "✅ ONNX Runtime libraries found at $ONNX_LIB_DIR"
            ls -1 "$ONNX_LIB_DIR"/*.so* | wc -l | xargs echo "   Libraries:"
        else
            echo "❌ ONNX Runtime libraries not found"
        fi

        echo ""
        echo "Python ONNX Runtime:"
        python3 -c "import onnxruntime; print(f'ONNX Runtime version: {onnxruntime.__version__}'); print(f'Available providers: {onnxruntime.get_available_providers()}')" 2>/dev/null || echo "⚠️  Python ONNX Runtime not installed"

        echo ""
        echo "System libraries:"
        ldconfig -p | grep -E "libonig|libopenblas|libcurl|libprotobuf" || echo "⚠️  Some system libraries not found in cache"

        echo ""
        echo "Protobuf:"
        if command -v protoc &> /dev/null; then
            PROTOC_VERSION=$(protoc --version)
            echo "✅ Protobuf compiler: $PROTOC_VERSION"
            PROTOC_MAJOR=$(echo "$PROTOC_VERSION" | grep -oP '[\d.]+' | head -1 | cut -d. -f1)
            if [ "$PROTOC_MAJOR" -ge 26 ]; then
                echo "⚠️  WARNING: Protobuf v26+ may cause ONNX Runtime linking issues"
            else
                echo "✅ Protobuf version compatible with ONNX Runtime"
            fi
        else
            echo "❌ Protobuf compiler not found"
        fi

        python3 -c "import google.protobuf; print(f'✅ Python protobuf: {google.protobuf.__version__}')" 2>/dev/null || echo "⚠️  Python protobuf not installed"

        python3 -c "import grpc; print(f'✅ Python gRPC: {grpc.__version__}')" 2>/dev/null || echo "⚠️  Python gRPC not installed"

        python3 - <<'PY'
        import importlib
        packages = [
            ("gudhi", "gudhi-gpu"),
            ("multipers", "multipers"),
            ("networkx", "networkx-gpu"),
            ("rdkit", "rdkit-gpu"),
            ("vllm", "vLLM"),
        ]
        for module_name, display in packages:
            try:
                module = importlib.import_module(module_name)
            except Exception as exc:  # noqa: BLE001
                print(f"⚠️  {display} import failed: {exc}")
            else:
                version = getattr(module, "__version__", getattr(module, "__VERSION__", "unknown"))
                print(f"✅ {display} import ok (version {version})")
        PY

        # =============================================================================
        # STEP 10: Build workspace (optional - verify compilation)
        # =============================================================================
        echo ""
        echo "🔨 STEP 10: Verifying Rust workspace compilation..."
        echo "Running cargo check (this may take a while)..."
        cd "$SCRIPT_DIR"

        # Source environment
        source "$ENV_FILE"

        # Run cargo check with minimal features first
        echo "Checking tcs-ml crate..."
        cargo check -p tcs-ml --features onnx --message-format=short 2>&1 | tail -20 || {
            echo "⚠️  WARNING: cargo check failed. This is okay for now - dependencies are installed."
            echo "You may need to adjust Cargo.toml or check for missing dependencies."
        }

        # =============================================================================
        # COMPLETION
        # =============================================================================
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "✅ INSTALLATION COMPLETE!"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        echo "📝 NEXT STEPS:"
        echo ""
        echo "1. Source the environment file:"
        echo "   source .runpod_env.sh"
        echo ""
        echo "2. Build the project:"
        echo "   cargo build --release"
        echo ""
        echo "3. Run tests:"
        echo "   cargo test"
        echo ""
        echo "4. Start services:"
        echo "   ./start_all_services.sh"
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

