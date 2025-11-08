        #!/bin/bash
        # Auto-generated RunPod environment file
        # Source this file: source .runpod_env.sh

        export LD_LIBRARY_PATH="/workspace/Niodoo-Final/third_party/onnxruntime-linux-x64-gpu-1.24.0/lib:${LD_LIBRARY_PATH}"
        export ORT_STRICT_VERSION_CHECK=0
        export RUSTONIG_SYSTEM_LIBONIG=1
        export RUSTFLAGS="-C link-arg=-Wl,-rpath,/workspace/Niodoo-Final/third_party/onnxruntime-linux-x64-gpu-1.24.0/lib"

        # Protobuf environment (for ONNX Runtime and gRPC)
        export PROTOC=$(which protoc 2>/dev/null || echo "")
        export PROTOC_INCLUDE=$(pkg-config --variable=includedir protobuf 2>/dev/null || echo "")
        if [ -n "$PROTOC_INCLUDE" ]; then
            export PKG_CONFIG_PATH="$PROTOC_INCLUDE/pkgconfig:$PKG_CONFIG_PATH"
        fi

        if [ -d "/usr/local/cuda-13.0" ]; then
            export CUDA_HOME=/usr/local/cuda-13.0
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

        # Source cargo environment
        if [ -f "$HOME/.cargo/env" ]; then
            source "$HOME/.cargo/env"
        fi

        # Source project cargo environment
        if [ -f "/workspace/Niodoo-Final/.cargo_env.sh" ]; then
            source "/workspace/Niodoo-Final/.cargo_env.sh"
        fi
        EOF

        chmod +x "/workspace/Niodoo-Final/.runpod_env.sh"
        echo "✅ Environment file created at /workspace/Niodoo-Final/.runpod_env.sh"

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
        if [ -d "/workspace/Niodoo-Final/third_party/onnxruntime-linux-x64-gpu-1.24.0/lib" ]; then
            echo "✅ ONNX Runtime libraries found at /workspace/Niodoo-Final/third_party/onnxruntime-linux-x64-gpu-1.24.0/lib"
            ls -1 "/workspace/Niodoo-Final/third_party/onnxruntime-linux-x64-gpu-1.24.0/lib"/*.so* | wc -l | xargs echo "   Libraries:"
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
            PROTOC_VERSION=libprotoc 3.21.12
            echo "✅ Protobuf compiler: 3.21.12"
            PROTOC_MAJOR=3
            if [ "3" -ge 26 ]; then
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
        cd "/workspace/Niodoo-Final"

        # Source environment
        source "/workspace/Niodoo-Final/.runpod_env.sh"

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

