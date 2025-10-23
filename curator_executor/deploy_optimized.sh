#!/bin/bash
# Optimized Deployment Script for NIODOO-TCS Framework
# Based on 2025 performance analysis recommendations

set -e

# Installation directory - can be overridden with INSTALL_DIR env var
# Defaults to per-user directory for portability
INSTALL_DIR="${INSTALL_DIR:-$HOME/.niodoo-tcs}"

echo "🚀 NIODOO-TCS Optimized Deployment"
echo "===================================="

# Detect hardware
if nvidia-smi | grep -q "RTX 5080"; then
    HARDWARE="laptop"
    GPU_NAME="RTX 5080-Q"
    POWER_LIMIT=150  # 150W TGP cap per analysis
    MEMORY_LIMIT="16GB"
    KV_CACHE_SIZE=256000
    BATCH_SIZE=2
    echo "💻 Detected: Laptop with RTX 5080-Q"
else
    HARDWARE="beelink"
    GPU_NAME="Quadro RTX 6000"
    POWER_LIMIT=260  # 260W TDP
    MEMORY_LIMIT="24GB"
    KV_CACHE_SIZE=128000
    BATCH_SIZE=4
    echo "🖥️ Detected: Beelink server with Quadro RTX 6000"
fi

# Set power limits to prevent thermal throttling
echo "⚡ Setting GPU power limit to ${POWER_LIMIT}W..."
sudo nvidia-smi -pl $POWER_LIMIT

# Check CUDA version
echo "🔍 Checking CUDA compatibility..."
if [ "$HARDWARE" = "laptop" ]; then
    # RTX 5080 needs CUDA 12.8 for Blackwell
    REQUIRED_CUDA="12.8"
else
    # Quadro RTX 6000 works with CUDA 11.8
    REQUIRED_CUDA="11.8"
fi

CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d'.' -f1-2)
echo "   Current CUDA: $CUDA_VERSION"
echo "   Required: $REQUIRED_CUDA"

# Configure environment variables
export HARDWARE_TYPE=$HARDWARE
export VLLM_ENDPOINT="http://localhost:8000"
export QDRANT_URL="http://localhost:6334"  # gRPC port for 15% performance boost
export KV_CACHE_SIZE=$KV_CACHE_SIZE
export BATCH_SIZE=$BATCH_SIZE

# Start Qdrant with optimized settings
echo "📦 Starting Qdrant with hyperspherical indexing..."
docker run -d \
    --name qdrant \
    --restart unless-stopped \
    -p 6333:6333 \
    -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage \
    -e QDRANT__SERVICE__GRPC_PORT=6334 \
    -e QDRANT__STORAGE__OPTIMIZERS__MEMMAP_THRESHOLD_KB=1024 \
    -e QDRANT__STORAGE__HNSW__M=16 \
    -e QDRANT__STORAGE__HNSW__EF_CONSTRUCT=100 \
    qdrant/qdrant:latest

# Wait for Qdrant
echo "⏳ Waiting for Qdrant to initialize..."
sleep 5

# Configure vLLM based on hardware
echo "🤖 Starting vLLM with hardware-optimized settings..."

if [ "$HARDWARE" = "laptop" ]; then
    # RTX 5080-Q configuration (150 tokens/s target)
    docker run -d \
        --name vllm \
        --restart unless-stopped \
        --gpus all \
        -v ~/.cache/huggingface:/root/.cache/huggingface \
        -p 8000:8000 \
        --ipc=host \
        vllm/vllm-openai:latest \
        --model Qwen/Qwen2.5-7B-Instruct-AWQ \
        --quantization awq \
        --max-model-len $KV_CACHE_SIZE \
        --gpu-memory-utilization 0.85 \
        --enforce-eager \
        --tensor-parallel-size 1 \
        --max-num-seqs 32 \
        --max-num-batched-tokens 4096
else
    # Quadro RTX 6000 configuration (60 tokens/s stable)
    docker run -d \
        --name vllm \
        --restart unless-stopped \
        --gpus all \
        -v ~/.cache/huggingface:/root/.cache/huggingface \
        -p 8000:8000 \
        --ipc=host \
        vllm/vllm-openai:latest \
        --model Qwen/Qwen2.5-7B-Instruct-AWQ \
        --quantization awq \
        --max-model-len $KV_CACHE_SIZE \
        --gpu-memory-utilization 0.9 \
        --tensor-parallel-size 1 \
        --max-num-seqs 64 \
        --max-num-batched-tokens 8192 \
        --enable-prefix-caching
fi

echo "⏳ Waiting for vLLM to load models..."
sleep 30

# Build the optimized curator-executor
echo "🔨 Building curator-executor with optimizations..."
cd "$INSTALL_DIR"
cargo build --release --features optimized

# Create systemd service with hardware-specific limits
echo "📝 Creating systemd service..."
sudo tee /etc/systemd/system/curator-executor-optimized.service > /dev/null <<EOF
[Unit]
Description=NIODOO-TCS Curator-Executor (Optimized)
After=network.target docker.service

[Service]
Type=simple
User=$USER
WorkingDirectory=$INSTALL_DIR
Environment="RUST_LOG=info"
Environment="HARDWARE_TYPE=$HARDWARE"
Environment="VLLM_ENDPOINT=http://localhost:8000"
Environment="QDRANT_URL=http://localhost:6334"
Environment="KV_CACHE_SIZE=$KV_CACHE_SIZE"
Environment="BATCH_SIZE=$BATCH_SIZE"
ExecStart=$INSTALL_DIR/target/release/curator_executor
Restart=on-failure
RestartSec=10

# Resource limits based on hardware
MemoryMax=$MEMORY_LIMIT
CPUQuota=200%
TasksMax=4096

# GPU access
SupplementaryGroups=video render

[Install]
WantedBy=multi-user.target
EOF

# Enable and start the service
sudo systemctl daemon-reload
sudo systemctl enable curator-executor-optimized
sudo systemctl restart curator-executor-optimized

# Monitor initial performance
echo "📊 Monitoring initial performance..."
sleep 5
sudo journalctl -u curator-executor-optimized -n 50 --no-pager

# Create performance monitoring script
cat > monitor_performance.sh <<'EOF'
#!/bin/bash
echo "📊 NIODOO-TCS Performance Monitor"
echo "=================================="

while true; do
    # GPU metrics
    echo -e "\n🎮 GPU Status:"
    nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw \
        --format=csv,noheader,nounits | column -t -s ','
    
    # Service status
    echo -e "\n📦 Service Status:"
    systemctl status curator-executor-optimized --no-pager | grep -E "Active:|Memory:|CPU:"
    
    # Qdrant metrics
    echo -e "\n💾 Qdrant Collections:"
    curl -s http://localhost:6333/collections | jq '.result[].name' 2>/dev/null || echo "Qdrant not responding"
    
    # vLLM health
    echo -e "\n🤖 vLLM Health:"
    curl -s http://localhost:8000/health | jq '.' 2>/dev/null || echo "vLLM not responding"
    
    # Recent logs
    echo -e "\n📜 Recent Activity:"
    sudo journalctl -u curator-executor-optimized -n 5 --no-pager | grep -E "Task|Coherence|Performance"
    
    sleep 10
done
EOF
chmod +x monitor_performance.sh

echo "✅ Deployment complete!"
echo ""
echo "📊 Performance Targets (per 2025 analysis):"
echo "   - Token Generation: ${GPU_NAME} @ $([ "$HARDWARE" = "laptop" ] && echo "150" || echo "60") tokens/s"
echo "   - KV Cache: ${KV_CACHE_SIZE} tokens"
echo "   - Batch Size: ${BATCH_SIZE}"
echo "   - Power Limit: ${POWER_LIMIT}W"
echo "   - Expected Gains: 40% adaptation, 95% retention"
echo ""
echo "🔧 Commands:"
echo "   Monitor: ./monitor_performance.sh"
echo "   Logs: sudo journalctl -u curator-executor-optimized -f"
echo "   Status: sudo systemctl status curator-executor-optimized"
echo ""
echo "🚀 System is running with 2025 optimizations!"