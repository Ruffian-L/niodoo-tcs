
#!/bin/bash
# NIODOO-TCS INTEGRATED PIPELINE RUNNER
# Sets all env vars and runs the full pipeline

cd /home/beelink/Niodoo-Final

# Load all environment vars
export QWEN_MODEL_PATH=/home/beelink/models/Qwen2.5-7B-Instruct-AWQ
export LD_LIBRARY_PATH=/home/beelink/Niodoo-Final/onnxruntime-linux-x64-1.16.3/lib:$LD_LIBRARY_PATH
export VLLM_ENDPOINT=http://localhost:8000
export QDRANT_URL=http://localhost:6333
export QDRANT_COLLECTION=experiences
export QDRANT_VECTOR_SIZE=896
export RUST_LOG=info

# Run the binary
./target/release/niodoo_real_integrated "$@"

