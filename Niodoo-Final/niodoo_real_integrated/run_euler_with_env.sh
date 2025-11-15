#!/bin/bash
# Load environment from parent directory before changing directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
if [ -f "$PARENT_DIR/niodoo_real_integrated.env" ]; then
    source "$PARENT_DIR/niodoo_real_integrated.env"
    echo "✅ Loaded environment from $PARENT_DIR/niodoo_real_integrated.env"
else
    echo "⚠️  Warning: niodoo_real_integrated.env not found at $PARENT_DIR/niodoo_real_integrated.env"
fi

export LD_LIBRARY_PATH=/home/beelink/niodoo-tcs/third_party/onnxruntime-linux-x64-gpu-1.23.2/lib:$LD_LIBRARY_PATH
export NIODOO_SKIP_SMOKE=1
cd "$SCRIPT_DIR"
exec cargo run --features cli_bins --bin euler_test -- --problems 5 --timeout 180
