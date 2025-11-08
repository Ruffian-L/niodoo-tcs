#!/bin/bash
# Run the end-to-end pipeline test
# This is THE test that validates the full pipeline works

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR/.."
cd "$ROOT_DIR"

echo "🚀 Running End-to-End Pipeline Test"
echo "===================================="
echo ""

# Build the test binary
echo "Building test binary..."
cd niodoo_real_integrated
cargo build --bin test_e2e_pipeline --release || {
    echo "❌ Build failed"
    exit 1
}

# Run the test
echo ""
echo "Running end-to-end test..."
echo ""
cargo run --bin test_e2e_pipeline --release

