#!/bin/bash
# Run all NIODOO Real Integrated tests
# This script sets TMPDIR to workspace to avoid /tmp disk space issues

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Set TMPDIR to workspace to avoid /tmp disk space issues
export TMPDIR="${PROJECT_ROOT}/tmp"
mkdir -p "$TMPDIR"

# Set CARGO_TARGET_DIR to workspace
export CARGO_TARGET_DIR="${PROJECT_ROOT}/target"

cd "$SCRIPT_DIR"

echo "🧪 Running all NIODOO Real Integrated tests"
echo "   TMPDIR: $TMPDIR"
echo "   CARGO_TARGET_DIR: $CARGO_TARGET_DIR"
echo ""

# Clean up any previous test artifacts
echo "🧹 Cleaning previous test artifacts..."
cargo clean --target-dir "$CARGO_TARGET_DIR" 2>/dev/null || true

# Run unit tests
echo ""
echo "📦 Running unit tests..."
cargo test --lib --tests --target-dir "$CARGO_TARGET_DIR" -- --nocapture || {
    echo "❌ Unit tests failed"
    exit 1
}

# Run integration tests
echo ""
echo "🔗 Running integration tests..."
cargo test --test '*' --target-dir "$CARGO_TARGET_DIR" -- --nocapture || {
    echo "❌ Integration tests failed"
    exit 1
}

# Run test binaries if they exist
echo ""
echo "🔧 Running test binaries..."
for test_bin in src/bin/test_*.rs src/bin/*_test.rs; do
    if [ -f "$test_bin" ]; then
        bin_name=$(basename "$test_bin" .rs)
        echo "   Running $bin_name..."
        cargo run --bin "$bin_name" --target-dir "$CARGO_TARGET_DIR" || {
            echo "   ⚠️  $bin_name failed (non-fatal)"
        }
    fi
done

echo ""
echo "✅ All tests completed!"

