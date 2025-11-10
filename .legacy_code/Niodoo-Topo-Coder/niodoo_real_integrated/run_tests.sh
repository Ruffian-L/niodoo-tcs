#!/bin/bash
# Run all NIODOO Real Integrated tests
# This script sets TMPDIR to workspace to avoid /tmp disk space issues

set -euo pipefail

CLEAN=0
SKIP_SERVICE_CHECKS="${SKIP_SERVICE_CHECKS:-0}"

while (($#)); do
    case "$1" in
        --clean)
            CLEAN=1
            shift
            ;;
        --skip-services)
            SKIP_SERVICE_CHECKS=1
            shift
            ;;
        --help|-h)
            echo "Usage: $(basename "$0") [--clean] [--skip-services]"
            echo "  --clean           Force cargo clean before running tests"
            echo "  --skip-services   Skip external service availability checks"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

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

if [ "$CLEAN" -eq 1 ]; then
    echo "   CLEAN: enabled (forced)"
else
    echo "   CLEAN: skipped (use --clean to force)"
fi

if [ "$SKIP_SERVICE_CHECKS" = "1" ]; then
    echo "   SERVICE CHECKS: skipped"
else
    echo "   SERVICE CHECKS: required"
fi

echo ""

if [ "$SKIP_SERVICE_CHECKS" != "1" ]; then
    echo "🔍 Verifying external services..."
    if ! "$PROJECT_ROOT/test_services.sh"; then
        echo "❌ Service availability check failed. Aborting tests."
        exit 1
    fi
else
    echo "⚠️  Skipping external service verification (requested)."
fi

if [ "$CLEAN" -eq 1 ]; then
    echo ""
    echo "🧹 Cleaning previous test artifacts (forced clean)..."
    cargo clean --target-dir "$CARGO_TARGET_DIR" 2>/dev/null || true
else
    echo ""
    echo "⚡️ Re-using incremental build cache (no cargo clean)."
fi

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

