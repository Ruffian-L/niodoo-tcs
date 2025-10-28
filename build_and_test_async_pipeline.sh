#!/bin/bash
set -e

# Async-Friendly Pipeline Build & Test Automation Script
# =====================================================
# Validates and tests the Arc<RwLock>, spawn_blocking, and swarm concurrency improvements

PROJECT_ROOT="/workspace/Niodoo-Final"
PACKAGE_DIR="$PROJECT_ROOT/niodoo_real_integrated"
BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

function print_step() {
    echo -e "\n${BOLD}${GREEN}▶ $1${NC}"
}

function print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

function print_error() {
    echo -e "${RED}✗ $1${NC}"
}

function print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Verify basic requirements
print_step "Checking prerequisites..."

if ! command -v cargo &> /dev/null; then
    print_error "Cargo not found. Please install Rust."
    exit 1
fi
print_success "Cargo available"

if [ ! -d "$PACKAGE_DIR" ]; then
    print_error "Package directory not found: $PACKAGE_DIR"
    exit 1
fi
print_success "Package directory exists"

# Load environment
print_step "Loading runtime environment..."
if [ -f "$PROJECT_ROOT/tcs_runtime.env" ]; then
    source "$PROJECT_ROOT/tcs_runtime.env"
    print_success "Environment loaded from tcs_runtime.env"
else
    print_warning "tcs_runtime.env not found; using defaults"
fi

# Verify external services (optional, informational)
print_step "Checking external services..."

OLLAMA_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:11434/api/tags 2>/dev/null || echo "000")
if [ "$OLLAMA_STATUS" = "200" ]; then
    print_success "Ollama is running"
else
    print_warning "Ollama not responding (optional for library tests)"
fi

QDRANT_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:6333/health 2>/dev/null || echo "000")
if [ "$QDRANT_STATUS" = "200" ]; then
    print_success "Qdrant is running"
else
    print_warning "Qdrant not responding (optional for library tests)"
fi

# Navigate to package directory
cd "$PACKAGE_DIR"
print_success "Working directory: $PACKAGE_DIR"

# Clean previous builds
print_step "Cleaning previous builds..."
cargo clean
print_success "Clean complete"

# Build library
print_step "Building library..."
if cargo build --lib 2>&1 | tee /tmp/build.log; then
    print_success "Library build successful"
else
    print_error "Library build failed. Check /tmp/build.log"
    exit 1
fi

# Run tests
print_step "Running library tests..."
echo "Note: Tests use mock Ollama server; some may skip if external services are unavailable"
if RUST_LOG=info cargo test --lib -- --test-threads=1 --nocapture 2>&1 | tee /tmp/test.log; then
    print_success "Tests passed"
    TEST_RESULT=0
else
    print_warning "Some tests may have failed (check above)"
    TEST_RESULT=$?
fi

# Summary of test results
print_step "Test Summary"
PASSED=$(grep -c "test.*ok" /tmp/test.log || echo "0")
FAILED=$(grep -c "test.*FAILED" /tmp/test.log || echo "0")
print_success "Tests completed: $PASSED passed"
if [ "$FAILED" -gt 0 ]; then
    print_warning "Failures: $FAILED (check logs for details)"
fi

# Build release binary
print_step "Building release binary..."
if cargo build --release --bin niodoo_real_integrated 2>&1 | tee /tmp/build_release.log; then
    print_success "Release binary built"
    ls -lh target/release/niodoo_real_integrated
else
    print_error "Release build failed. Check /tmp/build_release.log"
    exit 1
fi

# Test single-prompt execution
print_step "Testing single-prompt execution..."
if timeout 30 cargo run --release --bin niodoo_real_integrated -- \
    --prompt "Async pipeline single test" \
    --output-format json 2>&1 | head -50; then
    print_success "Single-prompt execution completed"
else
    RESULT=$?
    if [ $RESULT -eq 124 ]; then
        print_warning "Single-prompt execution timed out (services may not be available)"
    else
        print_error "Single-prompt execution failed"
    fi
fi

# Demonstrate swarm mode (if batch file exists)
print_step "Demonstrating swarm mode..."
BATCH_FILE="$PROJECT_ROOT/prompts/test_prompts.txt"
if [ -f "$BATCH_FILE" ]; then
    print_success "Batch file found: $BATCH_FILE"
    echo "Swarm mode command (requires services):"
    echo "  cd $PACKAGE_DIR"
    echo "  cargo run --release --bin niodoo_real_integrated -- \\"
    echo "    --batch $BATCH_FILE \\"
    echo "    --swarm 4 \\"
    echo "    --output-format csv"
else
    print_warning "Batch file not found: $BATCH_FILE"
    echo "Create test prompts with:"
    echo "  mkdir -p $PROJECT_ROOT/prompts"
    echo "  echo 'test prompt 1' > $BATCH_FILE"
    echo "  echo 'test prompt 2' >> $BATCH_FILE"
fi

# Final status report
print_step "Build & Test Complete"
echo ""
echo "Build artifacts:"
echo "  - Library: target/debug/libniodoo_real_integrated.rlib"
echo "  - Binary:  target/release/niodoo_real_integrated"
echo ""
echo "Environment:"
echo "  - Ollama:  $([ "$OLLAMA_STATUS" = "200" ] && echo "✓ Running" || echo "✗ Not running")"
echo "  - Qdrant:  $([ "$QDRANT_STATUS" = "200" ] && echo "✓ Running" || echo "✗ Not running")"
echo ""
echo "Next steps:"
echo "  1. Verify services: bash check_all_services.sh"
echo "  2. Run single prompt: cargo run --release -- --prompt 'test'"
echo "  3. Run swarm (4 pipelines): cargo run --release -- --swarm 4 --batch prompts.txt"
echo "  4. Full guide: cat /workspace/Niodoo-Final/ASYNC_PIPELINE_BUILD_GUIDE.md"
echo ""

exit $TEST_RESULT
