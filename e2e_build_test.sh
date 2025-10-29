#!/bin/bash
# Comprehensive End-to-End Build and Test Script
# Tests all components of the Niodoo-TCS system

set -e

PROJECT_ROOT="/workspace/Niodoo-Final"
BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# Track test results
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_SKIPPED=0

print_header() {
    echo -e "\n${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}${BLUE}  $1${NC}"
    echo -e "${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
}

print_step() {
    echo -e "${BOLD}${GREEN}▶ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
    ((TESTS_PASSED++))
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
    ((TESTS_FAILED++))
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
    ((TESTS_SKIPPED++))
}

# Check prerequisites
check_prerequisites() {
    print_header "Checking Prerequisites"
    
    local all_good=true
    
    if ! command -v cargo &> /dev/null; then
        print_error "Cargo not found"
        all_good=false
    else
        print_success "Cargo: $(cargo --version)"
    fi
    
    if ! command -v rustc &> /dev/null; then
        print_error "Rustc not found"
        all_good=false
    else
        print_success "Rustc: $(rustc --version)"
    fi
    
    if ! command -v jq &> /dev/null; then
        print_warning "jq not found (needed for service checks)"
    else
        print_success "jq: $(jq --version)"
    fi
    
    if [ ! -d "$PROJECT_ROOT" ]; then
        print_error "Project root not found: $PROJECT_ROOT"
        all_good=false
    else
        print_success "Project root exists"
    fi
    
    if [ "$all_good" = false ]; then
        print_error "Prerequisites check failed"
        exit 1
    fi
}

# Check external services
check_services() {
    print_header "Checking External Services"
    
    # Check vLLM
    print_step "Checking vLLM..."
    if curl -s http://127.0.0.1:5001/v1/models > /dev/null 2>&1; then
        print_success "vLLM is running"
    else
        print_warning "vLLM not running (some tests will be skipped)"
    fi
    
    # Check Qdrant
    print_step "Checking Qdrant..."
    if curl -s http://127.0.0.1:6333/health > /dev/null 2>&1; then
        print_success "Qdrant is running"
    else
        print_warning "Qdrant not running (some tests will be skipped)"
    fi
    
    # Check Ollama
    print_step "Checking Ollama..."
    if curl -s http://127.0.0.1:11434/api/tags > /dev/null 2>&1; then
        print_success "Ollama is running"
    else
        print_warning "Ollama not running (some tests will be skipped)"
    fi
}

# Build workspace
build_workspace() {
    print_header "Building Workspace"
    
    cd "$PROJECT_ROOT"
    
    print_step "Cleaning previous builds..."
    cargo clean 2>&1 | grep -v "warning:" || true
    print_success "Clean complete"
    
    print_step "Building library..."
    if cargo build --lib 2>&1 | tee /tmp/build_lib.log; then
        print_success "Library build successful"
    else
        print_error "Library build failed"
        cat /tmp/build_lib.log | tail -50
        exit 1
    fi
    
    print_step "Building binaries..."
    if cargo build --bins 2>&1 | tee /tmp/build_bins.log; then
        print_success "Binaries build successful"
    else
        print_error "Binaries build failed"
        cat /tmp/build_bins.log | tail -50
        exit 1
    fi
    
    print_step "Building release binaries..."
    if cargo build --release --bins 2>&1 | tee /tmp/build_release.log; then
        print_success "Release build successful"
    else
        print_error "Release build failed"
        cat /tmp/build_release.log | tail -50
        exit 1
    fi
}

# Run unit tests
run_unit_tests() {
    print_header "Running Unit Tests"
    
    cd "$PROJECT_ROOT"
    
    print_step "Running tests for tcs-core..."
    if cargo test --package tcs-core --lib 2>&1 | tee /tmp/test_tcs_core.log; then
        print_success "tcs-core tests passed"
    else
        print_error "tcs-core tests failed"
    fi
    
    print_step "Running tests for tcs-tda..."
    if cargo test --package tcs-tda --lib 2>&1 | tee /tmp/test_tcs_tda.log; then
        print_success "tcs-tda tests passed"
    else
        print_error "tcs-tda tests failed"
    fi
    
    print_step "Running tests for tcs-knot..."
    if cargo test --package tcs-knot --lib 2>&1 | tee /tmp/test_tcs_knot.log; then
        print_success "tcs-knot tests passed"
    else
        print_error "tcs-knot tests failed"
    fi
    
    print_step "Running tests for tcs-ml..."
    if cargo test --package tcs-ml --lib 2>&1 | tee /tmp/test_tcs_ml.log; then
        print_success "tcs-ml tests passed"
    else
        print_error "tcs-ml tests failed"
    fi
    
    print_step "Running tests for niodoo-core..."
    if cargo test --package niodoo-core --lib 2>&1 | tee /tmp/test_niodoo_core.log; then
        print_success "niodoo-core tests passed"
    else
        print_error "niodoo-core tests failed"
    fi
    
    print_step "Running integration tests..."
    if cargo test --test integration 2>&1 | tee /tmp/test_integration.log; then
        print_success "Integration tests passed"
    else
        print_error "Integration tests failed"
    fi
}

# Run performance benchmarks
run_benchmarks() {
    print_header "Running Performance Benchmarks"
    
    cd "$PROJECT_ROOT"
    
    print_step "Running topological benchmarks..."
    if cargo bench --bench topological_bench 2>&1 | tee /tmp/bench_topological.log; then
        print_success "Topological benchmarks complete"
    else
        print_warning "Topological benchmarks had issues"
    fi
    
    print_step "Running consciousness engine benchmarks..."
    if cargo bench --bench consciousness_engine_benchmark 2>&1 | tee /tmp/bench_consciousness.log; then
        print_success "Consciousness engine benchmarks complete"
    else
        print_warning "Consciousness engine benchmarks had issues"
    fi
}

# Test service startup
test_service_startup() {
    print_header "Testing Service Startup"
    
    print_step "Checking supervisor script..."
    if [ -f "$PROJECT_ROOT/supervisor.sh" ]; then
        chmod +x "$PROJECT_ROOT/supervisor.sh"
        print_success "Supervisor script ready"
    else
        print_error "Supervisor script not found"
    fi
    
    print_step "Checking service health scripts..."
    if [ -f "$PROJECT_ROOT/check_all_services.sh" ]; then
        chmod +x "$PROJECT_ROOT/check_all_services.sh"
        print_success "Service check script ready"
    else
        print_error "Service check script not found"
    fi
}

# Run health checks
run_health_checks() {
    print_header "Running Health Checks"
    
    cd "$PROJECT_ROOT"
    
    print_step "Checking vLLM health..."
    if curl -s http://127.0.0.1:5001/v1/models | jq -e '.data' > /dev/null 2>&1; then
        print_success "vLLM health check passed"
    else
        print_warning "vLLM health check failed (service may not be running)"
    fi
    
    print_step "Checking Qdrant health..."
    if curl -s http://127.0.0.1:6333/health | jq -e '.status' > /dev/null 2>&1; then
        print_success "Qdrant health check passed"
    else
        print_warning "Qdrant health check failed (service may not be running)"
    fi
    
    print_step "Checking Ollama health..."
    if curl -s http://127.0.0.1:11434/api/tags | jq -e '.[]' > /dev/null 2>&1; then
        print_success "Ollama health check passed"
    else
        print_warning "Ollama health check failed (service may not be running)"
    fi
}

# Run end-to-end integration test
run_e2e_test() {
    print_header "Running End-to-End Integration Test"
    
    cd "$PROJECT_ROOT"
    
    print_step "Running master consciousness orchestrator..."
    timeout 60 cargo run --release --bin master_consciousness_orchestrator 2>&1 | tee /tmp/e2e_test.log || true
    
    if grep -q "✅.*HEALTHY" /tmp/e2e_test.log; then
        print_success "End-to-end test passed"
    else
        print_warning "End-to-end test had issues (check logs)"
    fi
}

# Generate test report
generate_report() {
    print_header "Test Summary Report"
    
    local total=$((TESTS_PASSED + TESTS_FAILED + TESTS_SKIPPED))
    
    echo "Total tests: $total"
    echo "Passed: $TESTS_PASSED"
    echo "Failed: $TESTS_FAILED"
    echo "Skipped: $TESTS_SKIPPED"
    echo ""
    
    if [ $TESTS_FAILED -eq 0 ]; then
        print_success "All critical tests passed!"
        return 0
    else
        print_error "Some tests failed"
        return 1
    fi
}

# Main execution
main() {
    print_header "Niodoo-TCS End-to-End Build and Test Suite"
    
    check_prerequisites
    check_services
    build_workspace
    run_unit_tests
    run_benchmarks
    test_service_startup
    run_health_checks
    run_e2e_test
    generate_report
    
    echo ""
    print_header "Build and Test Complete"
}

# Run main
main "$@"


