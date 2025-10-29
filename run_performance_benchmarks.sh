#!/bin/bash
# Performance Benchmarking Suite for Niodoo-TCS
# Runs comprehensive benchmarks and generates performance reports

set -e

PROJECT_ROOT="/workspace/Niodoo-Final"
RESULTS_DIR="$PROJECT_ROOT/results/benchmarks"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_FILE="$RESULTS_DIR/benchmark_report_${TIMESTAMP}.txt"

mkdir -p "$RESULTS_DIR"

BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

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
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

log_result() {
    echo "$1" | tee -a "$REPORT_FILE"
}

# Check system resources
check_system_resources() {
    print_header "Checking System Resources"
    
    CPU_COUNT=$(nproc)
    TOTAL_MEM=$(free -m | awk 'NR==2{print $2}')
    AVAIL_MEM=$(free -m | awk 'NR==2{print $7}')
    
    log_result "CPU Cores: $CPU_COUNT"
    log_result "Total Memory: ${TOTAL_MEM}MB"
    log_result "Available Memory: ${AVAIL_MEM}MB"
    
    if [ "$AVAIL_MEM" -lt 2048 ]; then
        print_warning "Low available memory: ${AVAIL_MEM}MB"
    fi
    
    # Check GPU if available
    if command -v nvidia-smi &> /dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)
        log_result "GPU: $GPU_INFO"
        print_success "GPU detected"
    else
        print_warning "No GPU detected (using CPU)"
    fi
}

# Run Rust benchmarks
run_rust_benchmarks() {
    print_header "Running Rust Benchmarks"
    
    cd "$PROJECT_ROOT"
    
    print_step "Building benchmarks..."
    if cargo build --release --benches 2>&1 | tee /tmp/build_benches.log; then
        print_success "Benchmarks built"
    else
        print_error "Benchmark build failed"
        return 1
    fi
    
    # Run specific benchmarks
    BENCHES=(
        "topological_bench"
        "consciousness_engine_benchmark"
        "rag_optimization_benchmark"
        "sparse_gp_benchmark"
    )
    
    for bench in "${BENCHES[@]}"; do
        print_step "Running benchmark: $bench"
        if cargo bench --bench "$bench" 2>&1 | tee "/tmp/bench_${bench}.log"; then
            print_success "Benchmark $bench completed"
            log_result "Benchmark: $bench - PASSED"
        else
            print_warning "Benchmark $bench had issues"
            log_result "Benchmark: $bench - ISSUES"
        fi
    done
}

# Run TCS pipeline benchmarks
run_tcs_benchmarks() {
    print_header "Running TCS Pipeline Benchmarks"
    
    cd "$PROJECT_ROOT"
    
    print_step "Testing TCS orchestration..."
    
    # Create a test script
    cat > /tmp/test_tcs_pipeline.rs << 'EOF'
use tcs_pipeline::{TCSConfig, TCSOrchestrator};
use std::time::Instant;

#[tokio::main]
async fn main() {
    let mut config = TCSConfig::default();
    config.takens_dimension = 3;
    config.takens_delay = 2;
    config.takens_data_dim = 512;
    
    let mut orchestrator = TCSOrchestrator::with_config(16, config).unwrap();
    
    // Fill buffer
    for _ in 0..16 {
        orchestrator.ingest_sample(vec![0.1; 512]);
    }
    
    let start = Instant::now();
    let test_inputs = vec![
        "Test input for consciousness processing",
        "Analyze this topological structure",
        "Process cognitive patterns",
    ];
    
    for input in test_inputs {
        orchestrator.process(input).await.unwrap();
    }
    
    let elapsed = start.elapsed();
    println!("TCS Pipeline Benchmark: {:?}", elapsed);
}
EOF
    
    if cargo run --release --bin tcs_pipeline_test 2>&1 || rustc /tmp/test_tcs_pipeline.rs --edition 2021 -o /tmp/test_tcs_pipeline && /tmp/test_tcs_pipeline; then
        print_success "TCS pipeline benchmark completed"
        log_result "TCS Pipeline Benchmark - PASSED"
    else
        print_warning "TCS pipeline benchmark had issues"
        log_result "TCS Pipeline Benchmark - ISSUES"
    fi
}

# Memory profiling
run_memory_profiling() {
    print_header "Memory Profiling"
    
    cd "$PROJECT_ROOT"
    
    print_step "Checking binary sizes..."
    
    if [ -f "target/release/niodoo-consciousness" ]; then
        BINARY_SIZE=$(du -h target/release/niodoo-consciousness | cut -f1)
        log_result "Main binary size: $BINARY_SIZE"
        print_success "Binary size: $BINARY_SIZE"
    fi
    
    print_step "Checking workspace size..."
    WORKSPACE_SIZE=$(du -sh . | cut -f1)
    log_result "Workspace size: $WORKSPACE_SIZE"
}

# Performance metrics collection
collect_performance_metrics() {
    print_header "Collecting Performance Metrics"
    
    print_step "CPU usage..."
    CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1}')
    log_result "Current CPU usage: ${CPU_USAGE}%"
    
    print_step "Memory usage..."
    MEM_USAGE=$(free | grep Mem | awk '{printf("%.2f", $3/$2 * 100.0)}')
    log_result "Current memory usage: ${MEM_USAGE}%"
    
    print_step "Disk usage..."
    DISK_USAGE=$(df -h / | awk 'NR==2 {print $5}')
    log_result "Disk usage: $DISK_USAGE"
}

# Generate report
generate_report() {
    print_header "Generating Benchmark Report"
    
    echo "Benchmark Report Generated: $(date)" > "$REPORT_FILE"
    echo "===========================================" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    
    print_success "Report generated: $REPORT_FILE"
    
    # Show summary
    echo ""
    print_header "Benchmark Summary"
    cat "$REPORT_FILE"
}

# Main execution
main() {
    print_header "Niodoo-TCS Performance Benchmarking Suite"
    
    check_system_resources
    run_rust_benchmarks
    run_tcs_benchmarks
    run_memory_profiling
    collect_performance_metrics
    generate_report
    
    echo ""
    print_header "Benchmarking Complete"
    echo "Results saved to: $REPORT_FILE"
}

# Run main
main "$@"

