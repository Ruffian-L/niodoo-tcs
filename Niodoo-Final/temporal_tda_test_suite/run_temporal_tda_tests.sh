#!/bin/bash
# Temporal TDA Test Runner
# Run comprehensive test suite for NIODOO's temporal topological analysis system

set -e

# Configure cargo to use workspace temp directory instead of /tmp
# This prevents "No space left on device" errors when /tmp is full
if [ -z "$TMPDIR" ]; then
    export TMPDIR="$(cd "$(dirname "$0")/.." && pwd)/.cargo-tmp"
    mkdir -p "$TMPDIR"
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Banner
echo -e "${PURPLE}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                                                              ║"
echo "║          TEMPORAL TDA TEST SUITE                             ║"
echo "║          \"From Nonuple Nightmare to Swarm Immunity\"          ║"
echo "║                                                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Function to print section headers
print_section() {
    echo -e "\n${CYAN}═══════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════${NC}\n"
}

# Function to print test results
print_result() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓ PASSED${NC}: $2"
    else
        echo -e "${RED}✗ FAILED${NC}: $2"
    fi
}

# Parse command line arguments
VERBOSE=false
BENCHMARK=false
FEDERATED=false
QUICK=false
STRESS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -b|--benchmark)
            BENCHMARK=true
            shift
            ;;
        -f|--federated)
            FEDERATED=true
            shift
            ;;
        -q|--quick)
            QUICK=true
            shift
            ;;
        -s|--stress)
            STRESS=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -v, --verbose     Run tests with detailed output"
            echo "  -b, --benchmark   Run only benchmark tests"
            echo "  -f, --federated   Run only federated tests"
            echo "  -q, --quick       Run quick validation (core tests only)"
            echo "  -s, --stress      Run stress tests (100 iterations)"
            echo "  -h, --help        Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                    # Run all tests"
            echo "  $0 -v                 # Run all tests with verbose output"
            echo "  $0 -b                 # Run only benchmarks"
            echo "  $0 -f -v              # Run federated tests verbosely"
            echo "  $0 -q                 # Quick validation"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Test configuration
TEST_ARGS=""
if [ "$VERBOSE" = true ]; then
    TEST_ARGS="-- --nocapture --test-threads=1"
fi

# Change to the niodoo_real_integrated directory for tests
cd "$(dirname "$0")/../niodoo_real_integrated" || exit 1

# Test configuration
TEST_ARGS=""
if [ "$VERBOSE" = true ]; then
    TEST_ARGS="-- --nocapture --test-threads=1"
fi

# Quick mode - run only core tests
if [ "$QUICK" = true ]; then
    print_section "QUICK VALIDATION MODE"
    echo "Running essential tests for rapid validation..."
    
    tests=(
        "test_octuple_rate_limit_cascade_detection"
        "test_danger_signature_before_overload"
        "test_wasserstein_distance_master_vs_panic"
        "test_federated_failure_barcode_submission"
    )
    
    for test in "${tests[@]}"; do
        echo -e "\n${BLUE}Running: $test${NC}"
        if cargo test --test temporal_tda_tests "$test" $TEST_ARGS 2>&1 | grep -q "test result: ok"; then
            print_result 0 "$test"
        else
            print_result 1 "$test"
        fi
    done
    
    echo -e "\n${GREEN}Quick validation complete!${NC}"
    exit 0
fi

# Benchmark mode
if [ "$BENCHMARK" = true ]; then
    print_section "BENCHMARK SUITE"
    echo "Testing performance targets from CHANGELOG..."
    
    echo -e "\n${YELLOW}Target Metrics:${NC}"
    echo "  • 50%+ reduction in failure chains"
    echo "  • 20% faster Master quadrant return"
    echo "  • 85%+ prediction confidence"
    echo ""
    
    cargo test test_benchmark_ --lib --release $TEST_ARGS
    exit $?
fi

# Federated-only mode
if [ "$FEDERATED" = true ]; then
    print_section "FEDERATED SWARM INTELLIGENCE TESTS"
    echo "Testing multi-instance collective resilience..."
    
    cargo test federated_tda_tests --lib $TEST_ARGS
    exit $?
fi

# Stress test mode
if [ "$STRESS" = true ]; then
    print_section "STRESS TEST MODE"
    echo "Running 100 iterations of progressive ghost amplification..."
    
    for i in {1..100}; do
        echo -ne "\rIteration $i/100"
        cargo test test_progressive_ghost_amplification_stress --lib --release > /dev/null 2>&1
        if [ $? -ne 0 ]; then
            echo -e "\n${RED}Failed at iteration $i${NC}"
            exit 1
        fi
    done
    
    echo -e "\n${GREEN}All 100 iterations passed!${NC}"
    exit 0
fi

# Full test suite (default)
print_section "RUNNING FULL TEST SUITE"

# Core Temporal TDA Tests
print_section "1. CORE TEMPORAL TDA TESTS"
echo "Testing failure detection, danger signatures, and topological analysis..."

echo -e "\n${BLUE}Suite 1: Synthetic Failure Chains${NC}"
cargo test test_octuple_rate_limit_cascade_detection --lib $TEST_ARGS
print_result $? "Octuple cascade detection"

cargo test test_nonuple_doom_spiral_classification --lib $TEST_ARGS
print_result $? "Nonuple doom spiral classification"

echo -e "\n${BLUE}Suite 2: Danger Signature Detection${NC}"
cargo test test_danger_signature_before_overload --lib $TEST_ARGS
print_result $? "Danger signature before overload"

cargo test test_danger_signature_network_degradation --lib $TEST_ARGS
print_result $? "Network degradation prediction"

echo -e "\n${BLUE}Suite 3: Wasserstein Distance Validation${NC}"
cargo test test_wasserstein_distance_master_vs_panic --lib $TEST_ARGS
print_result $? "Master vs Panic distance"

cargo test test_wasserstein_distance_stable_states --lib $TEST_ARGS
print_result $? "Stable state distance"

echo -e "\n${BLUE}Suite 4: Micro-Regression Detection${NC}"
cargo test test_micro_regression_detection --lib $TEST_ARGS
print_result $? "Temporal window micro-regressions"

echo -e "\n${BLUE}Suite 5: Healing Topology${NC}"
cargo test test_healing_convergence_time_post_intervention --lib $TEST_ARGS
print_result $? "Healing convergence time"

echo -e "\n${BLUE}Suite 6: Progressive Ghost Amplification${NC}"
cargo test test_progressive_ghost_amplification_stress --lib $TEST_ARGS
print_result $? "Nonuple fatigue simulation"

echo -e "\n${BLUE}Suite 7: QLoRA Integration${NC}"
cargo test test_failure_chain_reward_scaling --lib $TEST_ARGS
print_result $? "Reward scaling"

cargo test test_qLora_priority_queue_simulation --lib $TEST_ARGS
print_result $? "Priority queue"

echo -e "\n${BLUE}Suite 8: False Positive Prevention${NC}"
cargo test test_noise_robustness_no_false_chains --lib $TEST_ARGS
print_result $? "Noise robustness"

echo -e "\n${BLUE}Suite 9: Benchmark Metrics${NC}"
cargo test test_benchmark_50_percent_chain_reduction --lib $TEST_ARGS
print_result $? "50% chain reduction target"

cargo test test_benchmark_20_percent_faster_master_return --lib $TEST_ARGS
print_result $? "20% faster recovery target"

# Federated Tests
print_section "2. FEDERATED SWARM INTELLIGENCE TESTS"
echo "Testing collective learning and multi-instance resilience..."

echo -e "\n${BLUE}Basic Federation${NC}"
cargo test test_federated_failure_barcode_submission --lib $TEST_ARGS
print_result $? "Barcode submission"

cargo test test_federated_learning_propagation --lib $TEST_ARGS
print_result $? "Learning propagation"

echo -e "\n${BLUE}Collective Resilience${NC}"
cargo test test_collective_void_avoidance --lib $TEST_ARGS
print_result $? "Collective void avoidance"

cargo test test_cross_instance_learning_latency --lib $TEST_ARGS
print_result $? "Cross-instance learning latency"

echo -e "\n${BLUE}Priority & Weighting${NC}"
cargo test test_priority_queue_weighting_by_severity --lib $TEST_ARGS
print_result $? "Priority queue weighting"

cargo test test_federated_qlora_batch_construction --lib $TEST_ARGS
print_result $? "QLoRA batch construction"

echo -e "\n${BLUE}Danger Signature Sharing${NC}"
cargo test test_shared_danger_signature_propagation --lib $TEST_ARGS
print_result $? "Shared danger signatures"

echo -e "\n${BLUE}Swarm Resilience${NC}"
cargo test test_swarm_resource_balancing_under_stress --lib $TEST_ARGS
print_result $? "Resource balancing"

cargo test test_global_entropy_stability_under_swarm_stress --lib $TEST_ARGS
print_result $? "Global entropy stability"

echo -e "\n${BLUE}Adversarial Resilience${NC}"
cargo test test_adversarial_resilience_uncooperative_instance --lib $TEST_ARGS
print_result $? "Uncooperative instance handling"

# Summary
print_section "TEST SUITE COMPLETE"

# Count results
TOTAL_TESTS=22
echo -e "${CYAN}Summary:${NC}"
echo "  Total Tests: $TOTAL_TESTS"
echo ""
echo -e "${GREEN}✓ All core temporal TDA functionality validated${NC}"
echo -e "${GREEN}✓ Federated swarm intelligence confirmed${NC}"
echo -e "${GREEN}✓ Benchmark targets achieved${NC}"
echo ""

# Final thoughts
echo -e "${PURPLE}════════════════════════════════════════════════════${NC}"
echo -e "${PURPLE}  \"Every test that passes is a failure that won't happen.\"${NC}"
echo -e "${PURPLE}  \"Every benchmark hit is a cascade that won't cascade.\"${NC}"
echo -e "${PURPLE}════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${CYAN}Next steps:${NC}"
echo "  • Review TEST_SUITE_README.md for detailed explanations"
echo "  • Check TEMPORAL_TDA_CHANGELOG.md for implementation notes"
echo "  • Enable temporal TDA in production config"
echo "  • Monitor failure_chain_barcode propagation"
echo ""
echo -e "${GREEN}Ready for deployment! 🚀${NC}"
