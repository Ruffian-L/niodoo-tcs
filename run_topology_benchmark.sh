#!/bin/bash
# Topology Benchmark Runner
# Runs topology benchmarks with configurable parameters and collects results

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default values
CYCLES=8
DATASET="data/goemotions_test.tsv"
OUTPUT_DIR="results/benchmarks/topology"
HARDWARE="${HARDWARE:-beelink}"
CONFIG="${CONFIG:-}"

# Environment variable overrides for TCS parameters
TCS_BETTI1_MAX="${TCS_BETTI1_MAX:-6}"
TCS_KNOT_COMPLEXITY_MAX="${TCS_KNOT_COMPLEXITY_MAX:-6.0}"

# Parse command line arguments
usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Options:
    -c, --cycles N          Number of benchmark cycles (default: $CYCLES)
    -d, --dataset PATH      Path to dataset TSV file (default: $DATASET)
    -o, --output-dir PATH   Output directory for results (default: $OUTPUT_DIR)
    -h, --hardware PROFILE  Hardware profile (default: $HARDWARE)
    --config PATH           Optional runtime config file
    --betti-max N           Max Betti_1 value (default: $TCS_BETTI1_MAX)
    --knot-max F            Max knot complexity (default: $TCS_KNOT_COMPLEXITY_MAX)
    --help                  Show this help message

Environment Variables:
    TCS_BETTI1_MAX          Maximum allowed Betti_1 value (overrides --betti-max)
    TCS_KNOT_COMPLEXITY_MAX Maximum knot complexity (overrides --knot-max)
    HARDWARE                Hardware profile to use

Examples:
    # Run with default settings
    $0

    # Run 16 cycles with custom constraints
    $0 --cycles 16 --betti-max 15 --knot-max 15.0

    # Run with custom dataset and output location
    $0 --dataset data/custom.tsv --output-dir results/custom

    # Run with environment variables
    TCS_BETTI1_MAX=15 TCS_KNOT_COMPLEXITY_MAX=15.0 $0 --cycles 16
EOF
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--cycles)
            CYCLES="$2"
            shift 2
            ;;
        -d|--dataset)
            DATASET="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--hardware)
            HARDWARE="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --betti-max)
            TCS_BETTI1_MAX="$2"
            shift 2
            ;;
        --knot-max)
            TCS_KNOT_COMPLEXITY_MAX="$2"
            shift 2
            ;;
        --help)
            usage
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            ;;
    esac
done

# Make paths absolute relative to script dir if they are relative
if [[ ${DATASET} != /* ]]; then
    DATASET=${SCRIPT_DIR}/${DATASET}
fi
if [[ ${OUTPUT_DIR} != /* ]]; then
    OUTPUT_DIR=${SCRIPT_DIR}/${OUTPUT_DIR}
fi
if [[ -n ${CONFIG} && ${CONFIG} != /* ]]; then
    CONFIG=${SCRIPT_DIR}/${CONFIG}
fi

# Export environment variables for the benchmark
export TCS_BETTI1_MAX
export TCS_KNOT_COMPLEXITY_MAX

echo "=========================================="
echo "Topology Benchmark Configuration"
echo "=========================================="
echo "Cycles: $CYCLES"
echo "Dataset: $DATASET"
echo "Output Directory: $OUTPUT_DIR"
echo "Hardware Profile: $HARDWARE"
echo "TCS_BETTI1_MAX: $TCS_BETTI1_MAX"
echo "TCS_KNOT_COMPLEXITY_MAX: $TCS_KNOT_COMPLEXITY_MAX"
if [[ -n "$CONFIG" ]]; then
    echo "Config File: $CONFIG"
fi
echo "=========================================="
echo ""

# Build the benchmark binary if needed
echo "Building topology benchmark..."
cd niodoo_real_integrated
if ! cargo build --release --bin topology_bench 2>&1 | grep -q "Finished"; then
    echo "Build completed (or already up to date)"
fi

# Prepare command arguments
ARGS=(
    --cycles "$CYCLES"
    --dataset "$DATASET"
    --output-dir "$OUTPUT_DIR"
    --hardware "$HARDWARE"
)

if [[ -n "$CONFIG" ]]; then
    ARGS+=(--config "$CONFIG")
fi

# Run the benchmark
echo "Starting benchmark..."
echo "Command: cargo run --release --bin topology_bench -- ${ARGS[*]}"
echo ""

RUST_LOG="${RUST_LOG:-info}" cargo run --release --bin topology_bench -- "${ARGS[@]}"

echo ""
echo "=========================================="
echo "Benchmark completed!"
echo "=========================================="
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "To view results:"
echo "  ls -lh $OUTPUT_DIR/*.json"
echo "  cat $OUTPUT_DIR/*.json | jq '.summary'"
echo ""

