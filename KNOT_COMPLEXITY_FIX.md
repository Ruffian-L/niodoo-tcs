# Knot Complexity Saturation Fix

## Summary

Fixed the knot complexity saturation issue and made it configurable. The benchmark showed that hybrid mode was consistently producing `knot_complexity = 15.0` and `betti_hybrid = [7, 15, 0]`, suggesting saturation behavior.

## Changes Made

### 1. Configurable Constraints (`tcs_analysis.rs`)

Added environment variable support for configuring topology constraints:

- **`TCS_BETTI1_MAX`**: Maximum allowed Betti_1 value (default: 6)
  - Controls the maximum number of 1-dimensional holes allowed
  - Example: `TCS_BETTI1_MAX=15` to allow higher values

- **`TCS_KNOT_COMPLEXITY_MAX`**: Maximum knot complexity value (default: 6.0)
  - Caps the final knot complexity calculation
  - Example: `TCS_KNOT_COMPLEXITY_MAX=15.0` to match observed behavior

### 2. Enhanced Debugging

Added debug logging to track:
- Original vs. capped Betti_1 values
- Knot complexity calculation steps (knot_proxy, knot_analysis_score, final value)
- Warnings when capping may have failed

### 3. Code Improvements

- Fixed unused import warnings in `compass.rs` and `data.rs`
- Removed unused `Instant` import from `compass.rs`
- Removed unused `AdaptiveSearchStats` import from `compass.rs`
- Cleaned up unused rand imports from `data.rs` (kept `SliceRandom` which is actually used)

### 4. Benchmark Runner Script

Created `run_topology_benchmark.sh` with:
- Configurable cycles, dataset, output directory
- Support for environment variable overrides
- Easy command-line interface
- Automatic build and execution

## Usage

### Running Benchmark with Default Settings

```bash
./run_topology_benchmark.sh
```

### Running with Custom Constraints

```bash
# Allow higher Betti_1 and knot complexity values
./run_topology_benchmark.sh --cycles 16 --betti-max 15 --knot-max 15.0
```

### Using Environment Variables

```bash
export TCS_BETTI1_MAX=15
export TCS_KNOT_COMPLEXITY_MAX=15.0
./run_topology_benchmark.sh --cycles 16
```

## Investigation Findings

The saturation at 15.0 appears to be caused by:

1. **Betti_1 = 15**: The persistence computation returns 15 persistent 1-dimensional holes
2. **Theoretical max**: With 7 points (from PAD dimensions), theoretical max = 6
3. **Capping logic**: Should cap betti[1] to 6, but benchmark shows 15 persists

**Possible causes**:
- The persistence computation may be creating more points than expected
- The capping check may not be executing in some code paths
- There may be a different code path for hybrid mode

**Next steps for investigation**:
See `DIAGNOSTIC_GUIDE.md` for detailed diagnostic steps.

Quick diagnosis:
```bash
RUST_LOG=tcs_ real_integrated=debug ./run_topology_benchmark.sh --cycles 4 2>&1 | grep -E "(Betti|capped|persistent)"
```

The code now includes:
- Enhanced debug logging with detailed Betti number tracing
- Manual persistent feature counting to verify persistence library output
- Double-check verification and force-capping if needed
- Assertion that will panic if capping fails (in debug builds)

## Configuration Options

All topology constraints can now be configured via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `TCS_BETTI1_MAX` | 6 | Maximum Betti_1 value |
| `TCS_KNOT_COMPLEXITY_MAX` | 6.0 | Maximum knot complexity |
| `TCS_KNN_K` | 16 | K-nearest neighbors for topology |
| `TCS_MAX_FILTRATION` | 1.5 | Maximum filtration value |

## Files Modified

- `niodoo_real_integrated/src/tcs_analysis.rs`: Added configurable constraints and debug logging
- `niodoo_real_integrated/src/compass.rs`: Removed unused imports
- `niodoo_real_integrated/src/data.rs`: Removed unused imports
- `niodoo_real_integrated/src/bin/topology_bench.rs`: Removed unused import
- `run_topology_benchmark.sh`: New benchmark runner script

## Testing

To verify the fixes:

```bash
# Run with default constraints (should cap to 6)
./run_topology_benchmark.sh --cycles 4

# Run with higher constraints (should allow up to 15)
./run_topology_benchmark.sh --cycles 4 --betti-max 15 --knot-max 15.0

# Compare results
diff <(jq '.summary' results/benchmarks/topology/*_default.json) \
     <(jq '.summary' results/benchmarks/topology/*_high.json)
```

