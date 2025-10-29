# Topology Benchmark Diagnostic Guide

## Understanding the Knot Complexity Saturation Issue

The benchmark shows that hybrid mode consistently produces:
- `betti_hybrid = [7, 15, 0]` (Betti_1 = 15)
- `knot_complexity_hybrid = 15.0`

This suggests saturation, but the code has capping logic that should limit Betti_1 to 6 (or `TCS_BETTI1_MAX`).

## Root Cause Hypothesis

With 7 points (from PAD dimensions), the theoretical maximum Betti_1 is 6. Getting 15 suggests:

1. **Multiple counting**: The persistence library might be counting features multiple times
2. **Graph complexity**: With kNN (k=16) on 7 points, we get a dense/complete graph with many edges, which could create many cycles
3. **Persistence library behavior**: The `rust_vr.rs` persistent reduction might be creating more persistent features than expected
4. **Capping not executing**: The capping logic might not be running due to a code path issue

## How to Diagnose

### Step 1: Run with Debug Logging

```bash
RUST_LOG=tcs_analysis=debug ./run_topology_benchmark.sh --cycles 4 2>&1 | tee debug.log
```

### Step 2: Look for Key Log Messages

**Before capping:**
```
Betti numbers before capping: [7, 15, 0], num_points=7, theoretical_max=6, constraint_max=6, max_allowed=6
```

**Persistent feature counts:**
```
Dimension 1: 15 persistent features (total features: XX)
```

**Capping confirmation:**
```
Betti_1 capped from 15 to 6 (max_allowed=6)
```

**If capping fails:**
```
Betti_1 capping failed during check: value=15 exceeds max_allowed=6. Force-capping now.
```

**Knot complexity calculation:**
```
Knot complexity calculation: betti[1]=6, knot_proxy=6.000, knot_analysis_score=X.XXX, final=X.XXX, max=6.000
```

### Step 3: Check Assertion Failures

If the assertion fails, you'll see:
```
thread 'main' panicked at 'Betti_1 assertion failed: 15 > 6 (theoretical_max=6, constraint_max=6, num_points=7)'
```

This would indicate the capping logic has a bug.

### Step 4: Verify in Results

After running, check the JSON results:
```bash
jq '.records[0] | {betti_hybrid, knot_complexity_hybrid}' results/benchmarks/topology/*.json
```

If betti_hybrid[1] is still 15 after capping, that means:
- Either the capping isn't executing
- Or the value is being overwritten after capping
- Or there's a different code path being used

## Expected Behavior After Fix

With default constraints (`TCS_BETTI1_MAX=6`):
- `betti_hybrid[1]` should be ≤ 6
- `knot_complexity_hybrid` should be ≤ 6.0
- Logs should show "Betti_1 capped from 15 to 6"

With higher constraints (`TCS_BETTI1_MAX=15`):
- `betti_hybrid[1]` can be up to 15
- `knot_complexity_hybrid` can be up to 15.0
- But it should still respect the theoretical maximum of `num_points - 1`

## Configuration to Match Observed Behavior

If you want to allow the observed behavior (betti[1] = 15) while investigating:

```bash
export TCS_BETTI1_MAX=15
export TCS_KNOT_COMPLEXITY_MAX=15.0
./run_topology_benchmark.sh --cycles 16
```

This will prevent capping and allow you to see the raw persistence computation results.

## Files to Check

1. **`tcs-core/src/topology/rust_vr.rs`**: The persistence reduction algorithm
   - Line 292-298: Creates persistent features with death=INFINITY
   - Check if it's creating more features than expected

2. **`niodoo_real_integrated/src/tcs_analysis.rs`**: The capping logic
   - Lines 183-254: Betti number computation and capping
   - Verify the capping is actually executing

3. **`niodoo_real_integrated/src/tcs_analysis.rs`**: Point cloud creation
   - Lines 299-330: `pad_to_points` function
   - Verify it creates exactly 7 points

## Quick Test

```bash
# Test with debug logging and custom constraints
RUST_LOG=tcs_analysis=debug \
  TCS_BETTI1_MAX=6 \
  TCS_KNOT_COMPLEXITY_MAX=6.0 \
  ./run_topology_benchmark.sh --cycles 2

# Check if capping happened
grep "Betti_1 capped" debug.log

# Check final values
jq '.records[].betti_hybrid[1]' results/benchmarks/topology/*.json | sort -u
```

If all values are ≤ 6, capping is working. If any are > 6, investigate further.

