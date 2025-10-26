# Experiment Status

## Current State

### Mu/Sigma Fix
✅ **Completed**: Fixed mu/sigma extraction in `torus.rs` to use variance-based calculation
- Changed from fixed indices (0-6, 7-13) to computing from actual embedding slices
- Each dimension now gets: mu = mean of slice, sigma = sqrt(variance of slice)

### Binaries
- `niodoo_real_integrated_stable` (8.2MB) - original behavior
- `niodoo_real_integrated_experimental` (8.2MB) - with mu/sigma fix

### Launch Attempted
- Started harness PID 205591 (stable binary)
- Error: `Tokenizer JSON not found; set TOKENIZER_JSON or QWEN_TOKENIZER`
- Process died immediately

## Issue Identified

Both binaries require tokenizer environment variables that aren't set. The rut_gauntlet binary needs:
- `TOKENIZER_JSON` or `QWEN_TOKENIZER` environment variable pointing to tokenizer.json

## Next Steps

1. **Find Tokenizer**: Locate tokenizer.json file in workspace
2. **Set Environment**: Export TOKENIZER_JSON or QWEN_TOKENIZER
3. **Verify Fix**: Run test to confirm mu/sigma vary between runs
4. **Launch Experiments**: Once environment is set, restart harness + sweep

## Commands Ready (pending tokenizer)

```bash
# Once tokenizer is found:
export TOKENIZER_JSON=/path/to/tokenizer.json

# Test the fix
./test_mu_sigma_fix.sh

# Launch experiments
./start_long_experiments.sh

# Monitor
tail -f harness.log
tail -f sweep.log
nvidia-smi
```

## Key Observation

Previous logs showed mu/sigma were identical:
```
mu: [0.0193..., 0.00411..., -0.1652..., -0.0376..., 0.0555..., -0.0669..., -0.00680...]
sigma: [0.05, 0.05, 0.05, 0.05, 0.0518..., 0.05, 0.05]
```

The fix should make these vary per run based on actual embedding variance.

