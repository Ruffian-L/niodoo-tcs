# GPU Experiment Launch Summary

## ✅ Completed Actions

### 1. Copied Stable Binary
- **Stable**: `niodoo_real_integrated_stable` (8.2MB, from Oct 26 12:15)
- **Experimental**: `niodoo_real_integrated_experimental` (8.2MB, rebuilt with mu/sigma fix)

### 2. Fixed Mu/Sigma Extraction
**File**: `niodoo_real_integrated/src/torus.rs` (lines 49-68)

**Problem**: mu and sigma were extracted from fixed embedding indices (0-6, 7-13), causing identical values across runs.

**Solution**: Now computing from actual embedding variance:
```rust
// Each dimension gets a slice of embedding
let slice_size = embedding.len() / 7;
for i in 0..7 {
    let start = i * slice_size;
    let end = if i == 6 { embedding.len() } else { (i + 1) * slice_size };
    let slice = &embedding[start..end];
    
    // mu = mean of slice
    mu[i] = slice.iter().map(|&x| x as f64).sum::<f64>() / slice.len() as f64;
    
    // sigma = sqrt(variance of slice)
    let variance = slice.iter()
        .map(|&x| { let diff = x as f64 - mu[i]; diff * diff })
        .sum::<f64>() / slice.len() as f64;
    sigma[i] = variance.sqrt().max(0.05);
}
```

### 3. Created Experiment Scripts
- `run_1000_iteration_harness.sh` - Long run with rotating prompts
- `topology_stress_sweep.sh` - Parameter sweep (125 combinations)
- `monitor_gpu.sh` - Continuous GPU monitoring
- `start_experiments.sh` - Master launcher
- `test_mu_sigma_fix.sh` - Quick verification test

## 🚀 Ready to Launch

### Quick Start
```bash
cd /workspace/Niodoo-Final
./start_experiments.sh
```

### Individual Components
```bash
# Test the fix first
./test_mu_sigma_fix.sh

# Start long harness
nohup cargo run -p niodoo_real_integrated --bin rut_gauntlet --release > harness.log 2>&1 &

# Start sweep
nohup ./topology_stress_sweep.sh > sweep.log 2>&1 &

# Monitor GPU
nohup ./monitor_gpu.sh > gpu_monitor.log 2>&1 &
```

## 📊 Expected Outputs

### Harness Logs (`harness_logs_*`)
- `iter_*.log` - Individual iteration logs
- `gpu_health.log` - GPU stats every 10 iterations
- `metrics_summary.log` - Cumulative metrics every 50 iterations

### Sweep Results (`sweep_*`)
- `sweep_results.csv` - All parameter combinations
- `probe_*.log` - Individual probe logs

### GPU Monitoring (`gpu_monitor_*`)
- `gpu_stats.csv` - Continuous GPU stats
- `thermal_warnings.log` - Temperature alerts

## 🔍 Key Observation Fixed

**Before**: mu and sigma identical across runs
```
mu: [0.0193..., 0.00411..., -0.1652..., -0.0376..., 0.0555..., -0.0669..., -0.00680...]
sigma: [0.05, 0.05, 0.05, 0.05, 0.0518..., 0.05, 0.05]
```

**After**: mu and sigma derived from actual embedding variance per run
- Each dimension gets a slice of the embedding
- mu = mean of slice
- sigma = sqrt(variance of slice)

## 📈 A/B Comparison

| Binary | Mu/Sigma Source | Entropy Behavior | Purpose |
|--------|----------------|------------------|---------|
| `niodoo_real_integrated_stable` | Fixed indices (0-6, 7-13) | May stagnate | Baseline reference |
| `niodoo_real_integrated_experimental` | Variance-based extraction | Should fluctuate | With fix applied |

## 🎯 Success Criteria

1. **Mu/Sigma Variation**: Experimental binary shows different mu/sigma across runs
2. **Entropy Dynamics**: Entropy fluctuates properly with varying sigma
3. **Topology Stability**: Sweep identifies parameter combinations that stabilize entropy
4. **GPU Utilization**: Experiments run within thermal limits (< 80°C)

## 🛠️ Monitoring Commands

```bash
# Check status
ps aux | grep niodoo
nvidia-smi

# Watch logs
tail -f harness.log
tail -f sweep.log
tail -f gpu_monitor.log

# Check metrics
find harness_logs_* -name "metrics_summary.log" -exec tail {} \;
cat sweep_*/sweep_results.csv | grep ",1,"  # Stabilized combinations
```

## 📝 Next Steps

1. Verify mu/sigma fix with test script
2. Launch experiments with master script
3. Monitor GPU utilization
4. Analyze results after completion
5. Compare stable vs experimental entropy behavior

---

**Launch Command**:
```bash
cd /workspace/Niodoo-Final && ./start_experiments.sh
```

