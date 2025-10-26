# Quick Experiment Commands

## Single Command to Launch Everything

```bash
cd /workspace/Niodoo-Final
./start_experiments.sh
```

This will:
1. Start GPU monitoring in background
2. Launch 1000-iteration harness (rotating prompts every 100 runs)
3. Run topology stress sweep (125 combinations × 10 iterations each)

## Individual Commands

### 1. Start Long Harness Run (Stable Binary)
```bash
cd /workspace/Niodoo-Final
nohup cargo run -p niodoo_real_integrated --bin rut_gauntlet --release > harness.log 2>&1 &
```

### 2. Run Topology Stress Sweep (Experimental Binary)
```bash
cd /workspace/Niodoo-Final
nohup ./topology_stress_sweep.sh > sweep.log 2>&1 &
```

### 3. Monitor GPU Continuously
```bash
cd /workspace/Niodoo-Final
nohup ./monitor_gpu.sh > gpu_monitor.log 2>&1 &
```

### 4. Check Status
```bash
# Check running processes
ps aux | grep niodoo

# Check GPU
nvidia-smi

# Watch logs
tail -f harness.log
tail -f sweep.log
tail -f gpu_monitor.log
```

### 5. A/B Test Experimental vs Stable
```bash
# Test experimental binary (with fixed mu/sigma extraction)
./niodoo_real_integrated_experimental --prompt "test prompt"

# Test stable binary (with old mu/sigma extraction)
./niodoo_real_integrated_stable --prompt "test prompt"
```

## Key Outputs

### Harness Logs
- `harness_logs_*/iter_*.log` - Individual iteration logs
- `harness_logs_*/gpu_health.log` - GPU health checks every 10 iterations
- `harness_logs_*/metrics_summary.log` - Cumulative metrics every 50 iterations

### Sweep Results
- `sweep_*/sweep_results.csv` - All parameter combinations and results
- `sweep_*/probe_*.log` - Individual probe logs

### GPU Monitoring
- `gpu_monitor_*/gpu_stats.csv` - Continuous GPU stats
- `gpu_monitor_*/thermal_warnings.log` - Temperature warnings

## Mu/Sigma Fix Summary

**Problem**: mu and sigma were identical across runs because they were extracted from fixed embedding indices (0-6 and 7-13).

**Solution**: Now computing mu/sigma from actual embedding variance:
- Each PAD dimension gets a slice of the embedding
- mu = mean of the slice
- sigma = sqrt(variance of the slice)

This ensures each run produces different mu/sigma values based on the actual embedding content.

## Expected Results

### Stable Binary
- mu/sigma identical across runs (OLD BEHAVIOR)
- Entropy might stagnate
- Metrics: baseline reference

### Experimental Binary  
- mu/sigma vary per run (NEW BEHAVIOR)
- Entropy should fluctuate properly
- Metrics: comparison point

## Post-Experiment Analysis

```bash
# Find stabilized parameter combinations
grep ",1," sweep_*/sweep_results.csv

# Check entropy trends
grep entropy harness_logs_*/metrics_summary.log

# Analyze GPU utilization
awk -F',' '{sum+=$2; count++} END {print sum/count}' gpu_monitor_*/gpu_stats.csv
```

