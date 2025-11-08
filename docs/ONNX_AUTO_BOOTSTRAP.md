# ONNX Runtime Auto-Bootstrap Guide

## Overview

The NIODOO system includes **automatic ONNX Runtime detection and configuration** to eliminate manual environment setup. This guide explains how it works and how to use it.

## Auto-Bootstrap Methods

### Method 1: Shell Script (Recommended for Manual Runs)

**Location**: `scripts/bootstrap_onnx.sh`

**Usage**:
```bash
# Source the bootstrap script before running any NIODOO binaries
source scripts/bootstrap_onnx.sh

# Or run it directly (it exports variables to current shell)
. scripts/bootstrap_onnx.sh

# Then run your binary
cargo run --release --bin soak_test -- --quick
```

**What it does**:
- Auto-detects ONNX Runtime GPU libraries (tries multiple versions: 1.24.0, 1.23.2, 1.18.1, 1.16.3)
- Falls back to CPU-only builds if GPU not found
- Configures `LD_LIBRARY_PATH` with all necessary paths
- Sets `ORT_DYLIB_PATH` and `ORT_DYLIB_DEFAULT_PATH` for ort crate
- Adds CUDA library paths automatically
- Adds cuDNN paths if available

### Method 2: Built-in Auto-Detection (Automatic)

**Location**: `niodoo_real_integrated/src/bin/soak_test.rs` (and other binaries)

**How it works**:
- Automatically runs on binary startup
- Searches multiple standard paths for ONNX Runtime
- Configures environment variables before ONNX Runtime is loaded
- No manual intervention needed

**Search Order**:
1. GPU builds (preferred):
   - `/workspace/Niodoo-Final/third_party/onnxruntime-linux-x64-gpu-1.24.0/lib`
   - `/workspace/Niodoo-Final/third_party/onnxruntime-linux-x64-gpu-1.23.2/lib`
   - `/workspace/Niodoo-Final/third_party/onnxruntime-linux-x64-gpu-1.18.1/lib`
   - `/workspace/Niodoo-Final/third_party/onnxruntime-linux-x64-gpu-1.16.3/lib`
   - `/workspace/onnxruntime-linux-x64-gpu-1.24.0/lib`
   - `/workspace/onnxruntime-linux-x64-gpu-1.23.2/lib`
   - `/workspace/onnxruntime-linux-x64-gpu-1.18.1/lib`

2. CPU builds (fallback):
   - `/workspace/Niodoo-Final/third_party/onnxruntime-linux-x64-1.18.1/lib`
   - `/workspace/Niodoo-Final/third_party/onnxruntime-linux-x64-1.16.3/lib`

### Method 3: Environment Files

**Location**: `.runpod_env.sh`, `config/h200.env`, `config/a100.env`, etc.

**Usage**:
```bash
# Source the environment file
source .runpod_env.sh

# Or for hardware-specific configs
source config/h200.env
```

**What they contain**:
- Pre-configured `LD_LIBRARY_PATH` with ONNX Runtime paths
- `ORT_STRICT_VERSION_CHECK=0` for compatibility
- CUDA paths
- Other runtime optimizations

## Environment Variables Set

The bootstrap process automatically sets:

| Variable | Purpose | Example |
|----------|---------|---------|
| `LD_LIBRARY_PATH` | Library search path | `/workspace/.../onnxruntime-linux-x64-gpu-1.18.1/lib:...` |
| `ORT_DYLIB_PATH` | Direct path to ONNX Runtime library | `/workspace/.../libonnxruntime.so` |
| `ORT_DYLIB_DEFAULT_PATH` | Default ONNX Runtime directory | `/workspace/.../onnxruntime-linux-x64-gpu-1.18.1/lib` |
| `ORT_STRICT_VERSION_CHECK` | Disable strict version checking | `0` |

## Custom Workspace Locations

If your project is in a non-standard location, set:

```bash
export WORKSPACE_ROOT=/path/to/your/project
```

The auto-detection will use this path instead of `/workspace/Niodoo-Final`.

## Troubleshooting

### Issue: "libonnxruntime.so: cannot open shared object file"

**Solution**: The auto-bootstrap should handle this, but if it doesn't:

1. **Check if ONNX Runtime exists**:
   ```bash
   find /workspace -name "libonnxruntime.so" 2>/dev/null
   ```

2. **Manually source bootstrap**:
   ```bash
   source scripts/bootstrap_onnx.sh
   ```

3. **Verify LD_LIBRARY_PATH**:
   ```bash
   echo $LD_LIBRARY_PATH
   ```

### Issue: "Unexpected input data type" (int64 vs float16)

**This is a different issue** - it's an ONNX model input type mismatch, not a library loading issue. The auto-bootstrap handles library loading correctly.

### Issue: Auto-detection not working

**Check**:
1. ONNX Runtime is in one of the standard paths
2. The library file exists: `libonnxruntime.so` or `libonnxruntime_providers_cuda.so`
3. Permissions are correct (readable)

**Manual override**:
```bash
export LD_LIBRARY_PATH="/your/custom/path:$LD_LIBRARY_PATH"
```

## Integration with CI/CD

For automated testing, the soak test and other binaries automatically bootstrap ONNX Runtime, so no manual setup is needed in CI pipelines.

## Summary

✅ **Automatic**: Binaries auto-detect and configure ONNX Runtime  
✅ **Multiple Methods**: Shell script, built-in detection, or environment files  
✅ **Fallback Support**: Tries GPU builds first, falls back to CPU  
✅ **No Manual Setup**: Works out of the box in standard configurations  

**You should never need to manually set `LD_LIBRARY_PATH` for ONNX Runtime** - the system handles it automatically!


