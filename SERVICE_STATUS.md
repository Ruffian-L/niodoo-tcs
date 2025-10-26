# Service Status & Restart Configuration

## Current Services

All scripts saved to `/workspace/Niodoo-Final` (network drive)

### Active Services
- ✅ **vLLM**: Running on port 8000
- ✅ **Qdrant**: Running on port 6333  
- ✅ **Ollama**: Running on port 11434

## Usage

### Check Status
```bash
cd /workspace/Niodoo-Final
./supervisor.sh status
```

### Start All Services
```bash
cd /workspace/Niodoo-Final
./supervisor.sh start
```

### Restart All Services
```bash
cd /workspace/Niodoo-Final
./restart-services.sh
```

### Start with Auto-Restart Monitor
```bash
cd /workspace/Niodoo-Final
./start-services.sh
```

## Problem Found & Fixed

### Issue: High CPU Usage from Rust Compilation
- **Problem**: Rust compiler (rustc) and linker (ld) were consuming 27-32% CPU continuously
- **Impact**: Slowing down entire runpod
- **Solution**: Killed the compilation processes (PIDs: 251389, 251540, 251544)

### Issue: No Auto-Restart on Pod Restart
- **Problem**: vLLM, Qdrant, and Ollama running as background processes without restart mechanism
- **Solution**: Created supervisor.sh script with monitoring and auto-restart
- **Features**:
  - Monitors all three services
  - Automatically restarts crashed services
  - Tracks PIDs
  - Logs to /tmp/

## Scripts Created

1. **supervisor.sh** - Main service manager with auto-restart
2. **restart-services.sh** - Quick restart command
3. **start-services.sh** - Start with background monitoring

## After Pod Restart

Run this command to start all services with auto-restart:
```bash
cd /workspace/Niodoo-Final && ./start-services.sh
```

Or manually:
```bash
cd /workspace/Niodoo-Final
./supervisor.sh start
```

## Log Files

- vLLM: `/tmp/vllm.log`
- Qdrant: `/tmp/qdrant.log`
- Ollama: `/tmp/ollama.log`
- Supervisor: `/tmp/supervisor.log`

## Verify Services

```bash
# Check if services are listening
curl http://127.0.0.1:8000/v1/models  # vLLM
curl http://127.0.0.1:6333/collections  # Qdrant
curl http://127.0.0.1:11434/api/tags  # Ollama
```

