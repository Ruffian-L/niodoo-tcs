# Service Restart Summary

## Date: 2025-01-26

## Status: 2/3 Services Operational

### ✅ Ollama - OPERATIONAL
- Status: Running on port 11434
- Model: qwen2.5:0.5b successfully pulled
- Test: `curl http://127.0.0.1:11434/api/embeddings` returns vector
- Vector sample: `[5.39, 11.35, -7.31, ...]`

### ✅ Qdrant - OPERATIONAL  
- Status: Running on port 6333
- Collection: experiences recreated with correct vector size
- Vector size: 896 (was 768, fixed via PUT request)
- Test: `curl http://127.0.0.1:6333/collections/experiences` returns collection info
- Previous issue: Stale file handle error - resolved by recreating collection

### ⚠️ vLLM - IN PROGRESS
- Status: Multiple processes running but not listening on port 8000
- Model path: `/workspace/models/hf_cache/models--Qwen--Qwen2.5-7B-Instruct-AWQ`
- Issue: Model loading timeout/failure
- Error: HF_HUB_ENABLE_HF_TRANSFER=1 but hf_transfer package not installed
- Fix applied: Set `HF_HUB_ENABLE_HF_TRANSFER=0` environment variable
- Current: Processes running but port 8000 not accepting connections
- Likely cause: Model still loading or initialization failure

## Actions Taken

1. **Killed vLLM stragglers**: Multiple instances running, cleaned up with `pkill -9 -f vllm`
2. **Restarted vLLM**: Used supervisor.sh with corrected model path
3. **Recreated Qdrant collection**: Deleted and recreated experiences collection with vector size 896
4. **Pulled Ollama model**: Successfully downloaded qwen2.5:0.5b
5. **Set environment variables**: Disabled hf_transfer for vLLM

## Environment Configuration

File: `tcs_runtime.env`
```
export VLLM_MODEL=/workspace/models/hf_cache/hub/models--Qwen--Qwen2.5-7B-Instruct-AWQ
export VLLM_ENDPOINT=http://127.0.0.1:8000
export OLLAMA_ENDPOINT=http://127.0.0.1:11434
export QDRANT_URL=http://127.0.0.1:6333
export QDRANT_COLLECTION=experiences
export QDRANT_VECTOR_SIZE=896
export RUST_LOG=info,tcs_core=debug
```

## Next Steps

### Immediate
1. Wait for vLLM to finish loading (may take 5-10 minutes for 7B model)
2. Verify vLLM endpoint: `curl http://127.0.0.1:8000/v1/models`
3. If still not responding, check logs: `tail -f /tmp/vllm.log`

### vLLM Troubleshooting Options
1. Check GPU memory: `nvidia-smi`
2. Try smaller model if available
3. Check for CUDA errors in logs
4. Verify model files are complete
5. Try restarting with `--trust-remote-code` flag

### Once All Services Up
1. Source environment: `source tcs_runtime.env`
2. Run smoke test: Look for `smoke_test_h200.sh` or equivalent
3. Monitor logs: `tail -f logs/smoke_test.log`
4. Verify no fallbacks to degraded modes

## Log Locations
- vLLM: `/tmp/vllm.log`
- Qdrant: `/tmp/qdrant.log`  
- Ollama: `/tmp/ollama.log`
- Supervisor: `/tmp/supervisor.log`

## Commands to Check Status

```bash
# Check all services
ps aux | grep -E "(vllm|qdrant|ollama)" | grep -v grep

# Test endpoints
curl http://127.0.0.1:8000/v1/models  # vLLM
curl http://127.0.0.1:6333/collections/experiences  # Qdrant
curl http://127.0.0.1:11434/api/embeddings -d '{"model":"qwen2.5:0.5b","prompt":"test"}'  # Ollama

# Check ports
lsof -i :8000  # vLLM
lsof -i :6333  # Qdrant
lsof -i :11434  # Ollama
```

