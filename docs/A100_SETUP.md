# A100 RunPod Setup Guide

This guide covers setting up the Niodoo pipeline on an NVIDIA A100-SXM4-80GB GPU instance.

## 1. Bootstrap the Environment

```bash
cd /workspace/Niodoo-Final
./scripts/start_a100_bootstrap.sh
```

This script:
- Validates A100 GPU detection (`nvidia-smi`)
- Locates CUDA installation (checks 13.0, 12.8, 12, 11.8)
- Wires ONNX Runtime GPU libraries into `LD_LIBRARY_PATH`
- Generates `config/a100.env` with optimized settings
- Optionally builds the workspace with GPU features

## 2. Source the A100 Environment

```bash
source config/a100.env
```

Key environment variables set:
- `HARDWARE=a100` - Hardware profile identifier
- `VLLM_GPU_MEMORY_UTILIZATION=0.85` - 85% of 80GB VRAM (~68GB)
- `VLLM_MAX_MODEL_LEN=32768` - 32K context length
- `VLLM_MAX_NUM_BATCHED_TOKENS=16384` - Large batch capacity
- `VLLM_MAX_NUM_SEQS=128` - High concurrency
- `QWEN_CUDA_MEM_LIMIT_MB=6144` - 6GB for embeddings (leaves room for training)

## 3. Start Services

```bash
./start_all_services.sh --hardware a100
```

This will:
- Start vLLM with A100-optimized settings
- Launch Qdrant vector database
- Start Ollama (if configured)

## A100-Specific Optimizations

### vLLM Configuration
- **Memory**: 85% utilization (~68GB) leaves room for training workloads
- **KV Cache**: FP16 (A100 doesn't support FP8 like H200/Hopper)
- **Attention**: Flash Attention enabled
- **Prefill**: Chunked prefill enabled for long sequences
- **DeepGEMM**: Disabled (A100 Ampere architecture)

### Embedding Memory
- ONNX Runtime GPU memory limit: 6GB
- Leaves ~62GB free for concurrent vLLM inference and training

### Training Considerations
- With vLLM using ~68GB, you have ~12GB free for LoRA training
- For QLoRA training, reduce `VLLM_GPU_MEMORY_UTILIZATION` to 0.70-0.75
- Consider stopping vLLM during intensive training runs

## Memory Breakdown (80GB A100)

| Component | Memory Usage |
|-----------|--------------|
| vLLM (85% util) | ~68GB |
| Embeddings (ONNX) | ~6GB |
| Training buffer | ~6GB |
| **Total** | ~80GB |

## Troubleshooting

### Out of Memory Errors
- Reduce `VLLM_GPU_MEMORY_UTILIZATION` to 0.75-0.80
- Lower `VLLM_MAX_NUM_BATCHED_TOKENS` to 8192
- Reduce `VLLM_MAX_NUM_SEQS` to 64

### CUDA Not Found
- Check CUDA installation: `ls /usr/local/cuda*`
- Verify `LD_LIBRARY_PATH` includes CUDA lib64
- Run bootstrap script again: `./scripts/start_a100_bootstrap.sh`

### ONNX Runtime CPU Fallback
- Verify ONNX Runtime GPU libs exist: `ls /workspace/Niodoo-Final/third_party/onnxruntime-linux-x64-gpu-*`
- Check `LD_LIBRARY_PATH` includes ONNX Runtime lib directory
- Set `QWEN_CUDA_MEM_LIMIT_MB` explicitly if needed

## Next Steps

- Monitor GPU utilization: `watch -n 1 nvidia-smi`
- Check vLLM logs: `tail -f /tmp/vllm_service.log`
- Verify services: `curl http://127.0.0.1:5001/v1/models`
