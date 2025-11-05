#!/usr/bin/env bash
# 64-cycle soak test with gRPC + vLLM curator

set -euo pipefail

cd /workspace/Niodoo-Final

# Source runtime environment first
if [[ -f "tcs_runtime.env" ]]; then
    set -a
    source tcs_runtime.env
    set +a
    echo "✅ Loaded environment from tcs_runtime.env"
fi

# Override with gRPC + vLLM curator settings
export QDRANT_USE_GRPC=true
export CURATOR_BACKEND=vllm
export CURATOR_VLLM_ENDPOINT=${VLLM_ENDPOINT:-http://127.0.0.1:5001}
export ENABLE_CURATOR=true

echo "🔥🔥🔥 64-CYCLE SOAK TEST 🔥🔥🔥"
echo "=================================="
echo "Config: gRPC Qdrant + vLLM Curator"
echo "QDRANT_USE_GRPC=$QDRANT_USE_GRPC"
echo "CURATOR_BACKEND=$CURATOR_BACKEND"
echo "ENABLE_CURATOR=$ENABLE_CURATOR"
echo "VLLM_ENDPOINT=${VLLM_ENDPOINT:-http://127.0.0.1:5001}"
echo "QDRANT_URL=${QDRANT_URL:-http://127.0.0.1:6333}"
echo ""

mkdir -p logs/soak/64cycle

# Check services
echo "Checking services..."
curl -sS http://127.0.0.1:6333/healthz > /dev/null || { echo "❌ Qdrant not running"; exit 1; }
curl -sS http://127.0.0.1:5001/v1/models > /dev/null || { echo "❌ vLLM not running"; exit 1; }
echo "✅ Services ready"
echo ""

# Run 64 cycles
echo "🚀 Starting 64-cycle soak test..."
echo ""

env QDRANT_USE_GRPC=true \
    CURATOR_BACKEND=vllm \
    CURATOR_VLLM_ENDPOINT="${VLLM_ENDPOINT:-http://127.0.0.1:5001}" \
    VLLM_ENDPOINT="${VLLM_ENDPOINT:-http://127.0.0.1:5001}" \
    QDRANT_URL="${QDRANT_URL:-http://127.0.0.1:6333}" \
    OLLAMA_ENDPOINT="${OLLAMA_ENDPOINT:-http://127.0.0.1:11434}" \
    TOKENIZER_JSON="${TOKENIZER_JSON:-}" \
    ENABLE_CURATOR=true \
    RUST_LOG=info \
./target/release/topology_bench \
  --cycles 64 \
  --dataset results/benchmarks/topology/curated_eval.tsv \
  2>&1 | tee logs/soak/64cycle/soak_64cycle.log

echo ""
echo "✅ 64-cycle soak test complete!"
echo ""
echo "📊 Summary:"
python3 << 'PY'
import csv,glob,statistics as s
files=sorted(glob.glob('results/benchmarks/topology/topology_benchmark_*.csv'))[-1:]
if files:
    bl_r=[];hy_r=[];bl_l=[];hy_l=[]
    for f in files:
      for r in csv.DictReader(open(f)):
        try:
          bl_r.append(float(r.get('rouge_baseline', 0))); hy_r.append(float(r.get('rouge_hybrid', 0)))
          bl_l.append(float(r.get('latency_baseline_ms', 0))); hy_l.append(float(r.get('latency_hybrid_ms', 0)))
        except: pass
    if hy_l:
        def p(v,q):
          v=sorted(v); i=(len(v)-1)*q; lo=int(i); hi=min(lo+1,len(v)-1); a=v[lo]; b=v[hi]; return a+(b-a)*(i-lo)
        print(f"N cycles: {len(hy_l)}")
        print(f"ROUGE mean: baseline={s.mean(bl_r):.3f}, hybrid={s.mean(hy_r):.3f}, delta={s.mean(hy_r)-s.mean(bl_r):.3f}")
        print(f"LATENCY mean: baseline={s.mean(bl_l):.0f}ms, hybrid={s.mean(hy_l):.0f}ms, delta={s.mean(hy_l)-s.mean(bl_l):.0f}ms")
        print(f"LATENCY p50/p95/p99 baseline: {p(bl_l,0.5):.0f}/{p(bl_l,0.95):.0f}/{p(bl_l,0.99):.0f}")
        print(f"LATENCY p50/p95/p99 hybrid:  {p(hy_l,0.5):.0f}/{p(hy_l,0.95):.0f}/{p(hy_l,0.99):.0f}")
PY

echo ""
echo "🎯 Check logs: logs/soak/64cycle/soak_64cycle.log"

