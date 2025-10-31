#!/usr/bin/env bash
# Big soak test: 4 parallel jobs, 100 cycles each

set -euo pipefail

cd /workspace/Niodoo-Final

# Set environment - MUST be exported before running topology_bench
export QDRANT_USE_GRPC=true
export CURATOR_BACKEND=vllm
export CURATOR_VLLM_ENDPOINT=http://127.0.0.1:5001
export VLLM_ENDPOINT=http://127.0.0.1:5001
export QDRANT_URL=http://127.0.0.1:6333
export ENABLE_CURATOR=true

echo "🔥 Big Soak Test: 4 jobs x 100 cycles"
echo "======================================"
echo "Config: gRPC Qdrant + vLLM Curator"
echo "QDRANT_USE_GRPC=$QDRANT_USE_GRPC"
echo "CURATOR_BACKEND=$CURATOR_BACKEND"
echo "ENABLE_CURATOR=$ENABLE_CURATOR"
echo ""

mkdir -p logs/soak/big

# Check services
echo "Checking services..."
curl -sS http://127.0.0.1:6333/healthz > /dev/null || { echo "❌ Qdrant not running"; exit 1; }
curl -sS http://127.0.0.1:5001/v1/models > /dev/null || { echo "❌ vLLM not running"; exit 1; }
echo "✅ Services ready"

# Run 4 parallel jobs, 100 cycles each - Pass env vars explicitly
for i in 1 2 3 4; do
  echo "Starting job $i..."
  env QDRANT_USE_GRPC=true \
      CURATOR_BACKEND=vllm \
      CURATOR_VLLM_ENDPOINT=http://127.0.0.1:5001 \
      VLLM_ENDPOINT=http://127.0.0.1:5001 \
      QDRANT_URL=http://127.0.0.1:6333 \
      ENABLE_CURATOR=true \
      RUST_LOG=info \
  ./target/release/topology_bench \
    --cycles 100 \
    --dataset results/benchmarks/topology/curated_eval.tsv \
    > logs/soak/big/soak_big_job${i}.log 2>&1 &
  echo $! > logs/soak/big/soak_big_job${i}.pid
  sleep 1
done

echo ""
echo "⏳ Running big soak test..."
echo "Monitor with: tail -f logs/soak/big/soak_big_job*.log"
echo "Check status: ps aux | grep topology_bench"

# Wait for all jobs
for pidfile in logs/soak/big/soak_big_job*.pid; do
  pid=$(cat "$pidfile")
  echo "Waiting for $(basename $pidfile) (pid=$pid)..."
  while kill -0 "$pid" 2>/dev/null; do
    sleep 60
  done
  echo "$(basename $pidfile) complete."
done

echo ""
echo "✅ Big soak test complete!"
echo ""
echo "📊 Summary:"
python3 << 'PY'
import csv,glob,statistics as s
files=sorted(glob.glob('results/benchmarks/topology/topology_benchmark_*.csv'))[-4:]
if not files:
    print("No CSV files found")
    exit(0)
bl_r=[];hy_r=[];bl_l=[];hy_l=[]
for f in files:
  try:
    for r in csv.DictReader(open(f)):
      try:
        bl_r.append(float(r.get('rouge_baseline', 0))); hy_r.append(float(r.get('rouge_hybrid', 0)))
        bl_l.append(float(r.get('latency_baseline_ms', 0))); hy_l.append(float(r.get('latency_hybrid_ms', 0)))
      except: pass
  except: pass
if not bl_r:
    print("No data found in CSV files")
    exit(0)
def p(v,q):
  v=sorted(v); i=(len(v)-1)*q; lo=int(i); hi=min(lo+1,len(v)-1); a=v[lo]; b=v[hi]; return a+(b-a)*(i-lo)
print('N cycles:',len(bl_r))
print('ROUGE mean baseline',round(s.mean(bl_r),3),'hybrid',round(s.mean(hy_r),3),'delta',round(s.mean(hy_r)-s.mean(bl_r),3))
print('LATENCY mean baseline',int(s.mean(bl_l)),'hybrid',int(s.mean(hy_l)),'delta',int(s.mean(hy_l)-s.mean(bl_l)))
print('LATENCY p50/p95/p99 baseline',int(p(bl_l,0.5)),int(p(bl_l,0.95)),int(p(bl_l,0.99)))
print('LATENCY p50/p95/p99 hybrid ',int(p(hy_l,0.5)),int(p(hy_l,0.95)),int(p(hy_l,0.99)))
PY

