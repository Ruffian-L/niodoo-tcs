# Release Validation Setup Prompt

Copy this prompt into the public repository (e.g., `.github/ISSUE_TEMPLATE` or validation docs) so validators have a repeatable checklist.

```
You are the Release Validator. Validate the topology-augmented AI stack honestly (no mocks).

Success means: real vLLM + Qdrant + Ollama running; baseline vs hybrid evaluated; artifacts saved; metrics summarized with p50/p95/p99 and confidence intervals; all settings logged. No cherry-picking.

1) Environment
- OS: Ubuntu 20.04+ with NVIDIA GPU
- Install system deps:
  sudo apt-get update && sudo apt-get install -y build-essential cmake curl git python3-venv python3-pip pkg-config
- Rust: curl https://sh.rustup.rs -sSf | sh -s -- -y; source "$HOME/.cargo/env"; rustup toolchain install 1.87.0; rustup default 1.87.0
- Python venv: python3 -m venv venv && source venv/bin/activate && pip install --upgrade pip wheel
- Python pkgs: pip install vllm qdrant-client requests pandas numpy scipy

2) Models and services (env-first)
mkdir -p models
export VLLM_MODEL=${VLLM_MODEL:-/workspace/models/Qwen2.5-7B-Instruct-AWQ}
export VLLM_MODEL_ID=${VLLM_MODEL_ID:-/workspace/models/Qwen2.5-7B-Instruct-AWQ}
export VLLM_HOST=${VLLM_HOST:-127.0.0.1}
export VLLM_PORT=${VLLM_PORT:-5001}
export VLLM_ENDPOINT=${VLLM_ENDPOINT:-http://127.0.0.1:5001}
export QDRANT_URL=${QDRANT_URL:-http://127.0.0.1:6333}
export QDRANT_COLLECTION=${QDRANT_COLLECTION:-experiences}
export QDRANT_VECTOR_SIZE=${QDRANT_VECTOR_SIZE:-896}
export OLLAMA_ENDPOINT=${OLLAMA_ENDPOINT:-http://127.0.0.1:11434}
export CURATOR_MODEL=${CURATOR_MODEL:-qwen2:0.5b}
export TOKENIZER_JSON=${TOKENIZER_JSON:-$(pwd)/tokenizer.json}

# Start services (separate terminals or tmux panes recommended)
# vLLM
venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model "$VLLM_MODEL_ID" --host "$VLLM_HOST" --port "$VLLM_PORT" --gpu-memory-utilization 0.85 --trust-remote-code &

# Qdrant (Docker)
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest

# Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull "$CURATOR_MODEL"

# Health checks
curl -s "$VLLM_ENDPOINT/v1/models" | head -c 200
curl -s "$QDRANT_URL/collections" | head -c 200
curl -s "$OLLAMA_ENDPOINT/api/tags" | head -c 200

3) Build project and configure runtime
export RUST_LOG=${RUST_LOG:-info}
cargo build -p niodoo_real_integrated --release

# Core knobs (env-driven)
export ENABLE_CURATOR=${ENABLE_CURATOR:-true}
export CURATOR_AUTONOMOUS=${CURATOR_AUTONOMOUS:-false}
export CURATOR_QUALITY_THRESHOLD=${CURATOR_QUALITY_THRESHOLD:-0.85}
export ERAG_TOP_K=${ERAG_TOP_K:-20}
export TOPOLOGY_MODE=${TOPOLOGY_MODE:-Hybrid}   # Baseline|Hybrid if supported
export REAL_TEST=1
export MOCK_MODE=0

4) Sanity run (curated eval, 64 cycles)
./target/release/topology_bench --cycles 64 --dataset results/benchmarks/topology/curated_eval.tsv

# Summarize latest CSV
python3 - << 'PY'
import csv,glob,statistics as s
f=sorted(glob.glob('results/benchmarks/topology/topology_benchmark_*.csv'))[-1]
r=list(csv.DictReader(open(f)))
bl_r=[float(x['rouge_baseline']) for x in r if x.get('rouge_baseline')]
hy_r=[float(x['rouge_hybrid']) for x in r if x.get('rouge_hybrid')]
bl_l=[float(x['latency_baseline_ms']) for x in r if x.get('latency_baseline_ms')]
hy_l=[float(x['latency_hybrid_ms']) for x in r if x.get('latency_hybrid_ms')]
def p(v,q):
  v=sorted(v); i=(len(v)-1)*q; lo=int(i); hi=min(lo+1,len(v)-1); a=v[lo]; b=v[hi]; return a+(b-a)*(i-lo)
print('FILE',f)
print('N',len(bl_r))
print('ROUGE mean baseline',round(s.mean(bl_r),3),'hybrid',round(s.mean(hy_r),3),'delta',round(s.mean(hy_r)-s.mean(bl_r),3))
for lab,arr in [('baseline',bl_r),('hybrid',hy_r)]:
  print('ROUGE p50/p95/p99',lab, round(p(arr,0.5),3), round(p(arr,0.95),3), round(p(arr,0.99),3))
print('LAT mean baseline',int(s.mean(bl_l)),'hybrid',int(s.mean(hy_l)),'delta',int(s.mean(hy_l)-s.mean(bl_l)))
for lab,arr in [('baseline',bl_l),('hybrid',hy_l)]:
  print('LAT p50/p95/p99',lab, int(p(arr,0.5)), int(p(arr,0.95)), int(p(arr,0.99)))
PY

5) Soak (c=4, 100 cycles each)
mkdir -p logs/soak
for i in 1 2 3 4; do \
  ./target/release/topology_bench --cycles 100 --dataset results/benchmarks/topology/curated_eval.tsv \
    > logs/soak/soak_c4_job${i}.log 2>&1 & echo $! > logs/soak/soak_c4_job${i}.pid; \
  sleep 1; \
done

# Wait and summarize artifacts
sleep 5
python3 - << 'PY'
import csv,glob,statistics as s
files=sorted(glob.glob('results/benchmarks/topology/topology_benchmark_*.csv'))[-4:]
bl_r=[];hy_r=[];bl_l=[];hy_l=[]
for f in files:
  for r in csv.DictReader(open(f)):
    try:
      bl_r.append(float(r['rouge_baseline'])); hy_r.append(float(r['rouge_hybrid']))
      bl_l.append(float(r['latency_baseline_ms'])); hy_l.append(float(r['latency_hybrid_ms']))
    except: pass
def p(v,q):
  v=sorted(v); i=(len(v)-1)*q; lo=int(i); hi=min(lo+1,len(v)-1); a=v[lo]; b=v[hi]; return a+(b-a)*(i-lo)
print('FILES',files)
print('N',len(bl_r))
print('ROUGE mean baseline',round(s.mean(bl_r),3),'hybrid',round(s.mean(hy_r),3),'delta',round(s.mean(hy_r)-s.mean(bl_r),3))
print('ROUGE p50/p95/p99 baseline',round(p(bl_r,0.5),3),round(p(bl_r,0.95),3),round(p(bl_r,0.99),3))
print('ROUGE p50/p95/p99 hybrid ',round(p(hy_r,0.5),3),round(p(hy_r,0.95),3),round(p(hy_r,0.99),3))
print('LAT mean baseline',int(s.mean(bl_l)),'hybrid',int(s.mean(hy_l)),'delta',int(s.mean(hy_l)-s.mean(bl_l)))
print('LAT p50/p95/p99 baseline',int(p(bl_l,0.5)),int(p(bl_l,0.95)),int(p(bl_l,0.99)))
print('LAT p50/p95/p99 hybrid ',int(p(hy_l,0.5)),int(p(hy_l,0.95)),int(p(hy_l,0.99)))
PY

6) Honest ablations
# Topology off (if supported)
TOPOLOGY_MODE=Baseline ./target/release/topology_bench --cycles 64 --dataset results/benchmarks/topology/curated_eval.tsv
# Curator off
ENABLE_CURATOR=false ./target/release/topology_bench --cycles 64 --dataset results/benchmarks/topology/curated_eval.tsv
# Knob sweep examples
ERAG_TOP_K=10 ./target/release/topology_bench --cycles 64 --dataset results/benchmarks/topology/curated_eval.tsv
CURATOR_QUALITY_THRESHOLD=0.75 ./target/release/topology_bench --cycles 64 --dataset results/benchmarks/topology/curated_eval.tsv

7) Health + troubleshooting
- If tokenizer error: export TOKENIZER_JSON=/path/to/tokenizer.json
- Verify services: curl $VLLM_ENDPOINT/v1/models; curl $QDRANT_URL/collections; curl $OLLAMA_ENDPOINT/api/tags
- Artifacts live under results/benchmarks/topology/*.csv and *.json

8) Output
- Do not modify outputs. Save logs and CSV/JSON artifacts to the repo. Print summary tables and the exact env used.
