import asyncio, aiohttp, time, os, json
from statistics import mean

ENDPOINT = os.environ.get("VLLM_ENDPOINT", "http://127.0.0.1:5001").rstrip("/") + "/v1/chat/completions"
MODEL = os.environ.get("VLLM_MODEL", "/workspace/models/Qwen2.5-7B-Instruct-AWQ")
TOTAL = int(os.environ.get("SOAK_TOTAL", "100"))
CONC = int(os.environ.get("SOAK_CONC", "4"))

PROMPTS = [
    "Design a resilient Qdrant write path with retries, jitter, idempotency.",
    "Draft Rust pseudo-code for a circuit breaker with half-open state and rolling window metrics.",
    "Explain AWQ vs GPTQ tradeoffs for 7B on a single 24GB GPU for latency/accuracy.",
    "Outline a topology-aware retrieval reranker using Betti numbers and spectral gap.",
    "Propose a token promotion policy to minimize rouge drop with context packing.",
]

sema = asyncio.Semaphore(CONC)
results = []

async def one(session, i):
    prompt = PROMPTS[i % len(PROMPTS)]
    payload = {
        "model": MODEL,
        "messages": [
            {"role":"system","content":"Be concise, technical. Use bullet points where helpful."},
            {"role":"user","content": prompt},
        ],
        "temperature": 0.6,
        "max_tokens": 256,
    }
    t0 = time.perf_counter()
    async with sema:
        try:
            async with session.post(ENDPOINT, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as r:
                ok = (r.status == 200)
                data = await r.json(content_type=None)
                t1 = time.perf_counter()
                lat_ms = (t1 - t0) * 1000.0
                usage = data.get("usage", {}) if isinstance(data, dict) else {}
                pt = usage.get("prompt_tokens", 0)
                ct = usage.get("completion_tokens", 0)
                results.append({"ok": ok, "ms": lat_ms, "pt": pt, "ct": ct, "status": r.status})
        except Exception as e:
            t1 = time.perf_counter()
            lat_ms = (t1 - t0) * 1000.0
            results.append({"ok": False, "ms": lat_ms, "pt": 0, "ct": 0, "status": str(e)})

async def main():
    connector = aiohttp.TCPConnector(limit_per_host=CONC)
    async with aiohttp.ClientSession(connector=connector) as session:
        await asyncio.gather(*(one(session, i) for i in range(TOTAL)))
    oks = [r for r in results if r["ok"]]
    fails = [r for r in results if not r["ok"]]
    lats = sorted(r["ms"] for r in results)
    def pct(q):
        idx = max(0, min(len(lats)-1, int(q*len(lats))-1))
        return lats[idx]
    p50, p95, p99 = pct(0.5), pct(0.95), pct(0.99)
    avg = mean(lats) if lats else 0
    total_ct = sum(r["ct"] for r in oks)
    total_pt = sum(r["pt"] for r in oks)
    print("SOAK RESULTS")
    print(f"endpoint={ENDPOINT}")
    print(f"model={MODEL}")
    print(f"total={TOTAL} ok={len(oks)} fail={len(fails)} success_rate={len(oks)/TOTAL:.2%}")
    print(f"lat_ms p50={p50:.0f} p95={p95:.0f} p99={p99:.0f} mean={avg:.0f}")
    print(f"tokens prompt={total_pt} completion={total_ct}")
    if fails:
        from collections import Counter
        statuses = Counter(str(r["status"]) for r in fails)
        print("failures:", dict(statuses))

if __name__ == "__main__":
    asyncio.run(main())
