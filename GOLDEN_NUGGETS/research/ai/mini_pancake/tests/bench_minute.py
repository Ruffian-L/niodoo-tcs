import os
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
	sys.path.insert(0, str(BASE_DIR))

import mini_pancake_quant as mpq


def bench_one(prompt: str = "Summarize: local test of capabilities in 2 sentences"):
	os.environ.setdefault("USE_LLAMA_CPP", "1")
	start = time.time()
	gen, _ = mpq.load_qwen()
	resp = mpq.falcon_answer(gen, "(no memory)", prompt)
	elapsed = time.time() - start
	print(f"[bench] elapsed {elapsed:.2f}s; resp head=\n{resp[:300]}")
	return elapsed


if __name__ == "__main__":
	print("[bench] starting 1-minute sanity bench…")
	outs = []
	for i in range(3):
		outs.append(bench_one(f"Trial {i+1}: say 'ok' once and finish."))
	print("[bench] times:", outs)
	print("[bench] done ✅")


