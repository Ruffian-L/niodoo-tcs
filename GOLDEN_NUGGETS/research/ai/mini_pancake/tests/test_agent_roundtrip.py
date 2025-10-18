import os
import sys
from pathlib import Path
import time

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
	sys.path.insert(0, str(BASE_DIR))

import mini_pancake_quant as mpq


def assert_true(cond, msg):
	if not cond:
		raise AssertionError(msg)


def test_prompt_roundtrip_cpu_safe():
	# Force GGUF if present; otherwise transformer fallback
	os.environ.setdefault("USE_LLAMA_CPP", "1")
	gen, _ = mpq.load_qwen()
	mem_key = f"unit-mem-{int(time.time())}"
	mpq.store_text(f"Note: {mem_key} loves noodles.", {"source": "unit:roundtrip"})
	ctx = mpq.recall(mem_key)
	resp = mpq.falcon_answer(gen, ctx, f"What does {mem_key} love?")
	print("[roundtrip] resp=\n", resp[:400])
	assert_true("noodles" in resp.lower() or mem_key in resp, "Agent did not appear to use memory context")


if __name__ == "__main__":
	print("[tests] starting roundtrip test…")
	test_prompt_roundtrip_cpu_safe()
	print("[tests] roundtrip passed ✅")


