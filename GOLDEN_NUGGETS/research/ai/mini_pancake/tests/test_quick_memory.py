import os
import sys
import tempfile
from pathlib import Path
import time

# Ensure local imports work when run from this folder
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
	sys.path.insert(0, str(BASE_DIR))

import mini_pancake_quant as mpq


def assert_true(condition: bool, msg: str):
	if not condition:
		raise AssertionError(msg)


def test_memory_store_and_recall():
	unique = f"purple-elephant-{int(time.time())}"
	mpq.store_text(f"This is a test note about {unique} hiding in the room.", {"source": "unit:test"})
	ctx = mpq.recall(unique)
	print("[memory] recall output:\n", ctx)
	assert_true(unique in ctx, "Expected unique token not found in recall output")


def test_indicator_extraction():
	sample = "Contact http://example.com/page and 198.51.100.23 now."
	inds = mpq.extract_indicators(sample)
	print("[indicators]", inds)
	assert_true("198.51.100.23" in inds.get("ips", []), "Expected IP not extracted")
	assert_true("example.com" in inds.get("domains", []), "Expected domain not extracted")


def test_harvest_and_recall_from_tempfile():
	with tempfile.TemporaryDirectory() as tmpd:
		tmpf = Path(tmpd) / "sample.log"
		content = "Suspicious connect to 198.51.100.23 from test harness."
		tmpf.write_text(content, encoding="utf-8")
		added = mpq.harvest_logs(paths=[str(tmpf)])
		print(f"[harvest] added {added}")
		ctx = mpq.recall("198.51.100.23")
		print("[recall-after-harvest]\n", ctx)
		assert_true("198.51.100.23" in ctx, "Expected harvested IP not recalled from memory")


if __name__ == "__main__":
	print("[tests] starting quick memory tests…")
	test_memory_store_and_recall()
	test_indicator_extraction()
	test_harvest_and_recall_from_tempfile()
	print("[tests] all quick tests passed ✅")


