#!/usr/bin/env python3
"""
Build a LoRA training batch from Euler test results enriched with TopoCoT telemetry.

Usage:
    python3 python_scripts/topocot_lora_batch.py --results-dir euler_test_results_YYYYMMDD_HHMMSS
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def load_results(results_dir: Path) -> Dict[str, Any]:
    results_path = results_dir / "euler_results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    with results_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_sample(problem: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    topocot = problem.get("topocot")
    topology = problem.get("topology_signature")
    reflection_summary = None
    if topocot:
        reflection_summary = topocot.get("reflection_summary")

    if not topocot or not topology or not reflection_summary:
        return None

    betti = topology.get("betti_numbers") or [0, 0, 0]
    if len(betti) < 3:
        betti = list(betti) + [0] * (3 - len(betti))

    features = [
        float(betti[0]),
        float(betti[1]),
        float(betti[2]),
        float(topology.get("spectral_gap") or 0.0),
        float(topology.get("persistence_entropy") or 0.0),
        float(topocot.get("thinking_depth") or 0.0),
        float(topocot.get("pivot_score") or 0.0),
    ]

    target = f"Slow down and structure the proof. {reflection_summary}"

    return {
        "problem_id": problem.get("problem_id"),
        "features": features,
        "target": target,
        "metadata": {
            "quality_score": problem.get("quality_score"),
            "gating_path": problem.get("gating_path"),
            "issues": topocot.get("issues", []),
            "mathematical_indicators": problem.get("mathematical_indicators"),
        },
    }


def build_batch(results_dir: Path) -> List[Dict[str, Any]]:
    payload = load_results(results_dir)
    batch: List[Dict[str, Any]] = []
    for problem in payload.get("results", []):
        sample = build_sample(problem)
        if sample:
            batch.append(sample)
    return batch


def save_batch(batch: Iterable[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in batch:
            handle.write(json.dumps(record))
            handle.write("\n")


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert Euler TopoCoT telemetry into a LoRA training batch."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Path to euler_test_results_<timestamp> directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/euler"),
        help="Directory to store the generated batch (default: analysis/euler).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    batch = build_batch(args.results_dir)
    if not batch:
        print("No TopoCoT telemetry found in results; nothing to export.")
        return 0

    timestamp = args.results_dir.name
    output_path = args.output_dir / f"{timestamp}_topocot_lora_batch.jsonl"
    save_batch(batch, output_path)
    print(f"Wrote {len(batch)} samples to {output_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

