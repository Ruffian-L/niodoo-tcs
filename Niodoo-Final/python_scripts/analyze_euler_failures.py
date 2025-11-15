#!/usr/bin/env python3
"""
Analyze Euler test results and extract failure details, including TopoCoT telemetry.

Usage:
    python3 python_scripts/analyze_euler_failures.py --results-dir euler_test_results_YYYYMMDD_HHMMSS
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_results(results_dir: Path) -> Dict[str, Any]:
    results_path = results_dir / "euler_results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    with results_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def summarise_problem(problem: Dict[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "problem_id": problem.get("problem_id"),
        "quality_score": problem.get("quality_score"),
        "gating_path": problem.get("gating_path"),
        "response_snippet": (problem.get("response") or "")[:360],
        "topology_signature": problem.get("topology_signature"),
        "pad_state": problem.get("pad_emotional_state"),
        "memory_retrieval_count": problem.get("memory_retrieval_count"),
        "processing_time_ms": problem.get("processing_time_ms"),
        "novel_topology": problem.get("novel_topology"),
        "extreme_emotion": problem.get("extreme_emotion"),
    }

    topocot = problem.get("topocot")
    if topocot:
        summary["topocot"] = {
            "score_overall": topocot.get("score_overall"),
            "score_completeness": topocot.get("score_completeness"),
            "score_consistency": topocot.get("score_consistency"),
            "score_actionability": topocot.get("score_actionability"),
            "issues": topocot.get("issues") or [],
            "thinking_depth": topocot.get("thinking_depth"),
            "pivot_score": topocot.get("pivot_score"),
            "reflection_summary": topocot.get("reflection_summary"),
            "raw_json": _truncate(topocot.get("raw_json")),
        }
    else:
        summary["topocot"] = None

    return summary


def _truncate(value: Optional[str], limit: int = 512) -> Optional[str]:
    if value is None:
        return None
    if len(value) <= limit:
        return value
    return value[: limit - 1] + "…"


def build_summary(results_dir: Path) -> Dict[str, Any]:
    payload = load_results(results_dir)
    per_problem = [summarise_problem(problem) for problem in payload.get("results", [])]

    summary: Dict[str, Any] = {
        "results_dir": str(results_dir),
        "summary": payload.get("summary", {}),
        "gating_analysis": payload.get("gating_analysis", {}),
        "intelligence_assessment": payload.get("intelligence_assessment", {}),
        "per_problem": per_problem,
    }
    return summary


def save_summary(summary: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarise Euler test failures and TopoCoT telemetry."
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
        help="Directory to store structured summaries (default: analysis/euler).",
    )
    args = parser.parse_args(argv)

    summary = build_summary(args.results_dir)
    output_dir = args.output_dir
    timestamp = summary.get("summary", {}).get("test_id") or args.results_dir.name
    output_path = output_dir / f"{timestamp}_summary.json"
    save_summary(summary, output_path)

    print(json.dumps(summary, indent=2))
    print(f"\nSummary written to {output_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())

