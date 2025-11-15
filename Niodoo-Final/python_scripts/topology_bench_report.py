#!/usr/bin/env python3
"""
Topology Benchmark Harness
==========================

Runs paired Euler intelligence batches in baseline vs hybrid topology modes,
collects key metrics, and renders scatter plots that correlate thinking_depth
with mathematical quality.  Outputs both the raw artifacts and a summary JSON
for downstream publishing.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
RUN_SCRIPT = WORKSPACE_ROOT / "run_euler_intelligence_test.sh"
VIS_ROOT = WORKSPACE_ROOT / "visualizations"

THINKING_DEPTH_REGEX = re.compile(r"thinking_depth=([\d.+\-eE]+)")


class BenchmarkRun:
    def __init__(self, mode: str, output_dir: Path, results: Dict[str, Any], log_path: Path):
        self.mode = mode
        self.output_dir = output_dir
        self.results = results
        self.log_path = log_path
        self.thinking_depths = self._extract_thinking_depths(log_path)

    @property
    def per_problem(self) -> List[Dict[str, Any]]:
        return list(self.results.get("results", []))

    @property
    def summary(self) -> Dict[str, Any]:
        return dict(self.results.get("summary", {}))

    def _extract_thinking_depths(self, log_path: Path) -> List[float]:
        if not log_path.exists():
            return []
        depths: List[float] = []
        with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                match = THINKING_DEPTH_REGEX.search(line)
                if match:
                    try:
                        depths.append(float(match.group(1)))
                    except ValueError:
                        continue
        return depths


def run_euler_batch(
    mode: str,
    problems: int,
    timeout: int,
    smoke: bool,
    extra_env: Optional[Dict[str, str]] = None,
) -> BenchmarkRun:
    if not RUN_SCRIPT.exists():
        raise FileNotFoundError(f"Euler harness not found at {RUN_SCRIPT}")

    env = os.environ.copy()
    env.update(
        {
            "TOPOLOGY_MODE": mode,
            "TDA_ENABLED": "true" if mode == "hybrid" else "false",
        }
    )
    if extra_env:
        env.update(extra_env)

    cmd = [
        str(RUN_SCRIPT),
        "--problems",
        str(problems),
        "--timeout",
        str(timeout),
    ]
    if smoke:
        cmd.append("--smoke")

    process = subprocess.run(
        cmd,
        cwd=str(WORKSPACE_ROOT),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    if process.returncode != 0:
        raise RuntimeError(
            f"Euler run failed for mode={mode} (exit {process.returncode}):\n{process.stderr}"
        )

    output_dir = _parse_output_dir(process.stdout)
    if output_dir is None:
        raise RuntimeError(
            "Unable to determine results directory from harness output.\n"
            "Output follows:\n"
            f"{process.stdout}"
        )

    results_path = output_dir / "euler_results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing results JSON at {results_path}")

    with results_path.open("r", encoding="utf-8") as handle:
        results = json.load(handle)

    log_path = output_dir / "euler_test.log"
    return BenchmarkRun(mode, output_dir, results, log_path)


def _parse_output_dir(stdout: str) -> Optional[Path]:
    for line in stdout.splitlines():
        if "Test Results Location" in line:
            tail = line.split(":", 1)[-1].strip()
            # Lines look like: "📁 Test Results Location: euler_test_results_..."
            rel_path = tail.rstrip("/ ")
            candidate = WORKSPACE_ROOT / rel_path
            if candidate.exists():
                return candidate
    return None


def build_scatter_plot(
    baseline: BenchmarkRun,
    hybrid: BenchmarkRun,
    target_dir: Path,
) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)

    baseline_depths = baseline.thinking_depths
    hybrid_depths = hybrid.thinking_depths

    baseline_scores = [entry["quality_score"] for entry in baseline.per_problem]
    hybrid_scores = [entry["quality_score"] for entry in hybrid.per_problem]

    fig, ax = plt.subplots(figsize=(8, 6))
    if baseline_depths and len(baseline_depths) == len(baseline_scores):
        ax.scatter(baseline_depths, baseline_scores, color="tab:blue", label="Baseline", alpha=0.7)
    if hybrid_depths and len(hybrid_depths) == len(hybrid_scores):
        ax.scatter(hybrid_depths, hybrid_scores, color="tab:orange", label="Hybrid", alpha=0.7)

    ax.set_xlabel("Thinking Depth (TopoReflection)")
    ax.set_ylabel("Quality Score")
    ax.set_title("Thinking Depth vs Quality — Euler Benchmark")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)

    plot_path = target_dir / "thinking_depth_vs_quality.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    return plot_path


def summarise_runs(baseline: BenchmarkRun, hybrid: BenchmarkRun) -> Dict[str, Any]:
    def aggregate(run: BenchmarkRun) -> Dict[str, Any]:
        summary = run.summary.copy()
        summary["thinking_depth_samples"] = len(run.thinking_depths)
        summary["thinking_depth_mean"] = (
            sum(run.thinking_depths) / len(run.thinking_depths)
            if run.thinking_depths
            else None
        )
        summary["quality_mean"] = (
            sum(entry["quality_score"] for entry in run.per_problem) / len(run.per_problem)
            if run.per_problem
            else None
        )
        return summary

    return {
        "baseline": aggregate(baseline),
        "hybrid": aggregate(hybrid),
        "delta": {
            "average_quality": (
                aggregate(hybrid)["quality_mean"] - aggregate(baseline)["quality_mean"]
                if aggregate(hybrid)["quality_mean"] is not None
                and aggregate(baseline)["quality_mean"] is not None
                else None
            ),
            "average_thinking_depth": (
                aggregate(hybrid)["thinking_depth_mean"] - aggregate(baseline)["thinking_depth_mean"]
                if aggregate(hybrid)["thinking_depth_mean"] is not None
                and aggregate(baseline)["thinking_depth_mean"] is not None
                else None
            ),
        },
    }


def write_summary(summary: Dict[str, Any], target_dir: Path) -> Path:
    summary_path = target_dir / "benchmark_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary_path


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run paired baseline vs hybrid topology Euler benchmarks and produce scatter plots.",
    )
    parser.add_argument("--problems", type=int, default=5, help="Number of Euler problems to run")
    parser.add_argument("--timeout", type=int, default=180, help="Per-problem timeout in seconds")
    parser.add_argument("--smoke", action="store_true", help="Use smoke mode for faster loops")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory to place visualizations (default: visualizations/topology_benchmark_<timestamp>)",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_argument_parser().parse_args(argv)

    timestamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (VIS_ROOT / f"topology_benchmark_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Benchmark] Running baseline topology (TOPOLOGY_MODE=baseline)")
    baseline_run = run_euler_batch(
        mode="baseline",
        problems=args.problems,
        timeout=args.timeout,
        smoke=args.smoke,
        extra_env={"TOPO_REFLECTION_DEPTH_THRESHOLD": "9999"},
    )
    print(f"[Benchmark] Results captured at {baseline_run.output_dir}")

    print(f"[Benchmark] Running hybrid topology (TOPOLOGY_MODE=hybrid)")
    hybrid_run = run_euler_batch(
        mode="hybrid",
        problems=args.problems,
        timeout=args.timeout,
        smoke=args.smoke,
        extra_env={"TOPO_REFLECTION_DEPTH_THRESHOLD": "0.7"},
    )
    print(f"[Benchmark] Results captured at {hybrid_run.output_dir}")

    plot_path = build_scatter_plot(baseline_run, hybrid_run, output_dir)
    summary = summarise_runs(baseline_run, hybrid_run)
    summary_path = write_summary(summary, output_dir)

    manifest = {
        "timestamp_utc": timestamp,
        "output_dir": str(output_dir),
        "baseline_results": str(baseline_run.output_dir),
        "hybrid_results": str(hybrid_run.output_dir),
        "visualizations": {"thinking_depth_vs_quality": str(plot_path)},
        "summary_file": str(summary_path),
    }

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"[Benchmark] Scatter plot written to {plot_path}")
    print(f"[Benchmark] Summary written to {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

