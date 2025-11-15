#!/usr/bin/env python
"""Batch topology feature extraction for stored hidden states.

The CLI now emits rich topology payloads (feature vectors, persistence diagrams,
Betti numbers, entropy, sheaf energy) suitable for TopKD supervision and
hybrid loss training. When multiple inputs are supplied the output is written as
JSON Lines, defaulting to `/workspace/Niodoo-Final/data/processed/`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ntokens import HiddenStateFeatureAdapter


def _load_tensor(path: Path) -> torch.Tensor:
    # RTX 5090 optimization: Load directly to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if path.suffix in {".pt", ".pth"}:
        data = torch.load(path, map_location=device)
        if isinstance(data, dict):
            for key in ("hidden_states", "activations", "embeddings"):
                if key in data:
                    tensor = data[key]
                    break
            else:
                raise ValueError("No hidden state tensor found in torch file")
        else:
            tensor = data
    elif path.suffix in {".npy", ".npz"}:
        arr = torch.from_numpy(np.load(path))  # type: ignore[name-defined]
        tensor = arr
    else:
        raise ValueError(f"Unsupported file extension: {path.suffix}")
    return tensor.float()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        type=Path,
        nargs="*",
        help="Hidden state tensor files or directories containing tensors.",
    )
    parser.add_argument("--pool-mode", choices=["mean", "cls"], default="mean")
    parser.add_argument("--betti-bins", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=(PROJECT_ROOT.parent / "data" / "processed" / "topology_features.jsonl"),
        help="Destination JSONL file (defaults to data/processed/topology_features.jsonl).",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=None,
        help="Pretty-print JSON when writing to stdout (ignored for JSONL files).",
    )
    return parser.parse_args()


def _iter_sources(inputs: Iterable[Path]) -> Iterable[Path]:
    supported_suffixes = {".pt", ".pth", ".npy", ".npz"}
    for path in inputs:
        if not path.exists():
            raise FileNotFoundError(path)
        if path.is_dir():
            for candidate in sorted(path.rglob("*")):
                if candidate.suffix.lower() in supported_suffixes:
                    yield candidate
        elif path.suffix.lower() in supported_suffixes:
            yield path
        else:
            raise ValueError(f"Unsupported file extension: {path.suffix}")


def _diagram_stats(diagram: np.ndarray) -> Dict[str, float]:
    if diagram.size == 0:
        return {
            "count": 0,
            "max_persistence": 0.0,
            "total_persistence": 0.0,
            "mean_persistence": 0.0,
            "max_lifetime": 0.0,
            "mean_lifetime": 0.0,
        }

    finite = diagram[np.isfinite(diagram[:, 1])]
    if finite.size == 0:
        finite = diagram

    births = finite[:, 0]
    deaths = finite[:, 1]
    lifetimes = np.maximum(deaths - births, 0.0)
    persistence = lifetimes

    count = float(finite.shape[0])
    if persistence.size == 0:
        max_persistence = total_persistence = mean_persistence = 0.0
    else:
        max_persistence = float(persistence.max())
        total_persistence = float(persistence.sum())
        mean_persistence = float(persistence.mean())

    if lifetimes.size == 0:
        max_lifetime = mean_lifetime = 0.0
    else:
        max_lifetime = float(lifetimes.max())
        mean_lifetime = float(lifetimes.mean())

    return {
        "count": count,
        "max_persistence": max_persistence,
        "total_persistence": total_persistence,
        "mean_persistence": mean_persistence,
        "max_lifetime": max_lifetime,
        "mean_lifetime": mean_lifetime,
    }


def _record_from_encoding(
    source: Path,
    adapter: HiddenStateFeatureAdapter,
    hidden_states: torch.Tensor,
) -> Dict[str, Any]:
    encoding = adapter.encode(hidden_states)
    feature_vector = adapter.extractor.from_encoding(encoding)
    vector_np = feature_vector.values.detach().cpu().numpy().astype(np.float32)

    diagrams: Dict[str, List[List[float]]] = {}
    diagram_summary: Dict[str, Dict[str, float]] = {}
    for dim, diagram in encoding.homology.diagrams.items():
        diag_np = diagram.astype(np.float32)
        diagrams[f"h{dim}"] = diag_np.tolist()
        diagram_summary[f"h{dim}"] = _diagram_stats(diag_np)

    sections = {
        name: tensor.detach().cpu().numpy().astype(np.float32).tolist()
        for name, tensor in feature_vector.sections.items()
    }

    betti_numbers = {f"b{dim}": int(count) for dim, count in encoding.homology.betti.items()}
    persistence_stats = {
        "entropy": float(encoding.homology.persistence_entropy),
        "sheaf_energy": float(sections.get("sheaf_energy", [0.0])[0] if sections.get("sheaf_energy") else 0.0),
        "num_features": float(sections.get("persistence_stats", [0.0, 0.0, 0.0, 0.0])[3])
        if sections.get("persistence_stats")
        else 0.0,
    }

    vector_norm = float(np.linalg.norm(vector_np))
    summary = {
        "vector_norm": vector_norm,
        "entropy": persistence_stats["entropy"],
        "sheaf_energy": persistence_stats["sheaf_energy"],
        "betti_max": max(betti_numbers.values()) if betti_numbers else 0.0,
        "betti_sum": float(sum(betti_numbers.values())) if betti_numbers else 0.0,
    }

    topology_text = (
        f"betti={betti_numbers} entropy={summary['entropy']:.4f} "
        f"sheaf={summary['sheaf_energy']:.4f} norm={summary['vector_norm']:.2f}"
    )

    return {
        "id": source.stem,
        "source_path": str(source.resolve()),
        "vector_dim": int(feature_vector.values.shape[0]),
        "vector": vector_np.tolist(),
        "sections": sections,
        "persistence_diagrams": diagrams,
        "betti_numbers": betti_numbers,
        "persistence_entropy": persistence_stats["entropy"],
        "sheaf_energy": persistence_stats["sheaf_energy"],
        "num_topological_features": persistence_stats["num_features"],
        "diagram_summary": diagram_summary,
        "summary": summary,
        "text": topology_text,
    }


def _ensure_output_parent(path: Path) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = _parse_args()
    adapter = HiddenStateFeatureAdapter(pool_mode=args.pool_mode)

    records: List[Dict[str, Any]] = []
    inputs = list(args.inputs)

    if not inputs:
        torch.manual_seed(0)
        sample = torch.randn(12, 128, 1024, device=args.device)
        records.append(_record_from_encoding(Path("synthetic"), adapter, sample))
    else:
        for tensor_path in _iter_sources(inputs):
            tensor = _load_tensor(tensor_path).to(args.device)
            records.append(_record_from_encoding(tensor_path, adapter, tensor))

    if args.output:
        _ensure_output_parent(args.output)
        with args.output.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        print(f"Wrote {len(records)} topology feature records to {args.output}")
    else:
        payload = records[0] if len(records) == 1 else records
        print(json.dumps(payload, indent=args.indent))


if __name__ == "__main__":
    main()
