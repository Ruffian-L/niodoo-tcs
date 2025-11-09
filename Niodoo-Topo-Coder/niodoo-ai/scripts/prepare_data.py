#!/usr/bin/env python
"""Pre-compute topology features and materialise tokenised datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

from datasets import Dataset
import numpy as np

from niodoo_ai import TopologyAugmentor, build_datasets, create_tokenizer, load_training_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Path to training YAML configuration.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Directory to store processed datasets (defaults to <output_dir>/processed).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_training_config(args.config)
    output_dir = args.output or (config.runtime.output_dir / "processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    augmentor = TopologyAugmentor()
    tokenizer = create_tokenizer(config)

    train_ds, eval_ds = build_datasets(config.data, tokenizer, augmentor)
    train_path = output_dir / "train"
    train_ds.save_to_disk(str(train_path))
    if eval_ds is not None:
        eval_path = output_dir / "eval"
        eval_ds.save_to_disk(str(eval_path))

    manifest = _build_manifest(train_ds, eval_ds)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Saved train dataset to {train_path}")
    if eval_ds is not None:
        print(f"Saved eval dataset to {eval_path}")
    print(f"Recorded topology manifest at {manifest_path}")


def _vector_dim(dataset: Dataset) -> Optional[int]:
    if len(dataset) == 0:
        return None
    expected = int(dataset[0]["topology_vector"].shape[-1])
    for row in dataset:
        if row["topology_vector"].shape[-1] != expected:
            raise ValueError("Inconsistent topology vector dimensionality detected")
    return expected


def _build_manifest(train_ds: Dataset, eval_ds: Optional[Dataset]) -> Dict[str, object]:
    def summarise(dataset: Dataset) -> Dict[str, object]:
        if len(dataset) == 0:
            return {"num_rows": 0}
        vector_dim = _vector_dim(dataset)
        sample_meta = dataset[0].get("topology_metadata", {})
        metadata_summary = _metadata_summary(dataset) if sample_meta else {}
        teacher_presence = float(sum(1 for row in dataset if row.get("teacher_key")) / len(dataset))
        return {
            "num_rows": len(dataset),
            "topology_vector_dim": vector_dim,
            "sample_metadata_keys": sorted(sample_meta.keys()),
            "topology_metadata_summary": metadata_summary,
            "teacher_key_coverage": teacher_presence,
        }

    manifest: Dict[str, object] = {"train": summarise(train_ds)}
    if eval_ds is not None:
        manifest["eval"] = summarise(eval_ds)
    return manifest


def _metadata_summary(dataset: Dataset) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for key in dataset[0].get("topology_metadata", {}).keys():
        values = [row["topology_metadata"].get(key) for row in dataset]
        numeric_values = [float(value) for value in values if isinstance(value, (int, float))]
        if not numeric_values:
            continue
        stats = {
            "mean": float(np.mean(numeric_values)),
            "std": float(np.std(numeric_values)),
            "min": float(np.min(numeric_values)),
            "max": float(np.max(numeric_values)),
        }
        summary[key] = stats
    return summary


if __name__ == "__main__":
    main()

