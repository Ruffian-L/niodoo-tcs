#!/usr/bin/env python
"""Evaluate a topology-aware fine-tuned model against the validation set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from niodoo_ai import evaluate_model, load_training_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Path to training YAML configuration.")
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Optional path to a fine-tuned checkpoint directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_training_config(args.config)
    metrics = evaluate_model(config, args.model)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()



