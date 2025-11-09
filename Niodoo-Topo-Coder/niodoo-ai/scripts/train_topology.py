#!/usr/bin/env python
"""Execute topology-aware QLoRA fine-tuning."""

from __future__ import annotations

import argparse
from pathlib import Path

from niodoo_ai import load_training_config, run_training


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Path to training YAML configuration.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_training_config(args.config)
    run_training(config)


if __name__ == "__main__":
    main()



