#!/usr/bin/env python
"""Run the minimal nToken encoder on random embeddings."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ntokens import MinimalNTokens


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    embeddings = torch.randn(4, 256, device=device)

    encoder = MinimalNTokens()
    result = encoder.encode(embeddings)

    print("=== nToken Encoding Summary ===")
    for key, value in result.summary().items():
        print(f"{key}: {value}")

    print("\nQuantized tensor shape:", tuple(result.quantized.values.shape))
    print("Sheaf patch vectors shape:", tuple(result.sheaf["patch_vectors"].shape))


if __name__ == "__main__":
    main()
