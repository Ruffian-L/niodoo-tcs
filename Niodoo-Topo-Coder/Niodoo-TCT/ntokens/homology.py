"""Persistent homology front-end built on top of ripser."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
import torch
from ripser import ripser


@dataclass
class HomologyResult:
    """Structured output for persistent homology computations."""

    diagrams: Dict[int, np.ndarray]
    betti: Dict[int, int]
    persistence_entropy: float

    def summary(self) -> Dict[str, float]:
        return {
            **{f"betti_{dim}": val for dim, val in self.betti.items()},
            "persistence_entropy": self.persistence_entropy,
        }


class PersistentHomologyBackend:
    """Wrapper around ripser with lightweight summary statistics."""

    def __init__(
        self,
        maxdim: int = 1,
        metric: str = "euclidean",
        coeff: int = 2,
        n_threads: Optional[int] = None,
    ) -> None:
        self.maxdim = maxdim
        self.metric = metric
        self.coeff = coeff
        self.n_threads = n_threads

    def compute(self, data: torch.Tensor | np.ndarray) -> HomologyResult:
        cloud = self._prepare_input(data)
        kwargs = {
            "maxdim": self.maxdim,
            "metric": self.metric,
            "coeff": self.coeff,
        }
        if self.n_threads is not None:
            # Only newer builds of ripser accept the n_threads kwarg.
            kwargs["n_threads"] = self.n_threads

        try:
            result = ripser(cloud, **kwargs)
        except TypeError:
            # Fallback for builds that do not expose the threading kwarg.
            kwargs.pop("n_threads", None)
            result = ripser(cloud, **kwargs)
        diagrams = {
            dim: dgm
            for dim, dgm in enumerate(result["dgms"])
            if dgm is not None and len(dgm)
        }
        betti = {dim: dgm.shape[0] for dim, dgm in diagrams.items()}
        entropy = self._persistence_entropy(diagrams.values())
        return HomologyResult(diagrams=diagrams, betti=betti, persistence_entropy=entropy)

    def _prepare_input(self, data: torch.Tensor | np.ndarray) -> np.ndarray:
        if isinstance(data, torch.Tensor):
            arr = data.detach().cpu().numpy()
        else:
            arr = np.asarray(data)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr.astype(np.float32)

    def _persistence_entropy(self, diagrams: Iterable[np.ndarray]) -> float:
        lifetimes = []
        for dgm in diagrams:
            if dgm.size == 0:
                continue
            finite_mask = np.isfinite(dgm[:, 1])
            lifetime = dgm[finite_mask, 1] - dgm[finite_mask, 0]
            lifetime = lifetime[lifetime > 0]
            if lifetime.size:
                lifetimes.append(lifetime)
        if not lifetimes:
            return 0.0
        lifetimes = np.concatenate(lifetimes)
        probs = lifetimes / lifetimes.sum()
        entropy = -np.sum(probs * np.log(probs + 1e-12))
        return float(entropy)
