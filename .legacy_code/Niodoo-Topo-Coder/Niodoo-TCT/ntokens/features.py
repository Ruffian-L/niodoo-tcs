"""Topology feature vectorisation utilities for nToken encodings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .encoder import MinimalNTokens, NTokensEncoding


def _flatten_lifetimes(diagrams: Iterable[np.ndarray]) -> np.ndarray:
    lifetimes: List[np.ndarray] = []
    for diagram in diagrams:
        if diagram.size == 0:
            continue
        finite = np.isfinite(diagram[:, 1])
        matched = diagram[finite]
        if matched.size == 0:
            continue
        lifespan = matched[:, 1] - matched[:, 0]
        lifespan = lifespan[lifespan > 0]
        if lifespan.size:
            lifetimes.append(lifespan)
    if not lifetimes:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(lifetimes)


def _sampling_domain(diagrams: Mapping[int, np.ndarray]) -> Tuple[float, float]:
    births: List[float] = []
    deaths: List[float] = []
    for diagram in diagrams.values():
        if diagram.size == 0:
            continue
        births.extend(diagram[:, 0].tolist())
        deaths.extend(diagram[:, 1].tolist())
    if not births or not deaths:
        return 0.0, 1.0
    min_birth = float(np.min(births))
    max_death = float(np.max(deaths))
    if not np.isfinite(max_death):
        max_death = float(np.max(births))
    if max_death <= min_birth:
        max_death = min_birth + 1.0
    return min_birth, max_death


def betti_curve(
    diagrams: Mapping[int, np.ndarray],
    n_bins: int = 32,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
) -> torch.Tensor:
    """Sample Betti curves for each homology dimension."""

    if n_bins < 2:
        raise ValueError("n_bins must be >= 2")

    start, end = _sampling_domain(diagrams)
    if min_value is not None:
        start = min_value
    if max_value is not None:
        end = max_value

    xs = np.linspace(start, end, num=n_bins, dtype=np.float32)
    curves: List[np.ndarray] = []
    for dim in sorted(diagrams.keys()):
        diagram = diagrams[dim]
        if diagram.size == 0:
            curves.append(np.zeros(n_bins, dtype=np.float32))
            continue
        counts = np.zeros_like(xs)
        births = diagram[:, 0]
        deaths = diagram[:, 1]
        for idx, x in enumerate(xs):
            alive = (births <= x) & ((deaths > x) | ~np.isfinite(deaths))
            counts[idx] = np.count_nonzero(alive)
        curves.append(counts.astype(np.float32))
    if not curves:
        curves.append(np.zeros(n_bins, dtype=np.float32))
    stacked = np.stack(curves, axis=0)
    return torch.from_numpy(stacked)


def persistence_statistics(diagrams: Mapping[int, np.ndarray]) -> Dict[str, float]:
    """Return scalar persistence statistics for homology diagrams."""

    stats: Dict[str, float] = {}
    total_lifetimes = _flatten_lifetimes(diagrams.values())
    stats["lifetime_sum"] = float(np.sum(total_lifetimes))
    stats["lifetime_mean"] = float(np.mean(total_lifetimes)) if total_lifetimes.size else 0.0
    stats["lifetime_max"] = float(np.max(total_lifetimes)) if total_lifetimes.size else 0.0
    stats["num_features"] = float(total_lifetimes.size)
    return stats


def sheaf_energy(sheaf_data: Mapping[str, torch.Tensor]) -> float:
    """Compute a simple energy statistic from sheaf restriction matrices."""

    restriction = sheaf_data.get("restriction")
    if restriction is None:
        return 0.0
    energy = torch.norm(restriction.float(), p=2)
    return float(energy.item())


@dataclass
class TopologyFeatureVector:
    """Container for a flattened feature vector and interpretable metadata."""

    values: torch.Tensor
    sections: Dict[str, torch.Tensor]


class TopologyFeatureExtractor:
    """Generate feature vectors from `NTokensEncoding` results."""

    def __init__(self, betti_bins: int = 32) -> None:
        self.betti_bins = betti_bins

    def from_encoding(self, encoding: NTokensEncoding) -> TopologyFeatureVector:
        curves = betti_curve(encoding.homology.diagrams, n_bins=self.betti_bins)
        stats = persistence_statistics(encoding.homology.diagrams)
        sheaf = torch.tensor([sheaf_energy(encoding.sheaf)], dtype=torch.float32)
        betti = torch.tensor(
            [float(encoding.homology.betti.get(dim, 0)) for dim in sorted(encoding.homology.betti.keys())],
            dtype=torch.float32,
        )
        entropy = torch.tensor([encoding.homology.persistence_entropy], dtype=torch.float32)

        sections: Dict[str, torch.Tensor] = {
            "betti_curve": curves.flatten(),
            "betti_numbers": betti,
            "persistence_stats": torch.tensor(
                [stats["lifetime_sum"], stats["lifetime_mean"], stats["lifetime_max"], stats["num_features"]],
                dtype=torch.float32,
            ),
            "entropy": entropy,
            "sheaf_energy": sheaf,
        }

        flat = torch.cat(list(sections.values()))
        return TopologyFeatureVector(values=flat, sections=sections)


class HiddenStateFeatureAdapter:
    """Helper that converts transformer hidden states into feature vectors."""

    def __init__(
        self,
        encoder: Optional[MinimalNTokens] = None,
        feature_extractor: Optional[TopologyFeatureExtractor] = None,
        pool_mode: str = "mean",
    ) -> None:
        if pool_mode not in {"mean", "cls"}:
            raise ValueError("pool_mode must be 'mean' or 'cls'")
        self.encoder = encoder or MinimalNTokens()
        self.extractor = feature_extractor or TopologyFeatureExtractor()
        self.pool_mode = pool_mode

    def _pool_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.dim() != 3:
            raise ValueError("hidden_states must have shape (layers, seq, hidden)")
        if self.pool_mode == "cls":
            pooled = hidden_states[-1, 0]
        else:
            pooled = hidden_states[-1].mean(dim=0)
        return pooled.unsqueeze(0)

    def _normalise(self, tensor: torch.Tensor) -> torch.Tensor:
        normed = F.layer_norm(tensor, tensor.shape[-1:])
        return normed

    def encode(self, hidden_states: torch.Tensor) -> NTokensEncoding:
        pooled = self._pool_hidden_states(hidden_states)
        normed = self._normalise(pooled)
        return self.encoder.encode(normed)

    def features(self, hidden_states: torch.Tensor) -> TopologyFeatureVector:
        encoding = self.encode(hidden_states)
        return self.extractor.from_encoding(encoding)


def collate_feature_vectors(vectors: Sequence[TopologyFeatureVector]) -> torch.Tensor:
    """Stack feature vectors into a single tensor."""

    if not vectors:
        raise ValueError("vectors must not be empty")
    first_dim = vectors[0].values.shape[0]
    for vector in vectors:
        if vector.values.shape[0] != first_dim:
            raise ValueError("all vectors must share the same dimensionality")
    return torch.stack([vec.values for vec in vectors], dim=0)




