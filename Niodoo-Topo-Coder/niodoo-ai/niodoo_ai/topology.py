"""Topology feature handling for fine-tuning datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence
import torch

try:
    from ntokens import HiddenStateFeatureAdapter, TopologyFeatureVector
except ModuleNotFoundError:  # pragma: no cover - fallback path for local dev
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    ntoken_path = repo_root / "Niodoo-TCT"
    if str(ntoken_path) not in sys.path:
        sys.path.insert(0, str(ntoken_path))
    from ntokens import HiddenStateFeatureAdapter, TopologyFeatureVector


@dataclass
class FeatureBundle:
    vector: torch.Tensor
    summary: Dict[str, float]
    text: str


class TopologyAugmentor:
    """Generate human-readable and numeric topology descriptors."""

    def __init__(
        self,
        pool_mode: str = "mean",
        decimal_places: int = 4,
        feature_bins: int = 32,
    ) -> None:
        self.adapter = HiddenStateFeatureAdapter(pool_mode=pool_mode)
        self.decimal_places = decimal_places
        self.feature_bins = feature_bins

    def _vector_from_feature_vector(self, feature_vector: TopologyFeatureVector) -> FeatureBundle:
        sections = feature_vector.sections
        summary = {
            "entropy": sections["entropy"].item() if "entropy" in sections else 0.0,
            "sheaf_energy": sections["sheaf_energy"].item() if "sheaf_energy" in sections else 0.0,
        }
        if "betti_numbers" in sections and sections["betti_numbers"].numel():
            summary["betti_max"] = float(sections["betti_numbers"].max().item())
            summary["betti_sum"] = float(sections["betti_numbers"].sum().item())
        text = self._format_text(feature_vector.values, summary)
        return FeatureBundle(vector=feature_vector.values, summary=summary, text=text)

    def _format_text(self, vector: torch.Tensor, summary: Dict[str, float]) -> str:
        rounded = [round(val, self.decimal_places) for val in vector[: min(8, vector.numel())].tolist()]
        pieces = [f"len={vector.numel()}"]
        for key, value in summary.items():
            pieces.append(f"{key}={value:.{self.decimal_places}f}")
        if rounded:
            pieces.append("head=" + ",".join(f"{val:.{self.decimal_places}f}" for val in rounded))
        return " ".join(pieces)

    def from_hidden_states(self, hidden_states: torch.Tensor) -> FeatureBundle:
        feature_vector = self.adapter.features(hidden_states)
        return self._vector_from_feature_vector(feature_vector)

    def from_tensor(self, tensor: torch.Tensor) -> FeatureBundle:
        if tensor.dim() != 1:
            raise ValueError("Expected flat topology feature tensor")
        summary = self._quick_summary(tensor)
        text = self._format_text(tensor, summary)
        return FeatureBundle(vector=tensor, summary=summary, text=text)

    def from_sequence(self, sequence: Sequence[float]) -> FeatureBundle:
        tensor = torch.tensor(sequence, dtype=torch.float32)
        return self.from_tensor(tensor)

    def from_file(self, path: Path) -> FeatureBundle:
        # Load directly to GPU if available - RTX 5090 optimization
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loaded = torch.load(path, map_location=device)
        if isinstance(loaded, torch.Tensor):
            if loaded.dim() == 1:
                return self.from_tensor(loaded)
            return self.from_hidden_states(loaded)
        if isinstance(loaded, dict):
            if "topology_features" in loaded:
                return self.from_sequence(loaded["topology_features"])
            if "hidden_states" in loaded:
                return self.from_hidden_states(loaded["hidden_states"])
            if "topology" in loaded:
                return self.from_payload(loaded["topology"])
        raise ValueError(f"Unrecognised topology payload in {path}")

    def from_payload(self, payload: Mapping[str, object]) -> FeatureBundle:
        vector_data = payload.get("vector") or payload.get("values")
        if vector_data is None:
            raise ValueError("Topology payload must include 'vector' or 'values'")
        if isinstance(vector_data, torch.Tensor):
            bundle = self.from_tensor(vector_data)
        else:
            bundle = self.from_sequence(list(vector_data))

        summary_override = payload.get("summary")
        if isinstance(summary_override, dict):
            bundle.summary.update({key: float(value) for key, value in summary_override.items()})

        if "persistence_entropy" in payload:
            bundle.summary["entropy"] = float(payload["persistence_entropy"])
        if "sheaf_energy" in payload:
            bundle.summary["sheaf_energy"] = float(payload["sheaf_energy"])

        betti_numbers = payload.get("betti_numbers")
        if isinstance(betti_numbers, Mapping):
            betti_values = [float(value) for value in betti_numbers.values()]
            if betti_values:
                bundle.summary["betti_max"] = max(betti_values)
                bundle.summary["betti_sum"] = sum(betti_values)

        text_override = payload.get("text")
        if isinstance(text_override, str) and text_override.strip():
            bundle.text = text_override.strip()
        else:
            bundle.text = self._format_text(bundle.vector, bundle.summary)

        return bundle

    def serialise(self, vector: torch.Tensor) -> List[float]:
        return vector.detach().cpu().tolist()

    def _quick_summary(self, tensor: torch.Tensor) -> Dict[str, float]:
        if tensor.numel() == 0:
            return {"mean": 0.0, "std": 0.0}
        return {
            "mean": float(torch.mean(tensor).item()),
            "std": float(torch.std(tensor).item()),
        }

    def ensure_bundle(
        self,
        existing_features: Optional[Iterable[float]] = None,
        hidden_states: Optional[torch.Tensor] = None,
        feature_path: Optional[Path] = None,
        topology_payload: Optional[Mapping[str, object]] = None,
    ) -> FeatureBundle:
        if topology_payload is not None:
            return self.from_payload(topology_payload)
        if existing_features is not None:
            return self.from_sequence(list(existing_features))
        if hidden_states is not None:
            return self.from_hidden_states(hidden_states)
        if feature_path is not None:
            return self.from_file(feature_path)
        raise ValueError("No topology information available to build bundle")

