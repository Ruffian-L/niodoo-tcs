"""
NIODOO TQFT Module - Thought-Knot Detection

Applies knot theory to analyze code execution trajectories and detect
persistent Betti-1 loops (cyclical dependencies).
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional

try:
    # Try to import Rust extension module
    from niodoo_real_integrated import tqft as _rust_tqft
    _RUST_AVAILABLE = True
except ImportError:
    _RUST_AVAILABLE = False
    _rust_tqft = None


class KnotSignature:
    """Represents a knot signature identifying architectural flaws."""
    def __init__(
        self,
        has_knot: bool,
        betti_derivative_norm: float,
        average_betti_1: float,
        persistent_loops: List[Tuple[float, float]],
        involved_modules: List[str],
    ):
        self.has_knot = has_knot
        self.betti_derivative_norm = betti_derivative_norm
        self.average_betti_1 = average_betti_1
        self.persistent_loops = persistent_loops
        self.involved_modules = involved_modules


def analyze_trajectory(trajectory_data: Dict[str, Any]) -> KnotSignature:
    """
    Analyze code trajectory and detect thought-knots.
    
    Args:
        trajectory_data: Dictionary containing:
            - trajectory_type: str ("CfgPath", "DfgPath", "CommitSequence", "ExecutionTrace")
            - points: List of dicts with keys: t (float), betti_0, betti_1, betti_2 (int), metadata (dict)
            - source_path: Optional[str]
        
    Returns:
        KnotSignature with thought-knot detection results
        
    Note:
        This function applies knot theory (Jones polynomial) to detect
        persistent Betti-1 loops that span multiple files/modules.
    """
    if not _RUST_AVAILABLE:
        raise NotImplementedError(
            "Rust FFI bindings not available. Build with: cargo build --features pyo3"
        )
    
    # Call Rust FFI
    result = _rust_tqft.analyze_trajectory(trajectory_data)
    
    return KnotSignature(
        has_knot=result.get("has_knot", False),
        betti_derivative_norm=result.get("betti_derivative_norm", 0.0),
        average_betti_1=result.get("average_betti_1", 0.0),
        persistent_loops=result.get("persistent_loops", []),
        involved_modules=result.get("involved_modules", []),
    )

