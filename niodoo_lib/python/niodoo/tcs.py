"""
NIODOO TCS (Topological Cognitive Signature) Module

Provides topological analysis functionality for analyzing data structures.
"""

import numpy as np
from typing import Dict, Any, Optional


class TopologicalSignature:
    """Represents a topological signature with Betti numbers and persistence metrics."""
    def __init__(
        self,
        betti_0: int,
        betti_1: int,
        betti_2: int,
        persistence_entropy: float,
        persistence_diagram: Optional[np.ndarray] = None,
    ):
        self.betti_0 = betti_0
        self.betti_1 = betti_1
        self.betti_2 = betti_2
        self.persistence_entropy = persistence_entropy
        self.persistence_diagram = persistence_diagram


try:
    # Try to import Rust extension module
    from niodoo_real_integrated import tcs as _rust_tcs
    _RUST_AVAILABLE = True
except ImportError:
    _RUST_AVAILABLE = False
    _rust_tcs = None


def analyze(matrix: np.ndarray) -> TopologicalSignature:
    """
    Analyze the topology of a matrix (adjacency matrix or point cloud).
    
    Args:
        matrix: Input matrix (adjacency matrix or distance matrix)
        
    Returns:
        TopologicalSignature with Betti numbers and persistence metrics
        
    Note:
        This function communicates with the NIODOO Rust backend via FFI.
        Uses giotto-tda for TDA computation via Rust-Orchestrated Hybrid bridge.
    """
    if not _RUST_AVAILABLE:
        raise NotImplementedError(
            "Rust FFI bindings not available. Build with: cargo build --features pyo3"
        )
    
    # Call Rust FFI
    result = _rust_tcs.analyze(matrix)
    
    # Convert result to TopologicalSignature
    betti_numbers = result.get("betti_numbers", [0, 0, 0])
    persistence_pairs = result.get("persistence_pairs", [])
    persistence_entropy = result.get("persistence_entropy", 0.0)
    
    # Convert persistence pairs to numpy array if available
    persistence_diagram = None
    if persistence_pairs:
        persistence_diagram = np.array(persistence_pairs)
    
    return TopologicalSignature(
        betti_0=betti_numbers[0] if len(betti_numbers) > 0 else 0,
        betti_1=betti_numbers[1] if len(betti_numbers) > 1 else 0,
        betti_2=betti_numbers[2] if len(betti_numbers) > 2 else 0,
        persistence_entropy=persistence_entropy,
        persistence_diagram=persistence_diagram,
    )



