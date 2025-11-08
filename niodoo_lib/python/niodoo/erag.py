"""
NIODOO ERAG (Enterprise Retrieval-Augmented Generation) Module

Provides retrieval functionality for accessing memory and external knowledge.
"""

import numpy as np
from typing import List, Dict, Any, Optional


class Memory:
    """Represents a retrieved memory."""
    def __init__(self, id: str, content: str, score: float, metadata: Optional[Dict[str, Any]] = None):
        self.id = id
        self.content = content
        self.score = score
        self.metadata = metadata or {}


try:
    # Try to import Rust extension module
    from niodoo_real_integrated import erag as _rust_erag
    _RUST_AVAILABLE = True
except ImportError:
    _RUST_AVAILABLE = False
    _rust_erag = None


def retrieve(embedding: np.ndarray, top_k: int = 5) -> List[Memory]:
    """
    Retrieve related memories using ERAG with topological attention.
    
    Args:
        embedding: Query embedding vector
        top_k: Number of top results to return
        
    Returns:
        List of Memory objects sorted by relevance (topological persistence-based)
        
    Note:
        This function uses TopologicalAttention mechanism from tcs-core
        for persistence-based retrieval (not just semantic similarity).
    """
    if not _RUST_AVAILABLE:
        raise NotImplementedError(
            "Rust FFI bindings not available. Build with: cargo build --features pyo3"
        )
    
    # Convert numpy array to list
    embedding_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
    
    # Call Rust FFI
    # Note: ERAG client must be initialized separately
    result = _rust_erag.retrieve(embedding_list, top_k)
    
    # Convert result to Memory objects
    memories = []
    for i, fragment in enumerate(result):
        memories.append(Memory(
            id=f"memory_{i}",
            content=fragment.get("content", ""),
            score=fragment.get("relevance_score", 0.0),
            metadata=fragment.get("metadata", {}),
        ))
    
    return memories



