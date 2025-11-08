"""
NIODOO Parser Module

Provides code parsing functionality for converting source code to graphs.
"""

import numpy as np
from typing import Union

try:
    # Try to import Rust extension module
    from niodoo_real_integrated import parser as _rust_parser
    _RUST_AVAILABLE = True
except ImportError:
    _RUST_AVAILABLE = False
    _rust_parser = None


def get_graph_from_file(code: str, language: str = "python") -> np.ndarray:
    """
    Parse code string and return adjacency matrix.
    
    Args:
        code: Source code string
        language: Language identifier ("python" or "typescript")
        
    Returns:
        numpy.ndarray: Adjacency matrix representing the code's control flow graph
        
    Raises:
        NotImplementedError: If Rust FFI bindings are not available
    """
    if not _RUST_AVAILABLE:
        raise NotImplementedError(
            "Rust FFI bindings not available. Build with: cargo build --features pyo3"
        )
    
    return _rust_parser.get_graph_from_file(code, language)


def get_graph_from_repo(path: str, language: str = "python") -> np.ndarray:
    """
    Parse repository and build global graph.
    
    Args:
        path: Path to repository root directory
        language: Language identifier ("python" or "typescript")
        
    Returns:
        numpy.ndarray: Adjacency matrix representing the repository's global graph
        
    Raises:
        NotImplementedError: If Rust FFI bindings are not available
    """
    if not _RUST_AVAILABLE:
        raise NotImplementedError(
            "Rust FFI bindings not available. Build with: cargo build --features pyo3"
        )
    
    return _rust_parser.get_graph_from_repo(path, language)

