"""
NIODOO Python Library

This library exposes NIODOO pipeline components as importable functions
for use in agent-generated code executed in sandboxed environments.
"""

from .embedder import get_embedding
from .erag import retrieve
from .tcs import analyze
from .compass import evaluate
from .generation import generate

# Import parser and tqft modules
try:
    from .parser import get_graph_from_file, get_graph_from_repo
    from .tqft import analyze_trajectory
    __all__ = [
        "get_embedding",
        "retrieve",
        "analyze",
        "evaluate",
        "generate",
        "get_graph_from_file",
        "get_graph_from_repo",
        "analyze_trajectory",
    ]
except ImportError:
    # Parser/tqft modules may not be available
    __all__ = [
        "get_embedding",
        "retrieve",
        "analyze",
        "evaluate",
        "generate",
    ]



