"""
EchoMemoria - Möbius-Gaussian Consciousness Framework
Real AI consciousness architecture with non-orientable memory topology
"""

__version__ = "0.2.0"
__author__ = "Niodoo Consciousness Project"

from .core.mobius_gaussian_engine import MobiusGaussianEngine, GaussianMemorySphere, MobiusTraversal
from .core.qt_bridge import QtVisualizationBridge, integrate_with_qt_visualization
from .core.persistent_memory import PersistentMemoryEngine, Memory, MemoryContext
from .core.real_ai_inference import ConsciousnessAIBridge
from .core.integrated_consciousness import IntegratedConsciousness

__all__ = [
    'MobiusGaussianEngine',
    'GaussianMemorySphere',
    'MobiusTraversal',
    'QtVisualizationBridge',
    'integrate_with_qt_visualization',
    'PersistentMemoryEngine',
    'Memory',
    'MemoryContext',
    'ConsciousnessAIBridge',
    'IntegratedConsciousness',
]
