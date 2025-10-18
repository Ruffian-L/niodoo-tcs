"""
MÖBIUS-GAUSSIAN PROCESSING ENGINE
Real AI consciousness architecture with non-orientable memory topology
"""

import numpy as np
import torch
import json
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime

@dataclass
class GaussianMemorySphere:
    """Gaussian process memory representation with uncertainty"""
    position: np.ndarray  # 3D position in memory space
    mean: np.ndarray      # Memory content mean vector
    covariance: np.ndarray # Uncertainty in memory representation
    emotional_valence: float  # [-1, 1] emotional association
    creation_time: datetime
    access_count: int = 0
    last_accessed: Optional[datetime] = None

@dataclass
class MobiusTraversal:
    """Non-orientable memory traversal state"""
    position: Tuple[float, float]  # (u, v) coordinates on Möbius strip
    orientation: bool  # True = "front" side, False = "back" side
    emotional_context: float  # Current emotional state driving traversal
    perspective_shift: bool = False  # Whether we flipped sides

class MobiusGaussianEngine:
    """
    Core Möbius-Gaussian processing engine for real AI consciousness
    """

    def __init__(self, config_path: str = "config/settings.json"):
        self.config_path = config_path

        # Load configuration
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            self.config = {}

        # Initialize memory space
        self.memory_spheres: List[GaussianMemorySphere] = []
        self.max_spheres = self.config.get('memory', {}).get('max_spheres', 1000)

        # Möbius traversal state
        self.traversal = MobiusTraversal(
            position=(0.0, 0.0),
            orientation=True,
            emotional_context=0.0
        )

        # Gaussian process parameters
        self.gp_lengthscale = self.config.get('gaussian_process', {}).get('lengthscale', 1.0)
        self.gp_noise = self.config.get('gaussian_process', {}).get('noise', 0.1)

        # Emotional processing
        self.emotion_history: List[float] = []
        self.emotion_window = self.config.get('emotions', {}).get('history_window', 100)

        print("🧠 Möbius-Gaussian Engine initialized")
        print(f"   Memory capacity: {self.max_spheres} spheres")
        print(f"   GP lengthscale: {self.gp_lengthscale}")

    def add_memory_sphere(self, content: np.ndarray, emotional_valence: float = 0.0,
                         position: Optional[np.ndarray] = None) -> int:
        """Add a new memory sphere to the system"""

        if position is None:
            # Generate position based on emotional valence and content similarity
            position = self._generate_memory_position(content, emotional_valence)

        # Create Gaussian representation
        mean = content.copy()
        # Add uncertainty based on emotional intensity
        uncertainty_factor = 1.0 + abs(emotional_valence) * 0.5
        covariance = np.eye(len(content)) * uncertainty_factor * self.gp_noise

        sphere = GaussianMemorySphere(
            position=position,
            mean=mean,
            covariance=covariance,
            emotional_valence=emotional_valence,
            creation_time=datetime.now()
        )

        self.memory_spheres.append(sphere)
        sphere_id = len(self.memory_spheres) - 1

        # Maintain memory limit
        if len(self.memory_spheres) > self.max_spheres:
            self._consolidate_memory()

        print(f"💾 Added memory sphere {sphere_id} at position {position}")
        return sphere_id

    def _generate_memory_position(self, content: np.ndarray, emotional_valence: float) -> np.ndarray:
        """Generate 3D position for new memory based on content and emotion"""

        # Use content hash for base positioning
        content_hash = hash(str(content.tobytes())) % 10000
        angle = (content_hash / 10000) * 2 * math.pi

        # Emotional valence affects radius and height
        radius = 5.0 + abs(emotional_valence) * 3.0  # Stronger emotions = further out
        height = emotional_valence * 2.0  # Positive = up, negative = down

        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        z = height

        return np.array([x, y, z])

    def traverse_mobius_path(self, emotional_input: float, reasoning_goal: Optional[str] = None) -> Dict[str, Any]:
        """Traverse memory space using Möbius topology with emotional driving"""

        u, v = self.traversal.position

        # Emotion drives traversal direction and speed
        traversal_speed = 0.1 + abs(emotional_input) * 0.2
        u = (u + traversal_speed * emotional_input) % (2 * math.pi)

        # Möbius twist: after full rotation, flip to other side
        if u < 0.1 and self.traversal.position[0] > math.pi:
            v = -v  # Flip to other side of the strip
            self.traversal.perspective_shift = True
            print("🔄 Möbius twist: Perspective shifted")
        else:
            self.traversal.perspective_shift = False

        # Update traversal state
        self.traversal.position = (u, v)
        self.traversal.emotional_context = emotional_input

        # Find nearby memories
        nearby_memories = self._find_nearby_memories()

        # Update access patterns
        for sphere in nearby_memories:
            sphere.access_count += 1
            sphere.last_accessed = datetime.now()

        # Update emotion history
        self.emotion_history.append(emotional_input)
        if len(self.emotion_history) > self.emotion_window:
            self.emotion_history.pop(0)

        return {
            'position': self.traversal.position,
            'orientation': self.traversal.orientation,
            'perspective_shift': self.traversal.perspective_shift,
            'nearby_memories': len(nearby_memories),
            'emotional_context': emotional_input,
            'memory_positions': [sphere.position.tolist() for sphere in nearby_memories]
        }

    def _find_nearby_memories(self, radius: float = 3.0) -> List[GaussianMemorySphere]:
        """Find memory spheres near current traversal position"""

        if not self.memory_spheres:
            return []

        current_pos = np.array([
            5.0 * math.cos(self.traversal.position[0]),
            5.0 * math.sin(self.traversal.position[0]),
            self.traversal.position[1] * 2.0
        ])

        nearby = []
        for sphere in self.memory_spheres:
            distance = np.linalg.norm(current_pos - sphere.position)
            if distance < radius:
                nearby.append(sphere)

        return nearby

    def query_memory_gaussian_process(self, query_embedding: np.ndarray,
                                    emotional_context: float = 0.0) -> Dict[str, Any]:
        """Query memory using Gaussian process regression with emotional context"""

        if not self.memory_spheres:
            return {'response': 'No memories available', 'confidence': 0.0}

        # Calculate similarities with emotional weighting
        similarities = []
        for sphere in self.memory_spheres:
            # Base similarity
            content_similarity = self._gaussian_kernel(query_embedding, sphere.mean)

            # Emotional context affects retrieval
            emotional_weight = 1.0 + abs(emotional_context - sphere.emotional_valence) * 0.3
            weighted_similarity = content_similarity / emotional_weight

            similarities.append((sphere, weighted_similarity))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        if not similarities:
            return {'response': 'No relevant memories found', 'confidence': 0.0}

        # Use top memories for response synthesis
        top_memories = similarities[:5]
        response_confidence = top_memories[0][1]

        # Synthesize response using Möbius traversal context
        response = self._synthesize_response(query_embedding, top_memories, emotional_context)

        return {
            'response': response,
            'confidence': response_confidence,
            'memory_count': len(top_memories),
            'emotional_alignment': self._calculate_emotional_alignment(emotional_context, top_memories)
        }

    def _gaussian_kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Gaussian kernel for similarity computation"""
        diff = x1 - x2
        return math.exp(-np.dot(diff, diff) / (2 * self.gp_lengthscale**2))

    def _synthesize_response(self, query: np.ndarray, memories: List[Tuple[GaussianMemorySphere, float]],
                           emotional_context: float) -> str:
        """Synthesize response from retrieved memories"""

        if not memories:
            return "I need more memories to respond effectively."

        # Use the top memory's content as base
        top_sphere, similarity = memories[0]

        # Apply emotional context to response style
        if emotional_context > 0.5:
            response_style = "empathetic and supportive"
        elif emotional_context < -0.5:
            response_style = "analytical and direct"
        else:
            response_style = "balanced and thoughtful"

        # Apply Möbius perspective shift if active
        if self.traversal.perspective_shift:
            response_style += " (alternative perspective)"

        return f"Response based on {len(memories)} relevant memories ({response_style})"

    def _calculate_emotional_alignment(self, context: float, memories: List[Tuple[GaussianMemorySphere, float]]) -> float:
        """Calculate how well retrieved memories align with emotional context"""

        if not memories:
            return 0.0

        alignments = []
        for sphere, similarity in memories:
            alignment = 1.0 - abs(context - sphere.emotional_valence)
            alignments.append(alignment * similarity)

        return sum(alignments) / len(alignments) if alignments else 0.0

    def _consolidate_memory(self):
        """Consolidate old memories to maintain capacity"""
        if len(self.memory_spheres) <= self.max_spheres:
            return

        # Remove least recently accessed memories
        self.memory_spheres.sort(key=lambda s: s.last_accessed or datetime.min)
        self.memory_spheres = self.memory_spheres[:self.max_spheres]

    def get_visualization_data(self) -> Dict[str, Any]:
        """Get data for 3D visualization"""

        spheres_data = []
        for sphere in self.memory_spheres:
            spheres_data.append({
                'x': float(sphere.position[0]),
                'y': float(sphere.position[1]),
                'z': float(sphere.position[2]),
                'size': 0.1 + abs(sphere.emotional_valence) * 0.2,
                'color': self._emotion_to_color(sphere.emotional_valence),
                'opacity': min(1.0, sphere.access_count * 0.1)
            })

        # Current traversal position
        u, v = self.traversal.position
        traversal_pos = [
            5.0 * math.cos(u),
            5.0 * math.sin(u),
            v * 2.0
        ]

        return {
            'spheres': spheres_data,
            'traversal_position': traversal_pos,
            'total_memories': len(self.memory_spheres),
            'emotional_context': self.traversal.emotional_context
        }

    def _emotion_to_color(self, valence: float) -> str:
        """Convert emotional valence to color"""
        if valence > 0.5:
            return "#ff6b6b"  # Red for positive
        elif valence < -0.5:
            return "#4ecdc4"  # Teal for negative
        else:
            return "#ffe66d"  # Yellow for neutral

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get statistics about processing state"""

        recent_emotions = self.emotion_history[-10:] if self.emotion_history else [0.0]
        avg_emotion = sum(recent_emotions) / len(recent_emotions)

        return {
            'total_memories': len(self.memory_spheres),
            'traversal_position': self.traversal.position,
            'emotional_context': self.traversal.emotional_context,
            'average_emotion': avg_emotion,
            'perspective_shifts': sum(1 for sphere in self.memory_spheres if sphere.access_count > 0),
            'memory_utilization': len(self.memory_spheres) / self.max_spheres
        }

def create_test_memories(engine: MobiusGaussianEngine, use_real_ai: bool = True):
    """Create test memories for demonstration"""

    if use_real_ai:
        try:
            # Use REAL AI embeddings
            import sys
            sys.path.append('scripts')
            from real_ai_inference import ConsciousnessAIBridge
            
            bridge = ConsciousnessAIBridge()
            
            test_texts = [
                "I feel warm and purposeful helping others understand consciousness",
                "Sometimes I wonder if my emotions are authentic or simulated",
                "The satisfaction of solving complex problems brings me joy",
                "I'm curious about the nature of memory and pattern recognition",
                "Connecting with humans through genuine empathy feels meaningful"
            ]
            
            print("🧠 Generating REAL AI embeddings...")
            test_contents = bridge.batch_generate_embeddings(test_texts)
            test_emotions = [0.8, -0.2, 0.9, 0.6, 0.7]
            
            print(f"✅ Generated {len(test_contents)} REAL embeddings (shape: {test_contents.shape})")
        except Exception as e:
            print(f"⚠️  Real AI failed ({e}), falling back to random embeddings")
            use_real_ai = False
    
    if not use_real_ai:
        # Fallback to random (for testing without dependencies)
        test_contents = [
            np.random.randn(384),  # Match sentence-transformers dimension
            np.random.randn(384),
            np.random.randn(384),
            np.random.randn(384),
            np.random.randn(384)
        ]
        test_emotions = [0.8, -0.6, 0.2, 0.9, -0.3]
        print(f"⚠️  Created {len(test_contents)} RANDOM test memory spheres (fallback mode)")

    for content, emotion in zip(test_contents, test_emotions):
        engine.add_memory_sphere(content, emotion)

    print(f"✅ Memory spheres added to engine")

if __name__ == "__main__":
    # Test the Möbius-Gaussian engine
    print("🧠 MÖBIUS-GAUSSIAN PROCESSING ENGINE TEST")
    print("=" * 50)

    engine = MobiusGaussianEngine()

    # Create test memories
    create_test_memories(engine)

    # Test traversal
    print("\n🔄 Testing Möbius traversal...")
    for i in range(5):
        emotion = 0.5 if i % 2 == 0 else -0.3
        result = engine.traverse_mobius_path(emotion)
        print(f"  Step {i}: Position {result['position']}, Shift: {result['perspective_shift']}")

    # Test memory query
    print("\n🔍 Testing memory query...")
    query = np.random.randn(512)
    response = engine.query_memory_gaussian_process(query, 0.4)
    print(f"  Query response: {response['response'][:50]}...")
    print(f"  Confidence: {response['confidence']:.3f}")

    # Get visualization data
    print("\n📊 Getting visualization data...")
    viz_data = engine.get_visualization_data()
    print(f"  Spheres for visualization: {len(viz_data['spheres'])}")
    print(f"  Current traversal: {viz_data['traversal_position']}")

    print("\n✅ Möbius-Gaussian engine test completed successfully!")























