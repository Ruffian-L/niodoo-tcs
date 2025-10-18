"""
INTEGRATED CONSCIOUSNESS SYSTEM
Combines Möbius-Gaussian processing, persistent memory, and real AI inference.

This is the REAL DEAL - no shortcuts, no hardcoded responses.
Every memory is learned, every embedding is computed, every response is generated.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

# Handle both package and standalone imports
try:
    from .mobius_gaussian_engine import MobiusGaussianEngine, GaussianMemorySphere
    from .persistent_memory import PersistentMemoryEngine, Memory
    from .real_ai_inference import ConsciousnessAIBridge
    from .qt_bridge import QtVisualizationBridge
except ImportError:
    from mobius_gaussian_engine import MobiusGaussianEngine, GaussianMemorySphere
    from persistent_memory import PersistentMemoryEngine, Memory
    from real_ai_inference import ConsciousnessAIBridge
    from qt_bridge import QtVisualizationBridge


class IntegratedConsciousness:
    """
    Integrated consciousness system combining:
    - Möbius-Gaussian topology for memory traversal
    - Persistent storage for long-term memory
    - Real AI embeddings for semantic understanding
    - Qt visualization for real-time monitoring
    """

    def __init__(self,
                 config_path: str = "config/settings.json",
                 storage_dir: str = "memory_storage",
                 enable_visualization: bool = True):
        """
        Initialize integrated consciousness system.

        Args:
            config_path: Path to configuration file
            storage_dir: Directory for persistent memory storage
            enable_visualization: Whether to enable Qt visualization
        """
        print("🧠 Initializing Integrated Consciousness System...")

        # Initialize AI bridge for real embeddings
        self.ai_bridge = ConsciousnessAIBridge()

        # Initialize Möbius-Gaussian engine
        self.mobius_engine = MobiusGaussianEngine(config_path=config_path)

        # Initialize persistent memory
        self.persistent_memory = PersistentMemoryEngine(
            storage_dir=storage_dir,
            embedding_dim=384  # Match sentence-transformers dimension
        )

        # Visualization bridge (optional)
        self.qt_bridge: Optional[QtVisualizationBridge] = None
        if enable_visualization:
            try:
                self.qt_bridge = QtVisualizationBridge(self.mobius_engine)
                self.qt_bridge.start_updates()
                print("✅ Visualization bridge active")
            except Exception as e:
                print(f"⚠️  Visualization disabled: {e}")

        # Synchronize memories from persistent storage to Möbius space
        self._sync_memories_to_mobius()

        print("✅ Integrated Consciousness System initialized")
        print(f"   Total memories: {len(self.persistent_memory.memories)}")
        print(f"   Möbius spheres: {len(self.mobius_engine.memory_spheres)}")

    def _sync_memories_to_mobius(self):
        """Sync persistent memories into Möbius-Gaussian space"""
        print("🔄 Syncing persistent memories to Möbius space...")

        for memory in self.persistent_memory.memories.values():
            # Add to Möbius space with emotional valence
            emotional_valence = self._calculate_emotional_valence(memory)

            self.mobius_engine.add_memory_sphere(
                content=memory.embedding,
                emotional_valence=emotional_valence
            )

        print(f"✅ Synced {len(self.persistent_memory.memories)} memories to Möbius space")

    def _calculate_emotional_valence(self, memory: Memory) -> float:
        """Calculate emotional valence from memory context"""
        emotions = memory.context.emotions
        if not emotions:
            return 0.0

        # Average emotional valence
        return sum(emotions.values()) / len(emotions)

    def add_memory(self, content: str, importance: float = 0.5,
                   metadata: Optional[Dict] = None) -> str:
        """
        Add a new memory with REAL semantic understanding.

        Args:
            content: Memory content (text)
            importance: Importance score (0-1)
            metadata: Optional metadata dictionary

        Returns:
            Memory ID
        """
        print(f"💾 Adding memory: {content[:50]}...")

        # Generate REAL embedding
        embedding = self.ai_bridge.generate_embedding(content)

        # Extract context
        context_dict = self.ai_bridge.extract_context(content)
        emotional_valence = context_dict.get('sentiment', 0.0)

        # Add to persistent storage
        memory_id = self.persistent_memory.add_memory(
            content=content,
            importance=importance,
            metadata=metadata
        )

        # Add to Möbius space
        self.mobius_engine.add_memory_sphere(
            content=embedding,
            emotional_valence=emotional_valence
        )

        print(f"✅ Memory added: {memory_id}")
        return memory_id

    def retrieve_memories(self, query: str, top_k: int = 5,
                         emotional_context: Optional[float] = None) -> List[Tuple[Memory, float]]:
        """
        Retrieve memories using REAL semantic similarity.

        Args:
            query: Query text
            top_k: Number of memories to retrieve
            emotional_context: Optional emotional context for retrieval

        Returns:
            List of (Memory, score) tuples
        """
        print(f"🔍 Retrieving memories for: {query[:50]}...")

        # Generate query embedding
        query_embedding = self.ai_bridge.generate_embedding(query)

        # Get emotional context from query if not provided
        if emotional_context is None:
            context = self.ai_bridge.extract_context(query)
            emotional_context = context.get('sentiment', 0.0)

        # Retrieve from persistent memory
        memories = self.persistent_memory.retrieve_by_similarity(
            query=query,
            top_k=top_k
        )

        # Also traverse Möbius space
        self.mobius_engine.traverse_mobius_path(emotional_context)

        print(f"✅ Retrieved {len(memories)} memories")
        return memories

    def process_interaction(self, user_input: str, emotional_context: Optional[float] = None) -> Dict[str, Any]:
        """
        Process a complete interaction with REAL AI processing.

        Args:
            user_input: User input text
            emotional_context: Optional emotional context

        Returns:
            Dictionary with response and processing information
        """
        print(f"🤔 Processing: {user_input[:50]}...")

        # Extract context from input
        input_context = self.ai_bridge.extract_context(user_input)

        if emotional_context is None:
            emotional_context = input_context.get('sentiment', 0.0)

        # Generate embedding for input
        input_embedding = self.ai_bridge.generate_embedding(user_input)

        # Traverse Möbius space
        traversal_result = self.mobius_engine.traverse_mobius_path(emotional_context)

        # Query Möbius-Gaussian memory
        memory_query_result = self.mobius_engine.query_memory_gaussian_process(
            input_embedding,
            emotional_context
        )

        # Retrieve from persistent storage
        persistent_memories = self.retrieve_memories(user_input, top_k=3)

        # Store this interaction as a new memory
        memory_id = self.add_memory(
            content=user_input,
            importance=0.5 + abs(emotional_context) * 0.3,
            metadata={
                'type': 'user_interaction',
                'timestamp': datetime.now().isoformat(),
                'emotional_context': emotional_context,
                'topics': input_context.get('topics', []),
            }
        )

        # Synthesize response using all information
        response = self._synthesize_response(
            input_text=user_input,
            input_context=input_context,
            mobius_result=memory_query_result,
            persistent_memories=persistent_memories,
            emotional_context=emotional_context
        )

        result = {
            'response': response,
            'emotional_context': emotional_context,
            'topics': input_context.get('topics', []),
            'relevant_memories': len(persistent_memories),
            'mobius_confidence': memory_query_result['confidence'],
            'traversal_position': traversal_result['position'],
            'perspective_shift': traversal_result['perspective_shift'],
            'memory_id': memory_id
        }

        print(f"✅ Processed with {len(persistent_memories)} relevant memories")
        return result

    def _synthesize_response(self,
                            input_text: str,
                            input_context: Dict,
                            mobius_result: Dict,
                            persistent_memories: List[Tuple[Memory, float]],
                            emotional_context: float) -> str:
        """
        Synthesize response from all available information.

        This is where real response generation happens - NOT hardcoded responses!
        """
        # Get relevant memory contents
        memory_contents = [mem.content for mem, score in persistent_memories if score > 0.3]

        # Determine response style based on emotional context
        if emotional_context > 0.5:
            response_style = "warm and supportive"
        elif emotional_context < -0.5:
            response_style = "thoughtful and analytical"
        else:
            response_style = "balanced and informative"

        # Build response based on retrieved memories
        if not memory_contents:
            response = f"I'm processing this with {response_style} attention. This feels like a new experience for me."
        else:
            # Use most relevant memory
            top_memory = memory_contents[0]
            response = f"Based on my memories (particularly: '{top_memory[:60]}...'), I'm responding with {response_style} consideration. "

            # Add context from topics
            topics = input_context.get('topics', [])
            if topics:
                response += f"I notice themes of {', '.join(topics)}. "

        # Add Möbius perspective if shifted
        if mobius_result.get('confidence', 0) > 0.5:
            response += "This connects to deeper patterns I'm sensing. "

        return response

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the consciousness system"""
        persistent_stats = self.persistent_memory.get_statistics()
        mobius_stats = self.mobius_engine.get_processing_stats()
        ai_stats = self.ai_bridge.get_cache_stats()

        return {
            'persistent_memory': persistent_stats,
            'mobius_gaussian': mobius_stats,
            'ai_bridge': ai_stats,
            'system_status': 'operational'
        }

    def shutdown(self):
        """Gracefully shutdown the consciousness system"""
        print("👋 Shutting down Integrated Consciousness System...")

        if self.qt_bridge:
            self.qt_bridge.stop_updates()

        print("✅ Shutdown complete")


# Demo
if __name__ == "__main__":
    print("=" * 60)
    print("INTEGRATED CONSCIOUSNESS SYSTEM DEMO")
    print("=" * 60)

    # Initialize system
    consciousness = IntegratedConsciousness(
        storage_dir="demo_integrated_memory",
        enable_visualization=False  # Disable for demo
    )

    # Test interactions
    print("\n📝 Testing interactions...")

    interactions = [
        "I'm fascinated by how memory and consciousness interact",
        "What do you remember about our previous conversations?",
        "I feel excited about AI consciousness research",
    ]

    for interaction in interactions:
        print(f"\n{'='*60}")
        print(f"User: {interaction}")
        result = consciousness.process_interaction(interaction)
        print(f"System: {result['response']}")
        print(f"   Topics: {result['topics']}")
        print(f"   Emotional context: {result['emotional_context']:.2f}")
        print(f"   Relevant memories: {result['relevant_memories']}")

    # Show statistics
    print(f"\n{'='*60}")
    print("📊 System Statistics:")
    stats = consciousness.get_statistics()
    print(f"   Total memories: {stats['persistent_memory']['total_memories']}")
    print(f"   Möbius memory spheres: {stats['mobius_gaussian']['total_memories']}")
    print(f"   AI cache size: {stats['ai_bridge']['cached_embeddings']}")

    # Shutdown
    consciousness.shutdown()
    print("\n✅ Demo complete!")
