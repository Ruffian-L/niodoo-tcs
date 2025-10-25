from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json
from pathlib import Path

@dataclass
class PipelineMetrics:
    start_time: datetime = field(default_factory=datetime.now)
    embedding_time: float = 0.0
    traversal_time: float = 0.0
    retrieval_time: float = 0.0
    generation_time: float = 0.0
    total_time: float = 0.0
    cache_hits: int = 0
    emotional_confidence: float = 0.0
    memory_count: int = 0
    rouge_score: float = 0.0  # Stub for future self-consistency

    def to_dict(self) -> Dict[str, Any]:
        end_time = datetime.now()
        self.total_time = (end_time - self.start_time).total_seconds() * 1000  # ms
        return {
            'embedding_time_ms': self.embedding_time,
            'traversal_time_ms': self.traversal_time,
            'retrieval_time_ms': self.retrieval_time,
            'generation_time_ms': self.generation_time,
            'total_time_ms': self.total_time,
            'cache_hits': self.cache_hits,
            'emotional_confidence': self.emotional_confidence,
            'memory_count': self.memory_count,
            'rouge_score': self.rouge_score
        }

@dataclass
class PipelineContext:
    """
    Centralized context for Niodoo pipeline stages.
    Manages state, caching, fallbacks, and metrics.
    """
    config_path: str = "config/settings.json"
    cache_dir: str = "cache"
    enable_fallbacks: bool = True
    max_retries: int = 3
    timeout_ms: int = 5000

    # Runtime state
    input_prompt: str = ""
    embedding: Optional[list] = None
    emotional_context: float = 0.0
    traversal_result: Optional[Dict] = None
    retrieved_memories: List[Tuple[Any, float]] = field(default_factory=list)
    generated_response: str = ""
    metrics: PipelineMetrics = field(default_factory=PipelineMetrics)

    # Caching (L1: in-memory)
    embedding_cache: Dict[str, list] = field(default_factory=dict)
    context_cache: Dict[str, Dict] = field(default_factory=dict)

    def get_cache_key(self, text: str) -> str:
        import hashlib
        return hashlib.md5(text.encode()).hexdigest()

    def cache_embedding(self, text: str, embedding: list):
        if self.embedding:
            key = self.get_cache_key(text)
            self.embedding_cache[key] = embedding
            self.metrics.cache_hits += 1

    def get_cached_embedding(self, text: str) -> Optional[list]:
        key = self.get_cache_key(text)
        return self.embedding_cache.get(key)

    def to_json_output(self) -> str:
        output = {
            'response': self.generated_response,
            'metrics': self.metrics.to_dict(),
            'emotional_context': self.emotional_context,
            'memory_count': len(self.retrieved_memories),
            'traversal_position': self.traversal_result.get('position') if self.traversal_result else None
        }
        return json.dumps(output, indent=2)

    def log_stage_time(self, stage: str, start_time: datetime):
        elapsed = (datetime.now() - start_time).total_seconds() * 1000
        if stage == 'embedding':
            self.metrics.embedding_time = elapsed
        elif stage == 'traversal':
            self.metrics.traversal_time = elapsed
        elif stage == 'retrieval':
            self.metrics.retrieval_time = elapsed
        elif stage == 'generation':
            self.metrics.generation_time = elapsed
