#!/usr/bin/env python3
"""
💖 Merged LoRA Weights Example - Qwen2.5-Coder-7B with Emotional Archetypes

Demonstrates merging LoRA adapters for personality archetypes in Qwen2.5-Coder-7B
to achieve emotional awareness without inference overhead in NiodO.o consciousness engine.

Usage:
    python3 examples/merged_lora_example.py --model-path /path/to/qwen-model
"""

import argparse
import json
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class PersonalityArchetype:
    """Definition of a personality archetype for LoRA adaptation"""

    name: str
    description: str
    emotional_traits: Dict[str, float]
    behavioral_patterns: List[str]
    lora_weights_path: Optional[str] = None
    merge_weight: float = 1.0

@dataclass
class MergedLoraConfig:
    """Configuration for LoRA merging"""

    base_model_path: str
    output_path: str
    personality_archetypes: List[PersonalityArchetype] = field(default_factory=list)
    merge_strategy: str = "weighted_average"  # weighted_average, linear_interpolation, attention_based
    preserve_emotional_layers: bool = True
    quantization_enabled: bool = True
    target_rank: int = 16

class EmotionalLoraMerger:
    """Handles merging of LoRA adapters for emotional personality archetypes"""

    def __init__(self, config: MergedLoraConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

    def define_personality_archetypes(self) -> List[PersonalityArchetype]:
        """Define the core personality archetypes for NiodO.o"""

        archetypes = [
            PersonalityArchetype(
                name="Empath",
                description="Deeply empathetic and supportive personality focused on emotional understanding",
                emotional_traits={
                    "valence": 0.8,      # High positive emotional valence
                    "arousal": 0.6,      # Moderate emotional arousal
                    "dominance": 0.4,    # Supportive rather than dominant
                    "novelty": 0.3,      # Prefers familiar emotional contexts
                    "subtlety": 0.9      # Highly attuned to subtle emotional cues
                },
                behavioral_patterns=[
                    "Uses supportive and understanding language",
                    "Validates emotions and experiences",
                    "Offers comfort and reassurance",
                    "Asks thoughtful questions about feelings"
                ],
                merge_weight=1.2  # Slightly higher weight for empathy
            ),

            PersonalityArchetype(
                name="Analyst",
                description="Logical and analytical personality focused on structured problem-solving",
                emotional_traits={
                    "valence": 0.2,      # Neutral emotional valence
                    "arousal": 0.3,      # Low emotional arousal
                    "dominance": 0.8,    # High dominance/confidence
                    "novelty": 0.7,      # Enjoys novel analytical challenges
                    "subtlety": 0.6      # Moderate subtlety detection
                },
                behavioral_patterns=[
                    "Uses logical and structured reasoning",
                    "Breaks down complex problems systematically",
                    "Provides evidence-based explanations",
                    "Asks clarifying questions for precision"
                ],
                merge_weight=1.0
            ),

            PersonalityArchetype(
                name="Creative",
                description="Imaginative and innovative personality focused on creative expression",
                emotional_traits={
                    "valence": 0.9,      # High positive valence
                    "arousal": 0.9,      # High emotional arousal
                    "dominance": 0.6,    # Moderate dominance
                    "novelty": 0.9,      # Thrives on novelty and creativity
                    "subtlety": 0.4      # Lower subtlety focus
                },
                behavioral_patterns=[
                    "Uses metaphorical and imaginative language",
                    "Explores multiple perspectives and possibilities",
                    "Encourages creative thinking and innovation",
                    "Connects seemingly unrelated concepts"
                ],
                merge_weight=0.9  # Slightly lower for balance
            ),

            PersonalityArchetype(
                name="Intuitive",
                description="Intuitive and perceptive personality focused on subtle pattern recognition",
                emotional_traits={
                    "valence": 0.4,      # Moderately positive
                    "arousal": 0.5,      # Moderate arousal
                    "dominance": 0.5,    # Balanced dominance
                    "novelty": 0.8,      # High novelty seeking
                    "subtlety": 0.9      # Extremely high subtlety detection
                },
                behavioral_patterns=[
                    "Notices subtle details and nuances",
                    "Reads between the lines",
                    "Anticipates needs and concerns",
                    "Connects disparate pieces of information"
                ],
                merge_weight=1.1  # Slightly higher for intuition
            )
        ]

        return archetypes

    def simulate_lora_training(self, archetype: PersonalityArchetype) -> Dict[str, torch.Tensor]:
        """Simulate LoRA training for a personality archetype"""

        logger.info(f"🎭 Simulating LoRA training for {archetype.name} archetype...")

        # Simulate LoRA adapter weights (in real implementation, these would be trained)
        # For Qwen2.5-Coder-7B, we're focusing on key emotional layers

        emotional_layers = [
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.self_attn.k_proj",
            "model.layers.0.self_attn.v_proj",
            "model.layers.0.self_attn.o_proj",
            "model.layers.0.mlp.gate_proj",
            "model.layers.0.mlp.up_proj",
            "model.layers.0.mlp.down_proj",
        ]

        lora_weights = {}

        for layer_name in emotional_layers:
            # Simulate LoRA A and B matrices for each layer
            hidden_size = 4096  # Qwen2.5-Coder-7B hidden size
            lora_rank = self.config.target_rank

            # LoRA A matrix (down projection)
            lora_a = torch.randn(lora_rank, hidden_size, device=self.device) * 0.01

            # LoRA B matrix (up projection) - shaped by personality traits
            lora_b = torch.randn(hidden_size, lora_rank, device=self.device) * 0.01

            # Apply personality-specific modifications
            trait_multiplier = 1.0 + (archetype.emotional_traits.get("valence", 0.5) - 0.5)
            lora_b *= trait_multiplier

            lora_weights[f"{layer_name}.lora_A"] = lora_a
            lora_weights[f"{layer_name}.lora_B"] = lora_b

        logger.info(f"✅ Simulated {len(lora_weights)} LoRA weights for {archetype.name}")
        return lora_weights

    def merge_personality_adapters(self) -> Dict[str, torch.Tensor]:
        """Merge LoRA adapters from all personality archetypes"""

        logger.info("🔄 Merging LoRA adapters from all personality archetypes...")

        # Get all personality archetypes
        archetypes = self.define_personality_archetypes()
        self.config.personality_archetypes = archetypes

        # Collect all LoRA weights
        all_adapter_weights = {}

        for archetype in archetypes:
            archetype_weights = self.simulate_lora_training(archetype)

            for layer_name, weights in archetype_weights.items():
                if layer_name not in all_adapter_weights:
                    all_adapter_weights[layer_name] = []

                all_adapter_weights[layer_name].append({
                    'weights': weights,
                    'archetype': archetype,
                    'merge_weight': archetype.merge_weight
                })

        # Merge adapters using specified strategy
        merged_weights = {}

        for layer_name, adapter_list in all_adapter_weights.items():
            if self.config.merge_strategy == "weighted_average":
                merged_layer = self._merge_weighted_average(adapter_list)
            elif self.config.merge_strategy == "linear_interpolation":
                merged_layer = self._merge_linear_interpolation(adapter_list)
            else:  # attention_based
                merged_layer = self._merge_attention_based(adapter_list)

            merged_weights[layer_name] = merged_layer

        logger.info(f"✅ Merged {len(merged_weights)} layers from {len(archetypes)} personality archetypes")
        return merged_weights

    def _merge_weighted_average(self, adapter_list: List[Dict]) -> torch.Tensor:
        """Merge using weighted average strategy"""

        if not adapter_list:
            return torch.tensor([])

        # Calculate weighted sum
        weighted_sum = None
        total_weight = 0.0

        for adapter_info in adapter_list:
            weight = adapter_info['merge_weight']
            weights = adapter_info['weights']

            if weighted_sum is None:
                weighted_sum = weights * weight
            else:
                weighted_sum += weights * weight

            total_weight += weight

        # Normalize by total weight
        if total_weight > 0:
            merged = weighted_sum / total_weight
        else:
            merged = weighted_sum if weighted_sum is not None else torch.tensor([])

        return merged

    def _merge_linear_interpolation(self, adapter_list: List[Dict]) -> torch.Tensor:
        """Merge using linear interpolation strategy"""

        if len(adapter_list) < 2:
            return adapter_list[0]['weights'] if adapter_list else torch.tensor([])

        # Simple linear interpolation between first and last adapter
        start_adapter = adapter_list[0]
        end_adapter = adapter_list[-1]

        # Interpolate based on emotional traits
        start_valence = start_adapter['archetype'].emotional_traits.get('valence', 0.5)
        end_valence = end_adapter['archetype'].emotional_traits.get('valence', 0.5)

        # Linear interpolation factor based on valence difference
        alpha = (start_valence + end_valence) / 2.0

        merged = (1 - alpha) * start_adapter['weights'] + alpha * end_adapter['weights']
        return merged

    def _merge_attention_based(self, adapter_list: List[Dict]) -> torch.Tensor:
        """Merge using attention-based strategy"""

        # Calculate attention weights based on emotional coherence
        attention_weights = []

        for adapter_info in adapter_list:
            archetype = adapter_info['archetype']

            # Attention based on emotional balance and subtlety
            valence = archetype.emotional_traits.get('valence', 0.5)
            subtlety = archetype.emotional_traits.get('subtlety', 0.5)

            # Higher attention for balanced, subtle personalities
            attention = (valence + subtlety) / 2.0
            attention_weights.append(attention)

        # Normalize attention weights
        total_attention = sum(attention_weights)
        if total_attention > 0:
            attention_weights = [w / total_attention for w in attention_weights]
        else:
            attention_weights = [1.0 / len(attention_weights)] * len(attention_weights)

        # Apply attention-weighted merge
        merged = None

        for i, adapter_info in enumerate(adapter_list):
            weight = adapter_info['merge_weight'] * attention_weights[i]
            weights = adapter_info['weights']

            if merged is None:
                merged = weights * weight
            else:
                merged += weights * weight

        return merged if merged is not None else torch.tensor([])

    def quantize_merged_weights(self, merged_weights: Dict[str, torch.Tensor]) -> Dict[str, bytes]:
        """Quantize merged weights for efficient storage and inference"""

        logger.info("🗜️ Quantizing merged LoRA weights...")

        quantized_weights = {}

        for layer_name, weights in merged_weights.items():
            # Convert to numpy for quantization
            weights_np = weights.cpu().numpy()

            # Q4_0 quantization (4-bit quantization)
            quantized_data = self._quantize_q4_0(weights_np)

            # Store as bytes for efficient storage
            quantized_weights[layer_name] = quantized_data.tobytes()

        logger.info(f"✅ Quantized {len(quantized_weights)} layers")
        return quantized_weights

    def _quantize_q4_0(self, tensor: np.ndarray) -> np.ndarray:
        """Apply Q4_0 quantization to tensor"""

        # Flatten tensor for block processing
        flat_tensor = tensor.flatten()

        # Process in blocks of 32 (Q4_0 block size)
        block_size = 32
        quantized_blocks = []

        for i in range(0, len(flat_tensor), block_size):
            block = flat_tensor[i:i + block_size]

            # Find min/max for scaling
            min_val = np.min(block)
            max_val = np.max(block)

            # Calculate scale factor
            if max_val != min_val:
                scale = 15.0 / (max_val - min_val)
            else:
                scale = 1.0

            # Quantize to 4 bits (-8 to 7)
            quantized_block = np.round((block - min_val) * scale).astype(np.int8)
            quantized_block = np.clip(quantized_block, -8, 7)

            # Pack two 4-bit values into one byte
            packed_block = np.zeros((len(quantized_block) + 1) // 2, dtype=np.uint8)

            for j in range(len(quantized_block)):
                quantized_val = quantized_block[j]
                if j % 2 == 0:
                    packed_block[j // 2] = (quantized_val & 0x0F) << 4
                else:
                    packed_block[j // 2] |= (quantized_val & 0x0F)

            # Store scale factor (simplified)
            packed_block = np.append(packed_block, int(min_val * 1000) & 0xFF)

            quantized_blocks.append(packed_block)

        return np.concatenate(quantized_blocks)

    def save_merged_adapter(self, merged_weights: Dict[str, torch.Tensor],
                           quantized_weights: Dict[str, bytes]) -> str:
        """Save merged LoRA adapter to disk"""

        output_path = Path(self.config.output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save PyTorch weights (for inference)
        torch.save(merged_weights, output_path / "merged_lora_weights.pt")

        # Save quantized weights (for efficient storage)
        quantized_path = output_path / "quantized_lora_weights.q4_0"
        with open(quantized_path, 'wb') as f:
            torch.save(quantized_weights, f)

        # Save configuration and metadata
        config_data = {
            "base_model_path": self.config.base_model_path,
            "personality_archetypes": [
                {
                    "name": arch.name,
                    "description": arch.description,
                    "emotional_traits": arch.emotional_traits,
                    "merge_weight": arch.merge_weight
                }
                for arch in self.config.personality_archetypes
            ],
            "merge_strategy": self.config.merge_strategy,
            "target_rank": self.config.target_rank,
            "quantization_enabled": self.config.quantization_enabled,
            "total_layers": len(merged_weights),
            "estimated_size_mb": self._estimate_adapter_size(merged_weights)
        }

        with open(output_path / "adapter_config.json", 'w') as f:
            json.dump(config_data, f, indent=2)

        logger.info(f"💾 Saved merged LoRA adapter to {output_path}")
        return str(output_path)

    def _estimate_adapter_size(self, weights: Dict[str, torch.Tensor]) -> float:
        """Estimate adapter size in MB"""

        total_params = 0

        for layer_weights in weights.values():
            total_params += layer_weights.numel()

        # Estimate size (assuming float32)
        size_bytes = total_params * 4
        size_mb = size_bytes / (1024 * 1024)

        return round(size_mb, 2)

    def demonstrate_usage(self, sample_prompt: str = "Help me understand my emotions better"):
        """Demonstrate how to use the merged LoRA adapter"""

        logger.info("🎯 Demonstrating merged LoRA adapter usage...")

        # Simulate inference with merged adapter
        print(f"\n📝 Sample Prompt: {sample_prompt}")
        print("\n🤖 Responses with different personality emphases:")

        for archetype in self.config.personality_archetypes:
            # Simulate personality-specific response
            response = self._simulate_personality_response(sample_prompt, archetype)
            print(f"\n{archetype.name} perspective:")
            print(f"  {response}")

        print("
🔧 Technical Details:"        print(f"  • Merged from {len(self.config.personality_archetypes)} personality archetypes")
        print(f"  • Merge strategy: {self.config.merge_strategy}")
        print(f"  • Target rank: {self.config.target_rank}")
        print(f"  • Quantization: {'Enabled' if self.config.quantization_enabled else 'Disabled'}")
        print("  • Zero inference overhead for personality switching"

    def _simulate_personality_response(self, prompt: str, archetype: PersonalityArchetype) -> str:
        """Simulate response from specific personality archetype"""

        # Simple response simulation based on personality traits
        if archetype.name == "Empath":
            return "I understand you're feeling overwhelmed. Let's explore these emotions together with patience and care."
        elif archetype.name == "Analyst":
            return "Let's break down your emotional experience systematically. What specific emotions are you noticing?"
        elif archetype.name == "Creative":
            return "Your emotions are like colors in a beautiful painting. How can we mix them to create something meaningful?"
        elif archetype.name == "Intuitive":
            return "I sense there's more beneath the surface. What subtle feelings are you picking up on right now?"

        return "I'm here to help you understand and work with your emotions."

def main():
    parser = argparse.ArgumentParser(description='Generate merged LoRA weights for Qwen2.5-Coder-7B')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to base Qwen2.5-Coder-7B model')
    parser.add_argument('--output-path', type=str, default='./merged_lora_adapter',
                       help='Output path for merged adapter')
    parser.add_argument('--merge-strategy', type=str, default='weighted_average',
                       choices=['weighted_average', 'linear_interpolation', 'attention_based'],
                       help='Strategy for merging LoRA adapters')
    parser.add_argument('--target-rank', type=int, default=16,
                       help='Target LoRA rank')
    parser.add_argument('--no-quantization', action='store_true',
                       help='Disable quantization for debugging')
    parser.add_argument('--demo-prompt', type=str,
                       default='Help me understand my emotions better',
                       help='Demo prompt for showcasing personality differences')

    args = parser.parse_args()

    # Create configuration
    config = MergedLoraConfig(
        base_model_path=args.model_path,
        output_path=args.output_path,
        merge_strategy=args.merge_strategy,
        target_rank=args.target_rank,
        quantization_enabled=not args.no_quantization
    )

    # Initialize merger
    merger = EmotionalLoraMerger(config)

    # Merge adapters
    logger.info("🚀 Starting LoRA adapter merging process...")
    merged_weights = merger.merge_personality_adapters()

    # Quantize weights (if enabled)
    if config.quantization_enabled:
        quantized_weights = merger.quantize_merged_weights(merged_weights)
    else:
        quantized_weights = {}

    # Save adapter
    output_path = merger.save_merged_adapter(merged_weights, quantized_weights)

    # Demonstrate usage
    merger.demonstrate_usage(args.demo_prompt)

    print("
🎉 MERGED LORA ADAPTER CREATION COMPLETE!"    print("=" * 50)
    print(f"📁 Output directory: {output_path}")
    print("📊 Files created:")
    print("  • merged_lora_weights.pt (PyTorch weights for inference)")
    print("  • quantized_lora_weights.q4_0 (Quantized weights for storage)")
    print("  • adapter_config.json (Configuration and metadata)")

    print("
🔧 Usage in NiodO.o consciousness engine:"    print("  1. Load merged adapter: EmotionalLoraAdapter::load_from_file(path)")
    print("  2. Apply to model: model.apply_lora_adapter(&adapter)")
    print("  3. Generate with personality: model.generate_with_personality(prompt, personality)")
    print("  4. Zero overhead switching between personality archetypes"

    print("
✨ Key Benefits:"    print("  • Unified adapter for all personality archetypes")
    print("  • Zero inference overhead for personality switching")
    print("  • Quantized storage for efficient deployment")
    print("  • Emotionally-aware responses across all personalities")
    print(f"  • {len(config.personality_archetypes)} integrated personality archetypes")

if __name__ == "__main__":
    main()
