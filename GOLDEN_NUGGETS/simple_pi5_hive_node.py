#!/usr/bin/env python3
"""
Pi5 Memory Palace Node - Ultimate ADHD Hive Brain (OFFLINE VERSION)
Role: Memory retrieval, context maintenance, historical analysis
Models: Phi-3-mini + Qwen2.5-1.5B (simulated)
Max Streams: 40
"""

import time
import random
import json
import os
from dataclasses import dataclass
from enum import Enum
from collections import deque

class MemoryStream(Enum):
    MEMORY_RECALL = "memory_recall"
    CONTEXT_MAINTENANCE = "context_maintenance"
    HISTORICAL_ANALYSIS = "historical_analysis"
    ANXIETY_SPIRAL = "anxiety_spiral"

@dataclass
class MemoryThought:
    thought_id: str
    content: str
    memory_type: str
    confidence: float
    timestamp: float = time.time()

class Pi5MemoryNode:
    def __init__(self):
        self.node_id = "pi5_memory_palace"
        self.max_streams = 40
        self.memory_streams = {}
        self.memory_palace = deque(maxlen=10000)
        self.active_thoughts = 0
        
        # Initialize memory streams
        for i in range(self.max_streams):
            stream_type = random.choice(list(MemoryStream))
            self.memory_streams[f"memory_stream_{i}"] = {
                "type": stream_type,
                "thoughts": deque(maxlen=100),
                "active": True
            }
        
        print(f"🧠 Pi5 Memory Palace initialized with {self.max_streams} streams")
    
    def process_memory_stimulus(self, stimulus: str) -> dict:
        """Process stimulus through memory palace"""
        start_time = time.time()
        
        # Generate thoughts across memory streams
        thoughts = []
        for stream_id, stream in self.memory_streams.items():
            if stream["active"]:
                thought = self._generate_memory_thought(stimulus, stream["type"])
                stream["thoughts"].append(thought)
                thoughts.append(thought)
                self.active_thoughts += 1
        
        # Memory palace analysis
        memory_insights = self._analyze_memory_patterns(thoughts)
        
        processing_time = time.time() - start_time
        
        return {
            "node": self.node_id,
            "thoughts_generated": len(thoughts),
            "memory_insights": memory_insights,
            "processing_time": processing_time,
            "active_streams": len([s for s in self.memory_streams.values() if s["active"]]),
            "memory_palace_size": len(self.memory_palace)
        }
    
    def _generate_memory_thought(self, stimulus: str, memory_type: MemoryStream) -> MemoryThought:
        """Generate a memory-based thought"""
        thought_id = f"pi5_mem_{int(time.time() * 1000)}"
        
        templates = {
            MemoryStream.MEMORY_RECALL: [
                "Memory palace recalls {} from 47 similar past experiences",
                "Long-term storage retrieves critical insight about {}",
                "Episodic memory links {} to breakthrough moment #238"
            ],
            MemoryStream.CONTEXT_MAINTENANCE: [
                "Contextual memory suggests {} previously solved similar problem",
                "Working memory maintains focus on {} while processing background data",
                "Semantic memory connects {} to 17 related concepts"
            ],
            MemoryStream.HISTORICAL_ANALYSIS: [
                "Historical analysis links {} to breakthrough moment #238",
                "Temporal memory shows {} follows pattern from 3 years ago",
                "Chronological analysis reveals {} matches historical trend #47"
            ],
            MemoryStream.ANXIETY_SPIRAL: [
                "Memory anxiety: What if {} doesn't work like last time?",
                "Historical worry: {} failed in 3 previous attempts, but maybe...",
                "Memory pressure: Must remember everything about {} or fail"
            ]
        }
        
        template = random.choice(templates[memory_type])
        content = template.format(stimulus[:30])
        
        return MemoryThought(
            thought_id=thought_id,
            content=content,
            memory_type=memory_type.value,
            confidence=random.uniform(0.6, 0.95)
        )
    
    def _analyze_memory_patterns(self, thoughts: list) -> list:
        """Analyze patterns in memory thoughts"""
        insights = []
        
        # Group by memory type
        memory_groups = {}
        for thought in thoughts:
            if thought.memory_type not in memory_groups:
                memory_groups[thought.memory_type] = []
            memory_groups[thought.memory_type].append(thought)
        
        # Generate insights
        for memory_type, group_thoughts in memory_groups.items():
            if len(group_thoughts) >= 3:
                insights.append({
                    "type": f"{memory_type}_pattern",
                    "strength": len(group_thoughts) / len(thoughts),
                    "thoughts": [t.thought_id for t in group_thoughts[:3]]
                })
        
        return insights

def main():
    """Main offline hive brain loop"""
    print("🚀 Starting Pi5 Memory Palace Node (OFFLINE)")
    print("✅ No internet required - running locally")
    
    memory_node = Pi5MemoryNode()
    
    # Create output directory
    os.makedirs("/home/pi/hive_brain_output", exist_ok=True)
    
    # Main processing loop
    stimulus_counter = 0
    while True:
        try:
            # Generate random stimulus for testing
            stimuli = [
                "distributed consciousness",
                "ADHD hive mind",
                "memory palace activation",
                "neural pattern recognition",
                "creative breakthrough",
                "system integration",
                "multi-node coordination",
                "emergent intelligence"
            ]
            
            stimulus = random.choice(stimuli)
            stimulus_counter += 1
            
            # Process stimulus
            result = memory_node.process_memory_stimulus(stimulus)
            
            # Save result to file
            output_file = f"/home/pi/hive_brain_output/memory_result_{stimulus_counter}.json"
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            
            print(f"[{time.strftime('%H:%M:%S')}] Processed stimulus #{stimulus_counter}: {stimulus}")
            print(f"  Thoughts generated: {result['thoughts_generated']}")
            print(f"  Active streams: {result['active_streams']}")
            
            # Sleep for a bit
            time.sleep(5)
            
        except KeyboardInterrupt:
            print("\n🛑 Pi5 Memory Palace shutting down...")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main()
