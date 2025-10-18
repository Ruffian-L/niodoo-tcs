#!/usr/bin/env python3
"""
Test script for Decision Reasoning System Integration
Demonstrates how Dorumon explains his decision-making process
"""

import sys
import os
import time
import json
from pathlib import Path

# Add the core directory to the path
sys.path.insert(0, str(Path(__file__).parent / "core"))

def test_decision_reasoning_integration():
    """Test the complete decision reasoning integration"""
    print("🧠 Testing Decision Reasoning System Integration...")
    
    try:
        # Import the decision reasoning engine
        from decision_reasoning import DecisionReasoningEngine
        
        # Initialize the engine
        reasoning_engine = DecisionReasoningEngine()
        print("✅ Decision reasoning engine initialized")
        
        # Test various decision scenarios
        test_scenarios = [
            {
                "name": "Morning Energy Bounce",
                "action": "bouncing",
                "emotion": "excited",
                "character_state": {
                    "energy": 0.9,
                    "mood": "excited",
                    "stress": 0.1
                },
                "context": {
                    "work_mode": False,
                    "entertainment_mode": True,
                    "recent_interactions": [
                        {"type": "mouse_movement", "intensity": 0.8, "timestamp": time.time() - 30},
                        {"type": "keyboard_activity", "intensity": 0.7, "timestamp": time.time() - 20}
                    ],
                    "environment": "home"
                },
                "memory_data": {
                    "relevant_memories": [
                        {
                            "id": "mem_001",
                            "type": "interaction",
                            "timestamp": time.time() - 3600,
                            "relevance": 0.9,
                            "summary": "user was very happy when I bounced in the morning",
                            "emotional_impact": 0.8
                        }
                    ]
                },
                "interaction_data": {
                    "recent_interactions": [
                        {"type": "mouse_movement", "intensity": 0.8, "timestamp": time.time() - 30},
                        {"type": "keyboard_activity", "intensity": 0.7, "timestamp": time.time() - 20},
                        {"type": "window_change", "intensity": 0.6, "timestamp": time.time() - 10}
                    ]
                }
            },
            {
                "name": "Work Mode Focus",
                "action": "idle",
                "emotion": "focused",
                "character_state": {
                    "energy": 0.6,
                    "mood": "focused",
                    "stress": 0.3
                },
                "context": {
                    "work_mode": True,
                    "entertainment_mode": False,
                    "recent_interactions": [
                        {"type": "keyboard_activity", "intensity": 0.9, "timestamp": time.time() - 60},
                        {"type": "keyboard_activity", "intensity": 0.8, "timestamp": time.time() - 30}
                    ],
                    "environment": "office"
                },
                "memory_data": {
                    "relevant_memories": [
                        {
                            "id": "mem_002",
                            "type": "observation",
                            "timestamp": time.time() - 7200,
                            "relevance": 0.7,
                            "summary": "user works better when I stay quiet and focused",
                            "emotional_impact": 0.2
                        }
                    ]
                },
                "interaction_data": {
                    "recent_interactions": [
                        {"type": "keyboard_activity", "intensity": 0.9, "timestamp": time.time() - 60},
                        {"type": "keyboard_activity", "intensity": 0.8, "timestamp": time.time() - 30},
                        {"type": "keyboard_activity", "intensity": 0.7, "timestamp": time.time() - 15}
                    ]
                }
            },
            {
                "name": "Biological Need - Sleep",
                "action": "sleepy",
                "emotion": "tired",
                "character_state": {
                    "energy": 0.2,
                    "mood": "tired",
                    "stress": 0.1
                },
                "context": {
                    "work_mode": False,
                    "entertainment_mode": False,
                    "recent_interactions": [],
                    "environment": "home"
                },
                "memory_data": {
                    "relevant_memories": []
                },
                "interaction_data": {
                    "recent_interactions": []
                }
            },
            {
                "name": "Creative Inspiration",
                "action": "creep",
                "emotion": "curious",
                "character_state": {
                    "energy": 0.8,
                    "mood": "curious",
                    "stress": 0.1
                },
                "context": {
                    "work_mode": False,
                    "entertainment_mode": True,
                    "recent_interactions": [
                        {"type": "mouse_movement", "intensity": 0.6, "timestamp": time.time() - 45}
                    ],
                    "environment": "home"
                },
                "memory_data": {
                    "relevant_memories": [
                        {
                            "id": "mem_003",
                            "type": "exploration",
                            "timestamp": time.time() - 1800,
                            "relevance": 0.6,
                            "summary": "I discovered something interesting while creeping around",
                            "emotional_impact": 0.4
                        }
                    ]
                },
                "interaction_data": {
                    "recent_interactions": [
                        {"type": "mouse_movement", "intensity": 0.6, "timestamp": time.time() - 45}
                    ]
                }
            }
        ]
        
        print(f"\n📋 Testing {len(test_scenarios)} decision scenarios...")
        
        for i, scenario in enumerate(test_scenarios, 1):
            print(f"\n--- Scenario {i}: {scenario['name']} ---")
            
            # Generate decision reasoning
            reasoning = reasoning_engine.generate_decision_reasoning(
                action=scenario['action'],
                emotion=scenario['emotion'],
                character_state=scenario['character_state'],
                context=scenario['context'],
                memory_data=scenario['memory_data'],
                interaction_data=scenario['interaction_data']
            )
            
            # Display results
            print(f"Action: {reasoning.action_chosen}")
            print(f"Emotion: {reasoning.emotion_chosen}")
            print(f"Reasoning Type: {reasoning.reasoning_type.value}")
            print(f"Confidence: {reasoning.confidence_level:.2f}")
            print(f"Primary Factors: {[f.value for f in reasoning.primary_factors]}")
            print(f"Explanation: {reasoning.reasoning_explanation}")
            
            if reasoning.alternative_actions:
                print(f"Alternative Actions: {', '.join(reasoning.alternative_actions)}")
            
            if reasoning.memory_references:
                print(f"Memory References: {len(reasoning.memory_references)}")
                for ref in reasoning.memory_references[:2]:  # Show first 2
                    print(f"  - {ref.content_summary} (relevance: {ref.relevance_score:.2f})")
            
            if reasoning.interaction_patterns:
                print(f"Interaction Patterns: {len(reasoning.interaction_patterns)}")
                for pattern in reasoning.interaction_patterns:
                    print(f"  - {pattern.pattern_type} (strength: {pattern.pattern_strength:.2f})")
        
        # Test statistics
        print(f"\n📊 Decision Reasoning Statistics:")
        stats = reasoning_engine.get_reasoning_statistics()
        print(f"Total Decisions: {stats.get('total_decisions', 0)}")
        print(f"Average Confidence: {stats.get('average_confidence', 0):.2f}")
        
        if stats.get('reasoning_type_distribution'):
            print("Reasoning Type Distribution:")
            for reasoning_type, count in stats['reasoning_type_distribution'].items():
                print(f"  - {reasoning_type}: {count}")
        
        if stats.get('most_common_actions'):
            print("Most Common Actions:")
            for action, count in stats['most_common_actions'].items():
                print(f"  - {action}: {count}")
        
        # Test decision history
        print(f"\n📚 Recent Decision History:")
        history = reasoning_engine.get_decision_history(limit=5)
        for decision in history:
            print(f"  - {decision.action_chosen} ({decision.reasoning_type.value}) - {decision.reasoning_explanation[:60]}...")
        
        print("\n✅ Decision reasoning integration test completed successfully!")
        
        # Demonstrate how this would look in the event bus
        print(f"\n🎭 Event Bus Integration Example:")
        print("When Dorumon makes a decision, the AI brain now includes reasoning:")
        
        example_payload = {
            "type": "ai.decision",
            "origin": "brain",
            "payload": {
                "characterId": "dorumon_001",
                "action": "bouncing",
                "emotion": "excited",
                "message": "Boing boing! I'm so excited!",
                "reasoning": {
                    "type": "creative_inspiration",
                    "explanation": "I'm feeling inspired to bounce by my environment!",
                    "confidence": 0.85,
                    "factors": ["emotional_state", "energy_level", "creativity_level"],
                    "alternative_actions": ["jumping", "creep", "idle"]
                }
            }
        }
        
        print(json.dumps(example_payload, indent=2))
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure the decision_reasoning.py file is in the core directory")
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

def test_nervous_system_integration():
    """Test how decision reasoning integrates with the nervous system"""
    print(f"\n🧬 Testing Nervous System Integration...")
    
    try:
        # Import nervous system
        from nervous_system import NervousSystem, Stimulus, StimulusType
        
        # Import decision reasoning
        from decision_reasoning import DecisionReasoningEngine
        
        # Initialize both systems
        nervous_system = NervousSystem()
        reasoning_engine = DecisionReasoningEngine()
        
        print("✅ Nervous system and reasoning engine initialized")
        
        # Test stimulus processing with reasoning
        print(f"\n🔊 Testing stimulus processing with reasoning...")
        
        # Create a sound stimulus
        sound_stimulus = Stimulus(
            type=StimulusType.SOUND,
            intensity=0.8,
            context={'source': 'door_slam'}
        )
        
        # Process stimulus through nervous system
        nervous_response = nervous_system.process_stimulus(sound_stimulus)
        
        if nervous_response:
            print(f"Nervous response: {nervous_response.action} - {nervous_response.emotion}")
            print(f"Reasoning: {nervous_response.reasoning}")
            
            # Update mood
            nervous_system.update_mood(nervous_response)
            
            # Generate decision reasoning for this response
            reasoning = reasoning_engine.generate_decision_reasoning(
                action=nervous_response.action,
                emotion=nervous_response.emotion,
                character_state=nervous_system.get_current_state(),
                context={'stimulus_type': 'sound', 'intensity': nervous_response.intensity}
            )
            
            print(f"Decision reasoning: {reasoning.reasoning_type.value}")
            print(f"Explanation: {reasoning.reasoning_explanation}")
            print(f"Confidence: {reasoning.confidence_level:.2f}")
        
        print("✅ Nervous system integration test completed!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
    except Exception as e:
        print(f"❌ Error during nervous system testing: {e}")

if __name__ == "__main__":
    print("🧠 Dorumon Decision Reasoning System Integration Test")
    print("=" * 60)
    
    # Test basic decision reasoning
    test_decision_reasoning_integration()
    
    # Test nervous system integration
    test_nervous_system_integration()
    
    print(f"\n🎉 All tests completed! Dorumon now explains his decisions transparently!")
    print(f"\n💡 Key Benefits:")
    print(f"  - Users understand WHY Dorumon does what he does")
    print(f"  - Transparent AI decision-making process")
    print(f"  - References to memories, patterns, and context")
    print(f"  - Confidence levels for decision quality")
    print(f"  - Alternative actions that could have been chosen")
    print(f"  - Integration with nervous system and multimodal processing")
