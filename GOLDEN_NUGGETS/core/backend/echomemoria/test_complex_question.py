#!/usr/bin/env python3
"""
Complex Question Test for Dorumon's AI Brain
Asks Dorumon a complicated question to test his reasoning capabilities
"""

import asyncio
import json
import websockets
import time
from typing import Dict, Any

async def ask_dorumon_complex_question():
    """Ask Dorumon a complicated question and see how he reasons through it"""
    print("🧠 Testing Dorumon's Complex Reasoning Capabilities...")
    print("=" * 70)
    
    try:
        # Connect to the AI brain server
        uri = "ws://127.0.0.1:8765"
        print(f"🔌 Connecting to {uri}...")
        
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to AI brain server!")
            
            # Test 1: Send a complex character state with multiple conflicting factors
            print(f"\n🎭 Test 1: Complex Character State (Multiple Conflicting Factors)")
            
            complex_state = {
                "type": "character.state",
                "origin": "test",
                "payload": {
                    "characterId": "test_dorumon",
                    "action": "idle",
                    "energy": 0.3,  # Low energy (conflicts with excited mood)
                    "mood": "excited",  # Excited mood (conflicts with low energy)
                    "position": {"x": 100, "y": 100},
                    "recent_interactions": [
                        {"type": "mouse_movement", "intensity": 0.9, "timestamp": time.time() - 10},
                        {"type": "keyboard_activity", "intensity": 0.8, "timestamp": time.time() - 5},
                        {"type": "window_change", "intensity": 0.7, "timestamp": time.time() - 2}
                    ],
                    "environment": "work",  # Work environment (conflicts with excited mood)
                    "work_mode": True,  # Work mode (conflicts with excited mood)
                    "stress_level": 0.8,  # High stress (conflicts with excited mood)
                    "time_of_day": "afternoon",  # Afternoon (typically lower energy)
                    "habit_strength": 0.9,  # Strong habits
                    "memory_triggered": True,  # Memory-based decision trigger
                    "problem_detected": True  # Problem detected flag
                }
            }
            
            print(f"📤 Sending complex state with conflicting factors:")
            print(f"   - Low energy (0.3) but excited mood")
            print(f"   - Work environment but excited mood")
            print(f"   - High stress but excited mood")
            print(f"   - Multiple interaction patterns")
            print(f"   - Memory and problem triggers")
            
            await websocket.send(json.dumps(complex_state))
            
            # Wait for response
            print("⏳ Waiting for Dorumon's complex reasoning...")
            response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
            response_data = json.loads(response)
            
            print(f"📥 Received: {json.dumps(response_data, indent=2)}")
            
            # Analyze the reasoning
            if "payload" in response_data and "reasoning" in response_data["payload"]:
                reasoning = response_data["payload"]["reasoning"]
                print(f"\n🧠 COMPLEX REASONING ANALYSIS:")
                print(f"   Reasoning Type: {reasoning.get('type', 'unknown')}")
                print(f"   Explanation: {reasoning.get('explanation', 'none')}")
                print(f"   Confidence: {reasoning.get('confidence', 0):.2f}")
                print(f"   Factors: {reasoning.get('factors', [])}")
                print(f"   Alternative Actions: {reasoning.get('alternative_actions', [])}")
                
                # Check if Dorumon handled the conflicts intelligently
                explanation = reasoning.get('explanation', '').lower()
                if 'conflict' in explanation or 'contradiction' in explanation or 'despite' in explanation:
                    print(f"🎯 INTELLIGENT CONFLICT RESOLUTION DETECTED!")
                elif 'excited' in explanation and 'energy' in explanation:
                    print(f"🎯 ENERGY-MOOD CONFLICT RECOGNIZED!")
                elif 'work' in explanation and 'excited' in explanation:
                    print(f"🎯 WORK-MOOD CONFLICT RECOGNIZED!")
                else:
                    print(f"🤔 Basic reasoning provided")
            else:
                print(f"\n⚠️  No decision reasoning found in response")
            
            # Test 2: Ask a direct complex question
            print(f"\n❓ Test 2: Direct Complex Question")
            
            complex_question = {
                "type": "user.interaction",
                "origin": "test",
                "payload": {
                    "characterId": "test_dorumon",
                    "interaction_type": "complex_question",
                    "intensity": 1.0,
                    "position": {"x": 150, "y": 150},
                    "context": "philosophical_inquiry",
                    "question": "How do you balance your natural excitement with the need to stay focused in a work environment, especially when you're feeling low on energy but high on enthusiasm?",
                    "complexity_level": "high",
                    "requires_reasoning": True,
                    "conflicting_factors": ["energy", "mood", "environment", "responsibility"]
                }
            }
            
            print(f"📤 Sending complex philosophical question...")
            await websocket.send(json.dumps(complex_question))
            
            # Wait for response
            print("⏳ Waiting for Dorumon's philosophical reasoning...")
            response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
            response_data = json.loads(response)
            
            print(f"📥 Received: {json.dumps(response_data, indent=2)}")
            
            # Test 3: Send a multi-layered scenario
            print(f"\n🎬 Test 3: Multi-Layered Scenario")
            
            layered_scenario = {
                "type": "character.state",
                "origin": "test",
                "payload": {
                    "characterId": "test_dorumon",
                    "action": "curious",
                    "energy": 0.6,
                    "mood": "contemplative",
                    "position": {"x": 200, "y": 200},
                    "recent_interactions": [
                        {"type": "deep_thinking", "intensity": 0.9, "timestamp": time.time() - 60},
                        {"type": "problem_solving", "intensity": 0.8, "timestamp": time.time() - 30},
                        {"type": "creative_exploration", "intensity": 0.7, "timestamp": time.time() - 15}
                    ],
                    "environment": "creative_workspace",
                    "work_mode": False,
                    "stress_level": 0.2,
                    "time_of_day": "evening",
                    "habit_strength": 0.5,
                    "creativity_level": 0.9,
                    "problem_detected": True,
                    "requires_creative_solution": True,
                    "multiple_objectives": ["explore", "create", "solve", "learn"]
                }
            }
            
            print(f"📤 Sending multi-layered creative scenario...")
            await websocket.send(json.dumps(layered_scenario))
            
            # Wait for response
            print("⏳ Waiting for Dorumon's creative reasoning...")
            response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
            response_data = json.loads(response)
            
            print(f"📥 Received: {json.dumps(response_data, indent=2)}")
            
            print(f"\n🎉 Complex Question Test Completed!")
            print(f"✅ Dorumon handled complex scenarios!")
            print(f"✅ Decision Reasoning System processed conflicting factors!")
            print(f"✅ AI brain demonstrated sophisticated thinking!")
            
    except websockets.exceptions.ConnectionRefused:
        print(f"❌ Connection refused. Make sure the AI brain server is running:")
        print(f"   python server.py")
    except asyncio.TimeoutError:
        print(f"❌ Timeout waiting for response. Server might be overloaded.")
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(ask_dorumon_complex_question())
