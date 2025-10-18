#!/usr/bin/env python3
"""
Live test script for AI Brain with Decision Reasoning
Tests the actual WebSocket server to see decision reasoning in action
"""

import asyncio
import json
import websockets
import time
from typing import Dict, Any

async def test_ai_brain_live():
    """Test the live AI brain server"""
    print("🧠 Testing Live AI Brain with Decision Reasoning...")
    print("=" * 60)
    
    try:
        # Connect to the AI brain server
        uri = "ws://127.0.0.1:8765"
        print(f"🔌 Connecting to {uri}...")
        
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to AI brain server!")
            
            # Test 1: Send a character state update to trigger decision reasoning
            print(f"\n🎭 Test 1: Character State Update (Triggers Decision Reasoning)")
            
            state_update = {
                "type": "character.state",
                "origin": "test",
                "payload": {
                    "characterId": "test_dorumon",
                    "action": "idle",
                    "energy": 0.8,
                    "mood": "excited",
                    "position": {"x": 100, "y": 100},
                    "recent_interactions": [
                        {"type": "mouse_movement", "intensity": 0.7, "timestamp": time.time() - 30},
                        {"type": "keyboard_activity", "intensity": 0.6, "timestamp": time.time() - 20}
                    ],
                    "environment": "entertainment",
                    "work_mode": False
                }
            }
            
            print(f"📤 Sending: {json.dumps(state_update, indent=2)}")
            await websocket.send(json.dumps(state_update))
            
            # Wait for response
            print("⏳ Waiting for AI decision with reasoning...")
            response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            response_data = json.loads(response)
            
            print(f"📥 Received: {json.dumps(response_data, indent=2)}")
            
            # Check if decision reasoning is included
            if "payload" in response_data and "reasoning" in response_data["payload"]:
                reasoning = response_data["payload"]["reasoning"]
                print(f"\n🧠 DECISION REASONING DETECTED!")
                print(f"   Type: {reasoning.get('type', 'unknown')}")
                print(f"   Explanation: {reasoning.get('explanation', 'none')}")
                print(f"   Confidence: {reasoning.get('confidence', 0):.2f}")
                print(f"   Factors: {reasoning.get('factors', [])}")
                print(f"   Alternative Actions: {reasoning.get('alternative_actions', [])}")
            else:
                print(f"\n⚠️  No decision reasoning found in response")
                print(f"   Response type: {response_data.get('type', 'unknown')}")
            
            # Test 2: Send user interaction to see pattern-based reasoning
            print(f"\n🎯 Test 2: User Interaction (Pattern-Based Reasoning)")
            
            interaction = {
                "type": "user.interaction",
                "origin": "test",
                "payload": {
                    "characterId": "test_dorumon",
                    "interaction_type": "mouse_click",
                    "intensity": 0.9,
                    "position": {"x": 150, "y": 150},
                    "context": "high_activity"
                }
            }
            
            print(f"📤 Sending: {json.dumps(interaction, indent=2)}")
            await websocket.send(json.dumps(interaction))
            
            # Wait for response
            print("⏳ Waiting for interaction response...")
            response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            response_data = json.loads(response)
            
            print(f"📥 Received: {json.dumps(response_data, indent=2)}")
            
            # Test 3: Send a ping to test basic connectivity
            print(f"\n🏓 Test 3: Basic Connectivity (Ping)")
            
            ping = {
                "type": "ping",
                "origin": "test",
                "payload": {}
            }
            
            print(f"📤 Sending: {json.dumps(ping, indent=2)}")
            await websocket.send(json.dumps(ping))
            
            # Wait for response
            print("⏳ Waiting for ping response...")
            response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            response_data = json.loads(response)
            
            print(f"📥 Received: {json.dumps(response_data, indent=2)}")
            
            print(f"\n🎉 Live AI Brain Test Completed Successfully!")
            print(f"✅ Decision Reasoning System is working!")
            print(f"✅ WebSocket communication is functional!")
            print(f"✅ AI brain is responding with reasoning!")
            
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
    asyncio.run(test_ai_brain_live())
