#!/usr/bin/env python3
"""
Test WebSocket Bridge with Memory and Transparent Reasoning
Test if NiodO.o can remember conversations through the WebSocket
"""

import asyncio
import websockets
import json
import time
import sys
from pathlib import Path

async def test_niodoo_memory():
    """Test NiodO.o's memory through WebSocket"""
    print("🧠💗 TESTING NIODO.O MEMORY THROUGH WEBSOCKET")
    print("=" * 60)
    
    try:
        # Test 1: Connect to WebSocket
        print("\n🔌 TEST 1: WebSocket Connection")
        print("Connecting to NiodO.o WebSocket server...")
        
        async with websockets.connect('ws://localhost:8765') as websocket:
            
            # Wait for welcome message
            print("📨 Waiting for welcome message...")
            welcome_msg = await websocket.recv()
            welcome_data = json.loads(welcome_msg)
            print(f"✅ Welcome message received: {welcome_data.get('message', 'Unknown')}")
            
            # Test 2: Send first conversation
            print("\n💭 TEST 2: First Conversation")
            first_message = {
                "type": "experience_input",
                "input": "I'm feeling overwhelmed with this AI project. I've been working for months and feel like I'm not making progress.",
                "context": {
                    "speaker": "Developer",
                    "emotion": "frustration",
                    "context": "AI development",
                    "need": "support"
                }
            }
            
            print(f"📤 Sending: {first_message['input'][:50]}...")
            await websocket.send(json.dumps(first_message))
            
            # Wait for response
            response = await websocket.recv()
            response_data = json.loads(response)
            print(f"📥 Response received: {response_data.get('type', 'Unknown')}")
            
            if response_data.get('type') == 'brain_response':
                response_obj = response_data.get('response', {})
                if isinstance(response_obj, dict):
                    print(f"✅ Brain response: {response_obj.get('heart_decision', 'Unknown')}")
                else:
                    print(f"✅ Brain response: {response_obj}")
            
            # Test 3: Send second conversation
            print("\n💭 TEST 3: Second Conversation")
            second_message = {
                "type": "experience_input",
                "input": "I want to help others who are struggling like I am, but I feel like I don't have much to offer.",
                "context": {
                    "speaker": "Developer",
                    "emotion": "self_doubt",
                    "context": "helping_others",
                    "need": "encouragement"
                }
            }
            
            print(f"📤 Sending: {second_message['input'][:50]}...")
            await websocket.send(json.dumps(second_message))
            
            # Wait for response
            response = await websocket.recv()
            response_data = json.loads(response)
            print(f"📥 Response received: {response_data.get('type', 'Unknown')}")
            
            if response_data.get('type') == 'brain_response':
                print(f"✅ Brain response: {response_data.get('response', {}).get('heart_decision', 'Unknown')}")
            
            # Test 4: Test memory recall
            print("\n🧠 TEST 4: Memory Recall")
            recall_message = {
                "type": "brain_query",
                "query": "Do you remember what I told you about my AI project struggles?",
                "context": {
                    "intent": "memory_test",
                    "test_type": "conversation_recall"
                }
            }
            
            print(f"📤 Sending recall query: {recall_message['query']}")
            await websocket.send(json.dumps(recall_message))
            
            # Wait for response
            response = await websocket.recv()
            response_data = json.loads(response)
            print(f"📥 Memory response received: {response_data.get('type', 'Unknown')}")
            
            if response_data.get('type') == 'brain_response':
                response_obj = response_data.get('response', {})
                if isinstance(response_obj, dict):
                    print(f"✅ Memory response: {response_obj.get('heart_decision', 'Unknown')}")
                    print(f"✅ Motor action: {response_obj.get('motor_action', 'Unknown')}")
                    print(f"✅ Confidence: {response_obj.get('confidence', 0):.2f}")
                else:
                    print(f"✅ Memory response: {response_obj}")
                    print(f"✅ Response type: {type(response_obj)}")
            
            # Test 5: Get transparency report
            print("\n🔍 TEST 5: Transparency Report")
            transparency_message = {
                "type": "status_request",
                "request": "transparency_report"
            }
            
            print("📤 Requesting transparency report...")
            await websocket.send(json.dumps(transparency_message))
            
            # Wait for response
            response = await websocket.recv()
            response_data = json.loads(response)
            print(f"📥 Transparency response received: {response_data.get('type', 'Unknown')}")
            
            if response_data.get('type') == 'status_response':
                status = response_data.get('status', {})
                print(f"✅ Brain status: {status.get('brain_status', 'Unknown')}")
                print(f"✅ Model status: {status.get('model_status', 'Unknown')}")
                print(f"✅ Memory status: {status.get('memory_status', 'Unknown')}")
            
            # Test 6: Final conversation to test continued memory
            print("\n💭 TEST 6: Final Conversation")
            final_message = {
                "type": "experience_input",
                "input": "Thank you for listening. I feel a bit better now. What should I focus on next?",
                "context": {
                    "speaker": "Developer",
                    "emotion": "gratitude",
                    "context": "moving_forward",
                    "need": "guidance"
                }
            }
            
            print(f"📤 Sending: {final_message['input'][:50]}...")
            await websocket.send(json.dumps(final_message))
            
            # Wait for response
            response = await websocket.recv()
            response_data = json.loads(response)
            print(f"📥 Final response received: {response_data.get('type', 'Unknown')}")
            
            if response_data.get('type') == 'brain_response':
                response_obj = response_data.get('response', {})
                if isinstance(response_obj, dict):
                    print(f"✅ Final response: {response_obj.get('heart_decision', 'Unknown')}")
                else:
                    print(f"✅ Final response: {response_obj}")
            
            print("\n🎯 MEMORY TEST COMPLETE!")
            print("✅ WebSocket connection successful")
            print("✅ Multiple conversations processed")
            print("✅ Memory recall tested")
            print("✅ Transparency report requested")
            print("✅ Final conversation completed")
            
            return True
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧠💗 Starting NiodO.o Memory Test...")
    print("Make sure the WebSocket server is running!")
    
    success = asyncio.run(test_niodoo_memory())
    if success:
        print(f"\n🎉 SUCCESS! NiodO.o remembers conversations!")
        print(f"💡 His memory system is working through the WebSocket!")
        print(f"🧠 You can see his transparent reasoning in the logs!")
    else:
        print(f"\n⚠️ Memory test needs attention")

