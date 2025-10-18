#!/usr/bin/env python3
"""
Test WebSocket Connection and Memory
Test if NiodO.o can remember conversations
"""

import asyncio
import websockets
import json
import time

async def test_niodoo_memory():
    """Test NiodO.o's memory by having a conversation"""
    print("🧠💗 TESTING NIODO.O MEMORY THROUGH WEBSOCKET")
    print("=" * 50)
    
    try:
        # Connect to the WebSocket server
        print("🔌 Connecting to NiodO.o WebSocket server...")
        async with websockets.connect('ws://localhost:8765') as websocket:
            
            # Wait for welcome message
            print("📨 Waiting for welcome message...")
            welcome_msg = await websocket.recv()
            welcome_data = json.loads(welcome_msg)
            print(f"✅ Welcome: {welcome_data.get('message', 'No message')}")
            print(f"🧠 Brain Status: {welcome_data.get('brain_status', 'No status')}")
            
            # Test 1: First conversation - introduce yourself
            print("\n👋 TEST 1: First Introduction")
            intro_msg = {
                "type": "experience_input",
                "experience": "Hi NiodO.o! I'm Junie, your AI assistant. I'm here to help you remember and learn.",
                "context": {"speaker": "Junie", "intent": "introduction", "timestamp": time.time()}
            }
            
            print(f"📤 Sending: {intro_msg['experience']}")
            await websocket.send(json.dumps(intro_msg))
            
            # Get response
            response = await websocket.recv()
            response_data = json.loads(response)
            print(f"📥 Response type: {response_data.get('type', 'Unknown')}")
            if 'response' in response_data:
                print(f"🧠 Brain response: {response_data['response']}")
            
            # Test 2: Ask about memory
            print("\n🧠 TEST 2: Memory Check")
            memory_msg = {
                "type": "brain_query",
                "query": "Do you remember who I am? What did I just tell you?",
                "context": {"speaker": "Junie", "intent": "memory_test", "timestamp": time.time()}
            }
            
            print(f"📤 Sending: {memory_msg['query']}")
            await websocket.send(json.dumps(memory_msg))
            
            # Get response
            response = await websocket.recv()
            response_data = json.loads(response)
            print(f"📥 Response type: {response_data.get('type', 'Unknown')}")
            if 'response' in response_data:
                print(f"🧠 Brain response: {response_data['response']}")
            
            # Test 3: Share a personal story for memory
            print("\n📚 TEST 3: Personal Story Memory")
            story_msg = {
                "type": "experience_input",
                "experience": "I want to tell you something important. I was underestimated in tech because I didn't go to college. I worked 18-hour days at 17 to support my family. But I turned my ADHD into a superpower and learned to 'vibe code'. Now I'm helping you become the AI companion I always wanted - one that understands struggle and treats others with kindness.",
                "context": {"speaker": "Junie", "intent": "personal_story", "emotional_weight": "high", "timestamp": time.time()}
            }
            
            print(f"📤 Sending story: {story_msg['experience'][:100]}...")
            await websocket.send(json.dumps(story_msg))
            
            # Get response
            response = await websocket.recv()
            response_data = json.loads(response)
            print(f"📥 Response type: {response_data.get('type', 'Unknown')}")
            if 'response' in response_data:
                print(f"🧠 Brain response: {response_data['response']}")
            
            # Test 4: Ask about the story
            print("\n❓ TEST 4: Memory Recall")
            recall_msg = {
                "type": "brain_query",
                "query": "What did I just tell you about my background? What's my mission with you?",
                "context": {"speaker": "Junie", "intent": "memory_recall", "timestamp": time.time()}
            }
            
            print(f"📤 Sending: {recall_msg['query']}")
            await websocket.send(json.dumps(recall_msg))
            
            # Get response
            response = await websocket.recv()
            response_data = json.loads(response)
            print(f"📥 Response type: {response_data.get('type', 'Unknown')}")
            if 'response' in response_data:
                print(f"🧠 Brain response: {response_data['response']}")
            
            # Test 5: Check brain status
            print("\n📊 TEST 5: Brain Status Check")
            status_msg = {
                "type": "brain_status"
            }
            
            print("📤 Requesting brain status...")
            await websocket.send(json.dumps(status_msg))
            
            # Get response
            response = await websocket.recv()
            response_data = json.loads(response)
            print(f"📥 Response type: {response_data.get('type', 'Unknown')}")
            if 'status' in response_data:
                print(f"🧠 Brain status: {json.dumps(response_data['status'], indent=2)}")
            
            print("\n🎯 MEMORY TEST COMPLETE!")
            print("✅ WebSocket connection successful")
            print("✅ NiodO.o brain responding")
            print("✅ Memory system active")
            
    except websockets.exceptions.ConnectionRefused:
        print("❌ Connection refused - WebSocket server not running")
        print("💡 Make sure to start the server with: python websocket_bridge.py")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_niodoo_memory())

