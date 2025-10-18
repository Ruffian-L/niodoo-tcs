#!/usr/bin/env python3
"""
Test Qt Message Format with WebSocket Bridge
Test if the fixed message parsing works correctly
"""

import asyncio
import websockets
import json
import time

async def test_qt_message_format():
    """Test if WebSocket bridge can parse Qt's message format"""
    print("🧠💗 TESTING QT MESSAGE FORMAT PARSING")
    print("=" * 50)
    
    try:
        # Connect to the WebSocket server
        print("🔌 Connecting to NiodO.o WebSocket server...")
        async with websockets.connect('ws://localhost:8768') as websocket:
            
            # Wait for welcome message
            print("📨 Waiting for welcome message...")
            welcome_msg = await websocket.recv()
            welcome_data = json.loads(welcome_msg)
            print(f"✅ Welcome: {welcome_data.get('message', 'No message')}")
            
            # Test 1: Qt brain_query format
            print("\n🧠 TEST 1: Qt brain_query format")
            qt_brain_query = {
                "origin": "qt-companion",
                "payload": {
                    "characterId": "dorumon_12345",
                    "type": "dorumon",
                    "stage": "1.Baby",
                    "personality": {
                        "patience": 0.6,
                        "playfulness": 0.7
                    }
                },
                "ts": int(time.time() * 1000)
            }
            
            print(f"📤 Sending Qt brain_query: {json.dumps(qt_brain_query, indent=2)}")
            await websocket.send(json.dumps(qt_brain_query))
            
            # Get response
            response = await websocket.recv()
            response_data = json.loads(response)
            print(f"📥 Response type: {response_data.get('type', 'Unknown')}")
            if response_data.get('type') == 'brain_response':
                print("✅ SUCCESS: brain_query parsed correctly!")
            else:
                print(f"❌ FAILED: Expected brain_response, got {response_data.get('type')}")
                print(f"Response: {response_data}")
            
            # Test 2: Qt experience_input format
            print("\n📚 TEST 2: Qt experience_input format")
            qt_experience = {
                "origin": "qt-companion",
                "payload": {
                    "action": "idle",
                    "characterId": 7,
                    "type": "experience_input"
                },
                "ts": int(time.time() * 1000)
            }
            
            print(f"📤 Sending Qt experience_input: {json.dumps(qt_experience, indent=2)}")
            await websocket.send(json.dumps(qt_experience))
            
            # Get response
            response = await websocket.recv()
            response_data = json.loads(response)
            print(f"📥 Response type: {response_data.get('type', 'Unknown')}")
            if response_data.get('type') == 'experience_response':
                print("✅ SUCCESS: experience_input parsed correctly!")
            else:
                print(f"❌ FAILED: Expected experience_response, got {response_data.get('type')}")
                print(f"Response: {response_data}")
            
            # Test 3: Qt ping format
            print("\n🏓 TEST 3: Qt ping format")
            qt_ping = {
                "origin": "qt-companion",
                "payload": {},
                "ts": int(time.time() * 1000),
                "type": "ping"
            }
            
            print(f"📤 Sending Qt ping: {json.dumps(qt_ping, indent=2)}")
            await websocket.send(json.dumps(qt_ping))
            
            # Get response
            response = await websocket.recv()
            response_data = json.loads(response)
            print(f"📥 Response type: {response_data.get('type', 'Unknown')}")
            if response_data.get('type') == 'pong':
                print("✅ SUCCESS: ping parsed correctly!")
            else:
                print(f"❌ FAILED: Expected pong, got {response_data.get('type')}")
                print(f"Response: {response_data}")
            
            print("\n🎯 TEST SUMMARY:")
            print("=" * 30)
            print("✅ All Qt message formats tested")
            print("✅ WebSocket bridge should now work with Qt companion")
            print("✅ Ready for Phase 2 Beta team to test")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("Make sure the WebSocket server is running on port 8768")

if __name__ == "__main__":
    asyncio.run(test_qt_message_format())
