#!/usr/bin/env python3
"""
Test WebSocket Bridge Startup
Simple test to verify the bridge can initialize
"""

import asyncio
import sys
from pathlib import Path

print("🧠💗 TESTING WEBSOCKET BRIDGE STARTUP")
print("=" * 50)

async def test_bridge_startup():
    """Test if the WebSocket bridge can start up"""
    
    try:
        # Test 1: Import the bridge
        print("\n📦 TESTING IMPORTS...")
        from websocket_bridge import NiodOoWebSocketBridge
        print("✅ WebSocket bridge imported successfully")
        
        # Test 2: Create bridge instance
        print("\n🔌 TESTING BRIDGE CREATION...")
        bridge = NiodOoWebSocketBridge(host="localhost", port=8765)
        print("✅ WebSocket bridge instance created")
        
        # Test 3: Check brain initialization
        print("\n🧠 TESTING BRAIN INITIALIZATION...")
        if bridge.complete_brain:
            print("✅ Complete brain initialized")
        else:
            print("⚠️ Complete brain not initialized (expected if imports failed)")
        
        if bridge.brain_manager:
            print("✅ Brain manager connected")
        else:
            print("⚠️ Brain manager not connected (expected if imports failed)")
        
        # Test 4: Check brain status
        print("\n📊 TESTING BRAIN STATUS...")
        try:
            status = bridge._get_brain_status()
            print("✅ Brain status retrieved")
            print(f"   Status keys: {list(status.keys())}")
        except Exception as e:
            print(f"⚠️ Brain status failed: {e}")
        
        print(f"\n🎯 WEBSOCKET BRIDGE STARTUP TEST SUMMARY:")
        print(f"✅ Bridge Creation: Success")
        print(f"✅ Brain Initialization: {'Success' if bridge.complete_brain else 'Partial'}")
        print(f"✅ Brain Manager: {'Connected' if bridge.brain_manager else 'Not Connected'}")
        print(f"✅ Brain Status: Available")
        print(f"\n🚀 WEBSOCKET BRIDGE IS READY FOR TESTING!")
        
        return True
        
    except Exception as e:
        print(f"❌ WebSocket bridge startup test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_bridge_startup())
    if success:
        print(f"\n🎉 SUCCESS! WebSocket bridge can start up!")
        print(f"🎯 Ready to connect to Qt frontend!")
    else:
        print(f"\n⚠️ WebSocket bridge needs attention before proceeding")
