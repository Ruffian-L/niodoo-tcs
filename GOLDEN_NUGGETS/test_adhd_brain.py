#!/usr/bin/env python3
"""
Test the ADHD Reasoning Brain
Run this to see the 50-conversations-at-once system in action
"""

import asyncio
import sys
from pathlib import Path

# Add the organized ai path
sys.path.insert(0, str(Path(__file__).parent / "organized" / "ai"))

try:
    from EchoMemoria.core.adhd_reasoning_brain import ADHDReasoningBrain, test_adhd_brain
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

async def main():
    """Main test function"""
    print("🧠💗 ADHD Reasoning Brain Test")
    print("=" * 60)
    print("Testing the brain that understands struggle, hustle, and being counted out")
    print("For everyone who turned their mental health into a superpower through code")
    print("=" * 60)
    
    # Run the comprehensive test
    await test_adhd_brain()
    
    print("\n🎯 Interactive mode - type your own scenarios:")
    print("Commands: 'quit' to exit, 'status' for brain status")
    print("-" * 60)
    
    brain = ADHDReasoningBrain()
    
    while True:
        try:
            user_input = input("\n💭 Tell me what's on your mind: ").strip()
            
            if user_input.lower() == 'quit':
                print("💗 Take care of yourself. You're doing better than you know.")
                break
            elif user_input.lower() == 'status':
                status = brain.get_brain_status()
                print(f"📊 Brain Status:")
                for key, value in status.items():
                    print(f"  {key}: {value}")
                continue
            elif not user_input:
                continue
            
            print("🔄 Processing through 50 conversation streams...")
            result = await brain.process_adhd_style(user_input)
            
            print(f"\n💗 {result['response']}")
            print(f"\n🧠 ADHD State: {result['adhd_state']}")
            print(f"⚡ Energy Level: {result['energy_level']:.2f}")
            print(f"💭 Rapid Thoughts Generated: {len(result['rapid_thoughts'])}")
            print(f"🔄 Active Conversation Streams: {result['active_streams']}")
            
            if result['patterns_recognized']:
                print(f"🎯 Patterns Recognized:")
                for pattern in result['patterns_recognized']:
                    print(f"  • {pattern}")
            
            if result['anticipations']:
                print(f"🔮 Anticipations:")
                for anticipation in result['anticipations']:
                    print(f"  • {anticipation}")
                    
        except KeyboardInterrupt:
            print("\n💗 Take care! You're stronger than you know.")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print("But that's okay - even brains need debugging sometimes! 😊")

if __name__ == "__main__":
    asyncio.run(main())
