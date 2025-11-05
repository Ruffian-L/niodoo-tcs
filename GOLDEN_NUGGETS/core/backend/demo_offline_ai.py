#!/usr/bin/env python3
"""
Offline AI Demonstration for EchoMemoria
Shows how the AI brain works with offline models
"""

import sys
import json
from pathlib import Path

# Add AI War Room scripts to path
scripts_path = Path(r"C:\AI_WarRoom\scripts")
if scripts_path.exists():
    sys.path.append(str(scripts_path))

try:
    from ai_commander import AICommander
    print("✅ AI Commander imported successfully!")
except ImportError as e:
    print(f"❌ Failed to import AI Commander: {e}")
    sys.exit(1)

def demo_offline_ai():
    """Demonstrate offline AI capabilities"""
    print("🚀 OFFLINE AI DEMONSTRATION FOR ECHOMEMORIA")
    print("=" * 50)
    
    # Initialize AI Commander
    commander = AICommander()
    
    # Show system status
    status = commander.get_system_status()
    print(f"📊 System Status:")
    print(f"   llama_cpp: {'✅' if status['llama_cpp_available'] else '❌'}")
    print(f"   Available models: {status['available_models']}")
    print(f"   Current model: {status['current_model'] or 'None'}")
    print(f"   Cache enabled: {'✅' if status['cache_enabled'] else '❌'}")
    
    # Test with a simple prompt
    print(f"\n🧪 Testing AI Decision Making...")
    
    # Simulate character context (like EchoMemoria would send)
    context = {
        "character_state": {
            "action": "idle",
            "energy": 0.8,
            "x": 500,
            "y": 500
        },
        "user_interaction": {
            "type": "drag_start",
            "position": {"x": 500, "y": 500}
        }
    }
    
    # Create a test prompt
    test_prompt = """You are Dorumon, a digital pet character. The user just started dragging you. What should you do?

Available Actions: idle, walk, sit, sleepy, hungry, foodTime, bouncing, jumping, creep, sprawl

Choose the most appropriate action and explain why. Format your response as:
ACTION: [action_name]
REASON: [brief explanation]
EMOTION: [emotion state]"""
    
    print(f"📝 Test Prompt: {test_prompt[:100]}...")
    
    if status['available_models']:
        # Test with available model
        model_name = status['available_models'][0]
        print(f"\n🤖 Testing with {model_name} model...")
        
        try:
            response = commander.execute_direct(test_prompt, model_name)
            print(f"✅ AI Response: {response}")
        except Exception as e:
            print(f"❌ Model test failed: {e}")
    else:
        print(f"\n💡 No models available yet. To get models:")
        print(f"   1. Install huggingface-cli: pip install huggingface_hub")
        print(f"   2. Download models:")
        print(f"      huggingface-cli download TheBloke/phi-3-mini-4k-instruct --local-dir C:\\AI_WarRoom\\models --include *.gguf")
        print(f"      huggingface-cli download TheBloke/mistral-7b-openorca --local-dir C:\\AI_WarRoom\\models --include *.gguf")
    
    print(f"\n🎯 Offline AI System Ready for EchoMemoria Integration!")
    print(f"   The AI brain can now make intelligent decisions without internet!")
    print(f"   Models will be loaded automatically when available.")

if __name__ == "__main__":
    demo_offline_ai()
