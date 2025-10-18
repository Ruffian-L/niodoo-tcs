#!/usr/bin/env python3
"""
Test Simple AI Model
"""

import sys
from pathlib import Path

# Add models directory to path
models_path = Path(r"C:\AI_WarRoom\models")
if models_path.exists():
    sys.path.append(str(models_path))

try:
    from simple_ai_model import get_simple_ai_decision
    print("✅ Simple AI model imported successfully!")
    
    # Test context
    test_context = {
        "character_state": {
            "action": "idle",
            "energy": 0.8,
            "x": 500,
            "y": 500
        },
        "user_interaction": {
            "type": "drag_start"
        }
    }
    
    print("🧪 Testing simple AI decision...")
    decision = get_simple_ai_decision(test_context)
    
    print("🤖 AI Decision:")
    print(f"   Action: {decision['action']}")
    print(f"   Reason: {decision['reason']}")
    print(f"   Emotion: {decision['emotion']}")
    print(f"   Source: {decision['source']}")
    print(f"   Confidence: {decision['confidence']}")
    
    print("\n🎯 Simple AI model is working!")
    
except ImportError as e:
    print(f"❌ Failed to import simple AI model: {e}")
except Exception as e:
    print(f"❌ Error testing simple AI: {e}")
