#!/usr/bin/env python3
"""
Simple Performance Test for NiodO.o Brain
Tests the optimized brain model performance
"""

import asyncio
import time
import sys
from pathlib import Path

# Add paths for imports
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir / "core"))

async def test_performance():
    """Test the optimized brain performance"""
    print("🚀 NIODO.O PERFORMANCE TEST")
    print("=" * 40)
    
    try:
        from brain_model_connections import BrainModelManager
        
        # Initialize manager
        manager = BrainModelManager()
        print("✅ Brain manager initialized")
        
        # Load models
        print("🔄 Loading models...")
        success = manager.load_brain_models()
        
        if success:
            print("✅ All 3 models loaded!")
            
            # Test single response time
            test_input = "Hello, how are you today?"
            print(f"\n🧪 Testing: {test_input}")
            
            start_time = time.time()
            result = await manager.process_through_all_brains(test_input)
            end_time = time.time()
            
            processing_time = end_time - start_time
            print(f"⚡ Processing time: {processing_time:.2f}s")
            print(f"🎯 Efficiency: {result.get('efficiency_guidance', 'N/A')[:100]}...")
            print(f"🧠 Motor: {result.get('motor_response', 'N/A')[:100]}...")
            print(f"✍️ LCARS: {result.get('lcars_response', 'N/A')[:100]}...")
            
            # Performance rating
            if processing_time < 5:
                rating = "🚀 EXCELLENT"
            elif processing_time < 10:
                rating = "✅ GOOD"
            elif processing_time < 20:
                rating = "⚠️ SLOW"
            else:
                rating = "🐌 TOO SLOW"
            
            print(f"\n📊 Performance Rating: {rating}")
            print(f"💡 Target: <5s, Current: {processing_time:.2f}s")
            
            # Clean up
            manager.unload_models()
            print("💾 Models unloaded")
            
        else:
            print("❌ Failed to load models")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_performance())
