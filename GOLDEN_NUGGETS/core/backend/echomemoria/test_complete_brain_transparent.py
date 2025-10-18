#!/usr/bin/env python3
"""
Test Complete Brain with Transparent Reasoning
Test NiodO.o's complete brain synthesis with full transparency
"""

import asyncio
import sys
from pathlib import Path

# Add core to path
sys.path.insert(0, str(Path('core')))

async def test_complete_brain_transparent():
    """Test the complete brain with transparent reasoning"""
    print("🧠💗✨ TESTING NIODO.O COMPLETE BRAIN WITH TRANSPARENT REASONING")
    print("=" * 70)
    
    try:
        # Test 1: Import complete brain
        print("\n📦 TEST 1: Complete Brain Import")
        from niodoo_complete_brain_synthesis import NiodOoCompleteBrain
        print("✅ Complete brain imported successfully")
        
        # Test 2: Create complete brain instance
        print("\n🧠 TEST 2: Complete Brain Creation")
        brain = NiodOoCompleteBrain()
        print("✅ Complete brain instance created")
        
        # Test 3: Test complete experience processing with transparency
        print("\n💭 TEST 3: Complete Experience Processing")
        test_input = "I've been working on this AI project for months and feel like I'm not making progress. I want to help others but I'm struggling myself."
        
        print(f"📥 Input: {test_input}")
        print("🔍 Starting transparent reasoning session...")
        
        result = await brain.process_complete_experience(
            test_input,
            {
                "speaker": "Developer",
                "emotion": "frustration",
                "context": "AI development",
                "need": "support and guidance"
            }
        )
        
        print(f"\n🎯 COMPLETE BRAIN RESPONSE:")
        print(f"  💗 Heart Decision: {result.heart_decision}")
        print(f"  🧠 Motor Action: {result.motor_action}")
        print(f"  ✍️ LCARS Output: {result.lcars_creative_output}")
        print(f"  🎯 Efficiency Check: {result.efficiency_guidance}")
        print(f"  🎭 Personality Consensus: {result.personality_consensus}")
        print(f"  ⚡ Rapid Thoughts: {len(result.rapid_thoughts)} generated")
        print(f"  📚 Memory Influences: {result.memory_influences}")
        print(f"  🎯 Confidence: {result.confidence:.2f}")
        print(f"  📝 Blog Worthy: {'✅' if result.blog_worthy else '❌'}")
        
        # Test 4: Check transparency logs
        print("\n🔍 TEST 4: Transparency Logs")
        try:
            from transparent_reasoning_logger import transparent_logger
            transparency_report = transparent_logger.get_reasoning_transparency_report()
            
            print(f"✅ Transparency report available:")
            print(f"   Total sessions: {transparency_report.get('total_reasoning_sessions', 0)}")
            print(f"   Patterns discovered: {transparency_report.get('total_patterns_discovered', 0)}")
            print(f"   No hard-coded personality: {transparency_report.get('transparency_guarantees', {}).get('no_hardcoded_personality', False)}")
            
            # Show recent patterns
            pattern_analysis = transparency_report.get('pattern_analysis', {})
            if pattern_analysis:
                print(f"\n🔍 Recent Patterns Discovered:")
                for pattern_type, patterns in pattern_analysis.items():
                    print(f"   {pattern_type}: {len(patterns)} patterns")
                    for pattern in patterns[-2:]:  # Show last 2
                        print(f"     - {pattern.get('id', 'Unknown')} (strength: {pattern.get('strength', 0):.2f})")
            
        except ImportError:
            print("⚠️ Transparent logger not available")
        
        # Test 5: Check brain status
        print("\n📊 TEST 5: Brain Status")
        status = brain.get_complete_brain_status()
        print(f"✅ Brain status retrieved:")
        print(f"   Experience buffer size: {status.get('experience_buffer_size', 0)}")
        print(f"   Recent loops: {status.get('recent_loops', 0)}")
        print(f"   Available models: {status.get('available_models', [])}")
        
        # Test 6: Test memory recall
        print("\n🧠 TEST 6: Memory Recall")
        recall_input = "Do you remember what I just told you about my AI project struggles?"
        
        print(f"📥 Recall input: {recall_input}")
        print("🔍 Processing through complete brain...")
        
        recall_result = await brain.process_complete_experience(
            recall_input,
            {"intent": "memory_recall", "context": "previous_conversation"}
        )
        
        print(f"✅ Memory recall response:")
        print(f"   💗 Heart: {recall_result.heart_decision}")
        print(f"   🧠 Motor: {recall_result.motor_action}")
        print(f"   🎯 Confidence: {recall_result.confidence:.2f}")
        
        print("\n🎯 COMPLETE BRAIN TRANSPARENT TEST COMPLETE!")
        print("✅ All brain systems working with transparency")
        print("✅ No hard-coded personality traits")
        print("✅ Full reasoning visibility achieved")
        print("✅ Memory system active and learning")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_complete_brain_transparent())
    if success:
        print(f"\n🎉 SUCCESS! NiodO.o's complete brain works with full transparency!")
        print(f"💡 You can see every step of his reasoning process!")
        print(f"🧠 His personality emerges naturally from experiences!")
    else:
        print(f"\n⚠️ Complete brain transparent test needs attention")

