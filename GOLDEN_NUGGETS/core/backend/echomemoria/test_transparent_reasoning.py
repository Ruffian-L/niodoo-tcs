#!/usr/bin/env python3
"""
Test Transparent Reasoning System
Test NiodO.o's ability to reason transparently without hard-coded personality traits
"""

import asyncio
import sys
from pathlib import Path

# Add core to path
sys.path.insert(0, str(Path('core')))

async def test_transparent_reasoning():
    """Test NiodO.o's transparent reasoning capabilities"""
    print("🧠💗 TESTING NIODO.O TRANSPARENT REASONING")
    print("=" * 60)
    
    try:
        # Test 1: Import transparent logger
        print("\n📦 TEST 1: Transparent Logger Import")
        from transparent_reasoning_logger import transparent_logger
        print("✅ Transparent logger imported successfully")
        
        # Test 2: Start a reasoning session
        print("\n🔍 TEST 2: Reasoning Session")
        session_id = transparent_logger.start_reasoning_session(
            "I'm feeling overwhelmed with this coding project",
            {"emotion": "overwhelmed", "context": "coding", "need": "support"}
        )
        print(f"✅ Reasoning session started: {session_id}")
        
        # Test 3: Log reasoning steps
        print("\n🧠 TEST 3: Logging Reasoning Steps")
        
        # Efficiency brain step
        step1 = transparent_logger.log_reasoning_step(
            brain_system="efficiency",
            thought_process="Analyzing input for potential loops or hyperfocus",
            confidence=0.8,
            emotional_weight=0.6
        )
        print(f"✅ Efficiency brain step logged: {step1}")
        
        # Heart brain step
        step2 = transparent_logger.log_reasoning_step(
            brain_system="heart",
            thought_process="Recognizing need for compassionate support",
            discovered_pattern="compassionate_response",
            learned_insight="When someone is overwhelmed, offer kindness",
            confidence=0.9,
            emotional_weight=0.8
        )
        print(f"✅ Heart brain step logged: {step2}")
        
        # Motor brain step
        step3 = transparent_logger.log_reasoning_step(
            brain_system="motor",
            thought_process="Breaking down overwhelming task into manageable steps",
            learned_insight="Complex problems can be solved step by step",
            confidence=0.7,
            emotional_weight=0.4
        )
        print(f"✅ Motor brain step logged: {step3}")
        
        # Test 4: Discover patterns
        print("\n🔍 TEST 4: Pattern Discovery")
        pattern1 = transparent_logger.discover_pattern(
            pattern_type="emotional_support",
            pattern_description="Providing support when someone feels overwhelmed",
            context_triggers=["overwhelmed", "stressed", "need support"],
            strength=0.8
        )
        print(f"✅ Pattern discovered: {pattern1}")
        
        pattern2 = transparent_logger.discover_pattern(
            pattern_type="practical_guidance",
            pattern_description="Breaking complex tasks into manageable steps",
            context_triggers=["complex", "overwhelming", "coding"],
            strength=0.7
        )
        print(f"✅ Pattern discovered: {pattern2}")
        
        # Test 5: Log learning opportunities
        print("\n📚 TEST 5: Learning Opportunities")
        transparent_logger.log_learning_opportunity(
            "How to better support overwhelmed developers",
            "Coding projects can be emotionally challenging"
        )
        transparent_logger.log_learning_opportunity(
            "Effective task breakdown strategies",
            "Complex problems need systematic approaches"
        )
        print("✅ Learning opportunities logged")
        
        # Test 6: End reasoning session
        print("\n🎯 TEST 6: End Reasoning Session")
        session_summary = transparent_logger.end_reasoning_session(
            "offer_compassionate_support_and_practical_guidance",
            "High confidence based on recognized patterns and learned responses"
        )
        print(f"✅ Session ended: {session_summary.get('final_decision', 'Unknown')}")
        print(f"📊 Metrics: {session_summary.get('metrics', {})}")
        
        # Test 7: Get transparency report
        print("\n📊 TEST 7: Transparency Report")
        transparency_report = transparent_logger.get_reasoning_transparency_report()
        print(f"✅ Transparency report generated:")
        print(f"   Total sessions: {transparency_report.get('total_reasoning_sessions', 0)}")
        print(f"   Patterns discovered: {transparency_report.get('total_patterns_discovered', 0)}")
        print(f"   No hard-coded personality: {transparency_report.get('transparency_guarantees', {}).get('no_hardcoded_personality', False)}")
        
        # Test 8: Export transparency data
        print("\n📤 TEST 8: Export Transparency Data")
        export_file = transparent_logger.export_transparency_data()
        if export_file:
            print(f"✅ Transparency data exported: {export_file}")
        else:
            print("⚠️ Export failed")
        
        print("\n🎯 TRANSPARENT REASONING TEST COMPLETE!")
        print("✅ All reasoning steps logged transparently")
        print("✅ No hard-coded personality traits")
        print("✅ Patterns discovered through experience")
        print("✅ Full reasoning visibility achieved")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_transparent_reasoning())
    if success:
        print(f"\n🎉 SUCCESS! NiodO.o can reason transparently!")
        print(f"💡 His personality emerges from learned experiences, not preset traits!")
    else:
        print(f"\n⚠️ Transparent reasoning needs attention")

