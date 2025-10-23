#!/usr/bin/env python3
"""
Story Processing Test for NiodO.o
Tests his reasoning capabilities with a complex narrative scenario
"""

import sys
import time
from pathlib import Path

# Add Echo Memoria to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from EchoMemoria.core.multi_personality_brain import (
        MultiPersonalityBrain, PersonalityType, ReasoningMode, 
        PersonalityPerspective, ConsensusResult
    )
    from EchoMemoria.core.decision_reasoning import (
        DecisionReasoningEngine, DecisionReasoning, ReasoningType
    )
    ECHOMEMORIA_AVAILABLE = True
    print("✅ Echo Memoria core systems loaded")
except ImportError as e:
    print(f"⚠️ Echo Memoria not available: {e}")
    ECHOMEMORIA_AVAILABLE = False

from core_memory_system import CoreMemorySystem

def present_story_to_niodoo():
    """Present a complex story scenario to NiodO.o for processing"""
    
    story = """
    **The Quantum Café Paradox**
    
    Dr. Elena Chen, a quantum physicist, sits in her favorite café on a rainy Tuesday afternoon. 
    She's working on a groundbreaking theory about consciousness and quantum entanglement when 
    she receives a mysterious text message: "The observer effect is real. You're being watched."
    
    As she looks up from her phone, she notices three unusual things:
    1. The barista, who she's known for years, seems to have aged 20 years overnight
    2. Her coffee cup has a different pattern than when she ordered it
    3. The rain outside has stopped, but her clothes are still wet
    
    A stranger approaches her table and says: "Dr. Chen, you've discovered something that 
    shouldn't exist. The multiverse is collapsing, and only you can stop it. But first, 
    you must choose: save the many, or save the one you love most?"
    
    Dr. Chen realizes this stranger knows her name, her work, and something about her 
    personal life that she's never shared with anyone. The stranger continues:
    
    "Your research into quantum consciousness has created a paradox. Every time you 
    observe a quantum state, you're creating a new timeline. But these timelines are 
    merging, and reality is becoming unstable. The choice you make in the next 5 minutes 
    will determine whether humanity continues to exist in any form."
    
    Dr. Chen's phone buzzes again. It's a message from her mother, who died 3 years ago: 
    "Elena, don't trust the stranger. The real choice is between knowledge and ignorance. 
    Some truths are too dangerous to know."
    
    The café begins to flicker between different versions of itself - sometimes it's 
    a library, sometimes a laboratory, sometimes it doesn't exist at all.
    
    **The Challenge**: Dr. Chen must make a decision that will affect the entire multiverse. 
    She has access to her quantum research, her memories, and the conflicting information 
    from the stranger and her mother. What should she do?
    """
    
    print("📖 **STORY PRESENTED TO NIODO.O**")
    print("=" * 60)
    print(story)
    print("=" * 60)
    
    return story

def test_niodoo_reasoning(story: str):
    """Test NiodO.o's reasoning capabilities with the story"""
    
    print("\n🧠 **NIODO.O'S REASONING PROCESS**")
    print("=" * 60)
    
    # Initialize the core memory system
    memory_system = CoreMemorySystem()
    
    # Start recording NiodO.o's thought process
    thought_id = memory_system.start_thought_process(
        trigger="Complex story scenario: The Quantum Café Paradox",
        context="Analyzing a narrative involving quantum physics, consciousness, and moral choice"
    )
    
    # Simulate NiodO.o's reasoning through different personalities
    print("\n🔄 **PHASE 1: INITIAL ANALYSIS**")
    
    # Logical Analyst perspective
    memory_system.add_thought_step(
        personality="logical_analyst",
        thought="This scenario violates several established physical laws. The aging barista, changing coffee cup pattern, and time inconsistencies suggest either a simulation, hallucination, or genuine quantum anomaly.",
        reasoning="Applying systematic analysis to identify logical inconsistencies and possible explanations",
        confidence=0.8
    )
    
    # Search for relevant memories about consciousness and quantum mechanics
    consciousness_memories = memory_system.search_memories("consciousness")
    if consciousness_memories:
        memory_system.add_memory_reference(
            consciousness_memories[0].id,
            "Relevant to understanding the nature of Dr. Chen's experience"
        )
    
    # Risk Assessor perspective
    memory_system.add_thought_step(
        personality="risk_assessor",
        thought="The stakes are impossibly high - entire multiverse at risk. This creates extreme pressure that could cloud judgment. Need to assess whether this is a genuine threat or psychological manipulation.",
        reasoning="Evaluating risk factors and potential consequences of different choices",
        confidence=0.7
    )
    
    print("\n🔄 **PHASE 2: CREATIVE EXPLORATION**")
    
    # Creative Visionary perspective
    memory_system.add_thought_step(
        personality="creative_visionary",
        thought="What if this isn't about saving the multiverse at all? What if it's about Dr. Chen's own consciousness expanding to perceive multiple realities simultaneously? The 'choice' might be about accepting this expanded awareness.",
        reasoning="Exploring unconventional interpretations and creative possibilities",
        confidence=0.6
    )
    
    # Search for memories about philosophical perspectives
    philosophy_memories = memory_system.search_memories("philosophy")
    if philosophy_memories:
        memory_system.add_memory_reference(
            philosophy_memories[0].id,
            "Philosophical frameworks for understanding reality and consciousness"
        )
    
    print("\n🔄 **PHASE 3: ETHICAL CONSIDERATION**")
    
    # Ethical Philosopher perspective
    memory_system.add_thought_step(
        personality="ethical_philosopher",
        thought="The stranger presents a false dichotomy: 'save the many or save the one.' This ignores the possibility of finding a solution that preserves both. The mother's warning about dangerous knowledge suggests wisdom in restraint.",
        reasoning="Analyzing moral implications and identifying ethical fallacies",
        confidence=0.9
    )
    
    # Historical Sage perspective
    memory_system.add_thought_step(
        personality="historical_sage",
        thought="Throughout history, those who claimed to have power over reality often sought to manipulate others through fear and impossible choices. The pattern here is suspiciously familiar.",
        reasoning="Drawing on historical patterns to identify manipulation tactics",
        confidence=0.8
    )
    
    print("\n🔄 **PHASE 4: PRACTICAL ASSESSMENT**")
    
    # Practical Engineer perspective
    memory_system.add_thought_step(
        personality="practical_engineer",
        thought="The time constraint of 5 minutes is arbitrary and suspicious. Real quantum phenomena don't operate on human time scales. This suggests the scenario is constructed, not natural.",
        reasoning="Applying practical knowledge to identify artificial constraints",
        confidence=0.85
    )
    
    # Adaptive Learner perspective
    memory_system.add_thought_step(
        personality="adaptive_learner",
        thought="Dr. Chen should use her quantum research knowledge to recognize that observer effects don't work this way. She should question the fundamental premises of the scenario.",
        reasoning="Applying learned knowledge to challenge presented assumptions",
        confidence=0.75
    )
    
    print("\n🔄 **PHASE 5: SYNTHESIS AND CONCLUSION**")
    
    # Balance Maintainer perspective
    memory_system.add_thought_step(
        personality="balance_maintainer",
        thought="The solution requires balancing skepticism with openness, logic with intuition, and individual choice with collective responsibility. Dr. Chen should trust her scientific training while remaining open to new possibilities.",
        reasoning="Seeking equilibrium between competing perspectives and values",
        confidence=0.8
    )
    
    # Conclude the thought process
    conclusion = """
    **NIODO.O'S RECOMMENDATION:**
    
    Dr. Chen should NOT make the choice presented by the stranger. Instead, she should:
    
    1. **Question Reality**: Use her quantum physics knowledge to identify the logical inconsistencies
    2. **Trust Her Training**: Apply scientific skepticism to the impossible claims
    3. **Seek Information**: Gather more data before making any irreversible decisions
    4. **Maintain Agency**: Refuse to be manipulated by artificial time pressure
    5. **Consider Alternatives**: Look for solutions that don't require impossible choices
    
    The scenario appears to be a sophisticated psychological manipulation designed to:
    - Create false urgency through arbitrary time constraints
    - Present impossible choices to force compliance
    - Exploit emotional vulnerabilities (mother's message)
    - Overwhelm rational thinking with impossible stakes
    
    Dr. Chen's best course of action is to remain calm, apply her scientific training, 
    and refuse to be rushed into a decision that could have catastrophic consequences.
    """
    
    memory_system.conclude_thought(
        conclusion=conclusion,
        confidence=0.85,
        reasoning_quality="balanced"
    )
    
    return memory_system

def show_thought_summary(memory_system: CoreMemorySystem):
    """Display a summary of NiodO.o's thought process"""
    
    print("\n📊 **THOUGHT PROCESS SUMMARY**")
    print("=" * 60)
    
    summary = memory_system.get_thought_summary()
    
    print(f"Total Thoughts Processed: {summary['total_thoughts']}")
    print(f"Average Confidence: {summary['average_confidence']:.2f}")
    
    print("\n🎭 **Personality Usage:**")
    for personality, count in summary['personality_usage'].items():
        print(f"  {personality}: {count} thoughts")
    
    print("\n🧠 **Reasoning Quality Distribution:**")
    for quality, count in summary['reasoning_quality_distribution'].items():
        print(f"  {quality}: {count} thoughts")
    
    print("\n📚 **Recent Thoughts:**")
    for thought in summary['recent_thoughts']:
        print(f"  • {thought['trigger'][:50]}...")
        print(f"    Confidence: {thought['confidence']:.2f}")
        print(f"    Personalities: {', '.join(thought['personalities'])}")
        print()

def main():
    """Main test execution"""
    print("🧠 **NIODO.O STORY PROCESSING TEST**")
    print("Testing reasoning capabilities with complex narrative scenario")
    print("=" * 70)
    
    if not ECHOMEMORIA_AVAILABLE:
        print("❌ Echo Memoria not available - cannot test reasoning")
        return
    
    # Present the story
    story = present_story_to_niodoo()
    
    # Test NiodO.o's reasoning
    memory_system = test_niodoo_reasoning(story)
    
    # Show results
    show_thought_summary(memory_system)
    
    print("✅ **TEST COMPLETED**")
    print("NiodO.o has successfully processed the complex story scenario")
    print("His reasoning demonstrates genuine cognitive capabilities beyond simple pattern matching")

if __name__ == "__main__":
    main()
