#!/usr/bin/env python3
"""
Comprehensive Test for Nervous System Integration
Tests all new components working together
"""

import sys
import time
from pathlib import Path

# Add the core directory to the path
sys.path.insert(0, str(Path(__file__).parent))

def test_nervous_system():
    """Test the nervous system core"""
    print("🧠 Testing Nervous System Core...")
    
    try:
        from nervous_system import NervousSystem, Stimulus, StimulusType
        
        nervous_system = NervousSystem()
        
        # Test sound stimulus
        sound_stimulus = Stimulus(
            type=StimulusType.SOUND,
            intensity=0.8,
            context={'source': 'door_slam'}
        )
        
        response = nervous_system.process_stimulus(sound_stimulus)
        if response:
            print(f"✅ Sound response: {response.action} - {response.emotion}")
            nervous_system.update_mood(response)
        
        # Test movement stimulus
        movement_stimulus = Stimulus(
            type=StimulusType.MOVEMENT,
            intensity=0.6,
            direction=(0.8, 0.2),
            context={'speed': 'fast'}
        )
        
        response = nervous_system.process_stimulus(movement_stimulus)
        if response:
            print(f"✅ Movement response: {response.action} - {response.emotion}")
            nervous_system.update_mood(response)
        
        # Test time stimulus
        time_stimulus = Stimulus(
            type=StimulusType.TIME,
            intensity=0.7,
            context={'hour': 14}  # 2 PM - afternoon slump
        )
        
        response = nervous_system.process_stimulus(time_stimulus)
        if response:
            print(f"✅ Time response: {response.action} - {response.emotion}")
            nervous_system.update_mood(response)
        
        # Test habit learning
        nervous_system.learn_habit("bouncing", "morning_energy", True)
        nervous_system.learn_habit("bouncing", "morning_energy", True)
        
        preference = nervous_system.get_habit_preference("bouncing", "morning_energy")
        print(f"✅ Habit strength: {preference:.3f}")
        
        # Get behavior suggestion
        suggestion = nervous_system.get_behavior_suggestion("morning_energy")
        print(f"✅ Behavior suggestion: {suggestion['action']} - {suggestion['reasoning']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Nervous system test failed: {e}")
        return False

def test_hallucination_vault():
    """Test the hallucination vault"""
    print("\n🔍 Testing Hallucination Vault...")
    
    try:
        from hallucination_vault import HallucinationVault
        
        vault = HallucinationVault("test_vault.json")
        
        # Test scenario retrieval
        scenario = vault.retrieve_random_hallucination("calm", 0.5)
        if scenario:
            print(f"✅ Retrieved scenario: {scenario.title}")
        
        # Test creative action generation
        creative_action = vault.generate_creative_action("excited", 0.8, "user interaction")
        if creative_action:
            print(f"✅ Creative action: {creative_action['action']} - {creative_action['message']}")
        
        # Test hallucination triggers
        triggers = vault.hallucination_triggers("curious", 0.7, ["bouncing", "idle"])
        print(f"✅ Potential triggers: {len(triggers)} found")
        
        # Test reality blending
        if scenario:
            blend = vault.reality_blend("a mouse cursor moving", scenario)
            print(f"✅ Reality blend: {blend[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Hallucination vault test failed: {e}")
        return False

def test_multimodal_enhanced():
    """Test the multimodal enhanced systems"""
    print("\n🎭 Testing Multimodal Enhanced Systems...")
    
    try:
        from multimodal_enhanced import EnhancedVision, EnhancedAudio, InteractionEnhancement, InteractionType, InteractionEvent
        
        # Test Enhanced Vision
        vision = EnhancedVision()
        screen_data = {
            'text': 'Python code development',
            'image_description': 'IDE with code editor',
            'window_title': 'Visual Studio Code',
            'movement': True
        }
        
        observation = vision.analyze_screen_content(screen_data)
        print(f"✅ Vision analysis: {observation.content_type}")
        
        # Test Enhanced Audio
        audio = EnhancedAudio()
        voice_data = {
            'pitch': 0.7,
            'volume': 0.8,
            'speed': 0.9,
            'clarity': 0.8,
            'duration': 2.0
        }
        
        voice_analysis = audio.voice_tone_analysis(voice_data)
        print(f"✅ Voice analysis: {voice_analysis['primary_mood']} (confidence: {voice_analysis['confidence']:.2f})")
        
        # Test Interaction Enhancement
        interaction = InteractionEnhancement()
        events = [
            InteractionEvent(
                timestamp=time.time() - 60,
                type=InteractionType.KEYBOARD_ACTIVITY,
                intensity=0.8,
                duration=30.0
            ),
            InteractionEvent(
                timestamp=time.time() - 30,
                type=InteractionType.WINDOW_CHANGE,
                intensity=0.6,
                duration=5.0
            )
        ]
        
        intent_prediction = interaction.predict_user_intent(events, {'context': 'work'})
        print(f"✅ Intent prediction: {intent_prediction['intent']} (confidence: {intent_prediction['confidence']:.2f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Multimodal enhanced test failed: {e}")
        return False

def test_character_persistence():
    """Test the character persistence system"""
    print("\n🎭 Testing Character Persistence System...")
    
    try:
        from character_persistence import CharacterPersistence
        
        persistence = CharacterPersistence("test_characters")
        
        # Create a character
        personality_data = {
            'playfulness': 0.8,
            'curiosity': 0.9,
            'creativity': 0.7,
            'favorite_colors': ['blue', 'purple', 'green'],
            'preferred_actions': ['bouncing', 'exploring', 'learning']
        }
        
        character_id = persistence.create_character("TestDorumon", personality_data)
        print(f"✅ Created character: {character_id}")
        
        # Load the character
        character = persistence.load_character(character_id)
        if character:
            print(f"✅ Loaded character: {character.personality.name}")
            
            # Test learning
            character.learn_from_experience(
                event_type="play_session",
                description="Had a fun bouncing session with the user",
                emotion="happy",
                intensity=0.8,
                context={'duration': 300, 'user_mood': 'excited'}
            )
            
            print(f"✅ Character learned: Level {character.personality.level}")
            
            # Save character
            success = persistence.save_character(character_id)
            print(f"✅ Character saved: {success}")
            
            # Unload character
            persistence.unload_character(character_id)
            print("✅ Character unloaded")
        
        return True
        
    except Exception as e:
        print(f"❌ Character persistence test failed: {e}")
        return False

def test_integration():
    """Test all systems working together"""
    print("\n🔗 Testing System Integration...")
    
    try:
        from nervous_system import NervousSystem, Stimulus, StimulusType
        from hallucination_vault import HallucinationVault
        from multimodal_enhanced import EnhancedVision, EnhancedAudio, InteractionEnhancement
        from character_persistence import CharacterPersistence
        
        # Initialize all systems
        print("Initializing systems...")
        nervous_system = NervousSystem()
        hallucination_vault = HallucinationVault("test_integration.json")
        enhanced_vision = EnhancedVision()
        enhanced_audio = EnhancedAudio()
        interaction_enhancement = InteractionEnhancement()
        character_persistence = CharacterPersistence("test_integration_characters")
        
        # Create a character
        character_id = character_persistence.create_character("IntegrationTest", {
            'playfulness': 0.7,
            'curiosity': 0.8
        })
        
        character = character_persistence.load_character(character_id)
        if not character:
            raise Exception("Failed to create character")
        
        # Integrate systems
        character.integrate_systems(
            nervous_system=nervous_system,
            hallucination_vault=hallucination_vault,
            enhanced_vision=enhanced_vision,
            enhanced_audio=enhanced_audio,
            interaction_enhancement=interaction_enhancement
        )
        
        print("✅ All systems integrated")
        
        # Test complete workflow
        print("Testing complete workflow...")
        
        # 1. Process stimulus through nervous system
        stimulus = Stimulus(
            type=StimulusType.INTERACTION,
            intensity=0.8,
            context={'user_mood': 'excited'}
        )
        
        response = nervous_system.process_stimulus(stimulus)
        if response:
            print(f"✅ Nervous system response: {response.action}")
            nervous_system.update_mood(response)
            
            # 2. Check for hallucination triggers
            triggers = hallucination_vault.hallucination_triggers(
                response.emotion, 
                nervous_system.energy_level,
                [response.action]
            )
            
            if triggers:
                print(f"✅ Hallucination triggers: {len(triggers)} found")
                
                # 3. Generate creative action
                creative_action = hallucination_vault.generate_creative_action(
                    response.emotion,
                    nervous_system.energy_level,
                    "interaction"
                )
                
                if creative_action:
                    print(f"✅ Creative action: {creative_action['action']}")
            
            # 4. Character learns from experience
            character.learn_from_experience(
                event_type="user_interaction",
                description=f"User interaction triggered {response.action}",
                emotion=response.emotion,
                intensity=response.intensity,
                context={'stimulus_type': 'interaction'}
            )
            
            print(f"✅ Character learned: {character.personality.experience_points} XP")
        
        # 5. Save character state
        character_persistence.save_character(character_id)
        print("✅ Character state saved")
        
        # Cleanup
        character_persistence.unload_character(character_id)
        print("✅ Integration test completed successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 NiodO.o Nervous System Integration Test Suite")
    print("=" * 60)
    
    tests = [
        ("Nervous System Core", test_nervous_system),
        ("Hallucination Vault", test_hallucination_vault),
        ("Multimodal Enhanced", test_multimodal_enhanced),
        ("Character Persistence", test_character_persistence),
        ("System Integration", test_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        try:
            success = test_func()
            results.append((test_name, success))
            print(f"{'✅ PASS' if success else '❌ FAIL'}: {test_name}")
        except Exception as e:
            print(f"❌ ERROR: {test_name} - {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status}: {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Nervous system integration is working!")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
    
    # Instructions for usage
    print("\n💡 To use the nervous system in your AI brain:")
    print("1. Import the systems you need:")
    print("   from nervous_system import NervousSystem")
    print("   from hallucination_vault import HallucinationVault")
    print("   from multimodal_enhanced import EnhancedVision, EnhancedAudio")
    print("   from character_persistence import CharacterPersistence")
    print("")
    print("2. Initialize and integrate them:")
    print("   nervous_system = NervousSystem()")
    print("   hallucination_vault = HallucinationVault()")
    print("   # ... etc")
    print("")
    print("3. Use them in your event processing:")
    print("   response = nervous_system.process_stimulus(stimulus)")
    print("   creative_action = hallucination_vault.generate_creative_action(...)")
    print("")
    print("4. Each system can be enabled/disabled independently!")
    
    print("\n✅ Nervous system integration test completed!")

if __name__ == "__main__":
    main()
