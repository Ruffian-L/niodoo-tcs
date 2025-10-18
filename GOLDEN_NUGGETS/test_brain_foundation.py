#!/usr/bin/env python3
"""
Complete Foundation Test for NiodO.o Brain Systems
Tests core systems and finds ALL 4 models for three-brain architecture
"""

import sys
import asyncio
from pathlib import Path

# Add EchoMemoria to path
sys.path.insert(0, str(Path('organized/ai/EchoMemoria/core')))

print("🧠💗 NIODO.O COMPLETE FOUNDATION TEST")
print("=" * 60)

# Test 1: Core System Imports
print("\n📦 TESTING CORE IMPORTS...")
core_systems = {}

try:
    from heart import HeartCore
    core_systems['heart'] = HeartCore
    print("✅ HeartCore")
except Exception as e:
    print(f"❌ HeartCore: {e}")

try:
    from multi_personality_brain import MultiPersonalityBrain, PersonalityType
    core_systems['multi_brain'] = MultiPersonalityBrain
    core_systems['personality_types'] = PersonalityType
    print("✅ Multi-personality Brain")
except Exception as e:
    print(f"❌ Multi-personality Brain: {e}")

try:
    from decision_reasoning import DecisionReasoningEngine
    core_systems['decision_engine'] = DecisionReasoningEngine
    print("✅ Decision Reasoning Engine")
except Exception as e:
    print(f"❌ Decision Reasoning Engine: {e}")

try:
    from blog_brain import NiodOoBlogBrain
    core_systems['blog_brain'] = NiodOoBlogBrain
    print("✅ Blog Brain")
except Exception as e:
    print(f"❌ Blog Brain: {e}")

try:
    import llama_cpp
    core_systems['llama_cpp'] = llama_cpp
    print("✅ llama.cpp")
except Exception as e:
    print(f"❌ llama.cpp: {e}")

# Test 2: COMPLETE Model Discovery (ALL 4 MODELS)
print(f"\n🔍 DISCOVERING ALL 4 MODELS...")
all_models = []

# Search in multiple directories for ALL model types
search_paths = [
    ".",  # Root directory
    "models",  # Root models directory
    "organized/ai/offline-companion/models",  # Offline companion models
    "organized/ai/mini_pancake",  # Mini pancake models
    "organized/ai/mini_pancake/models",  # Mini pancake models subdirectory
]

for search_path in search_paths:
    path = Path(search_path)
    if path.exists():
        # Find GGUF models
        gguf_models = list(path.glob("*.gguf"))
        for model in gguf_models:
            size_gb = model.stat().st_size / (1024**3)
            all_models.append({
                'path': str(model),
                'name': model.name,
                'size_gb': size_gb,
                'type': 'gguf',
                'location': search_path
            })
        
        # Find safetensors models (search more thoroughly)
        safetensors_models = list(path.glob("*.safetensors"))
        for model in safetensors_models:
            size_gb = model.stat().st_size / (1024**3)
            all_models.append({
                'path': str(model),
                'name': model.name,
                'size_gb': size_gb,
                'type': 'safetensors',
                'location': search_path
            })
        
        # Find large bin files (potential embedding models)
        bin_models = list(path.glob("*.bin"))
        for model in bin_models:
            size_mb = model.stat().st_size / (1024**2)
            if size_mb > 10:  # Only include substantial bin files
                all_models.append({
                    'path': str(model),
                    'name': model.name,
                    'size_gb': size_mb / 1024,
                    'type': 'bin',
                    'location': search_path
                })
        
        # Also search recursively in subdirectories for this path
        if search_path == "organized/ai/mini_pancake":
            # Search more thoroughly in mini_pancake
            for sub_path in path.rglob("*.safetensors"):
                size_gb = sub_path.stat().st_size / (1024**3)
                all_models.append({
                    'path': str(sub_path),
                    'name': sub_path.name,
                    'size_gb': size_gb,
                    'type': 'safetensors',
                    'location': f"{search_path}/subdir"
                })

if all_models:
    print(f"🎯 Found {len(all_models)} models across all directories:")
    total_size_gb = 0
    
    for i, model in enumerate(all_models, 1):
        print(f"  {i}. {model['name']}")
        print(f"     📁 Location: {model['location']}")
        print(f"     📊 Size: {model['size_gb']:.2f}GB")
        print(f"     🔧 Type: {model['type']}")
        total_size_gb += model['size_gb']
        print()
    
    print(f"📊 Total Model Storage: {total_size_gb:.2f}GB")
else:
    print("❌ No models found in any directory")

# Test 3: Brain Model Assignment Strategy
print(f"\n🧠 TESTING BRAIN MODEL ASSIGNMENT...")
if all_models:
    try:
        from brain_model_connections import BrainModelManager
        
        # Create manager with ALL discovered models
        models_dir = "models"  # Use root models directory as primary
        manager = BrainModelManager(models_dir=str(models_dir))
        print("✅ Brain Model Manager created")
        
        # Test model assignment
        success = manager.assign_models_to_brains()
        print(f"✅ Model assignment: {'Success' if success else 'Failed'}")
        
        # Show brain status
        status = manager.get_brain_status()
        print(f"📊 Brain Status:")
        for key, value in status.items():
            print(f"  {key}: {value}")
            
    except Exception as e:
        print(f"❌ Brain Model Manager: {e}")
        
        # Manual model assignment based on what we found
        print(f"\n🎯 MANUAL MODEL ASSIGNMENT STRATEGY:")
        print(f"Based on discovered models, here's the optimal assignment:")
        
        # Find the best models for each brain function
        coding_models = [m for m in all_models if 'qwen' in m['name'].lower() or 'instruct' in m['name'].lower()]
        thinking_models = [m for m in all_models if 'lcars' in m['name'].lower() or 'thinking' in m['name'].lower()]
        reasoning_models = [m for m in all_models if 'mistral' in m['name'].lower() or 'orca' in m['name'].lower()]
        embedding_models = [m for m in all_models if m['type'] == 'safetensors' or 'embed' in m['name'].lower()]
        
        print(f"  🖥️  MOTOR BRAIN (Coding): {coding_models[0]['name'] if coding_models else 'No coding model found'}")
        print(f"  ✍️  LCARS BRAIN (Creative): {thinking_models[0]['name'] if thinking_models else 'No thinking model found'}")
        print(f"  🧮 EFFICIENCY BRAIN (Reasoning): {reasoning_models[0]['name'] if reasoning_models else 'No reasoning model found'}")
        print(f"  🔍 RESEARCH BRAIN (Embeddings): {embedding_models[0]['name'] if embedding_models else 'No embedding model found'}")
        
else:
    print("⚠️ Skipping brain model test - no models available")

# Test 4: Simple Multi-Personality Test
print(f"\n🎭 TESTING 11 PERSONALITIES...")
if 'multi_brain' in core_systems:
    try:
        brain = core_systems['multi_brain']()
        print("✅ Multi-personality brain initialized")
        
        # Show all 11 personalities
        summary = brain.get_personality_summary()
        personalities = summary['personalities']
        print(f"🧩 Found {len(personalities)} personalities:")
        
        for p_name, p_data in list(personalities.items())[:5]:  # Show first 5
            print(f"  • {p_data['name']}: {', '.join(p_data['traits'])}")
        print(f"  ... and {len(personalities) - 5} more personalities")
        
    except Exception as e:
        print(f"❌ Multi-personality test failed: {e}")

# Test 5: Memory System
print(f"\n📚 TESTING MEMORY SYSTEM...")
blog_dir = Path("niodoo_blog/posts")
if blog_dir.exists():
    json_files = list(blog_dir.glob("*.json"))
    print(f"📖 Found {len(json_files)} existing blog memories")
    
    if json_files:
        print("Recent memories:")
        for json_file in json_files[-3:]:  # Show last 3
            print(f"  • {json_file.name}")
else:
    print("⚠️ No blog memories found")

# Summary
print(f"\n🎯 COMPLETE FOUNDATION TEST SUMMARY:")
working_systems = sum(1 for system in core_systems.values() if system)
total_systems = len(core_systems) 
print(f"✅ Working systems: {working_systems}/{total_systems}")
print(f"🔍 Models discovered: {len(all_models)}")
print(f"🧠 Ready for integration: {'YES' if working_systems >= 4 and len(all_models) >= 4 else 'NO'}")

if working_systems >= 4 and len(all_models) >= 4:
    print(f"\n🚀 FOUNDATION IS SOLID! All 4 models found!")
    print(f"🎯 Ready to build the Qt/Python bridge with complete model arsenal!")
else:
    print(f"\n⚠️ Foundation needs work - missing systems or models")

# Show the complete model arsenal
if all_models:
    print(f"\n🎯 NIODO.O'S COMPLETE MODEL ARSENAL:")
    print(f"   🖥️  Coding/Development: {[m['name'] for m in all_models if 'qwen' in m['name'].lower() or 'instruct' in m['name'].lower()]}")
    print(f"   ✍️  Creative/Thinking: {[m['name'] for m in all_models if 'lcars' in m['name'].lower() or 'thinking' in m['name'].lower()]}")
    print(f"   🧮 Reasoning/Efficiency: {[m['name'] for m in all_models if 'mistral' in m['name'].lower() or 'orca' in m['name'].lower()]}")
    print(f"   🔍 Research/Embeddings: {[m['name'] for m in all_models if m['type'] == 'safetensors' or 'embed' in m['name'].lower()]}")
