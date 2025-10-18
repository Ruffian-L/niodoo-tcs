#!/usr/bin/env python3
"""
Simple test of AI Commander's generate_response method
"""

from ai_commander import AICommander

def test_ai_commander():
    print("🧠 Testing AI Commander directly")
    print("=" * 40)
    
    # Initialize AI Commander
    commander = AICommander()
    
    # Load the model
    print("🔄 Loading model...")
    if not commander.load_model("_spydaz_web_lcars_artificial_human_r1_002-multi-lingual-thinking-q4_k_m.gguf"):
        print("❌ Failed to load model")
        return
    
    # Test simple generation
    print("\n🤖 Testing simple generation...")
    simple_prompt = "Write a short blog post about debugging code."
    
    print(f"Prompt: {simple_prompt}")
    response = commander.generate_response(simple_prompt, max_tokens=100)
    print(f"Response: {response}")
    
    # Test NiodO.o specific prompt
    print("\n🤖 Testing NiodO.o prompt...")
    niodoo_prompt = "Write a blog post as NiodO.o about helping debug code."
    
    print(f"Prompt: {niodoo_prompt}")
    response = commander.generate_response(niodoo_prompt, max_tokens=100)
    print(f"Response: {response}")

if __name__ == "__main__":
    test_ai_commander()
