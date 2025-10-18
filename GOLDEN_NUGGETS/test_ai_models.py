#!/usr/bin/env python3
"""
Test script to verify NiodO.o's AI models are working
"""

import os
import sys
from pathlib import Path

def test_models():
    """Test if our AI models are accessible and valid"""
    print("🧠💖 Testing NiodO.o AI Models...")
    print("=" * 40)
    
    # Check if we're in the right directory
    ai_models_dir = Path("ai_models")
    if not ai_models_dir.exists():
        print("❌ ai_models directory not found!")
        return False
    
    # List all models
    models = list(ai_models_dir.glob("*.gguf"))
    if not models:
        print("❌ No .gguf models found!")
        return False
    
    print(f"✅ Found {len(models)} AI models:")
    total_size = 0
    
    for model in models:
        size_mb = model.stat().st_size / (1024 * 1024)
        total_size += size_mb
        print(f"   📦 {model.name}: {size_mb:.1f} MB")
    
    print(f"\n💾 Total AI brain size: {total_size:.1f} MB")
    
    # Check if models are valid (not empty)
    valid_models = []
    for model in models:
        if model.stat().st_size > 1000000:  # > 1MB
            valid_models.append(model)
        else:
            print(f"⚠️  {model.name} seems too small, might be corrupted")
    
    print(f"\n🎯 Valid models: {len(valid_models)}/{len(models)}")
    
    if len(valid_models) == len(models):
        print("🎉 All AI models are ready!")
        print("🚀 NiodO.o can now think!")
        return True
    else:
        print("⚠️  Some models may have issues")
        return False

def test_python_env():
    """Test if Python environment can load AI libraries"""
    print("\n🐍 Testing Python Environment...")
    print("=" * 40)
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
    except ImportError:
        print("❌ PyTorch not available")
        return False
    
    try:
        import transformers
        print(f"✅ Transformers: {transformers.__version__}")
    except ImportError:
        print("❌ Transformers not available")
        return False
    
    try:
        import llama_cpp
        print(f"✅ llama-cpp-python: {llama_cpp.__version__}")
        return True
    except ImportError:
        print("❌ llama-cpp-python not available")
        print("💡 This is needed to run the GGUF models")
        return False

def main():
    """Main test function"""
    print("🧠💖 NIODO.O AI MODEL TEST SUITE")
    print("=" * 50)
    
    # Test 1: Check models exist
    models_ok = test_models()
    
    # Test 2: Check Python environment
    python_ok = test_python_env()
    
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS:")
    print(f"   AI Models: {'✅ READY' if models_ok else '❌ ISSUES'}")
    print(f"   Python Env: {'✅ READY' if python_ok else '❌ ISSUES'}")
    
    if models_ok and python_ok:
        print("\n🎉 SUCCESS! NiodO.o is ready to think!")
        print("🚀 Next step: Start the AI brain with:")
        print("   cd core/backend/echomemoria")
        print("   source ../../../niodo_env/bin/activate")
        print("   python3 websocket_bridge.py --server --port 8768")
    else:
        print("\n⚠️  Some issues detected. Check the output above.")
    
    return models_ok and python_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
