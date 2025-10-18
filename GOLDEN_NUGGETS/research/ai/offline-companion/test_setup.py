#!/usr/bin/env python3
"""
Test script to verify AI Commander setup
"""

def test_imports():
    """Test if all required packages can be imported"""
    print("🧪 Testing package imports...")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
    except ImportError as e:
        print(f"❌ PyTorch: {e}")
        return False
    
    try:
        import transformers
        print(f"✅ Transformers: {transformers.__version__}")
    except ImportError as e:
        print(f"❌ Transformers: {e}")
        return False
    
    try:
        import llama_cpp
        print(f"✅ Llama-cpp-python: {llama_cpp.__version__}")
    except ImportError as e:
        print(f"❌ Llama-cpp-python: {e}")
        return False
    
    try:
        import langchain
        print(f"✅ LangChain: {langchain.__version__}")
    except ImportError as e:
        print(f"❌ LangChain: {e}")
        return False
    
    try:
        import chromadb
        print(f"✅ ChromaDB: {chromadb.__version__}")
    except ImportError as e:
        print(f"❌ ChromaDB: {e}")
        return False
    
    try:
        import faiss
        print(f"✅ FAISS: {faiss.__version__}")
    except ImportError as e:
        print(f"❌ FAISS: {e}")
        return False
    
    try:
        import sentence_transformers
        print(f"✅ Sentence-Transformers: {sentence_transformers.__version__}")
    except ImportError as e:
        print(f"❌ Sentence-Transformers: {e}")
        return False
    
    return True

def test_ai_commander():
    """Test if AI Commander can be imported"""
    print("\n🧪 Testing AI Commander...")
    
    try:
        from ai_commander import AICommander
        print("✅ AI Commander imported successfully")
        
        # Test instantiation
        commander = AICommander()
        print("✅ AI Commander instantiated successfully")
        
        # Test model listing
        models = commander.list_models()
        print(f"✅ Model listing works: {len(models)} models found")
        
        return True
        
    except Exception as e:
        print(f"❌ AI Commander test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 AI Commander Setup Test")
    print("=" * 40)
    
    # Test imports
    imports_ok = test_imports()
    
    if imports_ok:
        # Test AI Commander
        commander_ok = test_ai_commander()
        
        if commander_ok:
            print("\n🎉 All tests passed! Your AI setup is ready.")
            print("\nNext steps:")
            print("1. Place a .gguf model file in models/ directory")
            print("2. Place .txt/.md files in documents/ for RAG")
            print("3. Run: python ai_commander.py chat")
        else:
            print("\n❌ AI Commander test failed")
    else:
        print("\n❌ Package import test failed")
    
    print("\n" + "=" * 40)

if __name__ == "__main__":
    main()
