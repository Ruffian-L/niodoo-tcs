#!/usr/bin/env python3

print("🧠 Testing NiodO.o AI Brain Dependencies...")

try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
except ImportError as e:
    print(f"❌ PyTorch: {e}")

try:
    import transformers
    print(f"✅ Transformers: {transformers.__version__}")
except ImportError as e:
    print(f"❌ Transformers: {e}")

try:
    import websockets
    print(f"✅ WebSockets: {websockets.__version__}")
except ImportError as e:
    print(f"❌ WebSockets: {e}")

try:
    import fastapi
    print(f"✅ FastAPI: {fastapi.__version__}")
except ImportError as e:
    print(f"❌ FastAPI: {e}")

try:
    import llama_cpp
    print(f"✅ llama-cpp-python: {llama_cpp.__version__}")
except ImportError as e:
    print(f"❌ llama-cpp-python: {e}")

print("\n🎉 Python environment test complete!")



