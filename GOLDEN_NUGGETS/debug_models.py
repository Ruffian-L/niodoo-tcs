#!/usr/bin/env python3
"""
Debug script to see exactly what models are being found
"""

from pathlib import Path

print("🔍 DEBUGGING MODEL DISCOVERY...")
print("=" * 50)

# Search in multiple directories for ALL model types
search_paths = [
    ".",  # Root directory (where qwen2-7b-instruct is)
    "models",  # Root models directory
    "organized/ai/offline-companion/models",  # Offline companion models
    "organized/ai/mini_pancake",  # Mini pancake models
]

all_models = []

for search_path in search_paths:
    path = Path(search_path)
    print(f"\n🔍 Searching in: {search_path}")
    print(f"   Path exists: {path.exists()}")
    
    if path.exists():
        # Find GGUF models
        gguf_models = list(path.glob("*.gguf"))
        print(f"   GGUF models found: {len(gguf_models)}")
        for model in gguf_models:
            size_gb = model.stat().st_size / (1024**3)
            print(f"     • {model.name} ({size_gb:.2f}GB)")
            all_models.append({
                'path': str(model),
                'name': model.name,
                'size_gb': size_gb,
                'type': 'gguf',
                'location': search_path
            })
        
        # Find safetensors models
        safetensors_models = list(path.glob("*.safetensors"))
        print(f"   Safetensors models found: {len(safetensors_models)}")
        for model in safetensors_models:
            size_gb = model.stat().st_size / (1024**3)
            print(f"     • {model.name} ({size_gb:.2f}GB)")
            all_models.append({
                'path': str(model),
                'name': model.name,
                'size_gb': size_gb,
                'type': 'safetensors',
                'location': search_path
            })
        
        # Find large bin files (potential embedding models)
        bin_models = list(path.glob("*.bin"))
        large_bin_models = [m for m in bin_models if m.stat().st_size > 10 * 1024 * 1024]  # > 10MB
        print(f"   Large bin files found: {len(large_bin_models)}")
        for model in large_bin_models:
            size_mb = model.stat().st_size / (1024**2)
            print(f"     • {model.name} ({size_mb:.1f}MB)")
            all_models.append({
                'path': str(model),
                'name': model.name,
                'size_gb': size_mb / 1024,
                'type': 'bin',
                'location': search_path
            })
    else:
        print(f"   ❌ Path does not exist")

print(f"\n🎯 TOTAL MODELS DISCOVERED: {len(all_models)}")
for i, model in enumerate(all_models, 1):
    print(f"  {i}. {model['name']} ({model['type']}) - {model['size_gb']:.2f}GB from {model['location']}")

# Also check for specific models we know exist
print(f"\n🔍 CHECKING FOR SPECIFIC KNOWN MODELS:")
known_models = [
    "qwen2-7b-instruct-q4_k_m.gguf",
    "mistral-7b-openorca.Q5_K_M.gguf", 
    "_spydaz_web_lcars_artificial_human_r1_002-multi-lingual-thinking-q4_k_m.gguf",
    "model.safetensors"
]

for model_name in known_models:
    found = False
    for search_path in search_paths:
        path = Path(search_path) / model_name
        if path.exists():
            size_gb = path.stat().st_size / (1024**3)
            print(f"  ✅ {model_name} found in {search_path} ({size_gb:.2f}GB)")
            found = True
            break
    if not found:
        print(f"  ❌ {model_name} NOT FOUND")
