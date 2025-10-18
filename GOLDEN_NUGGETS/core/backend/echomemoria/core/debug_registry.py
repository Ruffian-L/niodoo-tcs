#!/usr/bin/env python3
"""
Debug script for ModelRegistry
Shows detailed model discovery and combination logic
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from model_registry import ModelRegistry, ModelType, PerformanceTier

def debug_registry():
    print("🔍 DEBUGGING MODEL REGISTRY")
    print("=" * 40)
    
    registry = ModelRegistry()
    
    # Scan for models
    print("📁 Scanning for models...")
    registry.scan_for_models()
    
    # Show all discovered models
    print(f"\n📊 Found {len(registry.available_models)} models:")
    for name, model in registry.available_models.items():
        print(f"  • {name}")
        print(f"    Path: {model.path}")
        print(f"    Size: {model.size_gb:.1f}GB")
        print(f"    Type: {model.model_type.value}")
        print(f"    Tier: {model.performance_tier.value}")
        print(f"    Params: {model.estimated_params}B")
        print(f"    Context: {model.context_length}")
        print(f"    Max Tokens: {model.max_tokens}")
        print(f"    Threads: {model.threads}")
        print()
    
    # Test different performance tiers
    print("🚀 Testing Performance Tiers:")
    for tier in PerformanceTier:
        print(f"\n  {tier.value.upper()} MODE:")
        combination = registry.get_optimal_model_combination(tier, 8.0)
        if combination:
            for brain_type, model in combination.items():
                print(f"    {brain_type.value}: {model.name} ({model.size_gb:.1f}GB)")
        else:
            print(f"    ❌ No models found for {tier.value}")
    
    # Show system status
    print(f"\n💻 System Status:")
    status = registry.get_system_status()
    for key, value in status.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    debug_registry()
