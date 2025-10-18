#!/usr/bin/env python3
"""
Debug script for Model Registry
"""

from core.model_registry import ModelRegistry, ModelType, PerformanceTier

def debug_registry():
    print("🔍 DEBUGGING MODEL REGISTRY...")
    
    # Initialize registry
    registry = ModelRegistry()
    
    print(f"\n📊 Available Models:")
    for name, model in registry.available_models.items():
        print(f"  {name}:")
        print(f"    Type: {model.model_type.value}")
        print(f"    Tier: {model.performance_tier.value}")
        print(f"    Size: {model.size_gb:.1f}GB")
        print(f"    Params: {model.estimated_params}B")
    
    print(f"\n🎯 Testing Model Combinations:")
    
    for performance in ["tiny", "fast", "balanced", "power"]:
        print(f"\n  {performance.upper()} MODE:")
        
        # Get models by tier
        tier = PerformanceTier(performance)
        models_by_tier = registry.get_models_by_tier(tier)
        print(f"    Models in {performance} tier: {len(models_by_tier)}")
        
        for model in models_by_tier:
            print(f"      - {model.name} ({model.model_type.value})")
        
        # Test optimal combination
        available_ram = registry.get_system_status()["available_ram_gb"]
        print(f"    Available RAM: {available_ram:.1f}GB")
        
        optimal = registry.get_optimal_model_combination(performance, available_ram)
        print(f"    Optimal combination: {len(optimal)} models")
        
        for brain_type, model in optimal.items():
            print(f"      {brain_type.value}: {model.name}")
    
    print(f"\n🔧 System Status:")
    status = registry.get_system_status()
    for key, value in status.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    debug_registry()
