#!/usr/bin/env python3
"""Quick model inspection to find simpler models."""
import onnx
import os

model_dir = "/home/ruffian/Desktop/Niodoo-Final/models/qwen2.5-coder-0.5b-instruct-onnx/onnx"

models_to_check = [
    "model_int8.onnx",  # Quantized might be simpler
    "model_q4.onnx",    # Another quantized variant
    "model_quantized.onnx"  # Generic quantized
]

for model_name in models_to_check:
    model_path = os.path.join(model_dir, model_name)
    if os.path.exists(model_path):
        print(f"\n{'='*50}")
        print(f"INSPECTING: {model_name}")
        print(f"{'='*50}")
        
        try:
            model = onnx.load(model_path)
            
            print(f"Inputs ({len(model.graph.input)}):")
            for inp in model.graph.input:
                print(f"  - {inp.name}: {[d.dim_value for d in inp.type.tensor_type.shape.dim]}")
            
            print(f"\nOutputs ({len(model.graph.output)}):")
            for out in model.graph.output:
                print(f"  - {out.name}: {[d.dim_value for d in out.type.tensor_type.shape.dim]}")
                
            print(f"\nFile size: {os.path.getsize(model_path) / (1024*1024):.1f} MB")
            
        except Exception as e:
            print(f"Error loading {model_name}: {e}")