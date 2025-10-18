#!/usr/bin/env python3
"""
Qwen2.5 ONNX Model Inspector
Diagnoses input/output requirements for proper Rust integration
"""
import sys
import os
import numpy as np
try:
    import onnx
    import onnxruntime as ort
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("Install with: pip install onnx onnxruntime")
    sys.exit(1)

def inspect_model(model_path):
    print(f"🔍 Inspecting ONNX model: {model_path}")
    print("=" * 60)
    
    # Load model with ONNX
    try:
        model = onnx.load(model_path)
        print("✅ Model loaded successfully with ONNX")
    except Exception as e:
        print(f"❌ Failed to load model with ONNX: {e}")
        return
    
    # Inspect inputs
    print("\n📥 MODEL INPUTS:")
    for i, input_desc in enumerate(model.graph.input):
        name = input_desc.name
        shape = [dim.dim_value if dim.dim_value > 0 else f"dynamic:{dim.dim_param}" 
                for dim in input_desc.type.tensor_type.shape.dim]
        dtype = input_desc.type.tensor_type.elem_type
        
        # Map ONNX dtype to readable format
        dtype_map = {
            1: "float32", 2: "uint8", 3: "int8", 4: "uint16", 5: "int16",
            6: "int32", 7: "int64", 8: "string", 9: "bool", 10: "float16",
            11: "double", 12: "uint32", 13: "uint64"
        }
        dtype_str = dtype_map.get(dtype, f"unknown({dtype})")
        
        print(f"  {i+1}. {name}")
        print(f"     Shape: {shape}")
        print(f"     Type:  {dtype_str}")
    
    # Inspect outputs
    print("\n📤 MODEL OUTPUTS:")
    for i, output_desc in enumerate(model.graph.output):
        name = output_desc.name
        shape = [dim.dim_value if dim.dim_value > 0 else f"dynamic:{dim.dim_param}" 
                for dim in output_desc.type.tensor_type.shape.dim]
        dtype = output_desc.type.tensor_type.elem_type
        dtype_str = dtype_map.get(dtype, f"unknown({dtype})")
        
        print(f"  {i+1}. {name}")
        print(f"     Shape: {shape}")
        print(f"     Type:  {dtype_str}")
    
    # Try creating ONNX runtime session
    print("\n🏃 ONNX RUNTIME SESSION:")
    try:
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        print("✅ ONNX Runtime session created successfully")
        
        print("\nSession Input Details:")
        for input_meta in session.get_inputs():
            print(f"  - {input_meta.name}: {input_meta.shape} ({input_meta.type})")
            
        print("\nSession Output Details:")
        for output_meta in session.get_outputs():
            print(f"  - {output_meta.name}: {output_meta.shape} ({output_meta.type})")
            
    except Exception as e:
        print(f"❌ Failed to create ONNX Runtime session: {e}")
        
    # Generate Rust code template
    print("\n🦀 RUST INTEGRATION TEMPLATE:")
    print("```rust")
    print("// Based on model inspection:")
    
    for i, input_desc in enumerate(model.graph.input):
        name = input_desc.name
        print(f"// Input {i+1}: {name}")
        print(f"//   Expected shape: {shape}")
        print(f"//   Expected type: {dtype_str}")
    
    print("""
// Suggested input preparation:
let input_ids: Vec<i64> = tokenizer.encode(prompt, true)?
    .get_ids().iter().map(|&x| x as i64).collect();
let attention_mask: Vec<i64> = vec![1i64; input_ids.len()];

// Create tensors with proper shapes:
let input_ids_tensor = Array2::from_shape_vec((1, input_ids.len()), input_ids)?
    .into_dyn();
let attention_mask_tensor = Array2::from_shape_vec((1, attention_mask.len()), attention_mask)?
    .into_dyn();
```""")

def test_tokenizer_integration(model_dir):
    """Test if we can load the tokenizer"""
    import os
    tokenizer_path = os.path.join(model_dir, "tokenizer.json")
    
    print(f"\n🔤 TOKENIZER TEST:")
    print(f"Looking for tokenizer at: {tokenizer_path}")
    
    if not os.path.exists(tokenizer_path):
        print("❌ tokenizer.json not found")
        # Look for alternatives
        alternatives = ["tokenizer_config.json", "vocab.json", "merges.txt"]
        found_files = []
        for alt in alternatives:
            alt_path = os.path.join(model_dir, alt)
            if os.path.exists(alt_path):
                found_files.append(alt)
        
        if found_files:
            print(f"📁 Found alternative files: {found_files}")
        return
    
    try:
        # Try loading with HuggingFace tokenizers
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(tokenizer_path)
        print("✅ Tokenizer loaded successfully")
        
        # Test encoding
        test_prompt = "Write a Rust function"
        encoding = tokenizer.encode(test_prompt)
        print(f"Test encoding: '{test_prompt}'")
        print(f"  Token IDs: {encoding.ids[:10]}... ({len(encoding.ids)} total)")
        print(f"  Attention mask: {encoding.attention_mask[:10]}... ({len(encoding.attention_mask)} total)")
        
        return tokenizer
        
    except ImportError:
        print("⚠️  HuggingFace tokenizers not available")
        print("Install with: pip install tokenizers")
    except Exception as e:
        print(f"❌ Failed to load tokenizer: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 model_inspector.py <model_path>")
        print("Example: python3 model_inspector.py /path/to/model_fp16.onnx")
        sys.exit(1)
    
    model_path = sys.argv[1]
    model_dir = os.path.dirname(model_path)
    
    inspect_model(model_path)
    test_tokenizer_integration(model_dir)
    
    print(f"\n🎯 NEXT STEPS:")
    print("1. Use the Rust template above to fix tensor shapes")
    print("2. Ensure tokenizer.json is accessible to Rust code")
    print("3. Test with the corrected input preparation")