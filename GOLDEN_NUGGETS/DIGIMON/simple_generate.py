#!/usr/bin/env python3
"""
Simple image generator for Intel laptops
"""

from diffusers import StableDiffusionPipeline
import torch
import time
from pathlib import Path

def generate_image(prompt: str, output_path: str = None) -> bool:
    """Generate an image using the given prompt and save to output_path"""
    try:
        print("🚀 Starting Simple Image Generator...")
        
        model_id = "runwayml/stable-diffusion-v1-5"
        device = "cpu"
        
        print("📦 Loading model...")
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
            use_safetensors=False
        )
        
        print("🔧 Optimizing for CPU...")
        pipe.enable_attention_slicing()
        
        print("✅ Ready to generate!")
        print(f"🎨 Creating: {prompt}")
        
        start_time = time.time()
        
        # Generate image
        image = pipe(
            prompt,
            num_inference_steps=20,
            height=512,
            width=512,
            guidance_scale=7.5
        ).images[0]
        
        end_time = time.time()
        
        # Save image
        if output_path is None:
            output_path = "generated_digimon.png"
        
        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        image.save(str(output_path))
        
        print(f"✅ Done! Generated in {end_time-start_time:.1f} seconds")
        print(f"💾 Image saved as: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        return False

def main():
    """Main function for standalone execution"""
    # Enhanced companion-focused prompt  
    prompt = "a koromon desktop companion AI creature, fresh evolution stage, tiny round blob, sphere-like, bubble creature, innocent, curious, baby-like wonder personality traits, simple pure emotions, wide-eyed expressions, designed for desktop interaction and emotional bonding, perfect for animation and sprite work, like a newborn pokemon, slime-chan aesthetic, kawaii anime art style, clean vector-like design, smooth gradients, soft anime lighting, companion AI aesthetic, friendly and approachable, high quality digital artwork, professional anime character design, clean white background, sprite-ready format"
    
    success = generate_image(prompt)
    if success:
        print("🎉 Generation completed successfully!")
    else:
        print("💥 Generation failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())