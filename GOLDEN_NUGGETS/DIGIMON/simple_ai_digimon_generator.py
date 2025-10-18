#!/usr/bin/env python3
"""
Simple AI Digimon Generator
Basic version without complex animation dependencies
"""

import os
import sys
import json
import time
import asyncio
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
import argparse

logger = logging.getLogger(__name__)

class SimpleDigimonGenerator:
    """Simple AI Digimon generator"""
    
    def __init__(self):
        repo_root = Path(os.environ.get("PARANIODO_ROOT", Path(__file__).resolve().parents[1]))
        default_base = Path(__file__).resolve().parent
        self.base_path = Path(os.environ.get("DIGIMON_BASE_PATH", str(default_base)))
        self.venv_path = Path(os.environ.get("DIGIMON_VENV_PATH", str(self.base_path / "sdxl-env")))
        self.generated_path = Path(os.environ.get("DIGIMON_OUTPUT_DIR", str(self.base_path / "generated_digimon")))
        self.generated_path.mkdir(exist_ok=True)
        
        # Digimon types and their characteristics
        self.type_characteristics = {
            "beast": "with fur and fangs, wild appearance",
            "dragon": "with scales and claws, fierce expression", 
            "machine": "with mechanical parts, tech appearance",
            "plant": "with leaves and vines, nature appearance",
            "holy": "with wings and aura, divine appearance",
            "dark": "with shadow effects, mysterious appearance",
            "insect": "with exoskeleton, bug-like features",
            "bird": "with feathers and wings, aerial appearance"
        }
        
        logger.info("SimpleDigimonGenerator initialized")

    def _enhance_prompt(self, base_prompt: str, style: str = "watanabe") -> str:
        """Enhance prompt with Digimon-specific styling"""
        
        style_additions = {
            "standard": "in standard digimon art style, high quality digital artwork, clean background",
            "watanabe": "in classic Kenji Watanabe art style, original v-pet design, high quality digital artwork, clean background",
            "anime": "in anime art style, vibrant colors, dynamic pose, high quality digital artwork",
            "pixel": "in pixel art style, 16-bit retro graphics, clean sprites"
        }
        
        style_suffix = style_additions.get(style, style_additions["watanabe"])
        return f"{base_prompt}, {style_suffix}"

    async def generate_digimon_sprite(self, prompt: str, style: str = "watanabe") -> Tuple[str, bool]:
        """Generate a single Digimon sprite using AI"""
        
        # Enhance prompt with Digimon-specific elements
        enhanced_prompt = self._enhance_prompt(prompt, style)
        
        # Generate timestamp for unique filename
        timestamp = int(time.time())
        output_filename = f"digimon_{timestamp}.png"
        output_path = self.generated_path / output_filename
        
        try:
            # Use your existing simple_generate.py but with custom prompt
            success = await self._run_ai_generation(enhanced_prompt, str(output_path))
            
            if success and output_path.exists():
                logger.info(f"Generated sprite: {output_path}")
                return str(output_path), True
            else:
                logger.error(f"Failed to generate sprite for: {enhanced_prompt}")
                return "", False
                
        except Exception as e:
            logger.error(f"Error generating sprite: {e}")
            return "", False

    async def _run_ai_generation(self, prompt: str, output_path: str) -> bool:
        """Run AI generation using your existing system"""
        
        try:
            # Read the simple_generate.py script
            script_path = self.base_path / "simple_generate.py"
            
            if not script_path.exists():
                logger.error(f"simple_generate.py not found at {script_path}")
                return False
            
            with open(script_path, 'r') as f:
                script_content = f.read()
            
            # Create temporary script with custom prompt
            temp_script_content = script_content.replace(
                'prompt = "a cute digital monster, anime style, vibrant colors"',
                f'prompt = "{prompt}"'
            )
            
            # Write temporary script
            temp_script_path = self.base_path / "temp_generate.py"
            with open(temp_script_path, 'w') as f:
                f.write(temp_script_content)
            
            # Run with proper Python path from virtual environment if it exists
            python_path = self.venv_path / "bin" / "python"
            if not python_path.exists():
                python_path = Path(sys.executable)
            
            process = await asyncio.create_subprocess_exec(
                str(python_path),
                str(temp_script_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.base_path)
            )
            
            stdout, stderr = await process.communicate()
            
            # Clean up temporary script
            temp_script_path.unlink(missing_ok=True)
            
            if process.returncode == 0:
                # Check if the default output file was created and move it
                default_output = self.base_path / "generated_digimon.png"
                if default_output.exists():
                    default_output.rename(output_path)
                    logger.info(f"Generated and moved file to: {output_path}")
                    return True
                else:
                    logger.warning("Generation succeeded but output file not found")
                    # Check if the file is in the current directory
                    alt_output = Path("generated_digimon.png")
                    if alt_output.exists():
                        alt_output.rename(output_path)
                        logger.info(f"Found and moved alternate output to: {output_path}")
                        return True
                    return False
            else:
                logger.error(f"AI generation failed: {stderr.decode()}")
                return False
                
        except Exception as e:
            logger.error(f"Error running AI generation: {e}")
            return False

    def get_generation_statistics(self) -> Dict[str, Any]:
        """Get statistics about generated sprites"""
        generated_files = list(self.generated_path.glob("*.png"))
        
        return {
            "total_generated": len(generated_files),
            "generated_today": len([f for f in generated_files 
                                  if f.stat().st_mtime > time.time() - 86400]),
            "output_directory": str(self.generated_path),
            "latest_generation": max(generated_files, key=lambda f: f.stat().st_mtime).name if generated_files else "None"
        }

# CLI Interface
async def main():
    """Command line interface for Simple AI Digimon Generator"""
    
    parser = argparse.ArgumentParser(description="Simple AI Digimon Generator")
    parser.add_argument("--generate", type=str, help="Generate single Digimon with prompt")
    parser.add_argument("--style", type=str, default="watanabe", help="Art style")
    parser.add_argument("--stats", action="store_true", help="Show generation statistics")
    
    args = parser.parse_args()
    
    generator = SimpleDigimonGenerator()
    
    if args.generate:
        print(f"🎨 Generating Digimon: {args.generate}")
        sprite_path, success = await generator.generate_digimon_sprite(
            args.generate, args.style
        )
        if success:
            print(f"✅ Generated sprite: {sprite_path}")
        else:
            print("❌ Generation failed")
    
    elif args.stats:
        stats = generator.get_generation_statistics()
        print("📊 Generation Statistics:")
        print(json.dumps(stats, indent=2))
    
    else:
        print("Usage:")
        print("  python simple_ai_digimon_generator.py --generate 'fire dragon rookie level'")
        print("  python simple_ai_digimon_generator.py --stats")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())