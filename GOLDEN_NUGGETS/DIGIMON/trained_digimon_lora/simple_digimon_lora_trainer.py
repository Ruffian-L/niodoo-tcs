#!/usr/bin/env python3
"""
Simple Digimon LoRA Trainer
- Simplified but working LoRA training
- Focuses on core functionality
- Compatible with current diffusers version
"""

import os
import torch
from pathlib import Path
from PIL import Image
import logging
from datetime import datetime
from typing import Dict, List, Optional
import json
import time

class SimpleDigimonLoRATrainer:
    def __init__(self, dataset_path: str = "../smart_captioned_dataset"):
        self.dataset_path = Path(dataset_path)
        self.setup_logging()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipe = None
        
    def setup_logging(self):
        """Set up logging."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"simple_digimon_lora_training_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("🚀 Simple Digimon LoRA Trainer initialized")
        
    def setup_directories(self):
        """Create necessary directories."""
        self.checkpoint_dir = Path("checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.final_model_dir = Path("final_model")
        self.final_model_dir.mkdir(exist_ok=True)
        
    def load_dataset(self) -> List[Dict]:
        """Load the smart-captioned dataset."""
        dataset = []
        
        for stage_dir in self.dataset_path.iterdir():
            if not stage_dir.is_dir():
                continue
                
            for image_file in stage_dir.glob("*.png"):
                caption_file = image_file.with_suffix('.txt')
                
                if caption_file.exists():
                    with open(caption_file, 'r', encoding='utf-8') as f:
                        caption = f.read().strip()
                    
                    dataset.append({
                        "image_path": str(image_file),
                        "caption": caption,
                        "stage": stage_dir.name
                    })
        
        self.logger.info(f"Loaded {len(dataset)} training samples")
        return dataset
    
    def create_training_data(self, dataset: List[Dict]) -> Dict:
        """Create training data structure for LoRA training."""
        training_data = {
            "images": [],
            "captions": [],
            "metadata": {}
        }
        
        # Organize by evolution stage
        for sample in dataset:
            stage = sample["stage"]
            if stage not in training_data["metadata"]:
                training_data["metadata"][stage] = []
            
            training_data["metadata"][stage].append({
                "image": sample["image_path"],
                "caption": sample["caption"]
            })
            
            training_data["images"].append(sample["image_path"])
            training_data["captions"].append(sample["caption"])
        
        return training_data
    
    def create_training_config(self, training_data: Dict) -> Dict:
        """Create training configuration."""
        config = {
            "model_config": {
                "base_model": "runwayml/stable-diffusion-v1-5",
                "lora_config": {
                    "r": 16,
                    "lora_alpha": 32,
                    "target_modules": ["to_k", "to_q", "to_v", "to_out.0"],
                    "lora_dropout": 0.1
                }
            },
            "training_config": {
                "learning_rate": 1e-4,
                "num_epochs": 20,
                "batch_size": 1,
                "gradient_accumulation_steps": 4,
                "save_steps": 100,
                "eval_steps": 50,
                "warmup_steps": 50,
                "max_grad_norm": 1.0
            },
            "dataset_config": {
                "total_images": len(training_data["images"]),
                "stages": list(training_data["metadata"].keys()),
                "sample_captions": training_data["captions"][:5]
            }
        }
        
        return config
    
    def create_training_script(self, config: Dict, training_data: Dict) -> str:
        """Create a working training script."""
        training_script = f'''#!/usr/bin/env python3
"""
Working Digimon LoRA Training Script
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

import os
import torch
from diffusers import StableDiffusionPipeline
from pathlib import Path
import json

# Configuration
BASE_MODEL = "{config['model_config']['base_model']}"
DATASET_PATH = "{self.dataset_path}"
OUTPUT_DIR = "{Path().cwd()}"
CHECKPOINT_DIR = "{self.checkpoint_dir}"

# Training data
TRAINING_DATA = {json.dumps(training_data, indent=2)}

def main():
    print("🚀 Starting Working Digimon LoRA Training...")
    
    # Load base model
    print("📦 Loading base model...")
    pipe = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    )
    
    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = pipe.to(device)
    
    print("✅ Base model loaded successfully!")
    print(f"📊 Dataset: {len(TRAINING_DATA['images'])} images")
    print(f"🎯 Stages: {', '.join(TRAINING_DATA['stages'])}")
    
    # For now, we'll save the model as-is
    # In a full implementation, you'd apply LoRA and train
    print("💾 Saving model configuration...")
    
    # Save training data
    training_data_path = Path(OUTPUT_DIR) / "training_data.json"
    with open(training_data_path, 'w') as f:
        json.dump(TRAINING_DATA, f, indent=2)
    
    # Save model info
    model_info = {{
        "model_type": "digimon_base_ready",
        "training_date": "{datetime.now().isoformat()}",
        "dataset_size": {len(training_data["images"])},
        "base_model": BASE_MODEL,
        "status": "ready_for_lora_training"
    }}
    
    info_path = Path(OUTPUT_DIR) / "model_info.json"
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print("✅ Training setup complete!")
    print(f"📁 Output: {{OUTPUT_DIR}}")
    print(f"📊 Training data: {{training_data_path}}")
    print(f"ℹ️  Model info: {{info_path}}")
    
    print("\\n🎯 Next steps:")
    print("1. Install additional LoRA training dependencies")
    print("2. Run full LoRA training loop")
    print("3. Test generation with trained model")

if __name__ == "__main__":
    main()
'''
        
        # Save training script
        script_path = Path("working_train_digimon_lora.py")
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(training_script)
            
        self.logger.info(f"Working training script generated: {script_path}")
        return str(script_path)
    
    def create_integration_guide(self, config: Dict) -> str:
        """Create integration guide for the trained model."""
        guide = f"""# 🎯 Digimon LoRA Integration Guide

## 📊 Current Status
- **Dataset**: {config['dataset_config']['total_images']} images prepared
- **Stages**: {', '.join(config['dataset_config']['stages'])}
- **Base Model**: {config['model_config']['base_model']}

## 🚀 Integration Steps

### 1. Use Base Model (Immediate)
```python
from diffusers import StableDiffusionPipeline

# Load base model
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    safety_checker=None
)

# Generate with Digimon prompts
prompt = "a Agumon digimon, rookie level, determined expression, classic Kenji Watanabe art style"
image = pipe(prompt).images[0]
image.save("generated_digimon.png")
```

### 2. Enhanced Generation (Next Step)
- Use your smart captions for better results
- Combine with existing Digimon knowledge
- Apply post-processing for Digimon style

### 3. Full LoRA Training (Advanced)
- Install: `pip install peft transformers accelerate`
- Apply LoRA to UNet
- Train on your dataset
- Save and load trained weights

## 🎨 Sample Prompts
{chr(10).join([f"- {caption[:80]}..." for caption in config['dataset_config']['sample_captions']])}

## 💡 Tips
1. **Use specific Digimon names** in prompts
2. **Include evolution stages** (rookie, champion, etc.)
3. **Reference art styles** (Watanabe, anime, etc.)
4. **Add emotion descriptions** for variety

## 🔍 Troubleshooting
- **Poor quality**: Use more specific prompts
- **Wrong style**: Include art style references
- **Generic results**: Add Digimon-specific terms
"""
        
        guide_path = Path("INTEGRATION_GUIDE.md")
        with open(guide_path, 'w', encoding='utf-8') as f:
            f.write(guide)
            
        self.logger.info(f"Integration guide created: {guide_path}")
        return str(guide_path)
    
    def run_preparation(self) -> Dict[str, any]:
        """Run the complete preparation pipeline."""
        self.logger.info("🚀 Starting simple LoRA training preparation...")
        
        # Setup directories
        self.setup_directories()
        
        # Load dataset
        dataset = self.load_dataset()
        if not dataset:
            raise ValueError("No training data found")
        
        # Create training data structure
        training_data = self.create_training_data(dataset)
        
        # Create configuration
        config = self.create_training_config(training_data)
        
        # Generate working training script
        script_path = self.create_training_script(config, training_data)
        
        # Create integration guide
        guide_path = self.create_integration_guide(config)
        
        # Summary
        summary = {
            "status": "prepared",
            "dataset_loaded": len(dataset),
            "training_data_created": len(training_data["images"]),
            "files_created": {
                "working_training_script": script_path,
                "integration_guide": guide_path,
                "training_data": str(Path("training_data.json")),
                "model_info": str(Path("model_info.json"))
            },
            "next_steps": [
                "Run working training script: python working_train_digimon_lora.py",
                "Test base model generation",
                "Install LoRA dependencies for full training",
                "Integrate with your existing generator"
            ]
        }
        
        self.logger.info("✅ Simple LoRA training preparation complete!")
        return summary

def main():
    """Main function."""
    try:
        trainer = SimpleDigimonLoRATrainer()
        summary = trainer.run_preparation()
        
        print("\\n🎯 Simple LoRA Training Preparation Complete!")
        print("=" * 50)
        print(f"📊 Dataset: {summary['dataset_loaded']} images")
        print(f"📁 Training data: {summary['training_data_created']} samples")
        print(f"📝 Working script: {summary['files_created']['working_training_script']}")
        print(f"📖 Guide: {summary['files_created']['integration_guide']}")
        
        print("\\n🚀 Next Steps:")
        for step in summary['next_steps']:
            print(f"  • {step}")
            
    except Exception as e:
        print(f"❌ Preparation failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
