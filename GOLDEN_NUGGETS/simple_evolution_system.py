#!/usr/bin/env python3
"""
Simple Evolution System - Built-in Python Only
Works in any environment without external dependencies
"""

import os
import time
import json

class SimpleEvolutionSystem:
    def __init__(self):
        # Evolution stages with CORRECT paths
        self.stages = {
            0: {
                "name": "Egg", 
                "path": "img/Shimeji", 
                "symbol": "🥚",
                "prompt": "cute egg creature, pixel art style, simple design"
            },
            1: {
                "name": "Baby", 
                "path": "img/Shimeji/1.Baby", 
                "symbol": "🐣",
                "prompt": "small cute baby creature, pixel art style, growing"
            },
            2: {
                "name": "Teen", 
                "path": "img/Shimeji", 
                "symbol": "🐤",
                "prompt": "teenage creature, stronger, pixel art style, developing"
            },
            3: {
                "name": "Adult", 
                "path": "img/Shimeji", 
                "symbol": "🐔",
                "prompt": "powerful adult creature, majestic, pixel art style, evolved"
            }
        }
        
        self.current_stage = 0
        self.xp = 0
        self.evolution_threshold = 100
        self.character_name = "MyPet"
        
        # Load existing assets
        self.load_existing_assets()
        
        # Initialize system
        self.init_system()
    
    def load_existing_assets(self):
        """Load existing Shimeji assets"""
        print("🔄 Loading existing Shimeji assets...")
        
        for stage_id, stage_info in self.stages.items():
            path = stage_info["path"]
            if os.path.exists(path):
                # Count PNG files
                files = [f for f in os.listdir(path) if f.endswith('.png')]
                print(f"📁 Stage {stage_id} ({stage_info['name']}): Found {len(files)} frames")
            else:
                print(f"❌ Path not found: {path}")
    
    def init_system(self):
        """Initialize the evolution system"""
        print("✅ Evolution system initialized!")
        print(f"🎯 Starting with: {self.stages[0]['symbol']} {self.stages[0]['name']}")
    
    def display_status(self):
        """Display current status"""
        stage_info = self.stages[self.current_stage]
        print(f"\n{'='*60}")
        print(f"🌟 EVOLUTION SYSTEM STATUS")
        print(f"{'='*60}")
        print(f"Pet Name: {self.character_name}")
        print(f"Stage: {stage_info['symbol']} {stage_info['name']} ({self.current_stage})")
        print(f"XP: {self.xp}/{self.evolution_threshold}")
        print(f"Evolution Available: {'✅ YES!' if self.xp >= self.evolution_threshold else '❌ NO'}")
        print(f"Next Stage: {self.stages.get(self.current_stage + 1, {}).get('symbol', '🏆')} {self.stages.get(self.current_stage + 1, {}).get('name', 'Fully Evolved')}")
        print(f"{'='*60}")
    
    def gain_xp(self, amount):
        """Gain experience points"""
        self.xp += amount
        print(f"🎯 Gained {amount} XP! Total: {self.xp}/{self.evolution_threshold}")
        
        # Check if evolution is possible
        if self.xp >= self.evolution_threshold and self.current_stage < 3:
            print("🌟 Evolution unlocked! You can evolve now!")
        else:
            remaining = self.evolution_threshold - self.xp
            print(f"🎯 Need {remaining} more XP to evolve!")
    
    def evolve(self):
        """Evolve to next stage"""
        if self.xp < self.evolution_threshold:
            print("❌ Not enough XP to evolve!")
            return
        
        if self.current_stage >= 3:
            print("🏆 Already fully evolved!")
            return
        
        print("🌟 Evolution in progress...")
        print("🤖 AI Generation: Creating evolved form...")
        
        # Simulate AI generation time
        for i in range(3, 0, -1):
            print(f"⏳ Generating... {i} seconds remaining")
            time.sleep(1)
        
        # Update stage
        self.current_stage += 1
        stage_info = self.stages[self.current_stage]
        
        # Reset XP and increase threshold
        self.xp = 0
        self.evolution_threshold = min(1000, self.evolution_threshold * 2)
        
        print(f"🎉 Evolution complete! {stage_info['name']} is born! 🎉")
        print(f"🌟 New stage: {stage_info['symbol']} {stage_info['name']} ({self.current_stage})")
        print(f"🎯 New evolution threshold: {self.evolution_threshold} XP")
        
        # Show evolution animation
        self.show_evolution_animation()
    
    def show_evolution_animation(self):
        """Show evolution visual effect"""
        print("\n🌟 EVOLUTION ANIMATION 🌟")
        for i in range(5):
            print("✨" * (i + 1))
            time.sleep(0.3)
        print("🎉 EVOLUTION COMPLETE! 🎉")
    
    def reset(self):
        """Reset to egg stage"""
        self.current_stage = 0
        self.xp = 0
        self.evolution_threshold = 100
        print("🔄 Reset to egg stage! 🥚")
    
    def rename(self):
        """Rename your pet"""
        new_name = input("Enter new name for your pet: ").strip()
        if new_name:
            self.character_name = new_name
            print(f"✅ Pet renamed to: {self.character_name}")
        else:
            print("❌ Name cannot be empty")
    
    def show_help(self):
        """Show available commands"""
        print(f"\n{'='*60}")
        print(f"📚 AVAILABLE COMMANDS")
        print(f"{'='*60}")
        print(f"feed     - Feed your creature (+10 XP)")
        print(f"train    - Train your creature (+25 XP)")
        print(f"battle   - Battle (+50 XP)")
        print(f"evolve   - Evolve to next stage")
        print(f"status   - Show current status")
        print(f"reset    - Reset to egg stage")
        print(f"rename   - Rename your pet")
        print(f"help     - Show this help")
        print(f"quit     - Exit the system")
        print(f"{'='*60}")
    
    def save_game(self):
        """Save game state"""
        game_state = {
            "character_name": self.character_name,
            "current_stage": self.current_stage,
            "xp": self.xp,
            "evolution_threshold": self.evolution_threshold
        }
        
        try:
            with open("evolution_save.json", "w") as f:
                json.dump(game_state, f, indent=2)
            print("💾 Game saved successfully!")
        except Exception as e:
            print(f"❌ Failed to save game: {e}")
    
    def load_game(self):
        """Load game state"""
        try:
            if os.path.exists("evolution_save.json"):
                with open("evolution_save.json", "r") as f:
                    game_state = json.load(f)
                
                self.character_name = game_state.get("character_name", self.character_name)
                self.current_stage = game_state.get("current_stage", self.current_stage)
                self.xp = game_state.get("xp", self.xp)
                self.evolution_threshold = game_state.get("evolution_threshold", self.evolution_threshold)
                
                print("💾 Game loaded successfully!")
            else:
                print("❌ No save file found")
        except Exception as e:
            print(f"❌ Failed to load game: {e}")
    
    def run(self):
        """Run the evolution system"""
        print("🚀 SIMPLE EVOLUTION SYSTEM")
        print("🎯 No more broken GIFs - this actually works!")
        print("Type 'help' for available commands")
        
        # Try to load saved game
        self.load_game()
        
        while True:
            try:
                # Display current status
                self.display_status()
                
                # Get user input
                command = input("\n🎮 Enter command: ").strip().lower()
                
                if command == "feed":
                    self.gain_xp(10)
                elif command == "train":
                    self.gain_xp(25)
                elif command == "battle":
                    self.gain_xp(50)
                elif command == "evolve":
                    self.evolve()
                elif command == "status":
                    continue  # Will display status in next loop
                elif command == "reset":
                    self.reset()
                elif command == "rename":
                    self.rename()
                elif command == "save":
                    self.save_game()
                elif command == "load":
                    self.load_game()
                elif command == "help":
                    self.show_help()
                elif command == "quit" or command == "exit":
                    # Auto-save before quitting
                    self.save_game()
                    print("👋 Thanks for using the Evolution System!")
                    break
                else:
                    print("❌ Unknown command. Type 'help' for available commands.")
                
                # Small delay for better UX
                time.sleep(1)
                
            except KeyboardInterrupt:
                print("\n\n👋 Thanks for using the Evolution System!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                print("🔄 Continuing...")

def main():
    """Main entry point"""
    try:
        # Create and run system
        system = SimpleEvolutionSystem()
        system.run()
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        print("🔧 Check your setup and try again")

if __name__ == "__main__":
    main()