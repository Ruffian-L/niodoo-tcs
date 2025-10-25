"""
Qt Bridge for Möbius-Gaussian Visualization
Connects Python Möbius-Gaussian engine to Qt Quick 3D visualization
"""

import json
import threading
import time
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path

class QtVisualizationBridge:
    """
    Bridge between Python Möbius-Gaussian processing and Qt visualization
    """

    def __init__(self, mobius_engine, update_interval: float = 0.1):
        self.mobius_engine = mobius_engine
        self.update_interval = update_interval
        self.running = False
        self.update_thread = None

        # Visualization state
        self.last_update = 0
        self.visualization_data = {}

        print("🌉 Qt Visualization Bridge initialized")

    def start_updates(self):
        """Start the update thread for real-time visualization"""
        if self.running:
            return

        self.running = True
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        print("🔄 Qt bridge update loop started")

    def stop_updates(self):
        """Stop the update thread"""
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=1.0)
        print("⏹️ Qt bridge update loop stopped")

    def _update_loop(self):
        """Main update loop running in background thread"""
        while self.running:
            try:
                # Get current visualization data
                self.visualization_data = self.mobius_engine.get_visualization_data()

                # Save to JSON file for Qt to read
                self._save_visualization_data()

                time.sleep(self.update_interval)

            except Exception as e:
                print(f"⚠️ Qt bridge update error: {e}")
                time.sleep(1.0)

    def _save_visualization_data(self):
        """Save visualization data to JSON file for Qt"""
        try:
            viz_file = Path("data/visualization_state.json")
            viz_file.parent.mkdir(exist_ok=True)

            with open(viz_file, 'w') as f:
                json.dump(self.visualization_data, f, indent=2)

        except Exception as e:
            print(f"⚠️ Failed to save visualization data: {e}")

    def get_visualization_data(self) -> Dict[str, Any]:
        """Get current visualization data"""
        return self.visualization_data.copy()

    def trigger_manual_update(self):
        """Manually trigger a visualization update"""
        self.visualization_data = self.mobius_engine.get_visualization_data()
        self._save_visualization_data()
        print("📡 Manual visualization update triggered")

def integrate_with_qt_visualization(mobius_engine) -> QtVisualizationBridge:
    """Create and start Qt visualization bridge"""

    bridge = QtVisualizationBridge(mobius_engine)
    bridge.start_updates()

    return bridge

if __name__ == "__main__":
    # Test the Qt bridge
    print("🧪 Testing Qt Visualization Bridge")
    print("=" * 40)

    from core.mobius_gaussian_engine import MobiusGaussianEngine, create_test_memories

    # Create engine and test memories
    engine = MobiusGaussianEngine()
    create_test_memories(engine)

    # Create bridge
    bridge = integrate_with_qt_visualization(engine)

    # Test some operations
    print("🔄 Testing Möbius traversal...")
    for i in range(3):
        engine.traverse_mobius_path(0.5 if i % 2 == 0 else -0.3)
        time.sleep(0.5)

    # Get final visualization data
    viz_data = bridge.get_visualization_data()
    print(f"📊 Final visualization state: {len(viz_data.get('spheres', []))} spheres")

    # Stop bridge
    bridge.stop_updates()
    print("✅ Qt bridge test completed")























