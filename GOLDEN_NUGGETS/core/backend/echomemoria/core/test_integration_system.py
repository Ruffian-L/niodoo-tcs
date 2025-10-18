#!/usr/bin/env python3
"""
Integration System Test for EchoMemoria
Demonstrates how all components work together seamlessly

This test shows the complete orchestration of:
- System Integration Core (Julie)
- Movement-Physics Bridge (Julie)
- AI-Movement Coordinator (Julie)
- Performance Optimizer (Julie)
"""

import asyncio
import time
import logging
from typing import Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

async def test_complete_integration():
    """Test the complete integration system"""
    print("🎭 JULIE'S INTEGRATION SYSTEM TEST 🚀")
    print("=" * 50)
    
    try:
        # Phase 1: Initialize System Integration Core
        print("\n🔗 PHASE 1: Initializing System Integration Core...")
        from .system_integration import initialize_integration, start_integration, get_integration_status
        
        success = await initialize_integration()
        if not success:
            print("❌ Failed to initialize integration core")
            return
        
        print("✅ System Integration Core initialized successfully!")
        
        # Phase 2: Start the integration system
        print("\n🚀 PHASE 2: Starting Integration System...")
        await start_integration()
        print("✅ Integration system started!")
        
        # Phase 3: Test Movement-Physics Bridge
        print("\n🌉 PHASE 3: Testing Movement-Physics Bridge...")
        await test_movement_physics_bridge()
        
        # Phase 4: Test AI-Movement Coordinator
        print("\n🧠 PHASE 4: Testing AI-Movement Coordinator...")
        await test_ai_movement_coordinator()
        
        # Phase 5: Test Performance Optimizer
        print("\n⚡ PHASE 5: Testing Performance Optimizer...")
        await test_performance_optimizer()
        
        # Phase 6: Test Complete System Integration
        print("\n🎯 PHASE 6: Testing Complete System Integration...")
        await test_complete_system_integration()
        
        # Phase 7: Get System Status
        print("\n📊 PHASE 7: System Status Report...")
        await get_system_status_report()
        
        # Phase 8: Cleanup
        print("\n🛑 PHASE 8: System Cleanup...")
        from .system_integration import stop_integration
        await stop_integration()
        print("✅ Integration system stopped successfully!")
        
        print("\n🎉 INTEGRATION SYSTEM TEST COMPLETED SUCCESSFULLY! 🎉")
        
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        print(f"❌ Test failed with error: {e}")

async def test_movement_physics_bridge():
    """Test the movement-physics bridge"""
    try:
        from .movement_physics_bridge import (
            movement_physics_bridge, 
            MovementCommand, 
            BridgeMode
        )
        
        # Set up collision boundaries
        boundaries = [
            (0, 0, 800, 600),      # Screen boundary
            (200, 200, 100, 100),  # Obstacle 1
            (500, 300, 80, 80)     # Obstacle 2
        ]
        movement_physics_bridge.set_collision_boundaries(boundaries)
        print("✅ Collision boundaries set")
        
        # Test different bridge modes
        for mode in [BridgeMode.SMOOTH, BridgeMode.REAL_TIME, BridgeMode.ADAPTIVE]:
            movement_physics_bridge.set_bridge_mode(mode)
            print(f"✅ Bridge mode set to: {mode.value}")
        
        # Add test movement commands
        commands = [
            MovementCommand(
                command_id="test_walk",
                timestamp=time.time(),
                action_type="move",
                target_position=(400, 300),
                duration=2.0,
                emotion="happy"
            ),
            MovementCommand(
                command_id="test_jump",
                timestamp=time.time() + 2.0,
                action_type="jump",
                duration=1.0,
                emotion="excited"
            ),
            MovementCommand(
                command_id="test_sit",
                timestamp=time.time() + 3.0,
                action_type="pose",
                duration=1.5,
                emotion="calm",
                context={"pose": "sitting"}
            )
        ]
        
        for command in commands:
            movement_physics_bridge.add_movement_command(command)
            print(f"✅ Added command: {command.action_type}")
        
        # Simulate physics updates
        print("🔄 Simulating physics updates...")
        for i in range(10):
            delta_time = 0.016  # 60 FPS
            movement_physics_bridge.update(delta_time)
            
            # Get status
            status = movement_physics_bridge.get_movement_status()
            print(f"  Frame {i+1}: Position={status['position']}, Velocity={status['velocity']}")
            
            await asyncio.sleep(0.1)
        
        print("✅ Movement-Physics Bridge test completed")
        
    except Exception as e:
        logger.error(f"❌ Movement-Physics Bridge test failed: {e}")
        raise

async def test_ai_movement_coordinator():
    """Test the AI-movement coordinator"""
    try:
        from .ai_movement_coordinator import (
            ai_movement_coordinator,
            AIDecision,
            DecisionPriority,
            MovementContext
        )
        
        # Test context switching
        contexts = [
            MovementContext.WORK,
            MovementContext.PLAY,
            MovementContext.REST,
            MovementContext.FOCUS
        ]
        
        for context in contexts:
            ai_movement_coordinator.set_context(context)
            print(f"✅ Context set to: {context.value}")
            
            # Get context profile
            profile = ai_movement_coordinator.get_context_profile(context)
            if profile:
                print(f"  Movement speed: {profile.movement_speed}")
                print(f"  Animation intensity: {profile.animation_intensity}")
                print(f"  Energy efficiency: {profile.energy_efficiency}")
        
        # Create test AI decisions
        decisions = [
            AIDecision(
                decision_id="decision_work",
                timestamp=time.time(),
                action="walk",
                emotion="focused",
                context={"target": "workstation", "reason": "starting work"},
                priority=DecisionPriority.HIGH,
                confidence=0.9,
                reasoning="User is starting work, should move to workstation",
                target_position=(100, 100),
                duration=3.0
            ),
            AIDecision(
                decision_id="decision_play",
                timestamp=time.time() + 1.0,
                action="jump",
                emotion="excited",
                context={"reason": "celebration", "mood": "happy"},
                priority=DecisionPriority.NORMAL,
                confidence=0.8,
                reasoning="User accomplished something, celebrate!",
                duration=1.0
            ),
            AIDecision(
                decision_id="decision_rest",
                timestamp=time.time() + 2.0,
                action="sit",
                emotion="calm",
                context={"reason": "rest", "location": "comfortable_spot"},
                priority=DecisionPriority.LOW,
                confidence=0.7,
                reasoning="User seems tired, should rest",
                duration=2.0
            )
        ]
        
        # Add decisions to coordinator
        for decision in decisions:
            ai_movement_coordinator.add_ai_decision(decision)
            print(f"✅ Added AI decision: {decision.action} ({decision.emotion})")
        
        # Simulate coordinator updates
        print("🔄 Simulating coordinator updates...")
        for i in range(8):
            ai_movement_coordinator.update(0.016)
            
            # Get status
            status = ai_movement_coordinator.get_coordinator_status()
            print(f"  Update {i+1}: Context={status['current_context']}, Executions={status['execution_queue']}")
            
            await asyncio.sleep(0.1)
        
        print("✅ AI-Movement Coordinator test completed")
        
    except Exception as e:
        logger.error(f"❌ AI-Movement Coordinator test failed: {e}")
        raise

async def test_performance_optimizer():
    """Test the performance optimizer"""
    try:
        from .performance_optimizer import (
            performance_optimizer,
            OptimizationMode,
            start_performance_monitoring,
            stop_performance_monitoring
        )
        
        # Start performance monitoring
        print("📊 Starting performance monitoring...")
        start_performance_monitoring()
        
        # Test different optimization modes
        modes = [
            OptimizationMode.PERFORMANCE,
            OptimizationMode.BALANCED,
            OptimizationMode.EFFICIENCY,
            OptimizationMode.ADAPTIVE
        ]
        
        for mode in modes:
            performance_optimizer.set_optimization_mode(mode)
            print(f"✅ Optimization mode set to: {mode.value}")
        
        # Simulate performance data
        print("🔄 Simulating performance data...")
        frame_times = [15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 19.0, 18.0, 17.0, 16.0]
        
        for i, frame_time in enumerate(frame_times):
            performance_optimizer.update_frame_time(frame_time)
            print(f"  Frame {i+1}: Frame time = {frame_time:.1f}ms")
            await asyncio.sleep(0.2)
        
        # Get performance status
        from .performance_optimizer import get_performance_status
        status = get_performance_status()
        print(f"📊 Performance Status:")
        print(f"  FPS: {status['metrics'].fps:.1f}")
        print(f"  Memory: {status['metrics'].memory_usage_mb:.1f} MB")
        print(f"  CPU: {status['metrics'].cpu_usage_percent:.1f}%")
        print(f"  Performance Score: {status['metrics'].performance_score:.1f}")
        print(f"  System Health: {status['health'].status}")
        
        # Stop performance monitoring
        stop_performance_monitoring()
        print("✅ Performance monitoring stopped")
        
        print("✅ Performance Optimizer test completed")
        
    except Exception as e:
        logger.error(f"❌ Performance Optimizer test failed: {e}")
        raise

async def test_complete_system_integration():
    """Test the complete system integration"""
    try:
        print("🎯 Testing complete system integration...")
        
        # Test system communication
        print("🔗 Testing system communication...")
        
        # Simulate AI decision → Movement → Physics flow
        from .ai_movement_coordinator import ai_movement_coordinator
        from .movement_physics_bridge import movement_physics_bridge
        
        # Set up test scenario
        ai_movement_coordinator.set_context(MovementContext.PLAY)
        print("✅ Set context to PLAY mode")
        
        # Create a complex AI decision
        from .ai_movement_coordinator import AIDecision, DecisionPriority
        complex_decision = AIDecision(
            decision_id="complex_test",
            timestamp=time.time(),
            action="approach",
            emotion="excited",
            context={
                "target": "user",
                "reason": "greeting",
                "urgency": "high"
            },
            priority=DecisionPriority.HIGH,
            confidence=0.95,
            reasoning="User just arrived, should greet them enthusiastically",
            target_position=(600, 400),
            duration=2.5
        )
        
        # Add decision and let it flow through the system
        ai_movement_coordinator.add_ai_decision(complex_decision)
        print("✅ Added complex AI decision")
        
        # Simulate system updates
        print("🔄 Simulating complete system updates...")
        for i in range(15):
            # Update all systems
            ai_movement_coordinator.update(0.016)
            movement_physics_bridge.update(0.016)
            
            # Get status from all systems
            ai_status = ai_movement_coordinator.get_coordinator_status()
            bridge_status = movement_physics_bridge.get_movement_status()
            
            print(f"  Update {i+1}:")
            print(f"    AI Context: {ai_status['current_context']}")
            print(f"    Bridge Position: {bridge_status['position']}")
            print(f"    Current Command: {bridge_status['current_command']}")
            
            await asyncio.sleep(0.1)
        
        print("✅ Complete system integration test completed")
        
    except Exception as e:
        logger.error(f"❌ Complete system integration test failed: {e}")
        raise

async def get_system_status_report():
    """Get comprehensive system status report"""
    try:
        print("📊 Generating System Status Report...")
        print("-" * 40)
        
        # Get integration status
        from .system_integration import get_integration_status
        integration_status = get_integration_status()
        print("🔗 System Integration Core:")
        print(f"  State: {integration_status['state']}")
        print(f"  Mode: {integration_status['integration_mode']}")
        print(f"  Performance: {integration_status['performance']}")
        print(f"  System Health: {integration_status['system_health']}")
        
        # Get bridge status
        from .movement_physics_bridge import get_bridge_status
        bridge_status = get_bridge_status()
        print("\n🌉 Movement-Physics Bridge:")
        print(f"  Position: {bridge_status['position']}")
        print(f"  Velocity: {bridge_status['velocity']}")
        print(f"  Bridge Mode: {bridge_status['bridge_mode']}")
        print(f"  Performance: {bridge_status['performance']}")
        
        # Get coordinator status
        from .ai_movement_coordinator import get_coordinator_status
        coordinator_status = get_coordinator_status()
        print("\n🧠 AI-Movement Coordinator:")
        print(f"  Context: {coordinator_status['current_context']}")
        print(f"  Active Decisions: {coordinator_status['active_decisions']}")
        print(f"  Execution Queue: {coordinator_status['execution_queue']}")
        print(f"  Success Rate: {coordinator_status['success_rate']:.2f}")
        
        # Get performance status
        from .performance_optimizer import get_performance_status
        performance_status = get_performance_status()
        print("\n⚡ Performance Optimizer:")
        print(f"  FPS: {performance_status['metrics'].fps:.1f}")
        print(f"  Memory: {performance_status['metrics'].memory_usage_mb:.1f} MB")
        print(f"  CPU: {performance_status['metrics'].cpu_usage_percent:.1f}%")
        print(f"  Performance Score: {performance_status['metrics'].performance_score:.1f}")
        print(f"  System Health: {performance_status['health'].status}")
        print(f"  Optimization Mode: {performance_status['optimization']['mode']}")
        
        print("\n" + "=" * 50)
        print("🎉 SYSTEM STATUS REPORT COMPLETED! 🎉")
        
    except Exception as e:
        logger.error(f"❌ Failed to generate system status report: {e}")
        raise

def run_integration_test():
    """Run the complete integration test"""
    print("🚀 Starting EchoMemoria Integration System Test...")
    print("This test demonstrates the complete orchestration of all systems!")
    print()
    
    try:
        # Run the async test
        asyncio.run(test_complete_integration())
        
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        logger.error(f"Integration test failed: {e}")

if __name__ == "__main__":
    # Run the integration test
    run_integration_test()
