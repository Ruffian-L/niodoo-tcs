"""
🎭 JULIE'S INTEGRATION TEST SUITE 🚀
Tests all the brilliant systems working together - the ultimate digital orchestra!
"""

import time
import logging
import threading
from typing import Dict, Any, List
import json

# Import our brilliant integration systems
from .system_integration import IntegrationMaster, get_integration_master
from .movement_physics_bridge import MovementPhysicsBridge, get_movement_physics_bridge
from .ai_movement_coordinator import AIMovementCoordinator, get_ai_movement_coordinator
from .performance_optimizer import PerformanceOptimizer, get_performance_optimizer

# Configure logging for the integration test
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Julie_IntegrationTest")

class IntegrationTestSuite:
    """
    🎭 JULIE'S INTEGRATION TEST SUITE 🚀
    
    This test suite verifies that all systems work together seamlessly:
    - System Integration Core
    - Movement-Physics Bridge
    - AI-Movement Coordinator
    - Performance Optimizer
    
    Tests the complete pipeline:
    AI Brain → AI Coordinator → Movement System → Physics Bridge → Physics Engine
    """
    
    def __init__(self):
        self.integration_master = get_integration_master()
        self.movement_physics_bridge = get_movement_physics_bridge()
        self.ai_movement_coordinator = get_ai_movement_coordinator()
        self.performance_optimizer = get_performance_optimizer()
        
        # Test results
        self.test_results = {}
        self.test_start_time = None
        self.test_end_time = None
        
        # Mock systems for testing
        self.mock_systems = {}
        
        logger.info("🎭 Integration Test Suite initialized and ready to test!")
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all integration tests"""
        logger.info("🎭 Starting comprehensive integration test suite...")
        
        self.test_start_time = time.time()
        
        try:
            # Test 1: System Integration Core
            self._test_system_integration_core()
            
            # Test 2: Movement-Physics Bridge
            self._test_movement_physics_bridge()
            
            # Test 3: AI-Movement Coordinator
            self._test_ai_movement_coordinator()
            
            # Test 4: Performance Optimizer
            self._test_performance_optimizer()
            
            # Test 5: Full System Integration
            self._test_full_system_integration()
            
            # Test 6: Performance Under Load
            self._test_performance_under_load()
            
            # Test 7: Error Recovery
            self._test_error_recovery()
            
            # Test 8: System Health Monitoring
            self._test_system_health_monitoring()
            
        except Exception as e:
            logger.error(f"🎭 Test suite failed: {e}")
            self.test_results['overall_status'] = 'FAILED'
            self.test_results['error'] = str(e)
        
        finally:
            self.test_end_time = time.time()
            self._generate_test_report()
        
        return self.test_results
    
    def _test_system_integration_core(self):
        """Test the System Integration Core"""
        logger.info("🎭 Testing System Integration Core...")
        
        try:
            # Test initialization
            assert self.integration_master is not None, "Integration master not initialized"
            
            # Test system registration
            mock_system = MockSystem("test_system")
            self.integration_master.register_system("test_system", mock_system)
            
            # Test event sending
            self.integration_master.send_event("test_event", "test_source", {"data": "test"})
            
            # Test status retrieval
            status = self.integration_master.get_system_status()
            assert status is not None, "Failed to get system status"
            
            # Test orchestration start/stop
            self.integration_master.start_orchestration()
            time.sleep(0.1)  # Let it run briefly
            
            status = self.integration_master.get_system_status()
            assert status['orchestra_running'] == True, "Orchestra not running"
            
            self.integration_master.stop_orchestration()
            time.sleep(0.1)  # Let it stop
            
            status = self.integration_master.get_system_status()
            assert status['orchestra_running'] == False, "Orchestra not stopped"
            
            self.test_results['system_integration_core'] = 'PASSED'
            logger.info("✅ System Integration Core test PASSED")
            
        except Exception as e:
            self.test_results['system_integration_core'] = 'FAILED'
            logger.error(f"❌ System Integration Core test FAILED: {e}")
            raise
    
    def _test_movement_physics_bridge(self):
        """Test the Movement-Physics Bridge"""
        logger.info("🎭 Testing Movement-Physics Bridge...")
        
        try:
            # Test initialization
            assert self.movement_physics_bridge is not None, "Movement-Physics bridge not initialized"
            
            # Test movement request processing
            test_movement = {
                'type': 'walk',
                'direction': 'forward',
                'speed': 1.0,
                'duration': 2.0,
                'emotion': 'happy'
            }
            
            result = self.movement_physics_bridge.process_movement_request(test_movement)
            assert result is not None, "Movement request processing failed"
            assert hasattr(result, 'success'), "Movement result missing success attribute"
            
            # Test different movement types
            movement_types = ['walk', 'run', 'jump', 'sit', 'play']
            for movement_type in movement_types:
                test_movement['type'] = movement_type
                result = self.movement_physics_bridge.process_movement_request(test_movement)
                assert result.success, f"Movement type {movement_type} failed"
            
            # Test bridge status
            status = self.movement_physics_bridge.get_bridge_status()
            assert status is not None, "Failed to get bridge status"
            
            self.test_results['movement_physics_bridge'] = 'PASSED'
            logger.info("✅ Movement-Physics Bridge test PASSED")
            
        except Exception as e:
            self.test_results['movement_physics_bridge'] = 'FAILED'
            logger.error(f"❌ Movement-Physics Bridge test FAILED: {e}")
            raise
    
    def _test_ai_movement_coordinator(self):
        """Test the AI-Movement Coordinator"""
        logger.info("🎭 Testing AI-Movement Coordinator...")
        
        try:
            # Test initialization
            assert self.ai_movement_coordinator is not None, "AI-Movement coordinator not initialized"
            
            # Test AI decision processing
            test_decisions = [
                {
                    'action': 'walk',
                    'emotion': 'happy',
                    'context': {'work_mode': False, 'user_attention': True},
                    'priority': 1.5
                },
                {
                    'action': 'work',
                    'emotion': 'focused',
                    'context': {'work_mode': True, 'focus_required': True},
                    'priority': 2.0
                },
                {
                    'action': 'play',
                    'emotion': 'excited',
                    'context': {'play_mode': True, 'user_activity': 'social'},
                    'priority': 1.0
                }
            ]
            
            for decision in test_decisions:
                success = self.ai_movement_coordinator.process_ai_decision(decision)
                assert success, f"AI decision processing failed for {decision['action']}"
            
            # Test coordinator status
            status = self.ai_movement_coordinator.get_coordinator_status()
            assert status is not None, "Failed to get coordinator status"
            
            # Test context analysis
            assert 'context' in status, "Context analysis missing from status"
            
            self.test_results['ai_movement_coordinator'] = 'PASSED'
            logger.info("✅ AI-Movement Coordinator test PASSED")
            
        except Exception as e:
            self.test_results['ai_movement_coordinator'] = 'FAILED'
            logger.error(f"❌ AI-Movement Coordinator test FAILED: {e}")
            raise
    
    def _test_performance_optimizer(self):
        """Test the Performance Optimizer"""
        logger.info("🎭 Testing Performance Optimizer...")
        
        try:
            # Test initialization
            assert self.performance_optimizer is not None, "Performance optimizer not initialized"
            
            # Test performance level setting
            from .performance_optimizer import PerformanceLevel
            self.performance_optimizer.set_performance_level(PerformanceLevel.HIGH)
            
            # Test monitoring start/stop
            self.performance_optimizer.start_monitoring()
            time.sleep(0.5)  # Let it collect some data
            
            # Get performance summary
            summary = self.performance_optimizer.get_performance_summary()
            assert summary is not None, "Failed to get performance summary"
            assert 'current_metrics' in summary, "Performance metrics missing"
            
            # Test optimization recommendations
            recommendations = self.performance_optimizer.get_optimization_recommendations()
            assert isinstance(recommendations, list), "Optimization recommendations should be a list"
            
            # Stop monitoring
            self.performance_optimizer.stop_monitoring()
            
            self.test_results['performance_optimizer'] = 'PASSED'
            logger.info("✅ Performance Optimizer test PASSED")
            
        except Exception as e:
            self.test_results['performance_optimizer'] = 'FAILED'
            logger.error(f"❌ Performance Optimizer test FAILED: {e}")
            raise
    
    def _test_full_system_integration(self):
        """Test full system integration"""
        logger.info("🎭 Testing Full System Integration...")
        
        try:
            # Start all systems
            self.integration_master.start_orchestration()
            self.performance_optimizer.start_monitoring()
            
            # Connect all systems together
            self.movement_physics_bridge.connect_systems(
                physics_engine=MockPhysicsEngine(),
                movement_system=MockMovementSystem(),
                animation_system=MockAnimationSystem()
            )
            
            self.ai_movement_coordinator.connect_systems(
                movement_system=MockMovementSystem(),
                ai_brain=MockAIBrain(),
                context_analyzer=MockContextAnalyzer()
            )
            
            # Test complete pipeline
            test_ai_decision = {
                'action': 'walk',
                'emotion': 'happy',
                'context': {
                    'work_mode': False,
                    'user_attention': True,
                    'user_activity': 'social'
                },
                'priority': 1.5
            }
            
            # Process through AI coordinator
            success = self.ai_movement_coordinator.process_ai_decision(test_ai_decision)
            assert success, "AI decision processing failed in full integration"
            
            # Let systems process
            time.sleep(0.5)
            
            # Check system status
            integration_status = self.integration_master.get_system_status()
            performance_summary = self.performance_optimizer.get_performance_summary()
            
            assert integration_status is not None, "Integration status missing"
            assert performance_summary is not None, "Performance summary missing"
            
            # Stop systems
            self.integration_master.stop_orchestration()
            self.performance_optimizer.stop_monitoring()
            
            self.test_results['full_system_integration'] = 'PASSED'
            logger.info("✅ Full System Integration test PASSED")
            
        except Exception as e:
            self.test_results['full_system_integration'] = 'FAILED'
            logger.error(f"❌ Full System Integration test FAILED: {e}")
            raise
    
    def _test_performance_under_load(self):
        """Test system performance under load"""
        logger.info("🎭 Testing Performance Under Load...")
        
        try:
            # Start systems
            self.integration_master.start_orchestration()
            self.performance_optimizer.start_monitoring()
            
            # Generate load
            load_threads = []
            for i in range(5):
                thread = threading.Thread(target=self._generate_load, args=(i,))
                load_threads.append(thread)
                thread.start()
            
            # Let load run for a few seconds
            time.sleep(3.0)
            
            # Stop load threads
            for thread in load_threads:
                thread.join(timeout=1.0)
            
            # Check performance
            performance_summary = self.performance_optimizer.get_performance_summary()
            assert performance_summary is not None, "Performance summary missing under load"
            
            # Stop systems
            self.integration_master.stop_orchestration()
            self.performance_optimizer.stop_monitoring()
            
            self.test_results['performance_under_load'] = 'PASSED'
            logger.info("✅ Performance Under Load test PASSED")
            
        except Exception as e:
            self.test_results['performance_under_load'] = 'FAILED'
            logger.error(f"❌ Performance Under Load test FAILED: {e}")
            raise
    
    def _test_error_recovery(self):
        """Test system error recovery"""
        logger.info("🎭 Testing Error Recovery...")
        
        try:
            # Start systems
            self.integration_master.start_orchestration()
            
            # Simulate errors
            self.integration_master.send_event("invalid_event", "test_source", {})
            
            # Let error handling process
            time.sleep(0.5)
            
            # Check system status
            status = self.integration_master.get_system_status()
            assert status is not None, "System status missing after error"
            
            # Stop systems
            self.integration_master.stop_orchestration()
            
            self.test_results['error_recovery'] = 'PASSED'
            logger.info("✅ Error Recovery test PASSED")
            
        except Exception as e:
            self.test_results['error_recovery'] = 'FAILED'
            logger.error(f"❌ Error Recovery test FAILED: {e}")
            raise
    
    def _test_system_health_monitoring(self):
        """Test system health monitoring"""
        logger.info("🎭 Testing System Health Monitoring...")
        
        try:
            # Start systems
            self.integration_master.start_orchestration()
            self.performance_optimizer.start_monitoring()
            
            # Let health monitoring run
            time.sleep(1.0)
            
            # Check health status
            integration_status = self.integration_master.get_system_status()
            performance_summary = self.performance_optimizer.get_performance_summary()
            
            assert 'overall_health' in integration_status, "Overall health missing from integration status"
            assert 'system_health' in performance_summary, "System health missing from performance summary"
            
            # Stop systems
            self.integration_master.stop_orchestration()
            self.performance_optimizer.stop_monitoring()
            
            self.test_results['system_health_monitoring'] = 'PASSED'
            logger.info("✅ System Health Monitoring test PASSED")
            
        except Exception as e:
            self.test_results['system_health_monitoring'] = 'FAILED'
            logger.error(f"❌ System Health Monitoring test FAILED: {e}")
            raise
    
    def _generate_load(self, thread_id: int):
        """Generate load for performance testing"""
        try:
            for i in range(10):
                # Send events
                self.integration_master.send_event(
                    f"load_event_{thread_id}_{i}",
                    f"load_thread_{thread_id}",
                    {"data": f"load_data_{i}"}
                )
                
                # Process AI decisions
                test_decision = {
                    'action': 'walk',
                    'emotion': 'happy',
                    'context': {'work_mode': False},
                    'priority': 1.0
                }
                self.ai_movement_coordinator.process_ai_decision(test_decision)
                
                time.sleep(0.1)
                
        except Exception as e:
            logger.error(f"🎭 Load generation error in thread {thread_id}: {e}")
    
    def _generate_test_report(self):
        """Generate comprehensive test report"""
        try:
            # Calculate test duration
            duration = self.test_end_time - self.test_start_time if self.test_end_time else 0
            
            # Count passed/failed tests
            passed_tests = sum(1 for result in self.test_results.values() if result == 'PASSED')
            failed_tests = sum(1 for result in self.test_results.values() if result == 'FAILED')
            total_tests = len(self.test_results)
            
            # Determine overall status
            if failed_tests == 0:
                overall_status = 'ALL TESTS PASSED'
            else:
                overall_status = f'{failed_tests} TESTS FAILED'
            
            # Create report
            report = {
                'test_summary': {
                    'overall_status': overall_status,
                    'total_tests': total_tests,
                    'passed_tests': passed_tests,
                    'failed_tests': failed_tests,
                    'success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0,
                    'test_duration_seconds': duration
                },
                'test_results': self.test_results,
                'timestamp': time.time(),
                'test_name': "Julie's Integration Test Suite"
            }
            
            self.test_results['test_report'] = report
            
            # Log summary
            logger.info("🎭" + "="*50)
            logger.info(f"🎭 INTEGRATION TEST SUITE COMPLETED")
            logger.info(f"🎭 Overall Status: {overall_status}")
            logger.info(f"🎭 Passed: {passed_tests}/{total_tests}")
            logger.info(f"🎭 Duration: {duration:.2f} seconds")
            logger.info("🎭" + "="*50)
            
            # Save report to file
            self._save_test_report(report)
            
        except Exception as e:
            logger.error(f"🎭 Error generating test report: {e}")
    
    def _save_test_report(self, report: Dict[str, Any]):
        """Save test report to file"""
        try:
            filename = f"integration_test_report_{int(time.time())}.json"
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"🎭 Test report saved to: {filename}")
            
        except Exception as e:
            logger.error(f"🎭 Error saving test report: {e}")

# Mock system classes for testing
class MockSystem:
    """Mock system for testing"""
    def __init__(self, name: str):
        self.name = name
        self.health_check = lambda: {'state': 'healthy', 'metrics': {}}

class MockPhysicsEngine:
    """Mock physics engine for testing"""
    def apply_physics(self, params):
        return True
    
    def get_position(self):
        return (0.0, 0.0)
    
    def get_physics_state(self):
        return {'status': 'active', 'timestamp': time.time()}

class MockMovementSystem:
    """Mock movement system for testing"""
    def execute_movement(self, data):
        return True
    
    def execute_coordinated_movement(self, plan):
        return True
    
    def get_position(self):
        return (0.0, 0.0)
    
    def update_from_physics(self, feedback):
        pass

class MockAnimationSystem:
    """Mock animation system for testing"""
    def play_animation(self, animation_name):
        return True

class MockAIBrain:
    """Mock AI brain for testing"""
    def process_decision(self, decision):
        return True

class MockContextAnalyzer:
    """Mock context analyzer for testing"""
    def analyze_context(self, context):
        return {'work_mode': False, 'user_attention': True}

def run_integration_tests():
    """Run the complete integration test suite"""
    test_suite = IntegrationTestSuite()
    return test_suite.run_all_tests()

if __name__ == "__main__":
    # Run the integration test suite
    print("🎭 Starting Julie's Integration Test Suite...")
    results = run_integration_tests()
    
    # Display results
    print(f"\n🎭 Test Results:")
    for test_name, result in results.get('test_results', {}).items():
        if test_name != 'test_report':
            status_emoji = "✅" if result == 'PASSED' else "❌"
            print(f"{status_emoji} {test_name}: {result}")
    
    # Display summary
    if 'test_report' in results:
        report = results['test_report']
        summary = report['test_summary']
        print(f"\n🎭 Overall Status: {summary['overall_status']}")
        print(f"🎭 Success Rate: {summary['success_rate']:.1f}%")
        print(f"🎭 Duration: {summary['test_duration_seconds']:.2f} seconds")
    
    print("\n🎭 Integration Test Suite completed!")
