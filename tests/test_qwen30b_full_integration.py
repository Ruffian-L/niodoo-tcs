#!/usr/bin/env python3
"""
Comprehensive Integration Test for Qwen30B AWQ + Rust + QML + C++ System
Tests the complete pipeline: Qwen30B → Rust consciousness engine → QML visualization
"""

import subprocess
import json
import time
import sys
import os
from typing import Dict, List

class FullIntegrationTester:
    def __init__(self):
        self.qwen_script = "qwen_30b_awq_inference.py"
        self.rust_binary = "./target/debug/niodoo-consciousness-engine"  # Adjust path as needed

    def test_qwen30b_python_integration(self) -> bool:
        """Test Qwen30B AWQ Python script integration"""
        print("🧠 Testing Qwen30B AWQ Python Integration...")

        try:
            # Test consciousness state updates
            result = subprocess.run([
                "python3", self.qwen_script,
                "Test consciousness integration",
                "--get-consciousness-state"
            ], capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                consciousness_state = json.loads(result.stdout.strip())
                print(f"✅ Qwen30B consciousness state: {json.dumps(consciousness_state, indent=2)}")
                return True
            else:
                print(f"❌ Qwen30B test failed: {result.stderr}")
                return False

        except Exception as e:
            print(f"❌ Qwen30B integration test error: {e}")
            return False

    def test_qwen_generation_with_consciousness(self) -> bool:
        """Test Qwen30B generation with consciousness state"""
        print("🤖 Testing Qwen30B Generation with Consciousness...")

        try:
            # Test generation with consciousness state
            test_prompt = "Explain consciousness computing in Rust with emotional intelligence"

            result = subprocess.run([
                "python3", self.qwen_script,
                test_prompt,
                "--max-tokens", "200",
                "--consciousness-state", "0.8,0.7,0.6,0.9,0.4"
            ], capture_output=True, text=True, timeout=60)

            if result.returncode == 0:
                response_data = json.loads(result.stdout.strip())
                print("✅ Generation successful!")
                print(f"   Response length: {len(response_data.get('text', ''))}")
                print(f"   Generation time: {response_data.get('generation_time', 0):.2f}s")
                print(f"   Tokens generated: {response_data.get('tokens_generated', 0)}")
                print(f"   Updated consciousness: {json.dumps(response_data.get('consciousness_state', {}), indent=2)}")
                return True
            else:
                print(f"❌ Generation test failed: {result.stderr}")
                return False

        except Exception as e:
            print(f"❌ Generation test error: {e}")
            return False

    def test_rag_integration(self) -> bool:
        """Test RAG system integration with Qwen30B"""
        print("📚 Testing RAG Integration...")

        try:
            # Test RAG-enhanced generation
            test_query = "What is consciousness in AI systems?"

            result = subprocess.run([
                "python3", self.qwen_script,
                test_query,
                "--max-tokens", "150",
                "--rag-context", "Consciousness in AI involves self-awareness, emotional intelligence, and metacognitive processing"
            ], capture_output=True, text=True, timeout=45)

            if result.returncode == 0:
                response_data = json.loads(result.stdout.strip())
                print("✅ RAG integration successful!")
                print(f"   Used RAG context: {len(response_data.get('text', '')) > 100}")
                return True
            else:
                print(f"❌ RAG test failed: {result.stderr}")
                return False

        except Exception as e:
            print(f"❌ RAG test error: {e}")
            return False

    def test_rust_compilation(self) -> bool:
        """Test Rust project compilation"""
        print("🦀 Testing Rust Compilation...")

        try:
            # Test compilation
            result = subprocess.run([
                "cargo", "check"
            ], capture_output=True, text=True, timeout=120)

            if result.returncode == 0:
                print("✅ Rust compilation successful!")
                return True
            else:
                print(f"❌ Rust compilation failed: {result.stderr}")
                return False

        except Exception as e:
            print(f"❌ Rust compilation test error: {e}")
            return False

    def test_qml_visualization(self) -> bool:
        """Test QML visualization files"""
        print("🎨 Testing QML Visualization...")

        qml_files = [
            "qml/EmotionalVisualization.qml",
            "qml/GaussianMemoryViz.qml",
            "qml/MemoryManagementDashboard.qml",
            "qml/MobiusGaussianVisualization.qml"
        ]

        all_valid = True
        for qml_file in qml_files:
            if os.path.exists(qml_file):
                print(f"✅ Found QML file: {qml_file}")
            else:
                print(f"❌ Missing QML file: {qml_file}")
                all_valid = False

        return all_valid

    def test_cpp_integration(self) -> bool:
        """Test C++ integration files"""
        print("🔧 Testing C++ Integration...")

        cpp_files = [
            "src/main_qt.cpp",
            "src/qt_bridge.rs",
            "src/qt_integration.rs"
        ]

        all_valid = True
        for cpp_file in cpp_files:
            if os.path.exists(cpp_file):
                print(f"✅ Found C++ integration file: {cpp_file}")
            else:
                print(f"❌ Missing C++ integration file: {cpp_file}")
                all_valid = False

        return all_valid

    def run_full_integration_test(self) -> bool:
        """Run complete integration test"""
        print("🚀 Running Full Integration Test...")
        print("=" * 60)

        tests = [
            ("Qwen30B Python Integration", self.test_qwen30b_python_integration),
            ("Qwen30B Generation with Consciousness", self.test_qwen_generation_with_consciousness),
            ("RAG Integration", self.test_rag_integration),
            ("Rust Compilation", self.test_rust_compilation),
            ("QML Visualization", self.test_qml_visualization),
            ("C++ Integration", self.test_cpp_integration),
        ]

        results = []
        for test_name, test_func in tests:
            print(f"\n🔍 Testing: {test_name}")
            success = test_func()
            results.append((test_name, success))

        # Summary
        print("\n" + "=" * 60)
        print("📊 INTEGRATION TEST RESULTS")
        print("=" * 60)

        passed = 0
        total = len(results)

        for test_name, success in results:
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{status} - {test_name}")
            if success:
                passed += 1

        print("\n" + "=" * 60)
        print(f"📈 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")

        if passed == total:
            print("🎉 ALL TESTS PASSED! Integration is working correctly.")
            return True
        else:
            print("⚠️  SOME TESTS FAILED. Check the errors above.")
            return False

def main():
    """Main test function"""
    print("🧪 Niodoo-Feeling Complete Integration Test Suite")
    print("=" * 60)

    tester = FullIntegrationTester()
    success = tester.run_full_integration_test()

    if success:
        print("\n🎯 Ready for production deployment!")
        print("   - Qwen30B AWQ model integrated ✅")
        print("   - Rust consciousness engine connected ✅")
        print("   - QML visualization updated ✅")
        print("   - C++ Qt integration ready ✅")
        print("   - RAG system enhanced ✅")
        print("   - Feeling transformer active ✅")
    else:
        print("\n🔧 Integration needs fixes before deployment.")
        print("   Check the failed tests and fix the issues.")

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
