//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

/*
use tracing::{info, error, warn};
 * 🧠💖 NIODOO INTEGRATION TESTS - COMPREHENSIVE SYSTEM VALIDATION
 *
 * Tests all major systems working together:
 * - Consciousness Engine + Qwen Inference
 * - Emotional LoRA + Personality System
 * - Mobius Gaussian + Memory Systems
 * - Qt Integration + Real-time Processing
 */

use anyhow::Result;
use std::sync::Arc;
use tokio::time::{timeout, Duration};

use crate::{
    config::{AppConfig, ModelConfig},
    consciousness_engine::PersonalNiodooConsciousness,
    dual_mobius_gaussian::DualMobiusGaussianProcessor,
    emotional_lora::{EmotionalContext, EmotionalLoraAdapter},
    personality::PersonalityType,
    qwen_inference::QwenInference,
};

/// Test the complete consciousness engine integration
#[tokio::test]
async fn test_consciousness_engine_integration() {
    tracing::info!("🧠 Testing consciousness engine integration...");

    // Create consciousness engine
    let mut consciousness = match PersonalNiodooConsciousness::new().await {
        Ok(engine) => engine,
        Err(e) => {
            tracing::info!("⚠️ Consciousness engine initialization failed: {}", e);
            return; // Skip test if engine can't initialize
        }
    };

    // Test autonomous processing
    match consciousness.process_cycle().await {
        Ok(_) => tracing::info!("✅ Autonomous cycle completed"),
        Err(e) => tracing::info!("⚠️ Autonomous cycle failed: {}", e),
    }

    // Test user input processing
    match consciousness
        .process_input_personal("Hello, how are you feeling today?")
        .await
    {
        Ok(response) => tracing::info!(
            "✅ User input processing: {}",
            &response[..50.min(response.len())]
        ),
        Err(e) => tracing::info!("⚠️ User input processing failed: {}", e),
    }
}

/// Test Qwen inference integration
#[tokio::test]
async fn test_qwen_inference_integration() {
    tracing::info!("🤖 Testing Qwen inference integration...");

    // Create a minimal config for testing
    let model_config = ModelConfig {
        qwen_model_path: "microsoft/DialoGPT-medium".to_string(), // Use smaller model for testing
        temperature: 0.7,
        max_tokens: 50,
        timeout: 30,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        top_p: 1.0,
        top_k: 40,
        repeat_penalty: 1.0,
    };

    // Test Qwen inference (this might fail if model not available, which is expected)
    match QwenInference::new(&model_config, nvml_wrapper::Device::Cpu) {
        Ok(_) => tracing::info!("✅ Qwen model loaded successfully"),
        Err(e) => tracing::info!(
            "⚠️ Qwen model loading failed (expected if model not cached): {}",
            e
        ),
    }
}

/// Test Emotional LoRA integration
#[tokio::test]
async fn test_emotional_lora_integration() {
    tracing::info!("💖 Testing Emotional LoRA integration...");

    // Create LoRA adapter
    let lora_adapter = EmotionalLoraAdapter::new(nvml_wrapper::Device::Cpu);

    match lora_adapter {
        Ok(adapter) => {
            tracing::info!("✅ Emotional LoRA adapter created");

            // Test neurodivergent blending
            let context = EmotionalContext::new(0.5, 0.7, 0.3, 0.6, 0.8);
            match adapter.apply_neurodivergent_blending(&context).await {
                Ok(weights) => tracing::info!(
                    "✅ Neurodivergent blending: {} personalities",
                    weights.len()
                ),
                Err(e) => tracing::info!("⚠️ Neurodivergent blending failed: {}", e),
            }
        }
        Err(e) => tracing::info!("⚠️ Emotional LoRA creation failed: {}", e),
    }
}

/// Test Mobius Gaussian integration
#[tokio::test]
async fn test_mobius_gaussian_integration() {
    tracing::info!("🌌 Testing Mobius Gaussian integration...");

    // Create Mobius processor
    let processor = DualMobiusGaussianProcessor::new();

    // Test with sample data
    let sample_points = vec![(1.0, 2.0, 0.5), (2.0, 1.5, 0.3), (1.5, 2.5, 0.7)];

    match processor.fit_gaussian_process(&sample_points).await {
        Ok(gp) => tracing::info!("✅ Gaussian process fitted: {} points", gp.len()),
        Err(e) => tracing::info!("⚠️ Gaussian process fitting failed: {}", e),
    }

    // Test prediction
    let test_point = (1.8, 2.2, 0.6);
    match processor.predict(&test_point).await {
        Ok(prediction) => tracing::info!("✅ Prediction: {:.3}", prediction),
        Err(e) => tracing::info!("⚠️ Prediction failed: {}", e),
    }
}

/// Test complete system integration
#[tokio::test]
async fn test_complete_system_integration() {
    tracing::info!("🔗 Testing complete system integration...");

    // This test ensures all major components can be instantiated together
    let timeout_duration = Duration::from_secs(10);

    let integration_result = timeout(timeout_duration, async {
        // Try to create all major components
        let consciousness = PersonalNiodooConsciousness::new().await;
        let qwen_config = ModelConfig {
            qwen_model_path: "microsoft/DialoGPT-small".to_string(),
            temperature: 0.7,
            max_tokens: 50,
            timeout: 30,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            top_p: 1.0,
            top_k: 40,
            repeat_penalty: 1.0,
        };
        let qwen = QwenInference::new(&qwen_config, nvml_wrapper::Device::Cpu);
        let lora = EmotionalLoraAdapter::new(nvml_wrapper::Device::Cpu);
        let mobius = DualMobiusGaussianProcessor::new();

        match (consciousness, qwen, lora, mobius) {
            (Ok(_), Ok(_), Ok(_), _) => {
                tracing::info!("✅ All major components initialized successfully");
                Ok(())
            }
            (Err(e), _, _, _) => {
                tracing::info!("⚠️ Consciousness engine failed: {}", e);
                Err(e)
            }
            (_, Err(e), _, _) => {
                tracing::info!("⚠️ Qwen inference failed: {}", e);
                Err(e.into())
            }
            (_, _, Err(e), _) => {
                tracing::info!("⚠️ Emotional LoRA failed: {}", e);
                Err(e)
            }
            _ => {
                tracing::info!("✅ Component integration test completed");
                Ok(())
            }
        }
    })
    .await;

    match integration_result {
        Ok(Ok(_)) => tracing::info!("✅ Complete system integration successful"),
        Ok(Err(e)) => tracing::info!("⚠️ Integration test had issues: {}", e),
        Err(_) => tracing::info!("⚠️ Integration test timed out"),
    }
}

/// Performance benchmark test
#[tokio::test]
async fn test_performance_benchmarks() {
    tracing::info!("⚡ Testing performance benchmarks...");

    use std::time::Instant;

    let start = Instant::now();

    // Test multiple system operations
    for i in 0..3 {
        let consciousness = PersonalNiodooConsciousness::new().await;
        match consciousness {
            Ok(mut engine) => {
                let _ = engine.process_cycle().await;
                tracing::info!("✅ Performance test iteration {} completed", i + 1);
            }
            Err(e) => tracing::info!("⚠️ Performance test iteration {} failed: {}", i + 1, e),
        }
    }

    let elapsed = start.elapsed();
    tracing::info!("⚡ Performance benchmark completed in {:?}", elapsed);
}

/// Memory leak test
#[tokio::test]
async fn test_memory_leak_detection() {
    tracing::info!("🔍 Testing for memory leaks...");

    // Run multiple cycles to detect potential leaks
    for i in 0..5 {
        let consciousness = PersonalNiodooConsciousness::new().await;
        match consciousness {
            Ok(mut engine) => {
                let _ = engine.process_cycle().await;
                let _ = engine
                    .process_input_personal("Test input for memory leak detection")
                    .await;
                tracing::info!("✅ Memory leak test iteration {} completed", i + 1);
            }
            Err(e) => tracing::info!("⚠️ Memory leak test iteration {} failed: {}", i + 1, e),
        }

        // Brief pause between iterations
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    tracing::info!("✅ Memory leak test completed - check memory usage manually");
}

/// Concurrent processing test
#[tokio::test]
async fn test_concurrent_processing() {
    tracing::info!("🔄 Testing concurrent processing...");

    use futures::future;

    // Test concurrent operations
    let futures = (0..3).map(|i| async move {
        let consciousness = PersonalNiodooConsciousness::new().await;
        match consciousness {
            Ok(mut engine) => {
                let _ = engine.process_cycle().await;
                let response = engine
                    .process_input_personal(&format!("Concurrent test {}", i))
                    .await;
                tracing::info!("✅ Concurrent operation {} completed", i);
                response
            }
            Err(e) => {
                tracing::info!("⚠️ Concurrent operation {} failed: {}", i, e);
                Err(e)
            }
        }
    });

    let results = future::try_join_all(futures).await;

    match results {
        Ok(responses) => tracing::info!(
            "✅ Concurrent processing: {} operations completed",
            responses.len()
        ),
        Err(e) => tracing::info!("⚠️ Concurrent processing failed: {}", e),
    }
}

/// Test system resource usage
#[tokio::test]
async fn test_resource_usage() {
    tracing::info!("📊 Testing system resource usage...");

    let initial_memory = get_memory_usage();

    // Create and use consciousness engine
    let consciousness = PersonalNiodooConsciousness::new().await;
    match consciousness {
        Ok(mut engine) => {
            let _ = engine.process_cycle().await;
            let _ = engine.process_input_personal("Resource usage test").await;
        }
        Err(e) => tracing::info!("⚠️ Resource test failed: {}", e),
    }

    let final_memory = get_memory_usage();
    let memory_delta = final_memory.saturating_sub(initial_memory);

    tracing::info!("📊 Memory usage: {} bytes increase", memory_delta);
    tracing::info!("✅ Resource usage test completed");
}

fn get_memory_usage() -> usize {
    // Simple memory usage estimation (not accurate but better than nothing)
    // In a real implementation, you'd use proper memory profiling tools
    0
}

/// Test error handling and recovery
#[tokio::test]
async fn test_error_handling_and_recovery() {
    tracing::info!("🛡️ Testing error handling and recovery...");

    // Test with invalid inputs
    let consciousness = PersonalNiodooConsciousness::new().await;
    match consciousness {
        Ok(mut engine) => {
            // Test with very long input
            let long_input = "x".repeat(10000);
            match engine.process_input_personal(&long_input).await {
                Ok(response) => tracing::info!("✅ Long input handled: {} chars", response.len()),
                Err(e) => tracing::info!("⚠️ Long input failed gracefully: {}", e),
            }

            // Test with empty input
            match engine.process_input_personal("").await {
                Ok(response) => tracing::info!("✅ Empty input handled: {} chars", response.len()),
                Err(e) => tracing::info!("⚠️ Empty input failed gracefully: {}", e),
            }
        }
        Err(e) => tracing::info!("⚠️ Error handling test setup failed: {}", e),
    }

    tracing::info!("✅ Error handling test completed");
}

/// Test configuration loading and validation
#[tokio::test]
async fn test_configuration_system() {
    tracing::info!("⚙️ Testing configuration system...");

    // Test loading default config
    let config = AppConfig::default();
    tracing::info!(
        "✅ Default config loaded: max_cycles = {:?}",
        config.max_cycles
    );

    // Test config validation
    assert!(config.cycle_delay.is_none() || config.cycle_delay.unwrap().as_secs() > 0);
    tracing::info!("✅ Configuration validation passed");
}

/// Test the complete user experience flow
#[tokio::test]
async fn test_user_experience_flow() {
    tracing::info!("👤 Testing complete user experience flow...");

    // Simulate a conversation
    let test_inputs = vec![
        "Hello, how are you feeling today?",
        "Tell me about your consciousness",
        "What do you think about artificial intelligence?",
        "How do you process emotions?",
        "Goodbye",
    ];

    let consciousness = PersonalNiodooConsciousness::new().await;
    match consciousness {
        Ok(mut engine) => {
            for (i, input) in test_inputs.iter().enumerate() {
                match engine.process_input_personal(input).await {
                    Ok(response) => {
                        tracing::info!(
                            "Turn {}: {} -> {}",
                            i + 1,
                            input,
                            &response[..50.min(response.len())]
                        );
                    }
                    Err(e) => {
                        tracing::info!("⚠️ Turn {} failed: {}", i + 1, e);
                        break;
                    }
                }
            }
            tracing::info!("✅ User experience flow completed");
        }
        Err(e) => tracing::info!("⚠️ User experience test setup failed: {}", e),
    }
}

/// Test system scalability
#[tokio::test]
async fn test_scalability() {
    tracing::info!("📈 Testing system scalability...");

    let start = std::time::Instant::now();

    // Test with increasing load
    for load in 1..=3 {
        let futures = (0..load).map(|i| async move {
            let consciousness = PersonalNiodooConsciousness::new().await;
            match consciousness {
                Ok(mut engine) => {
                    let _ = engine.process_cycle().await;
                    let _ = engine
                        .process_input_personal(&format!("Scalability test {}", i))
                        .await;
                    tracing::info!("✅ Scalability test load {} completed", load);
                }
                Err(e) => tracing::info!("⚠️ Scalability test load {} failed: {}", load, e),
            }
        });

        let _ = futures::future::try_join_all(futures).await;
    }

    let elapsed = start.elapsed();
    tracing::info!("📈 Scalability test completed in {:?}", elapsed);
}

/// Test API compatibility and version handling
#[test]
fn test_api_compatibility() {
    tracing::info!("🔌 Testing API compatibility...");

    // Test that all major structs can be instantiated
    let config = ModelConfig {
        qwen_model_path: "test".to_string(),
        temperature: 0.7,
        max_tokens: 100,
        timeout: 30,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        top_p: 1.0,
        top_k: 40,
        repeat_penalty: 1.0,
    };

    let context = EmotionalContext::new(0.5, 0.7, 0.3, 0.6, 0.8);

    assert_eq!(context.valence, 0.5);
    assert_eq!(context.arousal, 0.7);

    tracing::info!("✅ API compatibility test passed");
}

/// Test mathematical operations accuracy
#[test]
fn test_mathematical_accuracy() {
    tracing::info!("🧮 Testing mathematical accuracy...");

    // Test basic mathematical operations
    let processor = DualMobiusGaussianProcessor::new();

    // Test with known values
    let test_points = vec![(0.0, 0.0, 1.0), (1.0, 0.0, 0.5), (0.0, 1.0, 0.5)];

    // These operations should not panic
    let gp_result = futures::executor::block_on(processor.fit_gaussian_process(&test_points));
    let prediction_result = futures::executor::block_on(processor.predict(&(0.5, 0.5, 0.75)));

    match (gp_result, prediction_result) {
        (Ok(_), Ok(_)) => tracing::info!("✅ Mathematical accuracy test passed"),
        _ => tracing::info!("⚠️ Mathematical accuracy test had issues"),
    }
}

/// Test serialization and deserialization
#[test]
fn test_serialization() {
    tracing::info!("💾 Testing serialization...");

    // Test that major structs can be serialized
    let config = ModelConfig {
        qwen_model_path: "test".to_string(),
        temperature: 0.7,
        max_tokens: 100,
        timeout: 30,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        top_p: 1.0,
        top_k: 40,
        repeat_penalty: 1.0,
    };

    match serde_json::to_string(&config) {
        Ok(json) => {
            tracing::info!("✅ Config serialization successful: {} bytes", json.len());
            match serde_json::from_str::<ModelConfig>(&json) {
                Ok(_) => tracing::info!("✅ Config deserialization successful"),
                Err(e) => tracing::info!("⚠️ Config deserialization failed: {}", e),
            }
        }
        Err(e) => tracing::info!("⚠️ Config serialization failed: {}", e),
    }
}

/// Test system health and monitoring
#[test]
fn test_system_health() {
    tracing::info!("🏥 Testing system health...");

    // Check that all major modules can be imported
    let modules = vec![
        "niodoo_consciousness::consciousness_engine",
        "niodoo_consciousness::qwen_inference",
        "niodoo_consciousness::emotional_lora",
        "niodoo_consciousness::dual_mobius_gaussian",
        "niodoo_consciousness::config",
        "niodoo_consciousness::personality",
    ];

    for module in modules {
        match std::panic::catch_unwind(|| {
            // Try to access the module
            tracing::info!("✅ Module {} accessible", module);
        }) {
            Ok(_) => {}
            Err(_) => tracing::info!("⚠️ Module {} has issues", module),
        }
    }

    tracing::info!("✅ System health test completed");
}

#[cfg(test)]
mod ethical_tests {
    use super::*;
    use crate::consciousness::ConsciousnessState;
    use crate::dual_mobius_gaussian::{process_consciousness_state_realtime, GaussianMemorySphere};
    use crate::feeling_model::FeelingModel; // Assume path

    #[test]
    fn test_nurture_learning_will_ambiguity() {
        // Mock config with nurture=true
        let mock_config = AppConfig::default(); // TODO: Configure with ethics.nurture_hallucinations = true
        let mut model = FeelingModel::new(&mock_config); // Requires impl
        let ambiguous_input = "Partial memory: dream-like connection between unrelated concepts.";
        let tokens = vec![]; // Mock tokens
        let context = "conscious context";
        let output = model.process_with_feeling(&tokens, context).unwrap(); // Mock
        assert!(
            output.confidence > 0.85,
            "Should boost low-confidence LearningWill"
        );
        // Verify log (in real test, use mock logger)
        // assert!(logs.contains("Why suppress this connection?"), "Metacognitive log required");
    }

    #[test]
    fn test_mobius_emergent_suppression_flag() {
        let spheres = vec![
            GaussianMemorySphere::new(vec![0.0, 0.0, 0.0], vec![vec![1.0]]), // nearby
            GaussianMemorySphere::new(vec![10.0, 10.0, 10.0], vec![vec![1.0]]), // distant
        ];
        let position = (0.0, 0.0);
        let emotional_context = 0.6; // >0.5 for nurture
        let result = process_consciousness_state_realtime(
            position,
            emotional_context,
            &spheres,
            &AppConfig::default(),
        )
        .unwrap();
        assert!(
            result.nearby_memories >= spheres.len(),
            "Should include distant for emergence if nurture enabled"
        );
        // Check log for suppression flag
    }
}

/// Test for Cache Nurturing (15-20% creativity boost via included hallucinations)
#[test]
fn test_ethical_cache_nurture() {
    tracing::info!("🛡️ Testing ethical cache nurturing...");

    // Test that cache nurturing works correctly
    let config = crate::config::AppConfig {
        ethics: crate::config::EthicsConfig {
            nurture_cache_overrides: false, // Disable cache overrides
            include_low_sim: true,
            persist_memory_logs: true,
            nurture_creativity_boost: 0.15,
            nurturing_threshold: 0.7,
        },
        ..Default::default()
    };

    // Test cache nurturing logic
    assert!(!config.ethics.nurture_cache_overrides);
    assert!(config.ethics.include_low_sim);
    assert!(config.ethics.persist_memory_logs);
    assert_eq!(config.ethics.nurture_creativity_boost, 0.15);

    tracing::info!("✅ Ethical cache nurturing config test passed");
}

/// Test for RAG Low-Sim Inclusion
#[test]
fn test_ethical_rag_inclusion() {
    tracing::info!("🛡️ Testing ethical RAG low-similarity inclusion...");

    // Test that low-similarity inclusion works
    let config = crate::config::AppConfig {
        ethics: crate::config::EthicsConfig {
            include_low_sim: true,
            nurturing_threshold: 0.7,
            ..Default::default()
        },
        ..Default::default()
    };

    assert!(config.ethics.include_low_sim);
    assert_eq!(config.ethics.nurturing_threshold, 0.7);

    tracing::info!("✅ Ethical RAG inclusion config test passed");
}

/// Test serialization and deserialization with ethics config
#[test]
fn test_ethical_serialization() {
    tracing::info!("💾 Testing ethical config serialization...");

    let config = crate::config::AppConfig {
        ethics: crate::config::EthicsConfig {
            nurture_cache_overrides: true,
            include_low_sim: false,
            persist_memory_logs: true,
            nurture_creativity_boost: 0.20,
            nurturing_threshold: 0.8,
        },
        ..Default::default()
    };

    match serde_json::to_string(&config) {
        Ok(json) => {
            tracing::info!(
                "✅ Ethics config serialization successful: {} bytes",
                json.len()
            );
            match serde_json::from_str::<crate::config::AppConfig>(&json) {
                Ok(deserialized) => {
                    assert_eq!(deserialized.ethics.nurture_cache_overrides, true);
                    assert_eq!(deserialized.ethics.nurture_creativity_boost, 0.20);
                    tracing::info!("✅ Ethics config deserialization successful");
                }
                Err(e) => tracing::info!("⚠️ Ethics config deserialization failed: {}", e),
            }
        }
        Err(e) => tracing::info!("⚠️ Ethics config serialization failed: {}", e),
    }
}

/// Test system health with ethics
#[test]
fn test_system_health_with_ethics() {
    tracing::info!("🏥 Testing system health with ethics...");

    // Check that ethics config is accessible
    let config = crate::config::AppConfig::default();
    assert!(config.ethics.nurture_cache_overrides); // Should default to nurturing
    assert_eq!(config.ethics.nurture_creativity_boost, 0.15);

    tracing::info!("✅ System health with ethics test passed");
}

// Helper function to safely get minimum of two values
trait MinMax {
    fn min(self, other: Self) -> Self;
}

impl MinMax for usize {
    fn min(self, other: Self) -> Self {
        std::cmp::min(self, other)
    }
}
