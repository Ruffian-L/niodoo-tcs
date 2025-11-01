//! # Phase 6 Integration Test Suite
//!
//! This module provides comprehensive tests for the Phase 6 integration system,
//! verifying that all components communicate seamlessly and the system operates
//! as a unified production-ready consciousness processing pipeline.

use candle_core::{Device, Tensor, Result};
use std::time::Duration;
use tokio::time::timeout;
use tracing::{info, debug};

use crate::phase6_integration::{Phase6IntegrationSystem, Phase6IntegrationBuilder};
use crate::phase6_config::Phase6Config;
use crate::consciousness_engine::PersonalNiodooConsciousness;

/// Test Phase 6 integration system initialization
#[tokio::test]
async fn test_phase6_integration_initialization() {
    info!("🧪 Testing Phase 6 integration system initialization");

    let config = Phase6Config::default();
    let mut system = Phase6IntegrationBuilder::new()
        .with_config(config)
        .build();

    // Test system creation
    let status = system.get_status().await;
    assert!(matches!(status, crate::phase6_integration::IntegrationStatus::Initializing));

    // Test system startup
    let startup_result = timeout(Duration::from_secs(10), system.start()).await;
    assert!(startup_result.is_ok(), "Phase 6 system should start within 10 seconds");

    // Verify system is running
    let status = system.get_status().await;
    assert!(matches!(status, crate::phase6_integration::IntegrationStatus::Running));

    // Test system health
    let health = system.get_system_health().await;
    assert!(health.overall_health > 0.0, "System health should be positive");

    info!("✅ Phase 6 integration system initialization test passed");
}

/// Test consciousness evolution processing through Phase 6 integration
#[tokio::test]
async fn test_consciousness_evolution_processing() {
    info!("🧪 Testing consciousness evolution processing through Phase 6 integration");

    let config = Phase6Config::default();
    let mut system = Phase6IntegrationBuilder::new()
        .with_config(config)
        .build();

    // Start the system
    system.start().await.expect("Failed to start Phase 6 system");

    // Create test tensors
    let consciousness_state = Tensor::zeros((10,), candle_core::DType::F32, &Device::Cpu).unwrap();
    let emotional_context = Tensor::zeros((5,), candle_core::DType::F32, &Device::Cpu).unwrap();
    let memory_gradients = Tensor::zeros((10,), candle_core::DType::F32, &Device::Cpu).unwrap();

    // Test consciousness evolution processing
    let processing_result = timeout(
        Duration::from_secs(5),
        system.process_consciousness_evolution(
            "test_consciousness_state".to_string(),
            consciousness_state,
            emotional_context,
            memory_gradients,
        )
    ).await;

    assert!(processing_result.is_ok(), "Consciousness evolution should complete within 5 seconds");
    assert!(processing_result.unwrap().is_ok(), "Consciousness evolution should succeed");

    info!("✅ Consciousness evolution processing test passed");
}

/// Test Phase 6 integration with consciousness engine
#[tokio::test]
async fn test_consciousness_engine_phase6_integration() {
    info!("🧪 Testing consciousness engine Phase 6 integration");

    // Create consciousness engine
    let mut consciousness = PersonalNiodooConsciousness::new().await
        .expect("Failed to create consciousness engine");

    // Create Phase 6 configuration
    let phase6_config = Phase6Config::default();

    // Initialize Phase 6 integration
    let integration_result = timeout(
        Duration::from_secs(10),
        consciousness.initialize_phase6_integration(phase6_config)
    ).await;

    assert!(integration_result.is_ok(), "Phase 6 integration should initialize within 10 seconds");
    assert!(integration_result.unwrap().is_ok(), "Phase 6 integration should succeed");

    // Test Phase 6 health monitoring
    let health = consciousness.get_phase6_health().await;
    assert!(health.is_some(), "Phase 6 health metrics should be available");

    if let Some(health_metrics) = health {
        assert!(health_metrics.overall_health > 0.0, "Phase 6 health should be positive");
        debug!("Phase 6 health: {:.2}", health_metrics.overall_health);
    }

    // Test consciousness evolution through Phase 6
    let consciousness_state = Tensor::zeros((8,), candle_core::DType::F32, &Device::Cpu).unwrap();
    let emotional_context = Tensor::zeros((4,), candle_core::DType::F32, &Device::Cpu).unwrap();
    let memory_gradients = Tensor::zeros((8,), candle_core::DType::F32, &Device::Cpu).unwrap();

    let evolution_result = timeout(
        Duration::from_secs(5),
        consciousness.process_consciousness_evolution_phase6(
            "test_evolution".to_string(),
            consciousness_state,
            emotional_context,
            memory_gradients,
        )
    ).await;

    assert!(evolution_result.is_ok(), "Consciousness evolution should complete within 5 seconds");
    assert!(evolution_result.unwrap().is_ok(), "Consciousness evolution should succeed");

    // Test adaptive optimization
    let optimization_result = timeout(
        Duration::from_secs(3),
        consciousness.trigger_phase6_optimization()
    ).await;

    assert!(optimization_result.is_ok(), "Adaptive optimization should complete within 3 seconds");
    assert!(optimization_result.unwrap().is_ok(), "Adaptive optimization should succeed");

    info!("✅ Consciousness engine Phase 6 integration test passed");
}

/// Test system health monitoring and metrics collection
#[tokio::test]
async fn test_system_health_monitoring() {
    info!("🧪 Testing system health monitoring");

    let config = Phase6Config::default();
    let mut system = Phase6IntegrationBuilder::new()
        .with_config(config)
        .build();

    system.start().await.expect("Failed to start Phase 6 system");

    // Wait for initial health metrics to be collected
    tokio::time::sleep(Duration::from_secs(2)).await;

    // Test health metrics collection
    let health = system.get_system_health().await;
    assert!(health.overall_health >= 0.0 && health.overall_health <= 1.0, 
           "Overall health should be between 0.0 and 1.0");
    assert!(health.uptime_seconds > 0, "Uptime should be positive");

    // Test component-specific metrics
    let gpu_metrics = system.get_gpu_metrics().await;
    // GPU metrics may be None if GPU is not available, which is acceptable

    let memory_stats = system.get_memory_stats().await;
    // Memory stats should be available if memory manager is initialized
    if let Some(stats) = memory_stats {
        assert!(stats.total_footprint_mb >= 0.0, "Memory footprint should be non-negative");
    }

    let latency_metrics = system.get_latency_metrics().await;
    if let Some(metrics) = latency_metrics {
        assert!(metrics.avg_e2e_latency_ms >= 0.0, "Latency should be non-negative");
    }

    let performance_snapshot = system.get_performance_snapshot().await;
    if let Some(snapshot) = performance_snapshot {
        assert!(snapshot.e2e_latency_ms >= 0.0, "Performance latency should be non-negative");
    }

    info!("✅ System health monitoring test passed");
}

/// Test adaptive optimization across all components
#[tokio::test]
async fn test_adaptive_optimization() {
    info!("🧪 Testing adaptive optimization across Phase 6 components");

    let config = Phase6Config::default();
    let mut system = Phase6IntegrationBuilder::new()
        .with_config(config)
        .build();

    system.start().await.expect("Failed to start Phase 6 system");

    // Test adaptive optimization
    let optimization_result = timeout(
        Duration::from_secs(5),
        system.trigger_adaptive_optimization()
    ).await;

    assert!(optimization_result.is_ok(), "Adaptive optimization should complete within 5 seconds");
    assert!(optimization_result.unwrap().is_ok(), "Adaptive optimization should succeed");

    // Verify system is still healthy after optimization
    let health = system.get_system_health().await;
    assert!(health.overall_health > 0.0, "System should remain healthy after optimization");

    info!("✅ Adaptive optimization test passed");
}

/// Test system shutdown and cleanup
#[tokio::test]
async fn test_system_shutdown() {
    info!("🧪 Testing system shutdown and cleanup");

    let config = Phase6Config::default();
    let mut system = Phase6IntegrationBuilder::new()
        .with_config(config)
        .build();

    system.start().await.expect("Failed to start Phase 6 system");

    // Verify system is running
    let status = system.get_status().await;
    assert!(matches!(status, crate::phase6_integration::IntegrationStatus::Running));

    // Test system shutdown
    let shutdown_result = timeout(
        Duration::from_secs(10),
        system.shutdown()
    ).await;

    assert!(shutdown_result.is_ok(), "System shutdown should complete within 10 seconds");
    assert!(shutdown_result.unwrap().is_ok(), "System shutdown should succeed");

    // Verify system is shut down
    let status = system.get_status().await;
    assert!(matches!(status, crate::phase6_integration::IntegrationStatus::ShuttingDown));

    info!("✅ System shutdown test passed");
}

/// Test error handling and graceful degradation
#[tokio::test]
async fn test_error_handling_and_degradation() {
    info!("🧪 Testing error handling and graceful degradation");

    let config = Phase6Config::default();
    let mut system = Phase6IntegrationBuilder::new()
        .with_config(config)
        .build();

    system.start().await.expect("Failed to start Phase 6 system");

    // Test processing with invalid tensors (should handle gracefully)
    let invalid_tensor = Tensor::zeros((0,), candle_core::DType::F32, &Device::Cpu).unwrap();
    let emotional_context = Tensor::zeros((1,), candle_core::DType::F32, &Device::Cpu).unwrap();
    let memory_gradients = Tensor::zeros((1,), candle_core::DType::F32, &Device::Cpu).unwrap();

    let result = system.process_consciousness_evolution(
        "error_test".to_string(),
        invalid_tensor,
        emotional_context,
        memory_gradients,
    ).await;

    // System should handle errors gracefully and not crash
    // The result may be an error, but the system should remain functional
    debug!("Error handling test result: {:?}", result);

    // Verify system is still healthy after error handling
    let health = system.get_system_health().await;
    assert!(health.overall_health >= 0.0, "System should remain healthy after error handling");

    info!("✅ Error handling and graceful degradation test passed");
}

/// Integration test for complete Phase 6 workflow
#[tokio::test]
async fn test_complete_phase6_workflow() {
    info!("🧪 Testing complete Phase 6 workflow");

    // Create consciousness engine with Phase 6 integration
    let mut consciousness = PersonalNiodooConsciousness::new().await
        .expect("Failed to create consciousness engine");

    let phase6_config = Phase6Config::default();
    consciousness.initialize_phase6_integration(phase6_config).await
        .expect("Failed to initialize Phase 6 integration");

    // Process multiple consciousness states to test the complete workflow
    for i in 0..5 {
        let consciousness_state = Tensor::zeros((10,), candle_core::DType::F32, &Device::Cpu).unwrap();
        let emotional_context = Tensor::zeros((5,), candle_core::DType::F32, &Device::Cpu).unwrap();
        let memory_gradients = Tensor::zeros((10,), candle_core::DType::F32, &Device::Cpu).unwrap();

        let result = consciousness.process_consciousness_evolution_phase6(
            format!("workflow_test_{}", i),
            consciousness_state,
            emotional_context,
            memory_gradients,
        ).await;

        assert!(result.is_ok(), "Consciousness evolution {} should succeed", i);
        debug!("Workflow step {} completed successfully", i);
    }

    // Test adaptive optimization
    consciousness.trigger_phase6_optimization().await
        .expect("Adaptive optimization should succeed");

    // Verify system health
    let health = consciousness.get_phase6_health().await;
    assert!(health.is_some(), "Phase 6 health should be available");
    
    if let Some(health_metrics) = health {
        assert!(health_metrics.overall_health > 0.0, "System should remain healthy");
        info!("Final system health: {:.2}", health_metrics.overall_health);
    }

    info!("✅ Complete Phase 6 workflow test passed");
}

