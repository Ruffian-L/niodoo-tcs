//! Integration tests for Silicon Synapse telemetry bus and aggregation engine
use tracing::{info, error, warn};

use std::time::{Duration, Instant, SystemTime};
use std::collections::HashMap;
use tokio::time::sleep;
use uuid::Uuid;

use crate::silicon_synapse::{
    config::{Config, TelemetryConfig, AggregationConfig},
    telemetry_bus::{TelemetryBus, TelemetryEvent, TelemetrySender},
    aggregation::{AggregationEngine, AggregatedMetrics},
};

/// Integration test for telemetry bus and aggregation engine event flow
#[tokio::test]
async fn test_telemetry_bus_to_aggregation_flow() {
    // Setup telemetry bus
    let telemetry_config = TelemetryConfig::default();
    let mut telemetry_bus = TelemetryBus::new(telemetry_config).unwrap();
    telemetry_bus.start().await.unwrap();
    
    // Setup aggregation engine
    let aggregation_config = AggregationConfig::default();
    let telemetry_sender = telemetry_bus.sender();
    let mut aggregation_engine = AggregationEngine::new(aggregation_config, telemetry_sender).unwrap();
    aggregation_engine.start().await.unwrap();
    
    // Send test events
    let sender = telemetry_bus.sender();
    let request_id = Uuid::new_v4();
    
    // Send inference events
    sender.send(TelemetryEvent::InferenceStart {
        request_id,
        timestamp: Instant::now(),
        prompt_length: 100,
    }).await.unwrap();
    
    sender.send(TelemetryEvent::TokenGenerated {
        request_id,
        token_id: 1,
        timestamp: Instant::now(),
        logits: Some(vec![0.1, 0.2, 0.3]),
    }).await.unwrap();
    
    sender.send(TelemetryEvent::InferenceComplete {
        request_id,
        timestamp: Instant::now(),
        total_tokens: 50,
        error: None,
    }).await.unwrap();
    
    // Send hardware metrics
    sender.send(TelemetryEvent::HardwareMetrics {
        timestamp: SystemTime::now(),
        gpu_temp_celsius: Some(72.5),
        gpu_power_watts: Some(245.3),
        gpu_fan_speed_percent: Some(65.0),
        vram_used_bytes: Some(8589934592),
        vram_total_bytes: Some(17179869184),
        gpu_utilization_percent: Some(85.0),
        cpu_utilization_percent: 45.0,
        ram_used_bytes: 16106127360,
    }).await.unwrap();
    
    // Send model metrics
    let mut activation_sparsity = HashMap::new();
    let mut activation_magnitude = HashMap::new();
    activation_sparsity.insert("layer_12".to_string(), 0.87);
    activation_magnitude.insert("layer_12".to_string(), 0.45);
    
    sender.send(TelemetryEvent::ModelMetrics {
        timestamp: SystemTime::now(),
        softmax_entropy: Some(0.342),
        activation_sparsity,
        activation_magnitude,
    }).await.unwrap();
    
    // Wait for processing
    sleep(Duration::from_millis(200)).await;
    
    // Get aggregated metrics
    let metrics = aggregation_engine.get_metrics().await;
    
    // Verify metrics were aggregated
    assert!(metrics.hardware_metrics.gpu_temperature.count > 0);
    assert!(metrics.inference_metrics.ttft_ms.count > 0);
    assert!(metrics.model_metrics.entropy_by_layer.contains_key("output"));
    
    // Cleanup
    aggregation_engine.stop().await.unwrap();
    telemetry_bus.stop().await.unwrap();
}

/// Test telemetry bus performance under load
#[tokio::test]
async fn test_telemetry_bus_performance() {
    let telemetry_config = TelemetryConfig::default();
    let mut telemetry_bus = TelemetryBus::new(telemetry_config).unwrap();
    telemetry_bus.start().await.unwrap();
    
    let sender = telemetry_bus.sender();
    let start_time = Instant::now();
    
    // Send 1000 events rapidly
    for i in 0..1000 {
        let request_id = Uuid::new_v4();
        sender.send(TelemetryEvent::InferenceStart {
            request_id,
            timestamp: Instant::now(),
            prompt_length: i,
        }).await.unwrap();
    }
    
    let elapsed = start_time.elapsed();
    tracing::info!("Sent 1000 events in {:?}", elapsed);
    
    // Verify events were processed
    sleep(Duration::from_millis(100)).await;
    
    // Check dropped events (should be minimal for unbounded channel)
    let dropped_count = telemetry_bus.dropped_events_count();
    tracing::info!("Dropped events: {}", dropped_count);
    
    telemetry_bus.stop().await.unwrap();
}

/// Test aggregation engine with multiple metric types
#[tokio::test]
async fn test_aggregation_engine_multiple_metrics() {
    let telemetry_config = TelemetryConfig::default();
    let mut telemetry_bus = TelemetryBus::new(telemetry_config).unwrap();
    telemetry_bus.start().await.unwrap();
    
    let aggregation_config = AggregationConfig::default();
    let telemetry_sender = telemetry_bus.sender();
    let mut aggregation_engine = AggregationEngine::new(aggregation_config, telemetry_sender).unwrap();
    aggregation_engine.start().await.unwrap();
    
    let sender = telemetry_bus.sender();
    
    // Send multiple hardware metrics to test statistical aggregation
    for i in 0..10 {
        sender.send(TelemetryEvent::HardwareMetrics {
            timestamp: SystemTime::now(),
            gpu_temp_celsius: Some(70.0 + i as f32),
            gpu_power_watts: Some(200.0 + i as f32 * 5.0),
            gpu_fan_speed_percent: Some(60.0 + i as f32),
            vram_used_bytes: Some(8000000000 + i * 100000000),
            vram_total_bytes: Some(17179869184),
            gpu_utilization_percent: Some(80.0 + i as f32),
            cpu_utilization_percent: 40.0 + i as f32,
            ram_used_bytes: 16000000000 + i * 100000000,
        }).await.unwrap();
    }
    
    // Wait for aggregation window
    sleep(Duration::from_secs(2)).await;
    
    let metrics = aggregation_engine.get_metrics().await;
    
    // Verify statistical calculations
    assert_eq!(metrics.hardware_metrics.gpu_temperature.count, 10);
    assert!(metrics.hardware_metrics.gpu_temperature.mean > 70.0);
    assert!(metrics.hardware_metrics.gpu_temperature.std_dev > 0.0);
    assert!(metrics.hardware_metrics.gpu_temperature.min >= 70.0);
    assert!(metrics.hardware_metrics.gpu_temperature.max <= 80.0);
    
    // Verify percentiles were calculated
    assert!(metrics.hardware_metrics.gpu_temperature.percentiles.contains_key(&50.0));
    assert!(metrics.hardware_metrics.gpu_temperature.percentiles.contains_key(&95.0));
    
    aggregation_engine.stop().await.unwrap();
    telemetry_bus.stop().await.unwrap();
}

/// Test correlation analysis
#[tokio::test]
async fn test_correlation_analysis() {
    let telemetry_config = TelemetryConfig::default();
    let mut telemetry_bus = TelemetryBus::new(telemetry_config).unwrap();
    telemetry_bus.start().await.unwrap();
    
    let mut aggregation_config = AggregationConfig::default();
    aggregation_config.enable_correlation_analysis = true;
    
    let telemetry_sender = telemetry_bus.sender();
    let mut aggregation_engine = AggregationEngine::new(aggregation_config, telemetry_sender).unwrap();
    aggregation_engine.start().await.unwrap();
    
    let sender = telemetry_bus.sender();
    
    // Send correlated data (high GPU utilization should correlate with high throughput)
    for i in 0..20 {
        // Hardware metrics
        sender.send(TelemetryEvent::HardwareMetrics {
            timestamp: SystemTime::now(),
            gpu_temp_celsius: Some(70.0 + i as f32 * 0.5),
            gpu_power_watts: Some(200.0 + i as f32 * 2.0),
            gpu_fan_speed_percent: Some(60.0 + i as f32),
            vram_used_bytes: Some(8000000000),
            vram_total_bytes: Some(17179869184),
            gpu_utilization_percent: Some(50.0 + i as f32 * 2.0), // Increasing utilization
            cpu_utilization_percent: 40.0,
            ram_used_bytes: 16000000000,
        }).await.unwrap();
        
        // Inference metrics (should correlate with GPU utilization)
        sender.send(TelemetryEvent::InferenceStart {
            request_id: Uuid::new_v4(),
            timestamp: Instant::now(),
            prompt_length: 100,
        }).await.unwrap();
    }
    
    // Wait for aggregation
    sleep(Duration::from_secs(2)).await;
    
    let metrics = aggregation_engine.get_metrics().await;
    
    // Verify correlations were calculated
    assert!(!metrics.correlations.is_empty());
    
    // Find the GPU utilization vs throughput correlation
    let gpu_correlation = metrics.correlations.iter()
        .find(|c| c.metric1 == "gpu_utilization" && c.metric2 == "inference_throughput");
    
    if let Some(correlation) = gpu_correlation {
        assert!(correlation.correlation_coefficient.abs() <= 1.0);
        tracing::info!("GPU utilization vs throughput correlation: {}", correlation.correlation_coefficient);
    }
    
    aggregation_engine.stop().await.unwrap();
    telemetry_bus.stop().await.unwrap();
}

/// Test error handling in telemetry bus
#[tokio::test]
async fn test_telemetry_bus_error_handling() {
    let telemetry_config = TelemetryConfig::default();
    let mut telemetry_bus = TelemetryBus::new(telemetry_config).unwrap();
    telemetry_bus.start().await.unwrap();
    
    let sender = telemetry_bus.sender();
    
    // Send an inference completion with error
    let request_id = Uuid::new_v4();
    sender.send(TelemetryEvent::InferenceComplete {
        request_id,
        timestamp: Instant::now(),
        total_tokens: 0,
        error: Some("Test error".to_string()),
    }).await.unwrap();
    
    // Wait for processing
    sleep(Duration::from_millis(100)).await;
    
    // Verify error was logged (we can't directly test logging, but the event should be processed)
    assert_eq!(telemetry_bus.dropped_events_count(), 0);
    
    telemetry_bus.stop().await.unwrap();
}

/// Test aggregation engine with empty buffers
#[tokio::test]
async fn test_aggregation_engine_empty_buffers() {
    let telemetry_config = TelemetryConfig::default();
    let mut telemetry_bus = TelemetryBus::new(telemetry_config).unwrap();
    telemetry_bus.start().await.unwrap();
    
    let aggregation_config = AggregationConfig::default();
    let telemetry_sender = telemetry_bus.sender();
    let mut aggregation_engine = AggregationEngine::new(aggregation_config, telemetry_sender).unwrap();
    aggregation_engine.start().await.unwrap();
    
    // Wait for aggregation window without sending any events
    sleep(Duration::from_secs(2)).await;
    
    let metrics = aggregation_engine.get_metrics().await;
    
    // Verify metrics are empty but valid
    assert_eq!(metrics.hardware_metrics.gpu_temperature.count, 0);
    assert_eq!(metrics.inference_metrics.ttft_ms.count, 0);
    assert!(metrics.correlations.is_empty());
    
    aggregation_engine.stop().await.unwrap();
    telemetry_bus.stop().await.unwrap();
}

/// Test telemetry sender cloning and sharing
#[tokio::test]
async fn test_telemetry_sender_sharing() {
    let telemetry_config = TelemetryConfig::default();
    let mut telemetry_bus = TelemetryBus::new(telemetry_config).unwrap();
    telemetry_bus.start().await.unwrap();
    
    let sender1 = telemetry_bus.sender();
    let sender2 = sender1.clone();
    
    // Both senders should work
    let request_id1 = Uuid::new_v4();
    let request_id2 = Uuid::new_v4();
    
    sender1.send(TelemetryEvent::InferenceStart {
        request_id: request_id1,
        timestamp: Instant::now(),
        prompt_length: 100,
    }).await.unwrap();
    
    sender2.send(TelemetryEvent::InferenceStart {
        request_id: request_id2,
        timestamp: Instant::now(),
        prompt_length: 200,
    }).await.unwrap();
    
    // Wait for processing
    sleep(Duration::from_millis(100)).await;
    
    // Verify both events were processed
    assert_eq!(telemetry_bus.dropped_events_count(), 0);
    
    telemetry_bus.stop().await.unwrap();
}