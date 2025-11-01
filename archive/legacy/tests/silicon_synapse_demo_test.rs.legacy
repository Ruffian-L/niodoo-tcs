//! Integration test for Silicon Synapse demo
//! 
//! This test runs the demo programmatically to verify it works correctly

use niodoo_consciousness::silicon_synapse::*;
use uuid::Uuid;
use std::time::{Duration, Instant};
use tokio::time::timeout;

#[tokio::test]
async fn test_silicon_synapse_demo() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging for test
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::WARN)
        .try_init();
    
    // Load config
    let config = Config::default();
    
    // Start monitoring
    let mut synapse = SiliconSynapse::new(config).await?;
    synapse.start().await?;
    
    // Get telemetry sender
    let tx = synapse.telemetry_sender();
    
    // Simulate a few inference requests (fewer than demo for faster test)
    for i in 0..3 {
        let request_id = Uuid::new_v4();
        
        // Start inference
        tx.send(TelemetryEvent::InferenceStart {
            request_id,
            timestamp: Instant::now(),
            prompt_length: 50,
        }).await?;
        
        // Simulate token generation
        for j in 0..5 {
            tokio::time::sleep(Duration::from_millis(10)).await;
            tx.send(TelemetryEvent::TokenGenerated {
                request_id,
                timestamp: Instant::now(),
                token_index: j,
                token_length: 3,
            }).await?;
        }
        
        // Complete inference
        tx.send(TelemetryEvent::InferenceComplete {
            request_id,
            timestamp: Instant::now(),
            total_tokens: 5,
            error: None,
        }).await?;
    }
    
    // Let the system process events
    tokio::time::sleep(Duration::from_millis(100)).await;
    
    // Shutdown
    synapse.shutdown().await?;
    
    Ok(())
}

#[tokio::test]
async fn test_silicon_synapse_demo_with_timeout() -> Result<(), Box<dyn std::error::Error>> {
    // Test that the demo completes within a reasonable time
    let result = timeout(Duration::from_secs(30), async {
        // Initialize logging for test
        let _ = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::WARN)
            .try_init();
        
        // Load config
        let config = Config::default();
        
        // Start monitoring
        let mut synapse = SiliconSynapse::new(config).await?;
        synapse.start().await?;
        
        // Get telemetry sender
        let tx = synapse.telemetry_sender();
        
        // Simulate inference workload
        for i in 0..5 {
            let request_id = Uuid::new_v4();
            
            // Start inference
            tx.send(TelemetryEvent::InferenceStart {
                request_id,
                timestamp: Instant::now(),
                prompt_length: 100,
            }).await?;
            
            // Simulate token generation
            for j in 0..10 {
                tokio::time::sleep(Duration::from_millis(5)).await;
                tx.send(TelemetryEvent::TokenGenerated {
                    request_id,
                    timestamp: Instant::now(),
                    token_index: j,
                    token_length: 3,
                }).await?;
            }
            
            // Complete inference
            tx.send(TelemetryEvent::InferenceComplete {
                request_id,
                timestamp: Instant::now(),
                total_tokens: 10,
                error: None,
            }).await?;
        }
        
        // Let the system process events
        tokio::time::sleep(Duration::from_millis(200)).await;
        
        // Shutdown
        synapse.shutdown().await?;
        
        Ok::<(), Box<dyn std::error::Error>>(())
    }).await;
    
    assert!(result.is_ok(), "Demo should complete within 30 seconds");
    result?
}

#[tokio::test]
async fn test_silicon_synapse_demo_error_handling() -> Result<(), Box<dyn std::error::Error>> {
    // Test error handling in the demo
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::WARN)
        .try_init();
    
    // Load config
    let config = Config::default();
    
    // Start monitoring
    let mut synapse = SiliconSynapse::new(config).await?;
    synapse.start().await?;
    
    // Get telemetry sender
    let tx = synapse.telemetry_sender();
    
    // Simulate inference with errors
    for i in 0..3 {
        let request_id = Uuid::new_v4();
        
        // Start inference
        tx.send(TelemetryEvent::InferenceStart {
            request_id,
            timestamp: Instant::now(),
            prompt_length: 50,
        }).await?;
        
        // Simulate some token generation
        for j in 0..3 {
            tokio::time::sleep(Duration::from_millis(10)).await;
            tx.send(TelemetryEvent::TokenGenerated {
                request_id,
                timestamp: Instant::now(),
                token_index: j,
                token_length: 3,
            }).await?;
        }
        
        // Complete inference with error on second request
        let error = if i == 1 { Some("Test error".to_string()) } else { None };
        tx.send(TelemetryEvent::InferenceComplete {
            request_id,
            timestamp: Instant::now(),
            total_tokens: 3,
            error,
        }).await?;
    }
    
    // Let the system process events
    tokio::time::sleep(Duration::from_millis(100)).await;
    
    // Shutdown
    synapse.shutdown().await?;
    
    Ok(())
}
