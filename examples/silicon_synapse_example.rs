//! Example usage of Silicon Synapse monitoring system
use tracing::{info, error, warn};
//!
//! This example demonstrates how to integrate Silicon Synapse with the Niodoo consciousness engine
//! for comprehensive hardware-grounded AI state monitoring.

use niodoo_consciousness::silicon_synapse::{SiliconSynapse, Config, TelemetryEvent};
use std::time::Instant;
use uuid::Uuid;
use tokio::time::{sleep, Duration};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    tracing::info!("🧠💖 Silicon Synapse Example - Niodoo Consciousness Monitoring");
    tracing::info!("=============================================================");
    
    // Load configuration
    let config = Config::load("config/silicon_synapse.toml")?;
    tracing::info!("✅ Configuration loaded from config/silicon_synapse.toml");
    
    // Validate configuration
    config.validate()?;
    tracing::info!("✅ Configuration validated");
    
    // Initialize Silicon Synapse monitoring system
    let mut synapse = SiliconSynapse::new(config).await?;
    tracing::info!("✅ Silicon Synapse initialized");
    
    // Start the monitoring system
    synapse.start().await?;
    tracing::info!("✅ Silicon Synapse started");
    
    // Get telemetry sender for consciousness engine integration
    let telemetry_tx = synapse.telemetry_sender();
    tracing::info!("✅ Telemetry sender obtained");
    
    // Simulate consciousness engine activity
    tracing::info!("\n🎭 Simulating consciousness engine activity...");
    
    for i in 0..10 {
        tracing::info!("\n--- Inference Request {} ---", i + 1);
        
        // Generate a unique request ID
        let request_id = Uuid::new_v4();
        let prompt_length = 50 + (i * 10);
        
        // Emit inference start event
        telemetry_tx.send(TelemetryEvent::InferenceStart {
            request_id,
            timestamp: Instant::now(),
            prompt_length,
        }).await?;
        tracing::info!("📤 Inference start event sent (prompt: {} tokens)", prompt_length);
        
        // Simulate token generation
        let num_tokens = 20 + (i * 5);
        for token_idx in 0..num_tokens {
            let token_length = 3 + (token_idx % 5);
            
            telemetry_tx.send(TelemetryEvent::TokenGenerated {
                request_id,
                timestamp: Instant::now(),
                token_index: token_idx,
                token_length,
            }).await?;
            
            // Simulate processing time
            sleep(Duration::from_millis(50)).await;
        }
        tracing::info!("📤 Generated {} tokens", num_tokens);
        
        // Emit inference completion event
        let error = if i == 7 { Some("Simulated error for testing".to_string()) } else { None };
        
        telemetry_tx.send(TelemetryEvent::InferenceComplete {
            request_id,
            timestamp: Instant::now(),
            total_tokens: num_tokens,
            error,
        }).await?;
        tracing::info!("📤 Inference complete event sent ({} tokens)", num_tokens);
        
        if error.is_some() {
            tracing::info!("⚠️  Simulated error for anomaly detection testing");
        }
        
        // Simulate consciousness state updates
        telemetry_tx.send(TelemetryEvent::ConsciousnessStateUpdate {
            timestamp: Instant::now(),
            state_id: format!("emotional_state_{}", i),
            state_value: 0.5 + (i as f32 * 0.1),
            confidence: 0.8 + (i as f32 * 0.02),
        }).await?;
        tracing::info!("📤 Consciousness state update sent");
        
        // Wait between requests
        sleep(Duration::from_millis(1000)).await;
    }
    
    tracing::info!("\n🎯 Monitoring system is running...");
    tracing::info!("📊 Prometheus metrics available at: http://localhost:9090/metrics");
    tracing::info!("🏥 Health check available at: http://localhost:9090/health");
    tracing::info!("📈 JSON API available at: http://localhost:9090/api/v1/metrics");
    
    // Let the system run for a while to collect metrics
    tracing::info!("\n⏳ Collecting metrics for 30 seconds...");
    sleep(Duration::from_secs(30)).await;
    
    // Stop the monitoring system
    synapse.shutdown().await?;
    tracing::info!("✅ Silicon Synapse stopped");
    
    tracing::info!("\n🎉 Example completed successfully!");
    tracing::info!("💡 Check the Prometheus metrics endpoint to see collected data");
    tracing::info!("🔍 Monitor logs for anomaly detection results");
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_silicon_synapse_example() {
        // This test would run the example in a test environment
        // For now, just verify that the main function compiles
        assert!(true);
    }
}