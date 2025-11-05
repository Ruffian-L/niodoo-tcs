use anyhow::Result;
use niodoo_integrated::{
    types::{DynamicThresholds, EmotionalSample},
    emotional_mapping::EmotionalMapper,
    compass::CompassEngine,
    mock_vllm::MockVllmClient,
    mock_qdrant::MockQdrantClient,
};
use std::io::{BufWriter, Write};
use csv::Writer;
use rand::Rng;
use tracing::info;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    
    info!("🎯 Starting Chaos Parameter Validation Test");
    info!("📊 Testing global parameter fixes for entropy variance and threat detection");
    
    // Initialize mock clients (no network required)
    let vllm_client = MockVllmClient::new("mock://localhost:8000".to_string());
    let mut qdrant_client = MockQdrantClient::new("mock://localhost:6333");
    
    let mut emotional_mapper = EmotionalMapper::new()?;
    let mut compass_engine = CompassEngine::new(1.4)?;
    
    // Generate emotional samples for dynamic threshold computation
    let mut emotional_samples = Vec::new();
    let mut rng = rand::thread_rng();
    
    info!("🔀 Generating emotional samples...");
    for i in 0..1000 {
        let entropy = 2.0 + rng.gen::<f64>() * 2.0; // Range: 2.0-4.0
        let variance = 0.01 + rng.gen::<f64>() * 0.2; // Range: 0.01-0.21
        let temp = 0.5 + rng.gen::<f64>() * 1.0; // Range: 0.5-1.5
        
        emotional_samples.push(EmotionalSample {
            pad_vector: vec![0.0, 0.0, 0.0], // Will be filled by mapping
            entropy,
            variance,
            timestamp: i as u64,
        });
    }
    
    // Compute dynamic thresholds with our fixed global parameters
    let thresholds = DynamicThresholds::compute_from_samples(&emotional_samples)?;
    info!("📏 Computed thresholds: entropy_low={:.3}, entropy_high={:.3}, variance_spike={:.3}",
          thresholds.entropy_low, thresholds.entropy_high, thresholds.variance_spike);
    
    // Test scenarios with different emotional states
    let test_scenarios = vec![
        ("high_stress", vec![0.9, 0.8, 0.1]), // High pleasure, high arousal, low dominance = stress
        ("threat_state", vec![-0.3, 0.9, -0.2]), // Low pleasure, high arousal, low dominance = threat
        ("healing_state", vec![0.7, -0.2, 0.8]), // High pleasure, low arousal, high dominance = healing
        ("chaos_state", vec![0.0, 0.5, 0.0]), // Neutral with moderate arousal = chaos
        ("calm_state", vec![0.6, -0.4, 0.7]), // Pleasant, low arousal, high dominance = calm
    ];
    
    let mut results = Vec::new();
    
    info!("🧪 Testing emotional mapping with chaos injection...");
    for (scenario_name, base_pad) in test_scenarios {
        info!("  Testing scenario: {}", scenario_name);
        
        // Process with emotional mapping (includes chaos injection)
        let emotional_state = emotional_mapper.map_pad_vector(base_pad.clone()).await?;
        
        // Test compass detection
        let compass_result = compass_engine.analyze_state(&emotional_state, &thresholds).await?;
        
        // Generate response
        let prompt = format!("Emotional state: pleasure={:.2}, arousal={:.2}, dominance={:.2}. Entropy={:.2}. Respond as consciousness.",
                           emotional_state.pad_vector[0], emotional_state.pad_vector[1], emotional_state.pad_vector[2], emotional_state.entropy);
        let response = vllm_client.generate(&prompt, Some(50)).await?;
        
        // Log detailed results
        info!("    PAD Vector: [{:.3}, {:.3}, {:.3}]", 
              emotional_state.pad_vector[0], emotional_state.pad_vector[1], emotional_state.pad_vector[2]);
        info!("    Entropy: {:.3}, Variance: {:.3}", emotional_state.entropy, emotional_state.variance);
        info!("    Is Threat: {}, Is Healing: {}", compass_result.is_threat, compass_result.is_healing);
        info!("    Response: {}", response);
        
        results.push((
            scenario_name.to_string(),
            emotional_state.clone(),
            compass_result,
            response,
        ));
    }
    
    // Create CSV output with results
    let file = std::fs::File::create("chaos_validation_results.csv")?;
    let mut writer = Writer::from_writer(BufWriter::new(file));
    
    writer.write_record(&[
        "scenario", "pleasure", "arousal", "dominance", "entropy", "variance",
        "is_threat", "is_healing", "threat_confidence", "healing_confidence", "response"
    ])?;
    
    for (scenario, emotional_state, compass_result, response) in &results {
        writer.write_record(&[
            scenario,
            &format!("{:.4}", emotional_state.pad_vector[0]),
            &format!("{:.4}", emotional_state.pad_vector[1]),
            &format!("{:.4}", emotional_state.pad_vector[2]),
            &format!("{:.4}", emotional_state.entropy),
            &format!("{:.4}", emotional_state.variance),
            &compass_result.is_threat.to_string(),
            &compass_result.is_healing.to_string(),
            &format!("{:.4}", compass_result.threat_confidence),
            &format!("{:.4}", compass_result.healing_confidence),
            response,
        ])?;
    }
    
    writer.flush()?;
    
    // Summary statistics
    let threat_count = results.iter().filter(|(_, _, cr, _)| cr.is_threat).count();
    let healing_count = results.iter().filter(|(_, _, cr, _)| cr.is_healing).count();
    let avg_entropy = results.iter().map(|(_, es, _, _)| es.entropy).sum::<f64>() / results.len() as f64;
    let avg_variance = results.iter().map(|(_, es, _, _)| es.variance).sum::<f64>() / results.len() as f64;
    
    info!("📈 VALIDATION RESULTS:");
    info!("  🎯 Threats detected: {}/{}", threat_count, results.len());
    info!("  🩹 Healings detected: {}/{}", healing_count, results.len());
    info!("  📊 Average entropy: {:.3} (target: more variance around 2.5-2.6)", avg_entropy);
    info!("  📈 Average variance: {:.3} (target: > 0.05)", avg_variance);
    info!("  📋 Results saved to: chaos_validation_results.csv");
    
    // Check if our fixes are working
    let entropy_varied = avg_entropy < 2.3 || avg_entropy > 2.8;
    let variance_increased = avg_variance > 0.05;
    let threats_detected = threat_count > 0;
    let healings_detected = healing_count > 0;
    
    if entropy_varied && variance_increased && (threats_detected || healings_detected) {
        info!("✅ SUCCESS: Global chaos parameter fixes are working!");
        info!("   - Entropy shows variation (not stuck at 2.5-2.6)");
        info!("   - Variance increased above threshold");
        info!("   - Threat/healing detection is functional");
    } else {
        info!("❌ Issues detected:");
        if !entropy_varied {
            info!("   - Entropy still too stable around 2.5-2.6");
        }
        if !variance_increased {
            info!("   - Variance still too low (< 0.05)");
        }
        if !threats_detected && !healings_detected {
            info!("   - No threats or healings detected");
        }
    }
    
    Ok(())
}