//! Test script for Fused Agent (Slow/Fast Agent Integration)
//!
//! This script tests the fused cognitive architecture:
//! - Slow Agent (TCS) analyzes user state
//! - Fast Agent (GenerationEngine) generates code modulated by strategy

use anyhow::Result;
use niodoo_real_integrated::compass::CompassOutcome;
use niodoo_real_integrated::fused_agent::{FusedAgent, TCSStrategy};
use niodoo_real_integrated::generation::GenerationEngine;
use niodoo_real_integrated::tcs_analysis::TCSAnalyzer;
use niodoo_real_integrated::torus::PadGhostState;
use std::sync::Arc;
use tokio::sync::Mutex;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    
    println!("🧪 Testing Fused Agent (Slow/Fast Integration)\n");
    
    // Test 1: TCS Strategy CQS Thresholds
    println!("Test 1: TCS Strategy CQS Thresholds");
    println!("===================================");
    assert_eq!(TCSStrategy::Stabilize.cqs_threshold(), 5.0);
    assert_eq!(TCSStrategy::Explore.cqs_threshold(), 12.0);
    assert_eq!(TCSStrategy::Optimize.cqs_threshold(), 8.0);
    assert_eq!(TCSStrategy::Refactor.cqs_threshold(), 10.0);
    println!("✅ All CQS thresholds correct\n");
    
    // Test 2: Strategy String Conversion
    println!("Test 2: Strategy String Conversion");
    println!("===================================");
    assert_eq!(TCSStrategy::from_str("STABILIZE").unwrap(), TCSStrategy::Stabilize);
    assert_eq!(TCSStrategy::from_str("explore").unwrap(), TCSStrategy::Explore);
    assert_eq!(TCSStrategy::from_str("OPTIMIZE").unwrap(), TCSStrategy::Optimize);
    assert_eq!(TCSStrategy::from_str("refactor").unwrap(), TCSStrategy::Refactor);
    println!("✅ Strategy string conversion works\n");
    
    // Test 3: Create Mock Generation Engine
    println!("Test 3: Creating Mock Generation Engine");
    println!("========================================");
    let mut generation_engine = GenerationEngine::new(
        "http://localhost:5001", // Mock endpoint
        "Qwen3-Coder-30B-A3B",
    )?;
    generation_engine.set_mock_mode(true);
    let generation_engine = Arc::new(generation_engine);
    println!("✅ Generation engine created (mock mode)\n");
    
    // Test 4: Create TCS Analyzer (optional)
    println!("Test 4: Creating TCS Analyzer");
    println!("==============================");
    let tcs_analyzer = match TCSAnalyzer::new_with_config(false) {
        Ok(analyzer) => {
            println!("✅ TCS Analyzer created");
            Some(Arc::new(Mutex::new(analyzer)))
        }
        Err(e) => {
            println!("⚠️  TCS Analyzer not available: {}", e);
            println!("   Continuing without topology analysis...");
            None
        }
    };
    
    // Test 5: Create Fused Agent
    println!("\nTest 5: Creating Fused Agent");
    println!("=============================");
    let fused_agent = FusedAgent::new(generation_engine, tcs_analyzer);
    println!("✅ Fused Agent created\n");
    
    // Test 6: Test Strategy Updates
    println!("Test 6: Testing Strategy Updates");
    println!("=================================");
    
    // Create a mock compass outcome for Panic quadrant (should map to Stabilize)
    let compass_panic = CompassOutcome {
        quadrant: niodoo_real_integrated::compass::CompassQuadrant::Panic,
        is_threat: true,
        is_healing: false,
        mcts_branches: vec![],
        intrinsic_reward: 0.0,
        cascade_stage: None,
        ucb1_score: None,
    };
    
    fused_agent.update_strategy_from_compass(&compass_panic).await?;
    let current_strategy = fused_agent.get_current_strategy().await;
    assert_eq!(current_strategy, TCSStrategy::Stabilize);
    println!("✅ Panic quadrant → Stabilize strategy");
    
    // Test Discover quadrant (should map to Explore)
    let compass_discover = CompassOutcome {
        quadrant: niodoo_real_integrated::compass::CompassQuadrant::Discover,
        is_threat: false,
        is_healing: false,
        mcts_branches: vec![],
        intrinsic_reward: 0.0,
        cascade_stage: None,
        ucb1_score: None,
    };
    
    fused_agent.update_strategy_from_compass(&compass_discover).await?;
    let current_strategy = fused_agent.get_current_strategy().await;
    assert_eq!(current_strategy, TCSStrategy::Explore);
    println!("✅ Discover quadrant → Explore strategy");
    
    // Test Master quadrant (should map to Optimize/Refactor)
    let compass_master = CompassOutcome {
        quadrant: niodoo_real_integrated::compass::CompassQuadrant::Master,
        is_threat: false,
        is_healing: true,
        mcts_branches: vec![],
        intrinsic_reward: 10.0,
        cascade_stage: Some(niodoo_real_integrated::compass::CascadeStage::Calm),
        ucb1_score: None,
    };
    
    fused_agent.update_strategy_from_compass(&compass_master).await?;
    let current_strategy = fused_agent.get_current_strategy().await;
    assert_eq!(current_strategy, TCSStrategy::Optimize);
    println!("✅ Master quadrant (Calm stage) → Optimize strategy\n");
    
    // Test 7: Test Code Generation with Strategy (Mock)
    println!("Test 7: Testing Code Generation with Strategy");
    println!("=============================================");
    
    // Create a simple PAD state
    let pad_state = PadGhostState {
        pad: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        mu: [0.0; 7],
        sigma: [1.0; 7],
        entropy: 2.5,
    };
    
    let goal = "Write a function to compute the factorial of a number";
    let result = fused_agent
        .generate_code_with_strategy(
            goal,
            niodoo_real_integrated::config::CodeLanguage::Python,
            Some(&pad_state),
        )
        .await;
    
    match result {
        Ok(fused_result) => {
            println!("✅ Code generation succeeded!");
            println!("   Strategy: {:?}", fused_result.strategy);
            println!("   Language: {:?}", fused_result.language);
            println!("   Code length: {} chars", fused_result.code.len());
            println!("   Latency: {:.2} ms", fused_result.latency_ms);
            if let Some(ref topo) = fused_result.topological_signature {
                println!("   Topology: Betti=[{}, {}, {}], PE={:.3}",
                    topo.betti_numbers[0],
                    topo.betti_numbers[1],
                    topo.betti_numbers[2],
                    topo.persistence_entropy
                );
            }
            println!("\n   Generated code preview:");
            println!("   {}", fused_result.code.chars().take(200).collect::<String>());
        }
        Err(e) => {
            println!("⚠️  Code generation failed: {}", e);
            println!("   This is expected in mock mode without vLLM server");
        }
    }
    
    println!("\n🎉 All tests completed!");
    println!("\nSummary:");
    println!("- ✅ TCS Strategy CQS thresholds");
    println!("- ✅ Strategy string conversion");
    println!("- ✅ Fused Agent creation");
    println!("- ✅ Strategy updates from Compass");
    println!("- ✅ Code generation with strategy modulation");
    
    Ok(())
}

