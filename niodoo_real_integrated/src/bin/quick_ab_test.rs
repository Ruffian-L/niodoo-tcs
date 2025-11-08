//! Quick A/B Test: What's Working vs What's Not
//! Tests each component individually to identify bottlenecks

use anyhow::Result;
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<()> {
    println!("🔥 QUICK A/B TEST: What's Working vs What's Broken\n");
    println!("{}", "=".repeat(60));
    
    let mut results = Vec::new();
    
    // Test 1: vLLM Direct Call
    println!("\n[TEST 1] Direct vLLM API Call");
    println!("{}", "-".repeat(60));
    let start = Instant::now();
    match test_vllm_direct().await {
        Ok(_) => {
            let elapsed = start.elapsed();
            println!("✅ PASS: vLLM direct call ({:.2}ms)", elapsed.as_secs_f64() * 1000.0);
            results.push(("vLLM Direct", true, elapsed));
        }
        Err(e) => {
            let elapsed = start.elapsed();
            println!("❌ FAIL: vLLM direct call - {}", e);
            results.push(("vLLM Direct", false, elapsed));
        }
    }
    
    // Test 2: Qdrant Connection
    println!("\n[TEST 2] Qdrant Connection");
    println!("{}", "-".repeat(60));
    let start = Instant::now();
    match test_qdrant().await {
        Ok(_) => {
            let elapsed = start.elapsed();
            println!("✅ PASS: Qdrant connection ({:.2}ms)", elapsed.as_secs_f64() * 1000.0);
            results.push(("Qdrant", true, elapsed));
        }
        Err(e) => {
            let elapsed = start.elapsed();
            println!("❌ FAIL: Qdrant connection - {}", e);
            results.push(("Qdrant", false, elapsed));
        }
    }
    
    // Test 3: Pipeline Initialization (with timeout)
    println!("\n[TEST 3] Pipeline Initialization");
    println!("{}", "-".repeat(60));
    let start = Instant::now();
    match tokio::time::timeout(
        std::time::Duration::from_secs(10),
        test_pipeline_init()
    ).await {
        Ok(Ok(_)) => {
            let elapsed = start.elapsed();
            println!("✅ PASS: Pipeline init ({:.2}ms)", elapsed.as_secs_f64() * 1000.0);
            results.push(("Pipeline Init", true, elapsed));
        }
        Ok(Err(e)) => {
            let elapsed = start.elapsed();
            println!("❌ FAIL: Pipeline init - {}", e);
            results.push(("Pipeline Init", false, elapsed));
        }
        Err(_) => {
            let elapsed = start.elapsed();
            println!("⏱️  TIMEOUT: Pipeline init took >10s");
            results.push(("Pipeline Init", false, elapsed));
        }
    }
    
    // Test 4: Code Generation (if pipeline works)
    if results.iter().any(|(name, ok, _)| name == &"Pipeline Init" && *ok) {
        println!("\n[TEST 4] Code Generation via Pipeline");
        println!("{}", "-".repeat(60));
        let start = Instant::now();
        match tokio::time::timeout(
            std::time::Duration::from_secs(30),
            test_code_generation()
        ).await {
            Ok(Ok(_)) => {
                let elapsed = start.elapsed();
                println!("✅ PASS: Code generation ({:.2}ms)", elapsed.as_secs_f64() * 1000.0);
                results.push(("Code Generation", true, elapsed));
            }
            Ok(Err(e)) => {
                let elapsed = start.elapsed();
                println!("❌ FAIL: Code generation - {}", e);
                results.push(("Code Generation", false, elapsed));
            }
            Err(_) => {
                let elapsed = start.elapsed();
                println!("⏱️  TIMEOUT: Code generation took >30s");
                results.push(("Code Generation", false, elapsed));
            }
        }
    } else {
        println!("\n[TEST 4] Code Generation via Pipeline");
        println!("{}", "-".repeat(60));
        println!("⏭️  SKIPPED: Pipeline init failed");
        results.push(("Code Generation", false, std::time::Duration::ZERO));
    }
    
    // Test 5: Fused Agent Strategy
    println!("\n[TEST 5] Fused Agent Strategy Mapping");
    println!("{}", "-".repeat(60));
    let start = Instant::now();
    match test_fused_agent_strategy().await {
        Ok(_) => {
            let elapsed = start.elapsed();
            println!("✅ PASS: Fused agent strategy ({:.2}ms)", elapsed.as_secs_f64() * 1000.0);
            results.push(("Fused Agent Strategy", true, elapsed));
        }
        Err(e) => {
            let elapsed = start.elapsed();
            println!("❌ FAIL: Fused agent strategy - {}", e);
            results.push(("Fused Agent Strategy", false, elapsed));
        }
    }
    
    // Summary
    println!("\n" + &"=".repeat(60));
    println!("SUMMARY");
    println!("=".repeat(60));
    
    let working: Vec<_> = results.iter().filter(|(_, ok, _)| *ok).collect();
    let broken: Vec<_> = results.iter().filter(|(_, ok, _)| !*ok).collect();
    
    println!("\n✅ WORKING ({}):", working.len());
    for (name, _, elapsed) in &working {
        println!("   {} - {:.2}ms", name, elapsed.as_secs_f64() * 1000.0);
    }
    
    println!("\n❌ BROKEN/TIMEOUT ({}):", broken.len());
    for (name, _, elapsed) in &broken {
        if elapsed.as_secs() > 0 {
            println!("   {} - TIMEOUT ({:.2}s)", name, elapsed.as_secs_f64());
        } else {
            println!("   {} - FAILED", name);
        }
    }
    
    println!("\n" + &"=".repeat(60));
    
    Ok(())
}

async fn test_vllm_direct() -> Result<()> {
    use reqwest::Client;
    
    let client = Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()?;
    
    let vllm_endpoint = std::env::var("VLLM_ENDPOINT")
        .unwrap_or_else(|_| "http://localhost:5001".to_string());
    
    // Test models endpoint
    let resp = client
        .get(&format!("{}/v1/models", vllm_endpoint))
        .send()
        .await?;
    
    if !resp.status().is_success() {
        return Err(anyhow::anyhow!("Models endpoint returned {}", resp.status()));
    }
    
    // Test completion
    let completion_resp = client
        .post(&format!("{}/v1/completions", vllm_endpoint))
        .json(&serde_json::json!({
            "model": "/workspace/models/Qwen3-Coder-30B-A3B",
            "prompt": "def hello",
            "max_tokens": 20,
            "temperature": 0.7
        }))
        .send()
        .await?;
    
    if !completion_resp.status().is_success() {
        return Err(anyhow::anyhow!("Completion endpoint returned {}", completion_resp.status()));
    }
    
    Ok(())
}

async fn test_qdrant() -> Result<()> {
    use reqwest::Client;
    
    let client = Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()?;
    
    let qdrant_url = std::env::var("QDRANT_URL")
        .unwrap_or_else(|_| "http://localhost:6333".to_string());
    
    let resp = client
        .get(&format!("{}/collections", qdrant_url))
        .send()
        .await?;
    
    if !resp.status().is_success() {
        return Err(anyhow::anyhow!("Collections endpoint returned {}", resp.status()));
    }
    
    Ok(())
}

async fn test_pipeline_init() -> Result<()> {
    use niodoo_real_integrated::config::CliArgs;
    use niodoo_real_integrated::pipeline::Pipeline;
    
    // Force real mode
    std::env::set_var("MOCK_MODE", "false");
    std::env::set_var("CODE_MODE_ENABLED", "true");
    std::env::set_var("CODE_MODE_LANGUAGE", "python");
    std::env::set_var("TOPOLOGY_MODE", "baseline"); // Use baseline to avoid TCS analyzer delays
    
    let args = CliArgs::default();
    let _pipeline = Pipeline::initialise(args).await?;
    
    Ok(())
}

async fn test_code_generation() -> Result<()> {
    use niodoo_real_integrated::config::CliArgs;
    use niodoo_real_integrated::pipeline::Pipeline;
    
    // Force real mode
    std::env::set_var("MOCK_MODE", "false");
    std::env::set_var("CODE_MODE_ENABLED", "true");
    std::env::set_var("CODE_MODE_LANGUAGE", "python");
    std::env::set_var("TOPOLOGY_MODE", "baseline");
    
    let args = CliArgs::default();
    let mut pipeline = Pipeline::initialise(args).await?;
    
    // Simple code generation test
    let result = pipeline.process_prompt("Write a function to add two numbers").await?;
    
    if result.hybrid_response.is_empty() {
        return Err(anyhow::anyhow!("Empty response from pipeline"));
    }
    
    println!("   Generated {} chars", result.hybrid_response.len());
    
    Ok(())
}

async fn test_fused_agent_strategy() -> Result<()> {
    use niodoo_real_integrated::compass::{CompassOutcome, CompassQuadrant};
    use niodoo_real_integrated::fused_agent::TCSStrategy;
    
    // Test strategy mapping
    let compass_panic = CompassOutcome {
        quadrant: CompassQuadrant::Panic,
        is_threat: true,
        is_healing: false,
        mcts_branches: vec![],
        intrinsic_reward: -1.0,
        cascade_stage: None,
        ucb1_score: None,
    };
    
    let strategy: TCSStrategy = (&compass_panic).into();
    if strategy != TCSStrategy::Stabilize {
        return Err(anyhow::anyhow!("Panic quadrant should map to Stabilize, got {:?}", strategy));
    }
    
    let compass_discover = CompassOutcome {
        quadrant: CompassQuadrant::Discover,
        is_threat: false,
        is_healing: true,
        mcts_branches: vec![],
        intrinsic_reward: 5.0,
        cascade_stage: None,
        ucb1_score: None,
    };
    
    let strategy: TCSStrategy = (&compass_discover).into();
    if strategy != TCSStrategy::Explore {
        return Err(anyhow::anyhow!("Discover quadrant should map to Explore, got {:?}", strategy));
    }
    
    Ok(())
}

