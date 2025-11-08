//! REAL Full Pipeline A/B Test
//! Actually runs niodoo_real_integrated pipeline end-to-end

use anyhow::Result;
use std::time::Instant;
use niodoo_real_integrated::config::CliArgs;
use niodoo_real_integrated::pipeline::Pipeline;
use reqwest;

async fn check_service(url: &str, _name: &str) -> bool {
    match reqwest::get(url).await {
        Ok(resp) => resp.status().is_success(),
        Err(_) => false,
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("🔥 REAL FULL PIPELINE A/B TEST");
    println!("{}", "=".repeat(70));
    println!("This test ACTUALLY runs the niodoo_real_integrated pipeline\n");
    
    // Pre-flight checks
    println!("[PRE-FLIGHT] Checking Required Services");
    println!("{}", "-".repeat(70));
    
    let qdrant_url = std::env::var("QDRANT_URL").unwrap_or_else(|_| "http://127.0.0.1:6333".to_string());
    let vllm_url = std::env::var("VLLM_URL").unwrap_or_else(|_| "http://127.0.0.1:5001/v1/models".to_string());
    
    let mut services_ok = true;
    
    println!("Checking Qdrant at {}...", qdrant_url);
    if check_service(&format!("{}/collections", qdrant_url), "Qdrant").await {
        println!("  ✅ Qdrant is running");
    } else {
        println!("  ❌ Qdrant is NOT running");
        println!("     Start Qdrant with: docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant");
        println!("     Or set QDRANT_URL to point to a running Qdrant instance");
        services_ok = false;
    }
    
    println!("Checking vLLM at {}...", vllm_url);
    if check_service(&vllm_url, "vLLM").await {
        println!("  ✅ vLLM is running");
    } else {
        println!("  ⚠️  vLLM is NOT running (may be optional for some tests)");
        println!("     Start vLLM with: python -m vllm.entrypoints.openai.api_server --model <model_path>");
        println!("     Or set VLLM_URL to point to a running vLLM instance");
    }
    
    if !services_ok {
        println!("\n❌ CRITICAL: Required services are not available!");
        println!("   Please start the required services and try again.");
        std::process::exit(1);
    }
    
    println!("\n{}", "=".repeat(70));
    
    // FORCE REAL MODE - NO MOCKS
    std::env::set_var("MOCK_MODE", "false");
    std::env::set_var("CODE_MODE_ENABLED", "true");
    std::env::set_var("CODE_MODE_LANGUAGE", "python");
    std::env::set_var("TOPOLOGY_MODE", "baseline"); // Baseline to avoid TCS delays
    
    let mut results = Vec::new();
    
    // Test A: Pipeline Initialization
    println!("[TEST A] Pipeline Initialization");
    println!("{}", "-".repeat(70));
    let start = Instant::now();
    match tokio::time::timeout(
        std::time::Duration::from_secs(20),
        Pipeline::initialise(CliArgs::default())
    ).await {
        Ok(Ok(mut pipeline)) => {
            let init_time = start.elapsed();
            println!("✅ PASS: Pipeline initialized ({:.2}ms)", init_time.as_secs_f64() * 1000.0);
            results.push(("Pipeline Init", true, init_time));
            
            // Test B: Full Pipeline Code Generation
            println!("\n[TEST B] Full Pipeline Code Generation");
            println!("{}", "-".repeat(70));
            let gen_start = Instant::now();
            match tokio::time::timeout(
                std::time::Duration::from_secs(60),
                pipeline.process_prompt("Write a simple function to add two numbers")
            ).await {
                Ok(Ok(cycle)) => {
                    let gen_time = gen_start.elapsed();
                    let response_len = cycle.hybrid_response.len();
                    println!("✅ PASS: Full pipeline generated code ({:.2}ms)", gen_time.as_secs_f64() * 1000.0);
                    println!("   Response length: {} chars", response_len);
                    println!("   Compass quadrant: {:?}", cycle.compass.quadrant);
                    println!("   Latency: {:.2}ms", cycle.latency_ms);
                    
                    // Extract code if present
                    if response_len > 0 {
                        if let Some(code_start) = cycle.hybrid_response.find("```python") {
                            if let Some(code_end) = cycle.hybrid_response[code_start..].find("```") {
                            let code = &cycle.hybrid_response[code_start+9..code_start+code_end];
                            let preview: String = code.chars().take(100).collect();
                            println!("   Code preview: {}...", preview);
                            }
                        }
                    }
                    
                    results.push(("Full Pipeline Generation", true, gen_time));
                    
                    // Test C: CQS Calculation on Generated Code
                    println!("\n[TEST C] CQS Calculation on Generated Code");
                    println!("{}", "-".repeat(70));
                    let cqs_start = Instant::now();
                    if let Some(code_start) = cycle.hybrid_response.find("```python") {
                        if let Some(code_end) = cycle.hybrid_response[code_start..].find("```") {
                            let code = &cycle.hybrid_response[code_start+9..code_start+code_end];
                            match niodoo_real_integrated::cqs_calculator::CQSCalculator::new()
                                .compute_cqs(code, niodoo_real_integrated::config::CodeLanguage::Python, 0) {
                                Ok(cqs) => {
                                    let cqs_time = cqs_start.elapsed();
                                    println!("✅ PASS: CQS calculated ({:.2}ms)", cqs_time.as_secs_f64() * 1000.0);
                                    println!("   CQS Score: {:.2}", cqs.score);
                                    println!("   Cyclomatic: {}", cqs.cyclomatic_complexity);
                                    println!("   Cognitive: {}", cqs.cognitive_complexity);
                                    results.push(("CQS Calculation", true, cqs_time));
                                }
                                Err(e) => {
                                    println!("❌ FAIL: CQS calculation - {}", e);
                                    results.push(("CQS Calculation", false, cqs_start.elapsed()));
                                }
                            }
                        } else {
                            println!("⚠️  SKIP: Could not extract code from response");
                            results.push(("CQS Calculation", false, cqs_start.elapsed()));
                        }
                    } else {
                        println!("⚠️  SKIP: No code found in response");
                        results.push(("CQS Calculation", false, cqs_start.elapsed()));
                    }
                }
                Ok(Err(e)) => {
                    println!("❌ FAIL: Pipeline generation - {}", e);
                    results.push(("Full Pipeline Generation", false, gen_start.elapsed()));
                }
                Err(_) => {
                    println!("⏱️  TIMEOUT: Pipeline generation took >60s");
                    results.push(("Full Pipeline Generation", false, std::time::Duration::from_secs(60)));
                }
            }
        }
        Ok(Err(e)) => {
            println!("❌ FAIL: Pipeline initialization - {}", e);
            results.push(("Pipeline Init", false, start.elapsed()));
        }
        Err(_) => {
            println!("⏱️  TIMEOUT: Pipeline initialization took >20s");
            results.push(("Pipeline Init", false, std::time::Duration::from_secs(20)));
        }
    }
    
    // Summary
    println!("\n{}", "=".repeat(70));
    println!("SUMMARY");
    println!("{}", "=".repeat(70));
    
    let working: Vec<_> = results.iter().filter(|(_, ok, _)| *ok).collect();
    let broken: Vec<_> = results.iter().filter(|(_, ok, _)| !*ok).collect();
    
    println!("\n✅ WORKING ({}):", working.len());
    for (name, _, elapsed) in &working {
        println!("   {:40} - {:7.2}ms", name, elapsed.as_secs_f64() * 1000.0);
    }
    
    println!("\n❌ BROKEN/TIMEOUT ({}):", broken.len());
    for (name, _, elapsed) in &broken {
        if elapsed.as_secs() > 10 {
            println!("   {:40} - TIMEOUT ({:.1}s)", name, elapsed.as_secs_f64());
        } else {
            println!("   {:40} - FAILED", name);
        }
    }
    
    println!("\n{}", "=".repeat(70));
    
    // Exit with error if critical tests failed
    if results.iter().any(|(name, ok, _)| name == &"Pipeline Init" && !*ok) {
        eprintln!("\n❌ CRITICAL: Pipeline initialization failed - cannot test full pipeline");
        std::process::exit(1);
    }
    
    Ok(())
}

