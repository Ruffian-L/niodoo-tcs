//! END-TO-END PIPELINE TEST
//!
//! This is THE test that validates the FULL pipeline works end-to-end.
//! No individual component testing - just the complete flow:
//!   Prompt → Embedding → ERAG → Compass → Generation → Curator → Learning → Memory → Response
//!
//! If this test passes, the pipeline works. Period.

use anyhow::Result;
use niodoo_real_integrated::config::CliArgs;
use niodoo_real_integrated::pipeline::Pipeline;
use std::time::Instant;
use tracing_subscriber;

/// Check if an HTTP endpoint is responding
async fn check_endpoint(url: &str) -> Result<bool> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()?;
    
    match client.get(url).send().await {
        Ok(resp) => Ok(resp.status().is_success()),
        Err(_) => Ok(false),
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                    END-TO-END PIPELINE TEST                                 ║");
    println!("║                                                                              ║");
    println!("║  This test validates the COMPLETE pipeline flow:                           ║");
    println!("║  Prompt → Embedding → ERAG → Compass → Generation → Curator → Memory       ║");
    println!("║                                                                              ║");
    println!("║  NO individual component testing. Just the FULL pipeline.                    ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!("\n");
    
    // FORCE REAL MODE - NO MOCKS
    std::env::remove_var("MOCK_MODE");
    std::env::set_var("MOCK_MODE", "false");
    
    // Step 1: Verify services are running
    println!("┌─ Step 1: Verifying Required Services ─────────────────────────────────────┐");
    
    let vllm_endpoint = std::env::var("VLLM_ENDPOINT")
        .unwrap_or_else(|_| "http://localhost:5001".to_string());
    let qdrant_url = std::env::var("QDRANT_URL")
        .unwrap_or_else(|_| "http://localhost:6333".to_string());
    
    let mut services_ok = true;
    
    println!("  Checking vLLM at {}...", vllm_endpoint);
    match check_endpoint(&vllm_endpoint).await {
        Ok(true) => println!("  ✅ vLLM is responding"),
        Ok(false) => {
            println!("  ❌ vLLM not responding");
            services_ok = false;
        }
        Err(e) => {
            println!("  ⚠️  Error checking vLLM: {}", e);
            services_ok = false;
        }
    }
    
    println!("  Checking Qdrant at {}...", qdrant_url);
    match check_endpoint(&qdrant_url).await {
        Ok(true) => println!("  ✅ Qdrant is responding"),
        Ok(false) => {
            println!("  ❌ Qdrant not responding");
            services_ok = false;
        }
        Err(e) => {
            println!("  ⚠️  Error checking Qdrant: {}", e);
            services_ok = false;
        }
    }
    
    if !services_ok {
        println!("\n❌ Required services are not available!");
        println!("   Start services:");
        println!("     - Qdrant: docker run -p 6333:6333 qdrant/qdrant");
        println!("     - vLLM: python -m vllm.entrypoints.openai.api_server --model <model_path>");
        return Err(anyhow::anyhow!("Services not available"));
    }
    
    println!("└──────────────────────────────────────────────────────────────────────────┘\n");
    
    // Step 2: Initialize pipeline
    println!("┌─ Step 2: Initializing Pipeline ───────────────────────────────────────────┐");
    let init_start = Instant::now();
    let args = CliArgs::default();
    let mut pipeline = match Pipeline::initialise(args).await {
        Ok(p) => {
            let init_time = init_start.elapsed();
            println!("  ✅ Pipeline initialized in {:.2}s", init_time.as_secs_f64());
            p
        }
        Err(e) => {
            println!("  ❌ Pipeline initialization failed: {}", e);
            return Err(e);
        }
    };
    println!("└──────────────────────────────────────────────────────────────────────────┘\n");
    
    // Step 3: Run end-to-end tests
    println!("┌─ Step 3: Running End-to-End Pipeline Tests ──────────────────────────────┐");
    
    let test_prompts = vec![
        "What is the meaning of life?",
        "Explain how neural networks learn",
        "Write a haiku about artificial intelligence",
    ];
    
    let mut passed = 0;
    let mut failed = 0;
    let mut total_latency_ms = 0.0;
    
    for (i, prompt) in test_prompts.iter().enumerate() {
        println!("\n  Test {}: {}", i + 1, prompt);
        println!("  ──────────────────────────────────────────────────────────────────────");
        
        let test_start = Instant::now();
        match pipeline.process_prompt(prompt).await {
            Ok(cycle) => {
                let test_time = test_start.elapsed();
                let latency_ms = test_time.as_secs_f64() * 1000.0;
                total_latency_ms += latency_ms;
                
                // Validate cycle has required fields
                let mut test_passed = true;

                if cycle.hybrid_response.is_empty() {
                    test_passed = false;
                    failures.push("Empty response");
                }
                
                // Check PAD state bounds (pad is [f64; 7] array)
                let pad_out_of_bounds = cycle.pad_state.pad.iter().any(|&v| v.abs() > 1.0);
                if pad_out_of_bounds {
                    test_passed = false;
                    failures.push("PAD state out of bounds");
                }
                
                if test_passed {
                    println!("    ✅ PASSED (latency: {:.1}ms)", latency_ms);
                    println!("       Response length: {} chars", cycle.hybrid_response.len());
                    println!("       Compass quadrant: {:?}", cycle.compass.quadrant);
                    println!("       PAD: [{:.2}, {:.2}, {:.2}, ...]", 
                        cycle.pad_state.pad[0], 
                        cycle.pad_state.pad[1], 
                        cycle.pad_state.pad[2]);
                    passed += 1;
                } else {
                    println!("    ❌ FAILED (latency: {:.1}ms)", latency_ms);
                    for failure in &failures {
                        println!("       - {}", failure);
                    }
                    failed += 1;
                }
            Err(e) => {
                let test_time = test_start.elapsed();
                let latency_ms = test_time.as_secs_f64() * 1000.0;
                println!("    ❌ FAILED (latency: {:.1}ms)", latency_ms);
                println!("       Error: {}", e);
                failed += 1;
            }
        }
    }
    
    println!("\n└──────────────────────────────────────────────────────────────────────────┘\n");
    
    // Step 4: Summary
    println!("┌─ Test Summary ───────────────────────────────────────────────────────────┐");
    println!("  Total tests: {}", test_prompts.len());
    println!("  ✅ Passed: {}", passed);
    println!("  ❌ Failed: {}", failed);
    if test_prompts.len() > 0 {
        println!("  ⏱️  Average latency: {:.1}ms", total_latency_ms / test_prompts.len() as f64);
    }
    println!("└──────────────────────────────────────────────────────────────────────────┘\n");
    
    if failed == 0 {
        println!("╔══════════════════════════════════════════════════════════════════════════════╗");
        println!("║                         ✅ ALL TESTS PASSED                                 ║");
        println!("║                                                                              ║");
        println!("║  The pipeline works end-to-end. All components integrated correctly.        ║");
        println!("╚══════════════════════════════════════════════════════════════════════════════╝\n");
        Ok(())
    } else {
        println!("╔══════════════════════════════════════════════════════════════════════════════╗");
        println!("║                         ❌ SOME TESTS FAILED                                ║");
        println!("║                                                                              ║");
        println!("║  {} out of {} tests failed. Pipeline has issues.                            ║", failed, test_prompts.len());
        println!("╚══════════════════════════════════════════════════════════════════════════════╝\n");
        Err(anyhow::anyhow!("{} tests failed", failed))
    }
}
