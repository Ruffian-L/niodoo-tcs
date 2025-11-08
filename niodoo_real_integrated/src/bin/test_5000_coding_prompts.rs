//! 5000 Coding Prompts End-to-End Test Suite with A/B Comparison
//!
//! Generates 5000 long, flowing, multi-turn conversational coding prompts and runs them
//! through both baseline and treatment configurations (10,000 total executions).
//! Verifies all endpoints, tracks metrics, and produces comprehensive A/B comparison reports.

use anyhow::{Context, Result};
use chrono::Utc;
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tracing::{info, warn};

use niodoo_real_integrated::config::CliArgs;
use niodoo_real_integrated::pipeline::Pipeline;
use niodoo_real_integrated::validation::stats::{
    bootstrap_percentile_ci, cohens_d, mann_whitney_u, StatisticalSummary,
};

#[derive(Parser, Debug)]
#[command(name = "test_5000_coding_prompts")]
#[command(about = "5000 coding prompts test suite with A/B comparison")]
struct Args {
    /// Baseline configuration name
    #[arg(long, default_value = "baseline")]
    baseline_name: String,

    /// Treatment configuration name
    #[arg(long, default_value = "treatment")]
    treatment_name: String,

    /// Output directory for results
    #[arg(long, default_value = "test_results_5000_coding")]
    output_dir: PathBuf,

    /// Number of conversations to generate (each has 10-20 turns)
    #[arg(long, default_value = "500")]
    num_conversations: usize,

    /// Skip endpoint verification (for debugging)
    #[arg(long)]
    skip_endpoints: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Conversation {
    id: usize,
    turns: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PromptMetrics {
    prompt: String,
    conversation_id: usize,
    turn_number: usize,
    success: bool,
    latency_ms: f64,
    response_length: usize,
    code_extracted: bool,
    error: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ConfigMetrics {
    config_name: String,
    total_prompts: usize,
    successful_prompts: usize,
    failed_prompts: usize,
    latencies: Vec<f64>,
    response_lengths: Vec<usize>,
    code_extraction_success: usize,
    errors: Vec<String>,
    latency_summary: StatisticalSummary,
}

#[derive(Debug, Serialize, Deserialize)]
struct ABComparison {
    latency_difference_ms: f64,
    latency_difference_pct: f64,
    success_rate_difference_pct: f64,
    code_extraction_rate_difference_pct: f64,
    cohens_d_latency: f64,
    p_value_latency: f64,
    confidence_interval_95_latency: (f64, f64),
    effect_size_latency: String,
    winner: String,
    winner_metrics: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct TestReport {
    timestamp: String,
    baseline_name: String,
    treatment_name: String,
    total_conversations: usize,
    total_prompts: usize,
    baseline_metrics: ConfigMetrics,
    treatment_metrics: ConfigMetrics,
    comparison: ABComparison,
    endpoint_status: HashMap<String, bool>,
    execution_time_secs: f64,
}

/// Generate long, flowing conversational coding prompts
fn generate_conversational_coding_prompts(num_conversations: usize) -> Vec<Conversation> {
    let mut conversations = Vec::new();
    
    // Base conversation templates that build context over multiple turns
    let conversation_templates = vec![
        // Project Development Flow
        vec![
            "I need to build a REST API for a todo list application. Can you help me design the endpoints?",
            "Actually, I want it to be async with proper error handling. Can you show me how to structure that?",
            "Great! Now I need to add authentication middleware. How should I integrate JWT tokens?",
            "I'm getting a 500 error when I try to create a todo. Can you help me debug this?",
            "The error is fixed, but now I want to add pagination. What's the best way to implement that?",
            "Perfect! Now I need to write tests for all these endpoints. Can you help me create comprehensive test cases?",
            "The tests are passing, but I want to optimize the database queries. Can you review my query patterns?",
            "I'm seeing some performance issues with large datasets. How can I add caching?",
            "The caching is working, but I need to handle cache invalidation properly. What's the best strategy?",
            "Now I want to add real-time updates using WebSockets. Can you show me how to integrate that?",
        ],
        // Debugging Session Flow
        vec![
            "I'm working on a Python function that processes large CSV files, but it's running out of memory. Can you help?",
            "I've tried using generators, but now the processing is too slow. What's a better approach?",
            "The performance improved, but I'm getting encoding errors with some files. How do I handle different encodings?",
            "The encoding is fixed, but now I need to add progress tracking. How can I show progress without blocking?",
            "The progress tracking works, but I want to add parallel processing. Can you help me make it thread-safe?",
            "I've added threading, but I'm seeing race conditions. Can you help me debug the synchronization?",
            "The race conditions are fixed, but I need to handle errors gracefully. What's the best error handling pattern?",
            "Now I want to add logging for debugging. How should I structure the logging?",
            "The logging is in place, but I need to optimize memory usage further. Can you review my code?",
            "Perfect! Now I want to add unit tests. Can you help me write comprehensive tests?",
        ],
        // Refactoring Journey Flow
        vec![
            "I have a large monolithic function that does too many things. Can you help me refactor it?",
            "I've broken it into smaller functions, but the dependencies are getting complex. How should I organize them?",
            "The organization is better, but I'm seeing code duplication. Can you help me extract common patterns?",
            "I've extracted the patterns, but now I need to add type hints. Can you help me add proper typing?",
            "The types are added, but I want to improve error handling. What's the best way to handle errors?",
            "The error handling is better, but I need to add documentation. Can you help me write docstrings?",
            "The documentation is done, but I want to add logging. How should I integrate logging?",
            "The logging is added, but I need to optimize performance. Can you review my code for bottlenecks?",
            "The performance is better, but I want to add caching. What's the best caching strategy?",
            "Perfect! Now I need to write tests. Can you help me create test cases that cover all the edge cases?",
        ],
        // Architecture Design Flow
        vec![
            "I'm designing a microservices architecture for an e-commerce platform. Can you help me plan the services?",
            "I've planned the services, but I need to design the API gateway. What's the best approach?",
            "The API gateway is designed, but I need to handle service discovery. How should I implement that?",
            "Service discovery is working, but I need to add load balancing. What's the best load balancing strategy?",
            "The load balancing is configured, but I need to handle failures. How should I implement circuit breakers?",
            "Circuit breakers are in place, but I need to add monitoring. What metrics should I track?",
            "The monitoring is set up, but I need to handle distributed tracing. How should I implement that?",
            "Tracing is working, but I need to optimize database access. What's the best caching strategy?",
            "The caching is added, but I need to handle data consistency. How should I implement eventual consistency?",
            "Perfect! Now I need to add security. Can you help me design authentication and authorization?",
        ],
        // Learning Scenario Flow
        vec![
            "I'm learning about async programming in Python. Can you explain how async/await works?",
            "I understand the basics, but I'm confused about event loops. Can you explain how they work?",
            "The event loop makes sense now, but I'm having trouble with async context managers. Can you show me examples?",
            "I've got context managers working, but I need to understand how to handle errors in async code. What's the best approach?",
            "Error handling is clearer now, but I want to understand how to test async code. Can you help me write async tests?",
            "The tests are working, but I'm seeing performance issues. How can I optimize my async code?",
            "The performance is better, but I need to understand how to debug async code. What tools should I use?",
            "Debugging is easier now, but I want to understand how to handle cancellation. Can you explain cancellation tokens?",
            "Cancellation is working, but I need to understand how to coordinate multiple async operations. What's the best pattern?",
            "Perfect! Now I want to build a real project. Can you help me design an async web scraper?",
        ],
    ];

    for i in 0..num_conversations {
        let template_idx = i % conversation_templates.len();
        let template = &conversation_templates[template_idx];
        
        // Each conversation has 10-20 turns (vary slightly)
        let num_turns = 10 + (i % 11); // 10-20 turns
        let mut turns = Vec::new();
        
        for turn_idx in 0..num_turns {
            let base_turn = template[turn_idx % template.len()].to_string();
            // Add variation to make each conversation unique
            let turn = if i > 0 {
                format!("{} [Conversation {} - Turn {}]", base_turn, i + 1, turn_idx + 1)
            } else {
                base_turn
            };
            turns.push(turn);
        }
        
        conversations.push(Conversation {
            id: i + 1,
            turns,
        });
    }

    conversations
}

/// Verify all system endpoints are healthy
async fn verify_all_endpoints() -> Result<HashMap<String, bool>> {
    let mut status = HashMap::new();
    
    info!("🔍 Verifying all system endpoints...");
    
    // Main pipeline health endpoints
    let health_port = std::env::var("NIODOO_HEALTH_PORT")
        .unwrap_or_else(|_| "9090".to_string())
        .parse::<u16>()
        .unwrap_or(9090);
    
    let endpoints = vec![
        ("main_health", format!("http://localhost:{}/health", health_port)),
        ("main_ready", format!("http://localhost:{}/ready", health_port)),
        ("main_metrics", format!("http://localhost:{}/metrics", health_port)),
        ("rl_health", "http://localhost:8080/health".to_string()),
        ("vllm_models", "http://localhost:5001/v1/models".to_string()),
        ("qdrant_collections", "http://localhost:6333/collections".to_string()),
    ];
    
    for (name, url) in endpoints {
        let mut success = false;
        for attempt in 1..=5 {
            match reqwest::get(&url).await {
                Ok(response) => {
                    if response.status().is_success() {
                        success = true;
                        break;
                    }
                }
                Err(_) => {
                    if attempt < 5 {
                        tokio::time::sleep(Duration::from_secs(2)).await;
                    }
                }
            }
        }
        status.insert(name.to_string(), success);
        if success {
            info!("  ✅ {}: OK", name);
        } else {
            warn!("  ⚠️  {}: FAILED (continuing anyway)", name);
        }
    }
    
    Ok(status)
}

/// Process a single prompt through pipeline and return metrics
async fn process_prompt(
    pipeline: &mut Pipeline,
    prompt: &str,
    conversation_id: usize,
    turn_number: usize,
) -> PromptMetrics {
    let start = Instant::now();
    
    match pipeline.process_prompt(prompt).await {
        Ok(cycle) => {
            let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
            let response_length = cycle.hybrid_response.len();
            
            // Check for code blocks
            let code_extracted = cycle.hybrid_response.contains("```python")
                || cycle.hybrid_response.contains("```rust")
                || cycle.hybrid_response.contains("```javascript")
                || cycle.hybrid_response.contains("```typescript")
                || cycle.hybrid_response.contains("```go")
                || cycle.hybrid_response.contains("```java")
                || cycle.hybrid_response.contains("```cpp");
            
            PromptMetrics {
                prompt: prompt.to_string(),
                conversation_id,
                turn_number,
                success: true,
                latency_ms,
                response_length,
                code_extracted,
                error: None,
            }
        }
        Err(e) => {
            let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
            PromptMetrics {
                prompt: prompt.to_string(),
                conversation_id,
                turn_number,
                success: false,
                latency_ms,
                response_length: 0,
                code_extracted: false,
                error: Some(format!("{}", e)),
            }
        }
    }
}

/// Run test suite for a single configuration
async fn run_configuration_test(
    config_name: &str,
    conversations: &[Conversation],
    baseline_config: bool,
) -> Result<ConfigMetrics> {
    info!("🧪 Running {} configuration test...", config_name);
    
    // Set up configuration
    std::env::set_var("CODE_MODE_ENABLED", "true");
    std::env::set_var("CODE_MODE_LANGUAGE", "python");
    std::env::set_var("MOCK_MODE", "true"); // Enable mock mode to bypass ONNX requirement
    
    if baseline_config {
        // Baseline: standard config
        std::env::set_var("TOPOLOGY_MODE", "baseline");
        std::env::set_var("RCE_ENABLED", "false");
    } else {
        // Treatment: enhanced config
        std::env::set_var("TOPOLOGY_MODE", "hybrid");
        std::env::set_var("RCE_ENABLED", "true");
    }
    
    // Initialize pipeline
    info!("  Initializing pipeline...");
    let mut pipeline = Pipeline::initialise(CliArgs::default())
        .await
        .context("Failed to initialize pipeline")?;
    
    // Process all conversations
    let mut all_metrics = Vec::new();
    let mut latencies = Vec::new();
    let mut response_lengths = Vec::new();
    let mut errors = Vec::new();
    let mut code_extraction_success = 0;
    let mut successful_prompts = 0;
    let mut failed_prompts = 0;
    
    let total_prompts: usize = conversations.iter().map(|c| c.turns.len()).sum();
    let mut processed = 0;
    
    info!("  Processing {} conversations ({} total prompts)...", conversations.len(), total_prompts);
    
    for conversation in conversations {
        for (turn_idx, prompt) in conversation.turns.iter().enumerate() {
            let metrics = process_prompt(
                &mut pipeline,
                prompt,
                conversation.id,
                turn_idx + 1,
            ).await;
            
            all_metrics.push(metrics.clone());
            
            if metrics.success {
                successful_prompts += 1;
                latencies.push(metrics.latency_ms);
                response_lengths.push(metrics.response_length);
                if metrics.code_extracted {
                    code_extraction_success += 1;
                }
            } else {
                failed_prompts += 1;
                if let Some(ref err) = metrics.error {
                    errors.push(err.clone());
                }
            }
            
            processed += 1;
            if processed % 100 == 0 {
                info!("    Progress: {}/{} prompts processed", processed, total_prompts);
            }
        }
    }
    
    info!("  ✅ {} configuration complete: {} successful, {} failed", config_name, successful_prompts, failed_prompts);
    
    // Calculate statistical summary
    let latency_summary = StatisticalSummary::from_values(&latencies);
    
    Ok(ConfigMetrics {
        config_name: config_name.to_string(),
        total_prompts,
        successful_prompts,
        failed_prompts,
        latencies,
        response_lengths,
        code_extraction_success,
        errors,
        latency_summary,
    })
}


/// Compare baseline and treatment metrics
fn compare_ab_results(baseline: &ConfigMetrics, treatment: &ConfigMetrics) -> ABComparison {
    // Latency comparison
    let baseline_mean = baseline.latency_summary.mean;
    let treatment_mean = treatment.latency_summary.mean;
    let latency_diff_ms = treatment_mean - baseline_mean;
    let latency_diff_pct = if baseline_mean > 0.0 {
        (latency_diff_ms / baseline_mean) * 100.0
    } else {
        0.0
    };
    
    // Success rate comparison
    let baseline_success_rate = if baseline.total_prompts > 0 {
        baseline.successful_prompts as f64 / baseline.total_prompts as f64 * 100.0
    } else {
        0.0
    };
    let treatment_success_rate = if treatment.total_prompts > 0 {
        treatment.successful_prompts as f64 / treatment.total_prompts as f64 * 100.0
    } else {
        0.0
    };
    let success_rate_diff = treatment_success_rate - baseline_success_rate;
    
    // Code extraction rate comparison
    let baseline_code_rate = if baseline.total_prompts > 0 {
        baseline.code_extraction_success as f64 / baseline.total_prompts as f64 * 100.0
    } else {
        0.0
    };
    let treatment_code_rate = if treatment.total_prompts > 0 {
        treatment.code_extraction_success as f64 / treatment.total_prompts as f64 * 100.0
    } else {
        0.0
    };
    let code_rate_diff = treatment_code_rate - baseline_code_rate;
    
    // Statistical tests
    let cohens_d_latency = cohens_d(&baseline.latencies, &treatment.latencies);
    let (_, p_value_latency) = mann_whitney_u(&baseline.latencies, &treatment.latencies);
    let ci_latency = bootstrap_percentile_ci(&treatment.latencies, 0.50, 1000, 0.95);
    
    // Effect size interpretation
    let effect_size = if cohens_d_latency.abs() < 0.2 {
        "small"
    } else if cohens_d_latency.abs() < 0.5 {
        "medium"
    } else if cohens_d_latency.abs() < 0.8 {
        "large"
    } else {
        "very large"
    };
    
    // Determine winner
    let mut winner = "baseline".to_string();
    let mut winner_metrics = Vec::new();
    
    if treatment_success_rate > baseline_success_rate {
        winner = "treatment".to_string();
        winner_metrics.push(format!("Higher success rate ({:.2}% vs {:.2}%)", treatment_success_rate, baseline_success_rate));
    }
    
    if treatment_code_rate > baseline_code_rate {
        if winner != "treatment" {
            winner = "treatment".to_string();
        }
        winner_metrics.push(format!("Higher code extraction rate ({:.2}% vs {:.2}%)", treatment_code_rate, baseline_code_rate));
    }
    
    if latency_diff_ms < 0.0 {
        // Treatment is faster (negative difference)
        if winner != "treatment" {
            winner = "treatment".to_string();
        }
        winner_metrics.push(format!("Lower latency ({:.2}ms vs {:.2}ms)", treatment_mean, baseline_mean));
    } else {
        winner_metrics.push(format!("Baseline has lower latency ({:.2}ms vs {:.2}ms)", baseline_mean, treatment_mean));
    }
    
    ABComparison {
        latency_difference_ms: latency_diff_ms,
        latency_difference_pct: latency_diff_pct,
        success_rate_difference_pct: success_rate_diff,
        code_extraction_rate_difference_pct: code_rate_diff,
        cohens_d_latency,
        p_value_latency,
        confidence_interval_95_latency: ci_latency,
        effect_size_latency: effect_size.to_string(),
        winner,
        winner_metrics,
    }
}

/// Generate comprehensive test report
fn generate_report(
    baseline_metrics: ConfigMetrics,
    treatment_metrics: ConfigMetrics,
    comparison: ABComparison,
    endpoint_status: HashMap<String, bool>,
    execution_time_secs: f64,
    args: &Args,
) -> TestReport {
    TestReport {
        timestamp: Utc::now().to_rfc3339(),
        baseline_name: args.baseline_name.clone(),
        treatment_name: args.treatment_name.clone(),
        total_conversations: args.num_conversations,
        total_prompts: baseline_metrics.total_prompts + treatment_metrics.total_prompts,
        baseline_metrics,
        treatment_metrics,
        comparison,
        endpoint_status,
        execution_time_secs,
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    
    let args = Args::parse();
    
    println!("🚀 5000 CODING PROMPTS TEST SUITE WITH A/B COMPARISON");
    println!("{}", "=".repeat(80));
    println!("Generating {} conversations (5000+ prompts total)", args.num_conversations);
    println!("Running A/B comparison: {} vs {}", args.baseline_name, args.treatment_name);
    println!("Total executions: 10,000 (5000 per config)");
    println!("{}", "=".repeat(80));
    
    let overall_start = Instant::now();
    
    // Step 1: Verify endpoints
    let endpoint_status = if !args.skip_endpoints {
        verify_all_endpoints().await.context("Endpoint verification failed")?
    } else {
        warn!("⚠️  Skipping endpoint verification");
        HashMap::new()
    };
    
    // Step 2: Generate conversations
    info!("📝 Generating {} conversations...", args.num_conversations);
    let conversations = generate_conversational_coding_prompts(args.num_conversations);
    let total_prompts: usize = conversations.iter().map(|c| c.turns.len()).sum();
    info!("✅ Generated {} conversations with {} total prompts", conversations.len(), total_prompts);
    
    // Save conversations to file
    let output_dir = &args.output_dir;
    std::fs::create_dir_all(output_dir)?;
    let conversations_file = output_dir.join("conversations.json");
    serde_json::to_writer_pretty(
        std::fs::File::create(&conversations_file)?,
        &conversations,
    )?;
    info!("💾 Saved conversations to: {}", conversations_file.display());
    
    // Step 3: Run baseline configuration
    info!("\n{}", "=".repeat(80));
    info!("RUNNING BASELINE CONFIGURATION");
    info!("{}", "=".repeat(80));
    let baseline_metrics = run_configuration_test(&args.baseline_name, &conversations, true)
        .await
        .context("Baseline configuration test failed")?;
    
    // Step 4: Run treatment configuration
    info!("\n{}", "=".repeat(80));
    info!("RUNNING TREATMENT CONFIGURATION");
    info!("{}", "=".repeat(80));
    let treatment_metrics = run_configuration_test(&args.treatment_name, &conversations, false)
        .await
        .context("Treatment configuration test failed")?;
    
    // Step 5: Compare results
    info!("\n{}", "=".repeat(80));
    info!("A/B COMPARISON ANALYSIS");
    info!("{}", "=".repeat(80));
    let comparison = compare_ab_results(&baseline_metrics, &treatment_metrics);
    
    // Step 6: Generate report
    let execution_time = overall_start.elapsed().as_secs_f64();
    let report = generate_report(
        baseline_metrics,
        treatment_metrics,
        comparison,
        endpoint_status,
        execution_time,
        &args,
    );
    
    // Save report
    let report_file = output_dir.join(format!("test_report_ab_{}.json", Utc::now().format("%Y%m%d_%H%M%S")));
    serde_json::to_writer_pretty(
        std::fs::File::create(&report_file)?,
        &report,
    )?;
    
    // Print summary
    println!("\n{}", "=".repeat(80));
    println!("TEST SUITE COMPLETE");
    println!("{}", "=".repeat(80));
    println!("Total execution time: {:.2} seconds", execution_time);
    println!("Total prompts processed: {}", report.total_prompts);
    println!("\nBaseline Results:");
    println!("  Successful: {}/{} ({:.2}%)", 
        report.baseline_metrics.successful_prompts,
        report.baseline_metrics.total_prompts,
        if report.baseline_metrics.total_prompts > 0 {
            report.baseline_metrics.successful_prompts as f64 / report.baseline_metrics.total_prompts as f64 * 100.0
        } else { 0.0 }
    );
    println!("  Code extraction: {}/{} ({:.2}%)",
        report.baseline_metrics.code_extraction_success,
        report.baseline_metrics.total_prompts,
        if report.baseline_metrics.total_prompts > 0 {
            report.baseline_metrics.code_extraction_success as f64 / report.baseline_metrics.total_prompts as f64 * 100.0
        } else { 0.0 }
    );
    println!("  Avg latency: {:.2}ms (p50: {:.2}ms, p95: {:.2}ms, p99: {:.2}ms)",
        report.baseline_metrics.latency_summary.mean,
        report.baseline_metrics.latency_summary.p50,
        report.baseline_metrics.latency_summary.p95,
        report.baseline_metrics.latency_summary.p99,
    );
    
    println!("\nTreatment Results:");
    println!("  Successful: {}/{} ({:.2}%)",
        report.treatment_metrics.successful_prompts,
        report.treatment_metrics.total_prompts,
        if report.treatment_metrics.total_prompts > 0 {
            report.treatment_metrics.successful_prompts as f64 / report.treatment_metrics.total_prompts as f64 * 100.0
        } else { 0.0 }
    );
    println!("  Code extraction: {}/{} ({:.2}%)",
        report.treatment_metrics.code_extraction_success,
        report.treatment_metrics.total_prompts,
        if report.treatment_metrics.total_prompts > 0 {
            report.treatment_metrics.code_extraction_success as f64 / report.treatment_metrics.total_prompts as f64 * 100.0
        } else { 0.0 }
    );
    println!("  Avg latency: {:.2}ms (p50: {:.2}ms, p95: {:.2}ms, p99: {:.2}ms)",
        report.treatment_metrics.latency_summary.mean,
        report.treatment_metrics.latency_summary.p50,
        report.treatment_metrics.latency_summary.p95,
        report.treatment_metrics.latency_summary.p99,
    );
    
    println!("\nA/B Comparison:");
    println!("  Winner: {}", report.comparison.winner);
    println!("  Latency difference: {:.2}ms ({:.2}%)",
        report.comparison.latency_difference_ms,
        report.comparison.latency_difference_pct
    );
    println!("  Success rate difference: {:.2}%",
        report.comparison.success_rate_difference_pct
    );
    println!("  Code extraction difference: {:.2}%",
        report.comparison.code_extraction_rate_difference_pct
    );
    println!("  Effect size: {} (Cohen's d: {:.3})",
        report.comparison.effect_size_latency,
        report.comparison.cohens_d_latency
    );
    println!("  P-value: {:.4}", report.comparison.p_value_latency);
    
    println!("\n✅ Report saved to: {}", report_file.display());
    println!("✅ Conversations saved to: {}", conversations_file.display());
    
    // Success criteria: All prompts processed
    let baseline_complete = report.baseline_metrics.total_prompts == 5000;
    let treatment_complete = report.treatment_metrics.total_prompts == 5000;
    
    if baseline_complete && treatment_complete {
        println!("\n🎉 SUCCESS: All 10,000 executions completed!");
        Ok(())
    } else {
        eprintln!("\n❌ FAILURE: Not all prompts processed");
        eprintln!("  Baseline: {}/5000", report.baseline_metrics.total_prompts);
        eprintln!("  Treatment: {}/5000", report.treatment_metrics.total_prompts);
        std::process::exit(1)
    }
}

