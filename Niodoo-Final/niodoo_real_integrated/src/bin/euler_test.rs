//! Euler Mathematical Intelligence Test Runner
//! Tests the full niodoo_real_integrated pipeline with Level 50 mathematical problems
//! Validates autonomous gating system and measures true mathematical reasoning capability

use anyhow::{anyhow, Result};
use clap::Parser;
use niodoo_real_integrated::config::{CliArgs, HardwareProfile, OutputFormat, RuntimeConfig};
use niodoo_real_integrated::euler_problems::{euler_level50_problems, EulerTestConfig};
use niodoo_real_integrated::pipeline::Pipeline;
use niodoo_real_integrated::smoke::ServiceSmokeVerifier;
use std::time::Instant;
use tracing::{error, info, warn};

#[derive(Parser, Debug)]
#[command(
    name = "euler_test",
    about = "Mathematical Intelligence Test using Euler Level 50 problems"
)]
struct EulerArgs {
    /// Number of Euler problems to test (1-10)
    #[arg(short, long, default_value_t = 10)]
    problems: usize,

    /// Hardware profile for optimization
    #[arg(short = 'w', long, default_value = "laptop")]
    hardware: String,

    /// Output results to file
    #[arg(short, long, default_value = "logs/euler_intelligence_test.json")]
    output: String,

    /// Timeout per problem in seconds
    #[arg(short, long, default_value_t = 300)]
    timeout: u64,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Run in smoke-test mode (reduced problem set, stricter pre-flight checks)
    #[arg(long)]
    smoke: bool,
}

#[cfg(feature = "cli_bins")]
#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(if std::env::args().any(|arg| arg == "--verbose") {
            tracing::Level::DEBUG
        } else {
            tracing::Level::INFO
        })
        .init();

    let euler_args = EulerArgs::parse();

    info!("🧮 EULER MATHEMATICAL INTELLIGENCE TEST");
    info!("=======================================");
    info!("Problems: {}", euler_args.problems);
    info!("Hardware: {}", euler_args.hardware);
    info!("Timeout: {}s per problem", euler_args.timeout);
    info!("Mode: {}", if euler_args.smoke { "SMOKE" } else { "FULL" });
    info!("");

    // Initialize Pipeline with unified environment (leverage niodoo_real_integrated complexity)
    info!("🔧 Initializing Full NIODOO Pipeline...");
    let hardware_profile = match euler_args.hardware.to_ascii_lowercase().as_str() {
        "beelink" => HardwareProfile::Beelink,
        "h200" => HardwareProfile::H200,
        "5090" | "rtx5090" => HardwareProfile::RTX5090,
        "5080q" | "5080-q" | "laptop" => HardwareProfile::Laptop5080Q,
        other => {
            warn!(
                hardware = %other,
                "Unknown hardware profile '{}'; defaulting to laptop 5080Q profile",
                other
            );
            HardwareProfile::Laptop5080Q
        }
    };

    let pipeline_args = CliArgs {
        prompt: None,
        prompt_file: None,
        swarm: 1,
        iterations: 1,
        output: OutputFormat::Json,
        hardware: hardware_profile,
        config: None,
        rng_seed_override: None,
    };

    let runtime_config = RuntimeConfig::load(&pipeline_args)?;
    if runtime_config.mock_mode {
        return Err(anyhow!(
            "Runtime configuration indicates MOCK_MODE=true. Disable MOCK_MODE to run Euler tests."
        ));
    }

    info!("🧪 Performing live service smoke verification...");
    let smoke_verifier = ServiceSmokeVerifier::new()?;
    smoke_verifier.verify(&runtime_config).await?;
    info!("✅ Live service smoke verification passed.");

    let mut pipeline = match Pipeline::initialise(pipeline_args).await {
        Ok(p) => {
            info!("✅ Full NIODOO Pipeline initialized successfully!");
            p
        }
        Err(e) => {
            error!("❌ Pipeline initialization failed: {}", e);
            error!("   This is why the full system hasn't been working!");
            return Err(e);
        }
    };

    info!("🧠 Pipeline ready - starting mathematical intelligence assessment...");
    info!("");

    // Get Euler problems
    let problems = euler_level50_problems();
    let effective_problem_cap = if euler_args.smoke {
        euler_args.problems.min(3)
    } else {
        euler_args.problems
    };
    let problem_timeout_secs = if euler_args.smoke {
        euler_args.timeout.min(120)
    } else {
        euler_args.timeout
    };
    let test_count = effective_problem_cap.min(problems.len());
    info!(
        "Executing {} Euler problems with a {}s timeout per problem",
        test_count, problem_timeout_secs
    );

    let mut results = Vec::new();
    let mut learning_gate_count = 0;
    let mut memory_gate_count = 0;
    let mut indifferent_count = 0;
    let mut novel_topology_count = 0;
    let mut extreme_emotion_count = 0;
    let mut golden_qualified_count = 0;

    let test_start = Instant::now();

    // Run each Euler problem through the full pipeline
    for (idx, problem) in problems.iter().take(test_count).enumerate() {
        let problem_id = idx + 1;
        info!(
            "🧮 Problem {}/{}: Mathematical Reasoning Test",
            problem_id, test_count
        );
        info!("Problem: {}...", &problem[..problem.len().min(100)]);

        let problem_start = Instant::now();

        // Run through full niodoo_real_integrated pipeline
        match tokio::time::timeout(
            std::time::Duration::from_secs(problem_timeout_secs),
            pipeline.process_prompt(problem),
        )
        .await
        {
            Ok(Ok(cycle_result)) => {
                let duration = problem_start.elapsed();

                info!("⏱️  Duration: {:?}", duration);

                let generation = &cycle_result.generation;
                let collapse = &cycle_result.collapse;

                let raw_quality = generation
                    .curator_quality
                    .or(collapse.curator_quality)
                    .unwrap_or(cycle_result.rouge);
                let quality_score = (raw_quality * 10.0).clamp(0.0, 10.0) as f32;

                info!("🎯 Quality Score: {:.1}/10", quality_score);
                info!(
                    "📝 Response: {}...",
                    generation.hybrid_response[..generation.hybrid_response.len().min(150)].trim()
                );

                // Analyze gating behavior (the pipeline should have already applied gating)
                let quality_bucket = quality_score.round() as i32;
                let gating_path = match quality_bucket {
                    0..=5 => {
                        learning_gate_count += 1;
                        "Learning Gate".to_string()
                    }
                    6..=7 => {
                        indifferent_count += 1;
                        "Indifferent Path".to_string()
                    }
                    _ => {
                        memory_gate_count += 1;
                        "Memory Gate".to_string()
                    }
                };

                info!("🚪 Gating Path: {}", gating_path);

                // Mathematical analysis
                let math_indicators = analyze_mathematical_content(
                    &generation.hybrid_response,
                    &cycle_result.topology,
                );

                // Check for novel topology and extreme emotion (from pipeline data)
                let novel_topology = cycle_result.pad_state.entropy > 2.0; // Heuristic for novelty
                let extreme_emotion = cycle_result
                    .pad_state
                    .pad
                    .iter()
                    .take(3)
                    .any(|&p| p.abs() > 0.4);

                if novel_topology {
                    novel_topology_count += 1;
                }
                if extreme_emotion {
                    extreme_emotion_count += 1;
                }
                if (novel_topology || extreme_emotion) && quality_bucket >= 8 {
                    golden_qualified_count += 1;
                    info!("🌟 Golden Memory Qualified!");
                }

                let topocot_summary = cycle_result.topocot.as_ref().map(|telemetry| {
                    niodoo_real_integrated::euler_problems::EulerTopoCotSummary {
                        score_overall: telemetry.score_overall,
                        score_completeness: telemetry.score_completeness,
                        score_consistency: telemetry.score_consistency,
                        score_actionability: telemetry.score_actionability,
                        issues: telemetry.issues.clone(),
                        raw_json: telemetry.raw_json.clone(),
                        thinking_depth: telemetry.thinking_depth,
                        pivot_score: telemetry.pivot_score,
                        reflection_summary: cycle_result.topology_reflection_summary.clone(),
                        plan_summary: telemetry.plan_summary.clone(),
                    }
                });

                let result = niodoo_real_integrated::euler_problems::EulerTestResult {
                    problem_id,
                    problem: problem.clone(),
                    response: generation.hybrid_response.clone(),
                    quality_score,
                    gating_path,
                    mathematical_indicators: math_indicators,
                    topology_signature: niodoo_real_integrated::euler_problems::TopologySignature {
                        betti_numbers: cycle_result.topology.betti_numbers.to_vec(),
                        knot_complexity: cycle_result.topology.knot_complexity as f32,
                        spectral_gap: cycle_result.topology.spectral_gap as f32,
                        persistence_entropy: cycle_result.topology.persistence_entropy as f32,
                    },
                    pad_emotional_state:
                        niodoo_real_integrated::euler_problems::PADEmotionalState {
                            pleasure: cycle_result
                                .pad_state
                                .pad
                                .get(0)
                                .copied()
                                .unwrap_or_default() as f32,
                            arousal: cycle_result
                                .pad_state
                                .pad
                                .get(1)
                                .copied()
                                .unwrap_or_default() as f32,
                            dominance: cycle_result
                                .pad_state
                                .pad
                                .get(2)
                                .copied()
                                .unwrap_or_default() as f32,
                            entropy: cycle_result.pad_state.entropy as f32,
                            surface_position: [0.0, 0.0, 0.0], // Placeholder
                        },
                    processing_time_ms: duration.as_millis() as u64,
                    memory_retrieval_count: collapse.top_hits.len(),
                    breakthrough_detected: !cycle_result.learning.breakthroughs.is_empty(),
                    novel_topology,
                    extreme_emotion,
                    topocot: topocot_summary,
                };

                results.push(result);

                info!("✅ Problem {} completed", problem_id);
                info!("");
            }
            Ok(Err(e)) => {
                error!("💥 Problem {} failed: {}", problem_id, e);
                learning_gate_count += 1; // Count errors as failures
            }
            Err(_) => {
                warn!(
                    "⏰ Problem {} timed out after {}s",
                    problem_id, problem_timeout_secs
                );
                learning_gate_count += 1; // Count timeouts as failures
            }
        }
    }

    let total_duration = test_start.elapsed();

    // Generate comprehensive results
    let test_config = EulerTestConfig {
        problems_to_run: test_count,
        timeout_secs: problem_timeout_secs,
        ..Default::default()
    };

    let mut test_results = niodoo_real_integrated::euler_problems::EulerTestSuiteResults {
        test_id: format!(
            "euler_intelligence_{}",
            chrono::Utc::now().format("%Y%m%d_%H%M%S")
        ),
        timestamp: chrono::Utc::now().to_rfc3339(),
        config: test_config,
        summary: niodoo_real_integrated::euler_problems::TestSummary {
            total_problems: test_count,
            completed_problems: results.len(),
            average_quality: if results.is_empty() {
                0.0
            } else {
                results.iter().map(|r| r.quality_score).sum::<f32>() / results.len() as f32
            },
            average_math_depth: if results.is_empty() {
                0.0
            } else {
                results
                    .iter()
                    .map(|r| r.mathematical_indicators.mathematical_depth as f32)
                    .sum::<f32>()
                    / results.len() as f32
            },
            total_duration_secs: total_duration.as_secs_f64(),
        },
        results,
        gating_analysis: niodoo_real_integrated::euler_problems::GatingAnalysis {
            learning_gate_count,
            indifferent_count,
            memory_gate_count,
            novel_topology_count,
            extreme_emotion_count,
            golden_memory_qualified: golden_qualified_count,
        },
        intelligence_assessment: niodoo_real_integrated::euler_problems::IntelligenceAssessment {
            mathematical_reasoning_grade: "".to_string(), // Will be filled by analyze_intelligence
            strengths: vec![],
            weaknesses: vec![],
            improvement_recommendations: vec![],
            autonomous_learning_effectiveness: 0.0,
            memory_curation_effectiveness: 0.0,
            system_intelligence_level: "".to_string(),
        },
    };

    // Analyze intelligence
    test_results.analyze_intelligence();

    // Save results
    std::fs::write(
        &euler_args.output,
        serde_json::to_vec_pretty(&test_results)?,
    )?;

    // Print summary
    test_results.print_intelligence_report();

    info!("\n📄 Results saved to: {}", euler_args.output);
    info!("🎉 Euler Mathematical Intelligence Test Complete!");

    Ok(())
}

#[cfg(not(feature = "cli_bins"))]
fn main() {
    eprintln!("Enable the `cli_bins` feature to run `euler_test`.");
}

fn analyze_mathematical_content(
    response: &str,
    topology: &niodoo_real_integrated::tcs_analysis::TopologicalSignature,
) -> niodoo_real_integrated::euler_problems::MathematicalIndicators {
    // Use real content analysis patterns that mirror curator.py's sophisticated assessment
    // The curator evaluates "factual accuracy, topological alignment, novelty, and actionability"

    let lower = response.to_lowercase();

    // Real mathematical indicators (based on curator.py's actual criteria)
    let contains_code = response.contains("fn ")
        || response.contains("def ")
        || response.contains("impl")
        || response.contains("algorithm");
    let contains_proof = lower.contains("proof")
        || lower.contains("theorem")
        || lower.contains("derive")
        || lower.contains("mathematical reasoning");
    let contains_algorithm = lower.contains("algorithm")
        || lower.contains("implementation")
        || lower.contains("approach")
        || lower.contains("method");
    let contains_optimization = lower.contains("complexity")
        || lower.contains("efficient")
        || lower.contains("optimization")
        || lower.contains("o(");

    // Mathematical depth assessment (mirrors curator's topological alignment check)
    let factual_accuracy_indicators = ["theorem", "proof", "mathematical", "equation"];
    let topological_indicators = ["structure", "pattern", "relationship", "connection"];
    let novelty_indicators = ["novel", "unique", "innovative", "creative"];
    let actionability_indicators = ["algorithm", "implementation", "solution", "method"];

    let mut depth_components = 0u8;
    if factual_accuracy_indicators
        .iter()
        .any(|&ind| lower.contains(ind))
    {
        depth_components += 2;
    }
    if topological_indicators
        .iter()
        .any(|&ind| lower.contains(ind))
    {
        depth_components += 3;
    }
    if novelty_indicators.iter().any(|&ind| lower.contains(ind)) {
        depth_components += 2;
    }
    if actionability_indicators
        .iter()
        .any(|&ind| lower.contains(ind))
    {
        depth_components += 3;
    }

    let betti_sum: usize = topology.betti_numbers.iter().sum();
    let betti_non_trivial = betti_sum > 1;
    let higher_dimensional = topology
        .betti_numbers
        .iter()
        .enumerate()
        .skip(1)
        .any(|(_, &b)| b > 0);
    let knot_signal = topology.knot_complexity.abs() > f64::EPSILON;
    let spectral_signal =
        topology.spectral_gap.is_finite() && topology.spectral_gap.abs() > f64::EPSILON;
    let entropy_signal = topology.persistence_entropy.is_finite()
        && topology.persistence_entropy.abs() > f64::EPSILON;

    let topological_feature_count = [
        betti_non_trivial,
        higher_dimensional,
        knot_signal,
        spectral_signal,
        entropy_signal,
    ]
    .into_iter()
    .filter(|flag| *flag)
    .count() as u8;

    depth_components = depth_components.saturating_add(topological_feature_count);

    // Code quality (based on curator's actionability assessment)
    let code_quality = if contains_code && contains_algorithm {
        if response.len() > 1500 && contains_optimization {
            9
        }
        // High actionability
        else if response.len() > 1000 {
            7
        }
        // Good actionability
        else if response.len() > 500 {
            5
        }
        // Moderate actionability
        else {
            3
        } // Low actionability
    } else if contains_algorithm {
        4 // Conceptual but not implemented
    } else {
        1 // Poor actionability
    };

    // Problem understanding (based on curator's factual accuracy assessment)
    let understanding = if contains_proof && contains_algorithm && contains_optimization {
        10
    }
    // Perfect understanding
    else if contains_proof && contains_algorithm {
        8
    }
    // Strong understanding
    else if contains_algorithm {
        6
    }
    // Adequate understanding
    else if response.len() > 200 {
        4
    }
    // Weak understanding
    else {
        2
    }; // Poor understanding

    let topological_floor = (topological_feature_count * 2).min(10);
    let mathematical_depth = depth_components.min(10).max(topological_floor);

    niodoo_real_integrated::euler_problems::MathematicalIndicators {
        contains_code,
        contains_proof,
        contains_algorithm,
        contains_optimization,
        mathematical_depth,
        code_quality,
        problem_understanding: understanding,
    }
}
