//! Validation Runner
//!
//! Executes all validation experiments and generates comprehensive report.

use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;
use tracing::info;

use niodoo_real_integrated::validation::{
    ablation_studies::{run_ablation_study, AblationConfig},
    topology_validation::betti_validation::validate_betti_numbers,
    benchmarks::baseline_rag::run_rag_baseline,
    learning_validation::forgetting_tests::test_forgetting,
    scale_testing::load_generator::generate_load,
    report_generator::{generate_report, save_report},
};

#[derive(Parser, Debug)]
#[command(name = "validation_runner")]
#[command(about = "Run comprehensive validation experiments to prove NIODOO")]
struct Args {
    /// Output directory for validation results
    #[arg(long, default_value = "validation_results")]
    output_dir: PathBuf,

    /// Run only specific experiment type
    #[arg(long)]
    experiment: Option<String>,

    /// Number of test prompts for ablation studies
    #[arg(long, default_value = "50")]
    ablation_prompts: usize,

    /// Number of interactions for scale testing
    #[arg(long, default_value = "1000")]
    scale_interactions: usize,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();
    std::fs::create_dir_all(&args.output_dir)?;

    info!("🚀 Starting NIODOO validation experiments");
    info!("Output directory: {}", args.output_dir.display());

    let experiment_type = args.experiment.as_deref();

    // Generate test prompts
    let test_prompts = generate_test_prompts(args.ablation_prompts);

    // Run ablation studies
    if experiment_type.is_none() || experiment_type == Some("ablation") {
        info!("📊 Running ablation studies...");
        run_ablation_experiments(&test_prompts, &args.output_dir).await?;
    }

    // Run topology validation
    if experiment_type.is_none() || experiment_type == Some("topology") {
        info!("🔬 Running topology validation...");
        run_topology_validation(&args.output_dir).await?;
    }

    // Run benchmarks
    if experiment_type.is_none() || experiment_type == Some("benchmarks") {
        info!("📈 Running comparative benchmarks...");
        run_benchmarks(&test_prompts, &args.output_dir).await?;
    }

    // Run learning validation
    if experiment_type.is_none() || experiment_type == Some("learning") {
        info!("🧠 Running learning validation...");
        run_learning_validation(&args.output_dir).await?;
    }

    // Run scale testing
    if experiment_type.is_none() || experiment_type == Some("scale") {
        info!("📏 Running scale testing...");
        run_scale_testing(args.scale_interactions, &test_prompts, &args.output_dir).await?;
    }

    // Generate comprehensive report
    if experiment_type.is_none() {
        info!("📝 Generating comprehensive validation report...");
        generate_comprehensive_report(&args.output_dir).await?;
    }

    info!("✅ Validation experiments completed!");
    info!("Results saved to: {}", args.output_dir.display());

    Ok(())
}

fn generate_test_prompts(count: usize) -> Vec<String> {
    let base_prompts = vec![
        "Explain quantum computing in simple terms".to_string(),
        "What is machine learning?".to_string(),
        "Describe the theory of relativity".to_string(),
        "How does a neural network work?".to_string(),
        "What is topological data analysis?".to_string(),
        "Explain Betti numbers in topology".to_string(),
        "What is persistent homology?".to_string(),
        "How does emotion affect decision making?".to_string(),
        "Describe the PAD emotional model".to_string(),
        "What is catastrophic forgetting in AI?".to_string(),
        "Explain continuous learning in neural networks".to_string(),
        "What is RAG (Retrieval-Augmented Generation)?".to_string(),
        "How does QLoRA fine-tuning work?".to_string(),
        "What is the difference between supervised and unsupervised learning?".to_string(),
        "Explain attention mechanisms in transformers".to_string(),
    ];

    let mut prompts = Vec::new();
    for i in 0..count {
        prompts.push(base_prompts[i % base_prompts.len()].clone());
    }
    prompts
}

async fn run_ablation_experiments(
    test_prompts: &[String],
    output_dir: &PathBuf,
) -> Result<()> {
    let components = vec!["topology", "erag", "compass", "learning", "curator"];
    let mut results = Vec::new();

    for component in components {
        info!("  Testing ablation: {}", component);
        
        let mut config = AblationConfig::default();
        match component {
            "topology" => config.topology_enabled = false,
            "erag" => config.erag_enabled = false,
            "compass" => config.compass_enabled = false,
            "learning" => config.learning_enabled = false,
            "curator" => config.curator_enabled = false,
            _ => {}
        }

        match run_ablation_study(component, config, test_prompts.to_vec(), output_dir.clone()).await {
            Ok(result) => {
                results.push(result);
                info!("    ✅ {} ablation completed", component);
            }
            Err(e) => {
                tracing::warn!(error = %e, component, "Ablation study failed");
            }
        }
    }

    // Save ablation results
    let ablation_file = output_dir.join("ablation_results.json");
    let json = serde_json::to_string_pretty(&results)?;
    std::fs::write(&ablation_file, json)?;

    Ok(())
}

async fn run_topology_validation(output_dir: &PathBuf) -> Result<()> {
    // Generate code samples with complexity labels
    let code_samples = vec![
        ("def hello(): print('world')".to_string(), 1),
        ("def factorial(n): return 1 if n <= 1 else n * factorial(n-1)".to_string(), 3),
        ("class Node: def __init__(self, val): self.val = val; self.next = None".to_string(), 2),
        ("def quicksort(arr): return arr if len(arr) <= 1 else quicksort([x for x in arr[1:] if x < arr[0]]) + [arr[0]] + quicksort([x for x in arr[1:] if x >= arr[0]])".to_string(), 5),
    ];

    match validate_betti_numbers(code_samples).await {
        Ok(result) => {
            let file = output_dir.join("topology_validation.json");
            let json = serde_json::to_string_pretty(&result)?;
            std::fs::write(&file, json)?;
            info!("  ✅ Topology validation completed");
        }
        Err(e) => {
            tracing::warn!(error = %e, "Topology validation failed");
        }
    }

    Ok(())
}

async fn run_benchmarks(
    test_prompts: &[String],
    output_dir: &PathBuf,
) -> Result<()> {
    info!("  Running standard RAG baseline...");
    match run_rag_baseline(test_prompts.to_vec()).await {
        Ok(result) => {
            let file = output_dir.join("benchmark_rag.json");
            let json = serde_json::to_string_pretty(&result)?;
            std::fs::write(&file, json)?;
            info!("  ✅ RAG baseline completed");
        }
        Err(e) => {
            tracing::warn!(error = %e, "RAG baseline failed");
        }
    }

    Ok(())
}

async fn run_learning_validation(output_dir: &PathBuf) -> Result<()> {
    // Initial tasks with expected accuracy
    let initial_tasks = vec![
        ("What is 2+2?".to_string(), 1.0),
        ("What is the capital of France?".to_string(), 1.0),
        ("Explain machine learning".to_string(), 0.8),
    ];

    // New tasks to learn
    let new_tasks = vec![
        "What is quantum entanglement?".to_string(),
        "Explain topological data analysis".to_string(),
    ];

    match test_forgetting(initial_tasks, new_tasks).await {
        Ok(result) => {
            let file = output_dir.join("learning_validation.json");
            let json = serde_json::to_string_pretty(&result)?;
            std::fs::write(&file, json)?;
            info!("  ✅ Learning validation completed");
        }
        Err(e) => {
            tracing::warn!(error = %e, "Learning validation failed");
        }
    }

    Ok(())
}

async fn run_scale_testing(
    interactions: usize,
    prompt_pool: &[String],
    output_dir: &PathBuf,
) -> Result<()> {
    info!("  Running scale test with {} interactions...", interactions);
    
    match generate_load(interactions, prompt_pool.to_vec()).await {
        Ok(result) => {
            let file = output_dir.join("scale_test.json");
            let json = serde_json::to_string_pretty(&result)?;
            std::fs::write(&file, json)?;
            info!("  ✅ Scale testing completed");
        }
        Err(e) => {
            tracing::warn!(error = %e, "Scale testing failed");
        }
    }

    Ok(())
}

async fn generate_comprehensive_report(output_dir: &PathBuf) -> Result<()> {
    // Load all results
    let ablation_file = output_dir.join("ablation_results.json");
    let topology_file = output_dir.join("topology_validation.json");
    let benchmark_file = output_dir.join("benchmark_rag.json");
    let learning_file = output_dir.join("learning_validation.json");
    let scale_file = output_dir.join("scale_test.json");

    let ablation_results: Vec<niodoo_real_integrated::validation::ablation_studies::AblationResult> = if ablation_file.exists() {
        serde_json::from_str(&std::fs::read_to_string(&ablation_file)?).unwrap_or_default()
    } else {
        Vec::new()
    };

    let topology_results: Vec<niodoo_real_integrated::validation::topology_validation::TopologyValidationResult> = if topology_file.exists() {
        vec![serde_json::from_str(&std::fs::read_to_string(&topology_file)?).unwrap_or_else(|_| {
            niodoo_real_integrated::validation::topology_validation::TopologyValidationResult {
                experiment_name: "placeholder".to_string(),
                correlation: 0.0,
                improvement_pct: 0.0,
                statistical_significance: 1.0,
                timestamp: chrono::Utc::now().to_rfc3339(),
            }
        })]
    } else {
        Vec::new()
    };

    let benchmark_results: Vec<niodoo_real_integrated::validation::benchmarks::BenchmarkResult> = if benchmark_file.exists() {
        vec![serde_json::from_str(&std::fs::read_to_string(&benchmark_file)?).unwrap_or_else(|_| {
            niodoo_real_integrated::validation::benchmarks::BenchmarkResult {
                system_name: "placeholder".to_string(),
                test_suite: "placeholder".to_string(),
                metrics: niodoo_real_integrated::validation::benchmarks::BenchmarkMetrics {
                    accuracy: 0.0,
                    latency_ms: 0.0,
                    rouge_score: 0.0,
                    memory_usage_mb: 0.0,
                },
                timestamp: chrono::Utc::now().to_rfc3339(),
            }
        })]
    } else {
        Vec::new()
    };

    let learning_results: Vec<niodoo_real_integrated::validation::learning_validation::LearningValidationResult> = if learning_file.exists() {
        vec![serde_json::from_str(&std::fs::read_to_string(&learning_file)?).unwrap_or_else(|_| {
            niodoo_real_integrated::validation::learning_validation::LearningValidationResult {
                test_name: "placeholder".to_string(),
                forgetting_rate: 0.0,
                improvement_rate: 0.0,
                breakthrough_precision: None,
                safety_score_delta: None,
                timestamp: chrono::Utc::now().to_rfc3339(),
            }
        })]
    } else {
        Vec::new()
    };

    let scale_results: Vec<niodoo_real_integrated::validation::scale_testing::ScaleTestResult> = if scale_file.exists() {
        vec![serde_json::from_str(&std::fs::read_to_string(&scale_file)?).unwrap_or_else(|_| {
            niodoo_real_integrated::validation::scale_testing::ScaleTestResult {
                interaction_count: 0,
                metrics: niodoo_real_integrated::validation::scale_testing::ScaleMetrics {
                    rouge_scores: Vec::new(),
                    latency_ms: Vec::new(),
                    memory_usage_mb: 0.0,
                    improvement_rate: 0.0,
                    stability_score: 0.0,
                },
                timestamp: chrono::Utc::now().to_rfc3339(),
            }
        })]
    } else {
        Vec::new()
    };

    // Generate report
    let report = generate_report(
        ablation_results,
        topology_results,
        benchmark_results,
        learning_results,
        scale_results,
        Vec::new(), // ROI results - would be populated separately
        Vec::new(), // Terminology results - would be populated separately
    );

    // Save report
    let report_file = output_dir.join("validation_report.json");
    save_report(&report, report_file.clone())?;

    info!("📊 Comprehensive report generated: {}", report_file.display());
    info!("");
    info!("Overall Assessment:");
    info!("  Topology improves understanding: {}", report.overall_assessment.topology_improves_understanding);
    info!("  ERAG improves context: {}", report.overall_assessment.erag_improves_context);
    info!("  Learning works without forgetting: {}", report.overall_assessment.learning_works_without_forgetting);
    info!("  System scales: {}", report.overall_assessment.system_scales);
    info!("  Minimum viable proof: {}", report.overall_assessment.minimum_viable_proof);
    info!("  Strong proof: {}", report.overall_assessment.strong_proof);
    info!("");
    info!("Summary: {}", report.overall_assessment.summary);

    Ok(())
}

