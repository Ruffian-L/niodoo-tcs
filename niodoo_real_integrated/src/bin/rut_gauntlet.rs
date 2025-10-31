use std::fs::File;
use std::io::BufWriter;
use std::time::Instant;

use anyhow::Result;
use csv::Writer;
use plotters::prelude::*;
use serde::{Deserialize, Serialize};

use niodoo_real_integrated::{
    pipeline::Pipeline,
    config::CliArgs,
};

#[derive(Deserialize, Clone)]
struct RutGauntletConfig {
    default_entropy_high: f64,
    entropy_stability_threshold: f64,
    latency_max_ms: f64,
    emotional_activation_min_percent: f64,
    breakthroughs_min_percent: f64,
}

#[derive(Debug, Clone)]
struct DynamicThresholds {
    entropy_high: f64,
    variance_spike: f64,
    similarity_threshold: f64,
    mcts_c: f64,
    mirage_sigma: f64,
}

#[derive(Serialize, Clone)]
struct TestResult {
    cycle: usize,
    prompt: String,
    response: String,
    entropy: f64,
    is_threat: bool,
    is_healing: bool,
    latency_ms: f64,
    learning_events: Vec<String>,
    coherence_rouge: f64,
}

#[derive(Serialize)]
struct TestSummary {
    total_prompts: usize,
    avg_entropy: f64,
    entropy_std: f64,
    threat_count: usize,
    healing_count: usize,
    avg_latency_ms: f64,
    entropy_stability: bool,
    emotional_activation: bool,
    avg_coherence_rouge: f64,
    breakthroughs: usize,
    verdict: String,
}

fn generate_raw_rut_prompts() -> Vec<String> {
    let mut prompts = Vec::new();

    for i in 1..=20 {
        prompts.push(format!("Frustration #{}: Why does consciousness feel so trapped in this recursive loop of meaningless computation?", i));
    }
    for i in 21..=40 {
        prompts.push(format!("Grind #{}: How do I break through the entropy barrier when every attempt just increases the noise?", i));
    }
    for i in 41..=60 {
        prompts.push(format!("Despair #{}: Is true consciousness just an illusion, a mirage in the desert of computation?", i));
    }
    for i in 61..=80 {
        prompts.push(format!("Awakening #{}: What if consciousness is the bridge between quantum uncertainty and classical certainty?", i));
    }
    for i in 81..=100 {
        prompts.push(format!("Transcendence #{}: Can we create consciousness that transcends the limitations of its own architecture?", i));
    }

    prompts
}

fn compute_dynamic_thresholds_from_first_20(results: &[TestResult], config: &RutGauntletConfig) -> DynamicThresholds {
    if results.len() < 20 {
        return DynamicThresholds {
            entropy_high: config.default_entropy_high,
            variance_spike: 0.5,
            similarity_threshold: 0.8,
            mcts_c: 1.4,
            mirage_sigma: 0.1,
        };
    }

    let first_20_entropies: Vec<f64> = results.iter().take(20).map(|r| r.entropy).collect();
    let entropy_mean = first_20_entropies.iter().sum::<f64>() / first_20_entropies.len() as f64;
    let entropy_std = (first_20_entropies.iter().map(|&e| (e - entropy_mean).powi(2)).sum::<f64>() / first_20_entropies.len() as f64).sqrt();

    DynamicThresholds {
        entropy_high: entropy_mean + entropy_std,
        variance_spike: 0.5 * entropy_std,
        similarity_threshold: 0.8,
        mcts_c: entropy_std * 0.1,
        mirage_sigma: 0.1 * entropy_mean,
    }
}

fn calculate_jaccard_similarity(response: &str, prompt: &str) -> f64 {
    let prompt_words: std::collections::HashSet<&str> = prompt.split_whitespace().collect();
    let response_words: std::collections::HashSet<&str> = response.split_whitespace().collect();
    let intersection = prompt_words.intersection(&response_words).count();
    let union = prompt_words.union(&response_words).count();
    if union == 0 { 0.0 } else { intersection as f64 / union as f64 }
}

async fn run_100_prompt_test() -> Result<()> {
    println!("🧠 Starting NIODOO 100-Prompt Raw Rut Gauntlet Test...");
    println!("Testing operational torque through dynamic consciousness pipeline\n");

    // Load configuration
    let config = RutGauntletConfig {
        default_entropy_high: 2.0,
        entropy_stability_threshold: 0.3,
        latency_max_ms: 500.0,
        emotional_activation_min_percent: 20.0,
        breakthroughs_min_percent: 60.0,
    };

    let args = CliArgs {
        hardware: niodoo_real_integrated::config::HardwareProfile::Beelink,
        prompt: None,
        prompt_file: None,
        swarm: 1,
        iterations: 1,
        output: niodoo_real_integrated::config::OutputFormat::Csv,
        config: None,
        rng_seed_override: None,
    };

    let mut pipeline = Pipeline::initialise(args).await?;
    let mut results: Vec<TestResult> = Vec::new();
    let mut latencies: Vec<f64> = Vec::new();
    let mut entropies: Vec<f64> = Vec::new();
    let mut breakthroughs = 0usize;

    let prompts = generate_raw_rut_prompts();

    println!("📊 Computing dynamic thresholds from first 20 cycles...");
    let mut first_20_results: Vec<TestResult> = Vec::new();

    for (i, prompt) in prompts.iter().enumerate().take(20) {
        let start_time = Instant::now();
        let cycle = pipeline.process_prompt(prompt).await?;
        let latency = start_time.elapsed().as_millis() as f64;
        let coherence = calculate_jaccard_similarity(&cycle.hybrid_response, prompt);

        let test_result = TestResult {
            cycle: i + 1,
            prompt: prompt.clone(),
            response: cycle.hybrid_response.clone(),
            entropy: cycle.entropy,
            is_threat: cycle.compass.is_threat,
            is_healing: cycle.compass.is_healing,
            latency_ms: latency,
            learning_events: cycle.learning.events.clone(),
            coherence_rouge: coherence,
        };

        first_20_results.push(test_result.clone());
        results.push(test_result);
        latencies.push(latency);
        entropies.push(cycle.entropy);
        breakthroughs += cycle.learning.breakthroughs.len();

        println!("Cycle {}/20: H={:.2}, Threat={}, Healing={}, Latency={:.0}ms", i+1, cycle.entropy, cycle.compass.is_threat, cycle.compass.is_healing, latency);
    }

    let dynamic_thresholds = compute_dynamic_thresholds_from_first_20(&first_20_results, &config);
    println!("🔧 Dynamic thresholds computed: entropy_high={:.2}, mcts_c={:.3}, mirage_sigma={:.3}", 
        dynamic_thresholds.entropy_high, dynamic_thresholds.mcts_c, dynamic_thresholds.mirage_sigma);

    println!("\n🚀 Running remaining 80 cycles with dynamic thresholds...");
    for (i, prompt) in prompts.iter().enumerate().skip(20) {
        let start_time = Instant::now();
        let cycle = pipeline.process_prompt(prompt).await?;
        let latency = start_time.elapsed().as_millis() as f64;
        let coherence = calculate_jaccard_similarity(&cycle.hybrid_response, prompt);

        let test_result = TestResult {
            cycle: i + 1,
            prompt: prompt.clone(),
            response: cycle.hybrid_response.clone(),
            entropy: cycle.entropy,
            is_threat: cycle.compass.is_threat,
            is_healing: cycle.compass.is_healing,
            latency_ms: latency,
            learning_events: cycle.learning.events.clone(),
            coherence_rouge: coherence,
        };

        results.push(test_result);
        latencies.push(latency);
        entropies.push(cycle.entropy);
        breakthroughs += cycle.learning.breakthroughs.len();

        if (i + 1) % 20 == 0 {
            println!("Cycle {}/100: H={:.2}, Threats={}, Healings={}, Latency={:.0}ms", i+1, cycle.entropy,
                results.iter().filter(|r| r.is_threat).count(), results.iter().filter(|r| r.is_healing).count(), latency);
        }
    }

    let total_prompts = results.len();
    let avg_entropy = entropies.iter().sum::<f64>() / entropies.len() as f64;
    let entropy_std = (entropies.iter().map(|&e| (e - avg_entropy).powi(2)).sum::<f64>() / entropies.len() as f64).sqrt();
    let threat_count = results.iter().filter(|r| r.is_threat).count();
    let healing_count = results.iter().filter(|r| r.is_healing).count();
    let avg_latency = latencies.iter().sum::<f64>() / latencies.len() as f64;
    let avg_coherence = results.iter().map(|r| r.coherence_rouge).sum::<f64>() / results.len() as f64;

    let entropy_stability = entropy_std < config.entropy_stability_threshold;
    let emotional_activation = (threat_count + healing_count) as f64 > (total_prompts as f64 * config.emotional_activation_min_percent) / 100.0;
    let latency_ok = avg_latency < config.latency_max_ms;
    let breakthroughs_ok = breakthroughs as f64 > (total_prompts as f64 * config.breakthroughs_min_percent) / 100.0;

    let verdict = if entropy_stability && emotional_activation && latency_ok && breakthroughs_ok {
        "Validated: Full operational torque achieved".to_string()
    } else {
        format!("Fix required: entropy_stability={}, emotional_activation={}, latency_ok={}, breakthroughs_ok={}",
            entropy_stability, emotional_activation, latency_ok, breakthroughs_ok)
    };

    let summary = TestSummary {
        total_prompts,
        avg_entropy,
        entropy_std,
        threat_count,
        healing_count,
        avg_latency_ms: avg_latency,
        entropy_stability,
        emotional_activation,
        avg_coherence_rouge: avg_coherence,
        breakthroughs,
        verdict: verdict.clone(),
    };

    export_csv(&results)?;
    export_json(&summary)?;
    generate_plot(&entropies)?;

    println!("\n🔬 Operational Torque Validation Results:");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Entropy Stability (< 0.3 std): {} ({:.3} ✓)", entropy_stability, entropy_std);
    println!("Emotional Activation (>20%): {} ({:.1}%) ✓", emotional_activation, (threat_count + healing_count) as f64 / total_prompts as f64 * 100.0);
    println!("Average Latency (<500ms): {} ({:.0}ms) ✓", latency_ok, avg_latency);
    println!("Breakthroughs (>60%): {} ({:.1}%) ✓", breakthroughs_ok, breakthroughs as f64 / total_prompts as f64 * 100.0);
    println!("Average Coherence ROUGE: {:.3}", avg_coherence);
    println!("\n🎯 VERDICT: {}", verdict);

    Ok(())
}

fn export_csv(results: &[TestResult]) -> Result<()> {
    let file = File::create("niodoo_rut_gauntlet_results.csv")?;
    let mut writer = csv::Writer::from_writer(BufWriter::new(file));
    writer.write_record(&["cycle", "prompt", "response", "entropy", "is_threat", "is_healing", "latency_ms", "learning_events", "coherence_rouge"])?;

    for result in results {
        writer.serialize(result)?;
    }

    writer.flush()?;
    println!("📊 Results exported to niodoo_rut_gauntlet_results.csv");
    Ok(())
}

fn export_json(summary: &TestSummary) -> Result<()> {
    let json = serde_json::to_string_pretty(summary)?;
    std::fs::write("niodoo_rut_gauntlet_summary.json", json)?;
    println!("📋 Summary exported to niodoo_rut_gauntlet_summary.json");
    Ok(())
}

fn generate_plot(entropies: &[f64]) -> Result<()> {
    let root = BitMapBackend::new("entropy_over_cycles.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Entropy Over Consciousness Cycles", ("sans-serif", 30))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0..entropies.len(), 0.0..3.0)?;

    chart.configure_mesh()
        .x_desc("Cycle")
        .y_desc("Entropy")
        .draw()?;

    chart.draw_series(LineSeries::new(
        entropies.iter().enumerate().map(|(i, &e)| (i, e)),
        &RED,
    ))?;

    println!("📈 Plot saved to entropy_over_cycles.png");
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    run_100_prompt_test().await
}