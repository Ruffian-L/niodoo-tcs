use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use niodoo_cli::compass::{CompassConfig, CompassEngine};
use niodoo_cli::context::augment_prompt_with_memory;
use niodoo_cli::embedding::LocalEmbedder;
use niodoo_cli::erag::EragService;
use niodoo_cli::experience::Experience;
use niodoo_cli::memory::ExperienceStore;
use niodoo_cli::security::{PromptSecurityManager, SecurityConfig};
use niodoo_cli::tcs_analysis::TCSAnalyzer;
use niodoo_cli::torus::{TorusConfig, TorusProjector};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

#[derive(Parser, Debug)]
#[command(author, version, about = "Niodoo System 2 loop (Curator + Learning)")]
struct Args {
    /// Number of iterations to run
    #[arg(long, default_value_t = 5)]
    iterations: usize,

    /// ERAG configuration path
    #[arg(long, default_value = "config/erag.toml")]
    erag_config: String,

    /// Experience store configuration path
    #[arg(long, default_value = "config/system2_memory.toml")]
    memory_config: String,

    /// Prompt security configuration path
    #[arg(long, default_value = "config/security.toml")]
    security_config: String,

    /// Log file path
    #[arg(long, default_value = "logs/system2_loop.log")]
    log_file: PathBuf,

    /// Baseline output file
    #[arg(long, default_value = "baselines/system2.json")]
    baseline_file: PathBuf,

    /// Consciousness compass filter
    #[arg(long, default_value = "Discover")]
    compass: String,

    /// Learning loop configuration path
    #[arg(long, default_value = "config/learning_loop.toml")]
    learning_config: String,

    /// Torus projection configuration path
    #[arg(long, default_value = "config/torus.toml")]
    torus_config: String,

    /// Compass configuration path
    #[arg(long, default_value = "config/compass.toml")]
    compass_config: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct CuratorOutput {
    rouge_l: f32,
    quality_score: Option<i32>,
    feedback: String,
    #[allow(dead_code)]
    raw_completion: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
struct LearningLoopOutput {
    status: String,
    buffer_count: usize,
    #[serde(default)]
    triggered_training: bool,
    #[serde(default)]
    reason: Option<String>,
    #[serde(default)]
    training_summary: Option<Value>,
    #[serde(default)]
    adapter_path: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    tracing_subscriber::fmt::try_init().ok();

    let mut log_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&args.log_file)
        .with_context(|| format!("failed to open log file {:?}", args.log_file))?;

    let mut samples_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open("logs/system2_samples.jsonl")
        .context("failed to open samples log logs/system2_samples.jsonl")?;

    writeln!(
        log_file,
        "[system2] starting loop with {} iterations",
        args.iterations
    )?;

    let security_config = SecurityConfig::from_file(&args.security_config)?;
    let security = PromptSecurityManager::new(security_config.clone())?;
    security.audit_config_snapshot(&security_config);

    let torus_config = TorusConfig::from_file(&args.torus_config)?;
    let mut torus_projector = TorusProjector::new(torus_config);
    let mut tcs_analyzer = TCSAnalyzer::new()?;
    let compass_config = CompassConfig::from_file(&args.compass_config)?;
    let compass_engine = CompassEngine::new(compass_config);

    let erag = EragService::initialise(&args.erag_config).await?;
    let experience_store = ExperienceStore::initialise(&args.memory_config).await?;
    let embedder = LocalEmbedder::from_env()?;

    let granite_endpoint = resolve_endpoint(None)?;
    let http_client = reqwest::Client::new();

    let tasks = default_tasks();

    let mut latency_ms = Vec::new();
    let mut quality = Vec::new();
    let mut rouges = Vec::new();
    let mut final_buffer_count = 0usize;
    let mut trainings_triggered = 0usize;
    let mut last_training_summary: Option<Value> = None;

    for iteration in 0..args.iterations {
        let raw_prompt = &tasks[iteration % tasks.len()];
        let secured_prompt = security.enforce_prompt(raw_prompt)?;

        writeln!(
            log_file,
            "[system2] iteration {} prompt_raw: {}",
            iteration + 1,
            raw_prompt
        )?;
        if raw_prompt != &secured_prompt {
            writeln!(
                log_file,
                "[system2] iteration {} prompt_sanitized: {}",
                iteration + 1,
                secured_prompt
            )?;
        }

        // STAGE 2: Embed the prompt (for topology analysis)
        let prompt_embedding = embedder.embed(&secured_prompt)?;

        // STAGE 3: Project embedding onto k-twisted torus manifold to get PAD state
        let pad_state = torus_projector.project(&prompt_embedding)?;
        writeln!(
            log_file,
            "[system2] PAD state: P={:.3} A={:.3} D={:.3} entropy={:.3} surface=[{:.2}, {:.2}, {:.2}]",
            pad_state.pleasure(),
            pad_state.arousal(),
            pad_state.dominance(),
            pad_state.entropy,
            pad_state.surface_position[0],
            pad_state.surface_position[1],
            pad_state.surface_position[2]
        )?;

        // STAGE 4: Compute topological signature from token-level embeddings
        // Generate point cloud from CRDT-synced promoted tokens via FastAPI service
        // Falls back to local tokenizer or word-based if service unavailable
        let topology = tcs_analyzer.analyze_prompt_text(
            &secured_prompt,
            &embedder,
            &pad_state.coordinates,
        ).await?;
        writeln!(
            log_file,
            "[system2] Topology: β₀={} β₁={} β₂={} entropy={:.3} complexity={:.3}",
            topology.betti_0(),
            topology.betti_1(),
            topology.betti_2(),
            topology.persistence_entropy,
            topology.complexity()
        )?;

        // STAGE 5: Compute consciousness compass quadrant
        let compass_state = compass_engine.compute_quadrant(&pad_state, &topology);
        writeln!(
            log_file,
            "[system2] Compass: {} (confidence={:.2}) - {}",
            compass_state.quadrant.as_str(),
            compass_state.confidence,
            compass_engine.strategic_advice(compass_state.quadrant)
        )?;

        // STAGE 6: ERAG memory retrieval with computed compass filter
        let compass_filter = if args.compass == "auto" {
            // Use computed compass quadrant
            Some(compass_state.quadrant.as_str())
        } else {
            // Use CLI override (for baseline comparison)
            Some(args.compass.as_str())
        };

        let start = Instant::now();
        let (augmented_prompt, memories) =
            augment_prompt_with_memory(&erag, &secured_prompt, compass_filter).await?;

        // STAGE 7: Generate with Granite
        let completion =
            generate_completion(&http_client, &granite_endpoint, &augmented_prompt).await?;
        let latency = start.elapsed().as_millis() as u64;
        latency_ms.push(latency);

        writeln!(
            log_file,
            "[system2] latency_ms={} token_usage={:?}",
            latency, completion.usage
        )?;

        let text = completion
            .choices
            .first()
            .map(|choice| choice.text.trim().to_string())
            .unwrap_or_default();

        // POST-GENERATION: Curator scoring
        let curator = run_curator(&secured_prompt, &text)?;
        writeln!(
            log_file,
            "[system2] quality score {:?} rouge {:.3} feedback: {}",
            curator.quality_score,
            curator.rouge_l,
            curator.feedback.replace('\n', " ")
        )?;

        quality.push(curator.quality_score.unwrap_or(0) as f32);
        rouges.push(curator.rouge_l);

        let context_text = if memories.is_empty() {
            String::new()
        } else {
            augmented_prompt
                .splitn(2, "\n\nUser Prompt:\n")
                .next()
                .unwrap_or_default()
                .to_string()
        };

        // Store experience with full metadata
        let mut embedding = prompt_embedding.clone();
        match embedding.len().cmp(&experience_store.vector_size()) {
            std::cmp::Ordering::Less => embedding.resize(experience_store.vector_size(), 0.0),
            std::cmp::Ordering::Greater => embedding.truncate(experience_store.vector_size()),
            std::cmp::Ordering::Equal => {}
        }

        let experience = Experience::new(
            secured_prompt.clone(),
            text.clone(),
            context_text,
            "general_task".to_string(),
            curator.quality_score,
            curator.rouge_l,
            curator.feedback.clone(),
            curator
                .quality_score
                .map(|q| q as f64 / 10.0)
                .unwrap_or(0.0),
            Some(json!({
                "latency_ms": latency,
                "memory_hits": memories.len(),
                "pad_state": {
                    "coordinates": pad_state.coordinates,
                    "entropy": pad_state.entropy,
                    "pleasure": pad_state.pleasure(),
                    "arousal": pad_state.arousal(),
                    "dominance": pad_state.dominance(),
                    "surface_position": pad_state.surface_position,
                },
                "topology": {
                    "betti_numbers": topology.betti_numbers,
                    "persistence_entropy": topology.persistence_entropy,
                    "complexity": topology.complexity(),
                },
                "compass": {
                    "quadrant": compass_state.quadrant.as_str(),
                    "confidence": compass_state.confidence,
                },
            })),
        );

        experience_store.upsert(&experience, &embedding).await?;

        let sample_payload = json!({
            "prompt": secured_prompt.clone(),
            "prompt_raw": raw_prompt,
            "response": text,
            "feedback": curator.feedback.clone(),
            "quality_score": curator.quality_score,
            "rouge_l": curator.rouge_l,
        });
        let learning = run_learning_loop(&args.learning_config, &sample_payload)?;
        writeln!(
            log_file,
            "[system2] learning status {} buffer {} reason {}",
            learning.status,
            learning.buffer_count,
            learning.reason.clone().unwrap_or_else(|| "-".to_string())
        )?;

        let sample_line = serde_json::to_string(&json!({
            "iteration": iteration + 1,
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "prompt_raw": raw_prompt,
            "prompt": secured_prompt,
            "augmented_prompt": augmented_prompt,
            "response": text,
            "curator": curator,
            "learning": learning,
        }))?;
        writeln!(samples_file, "{}", sample_line)?;
        final_buffer_count = learning.buffer_count;
        if learning.triggered_training {
            trainings_triggered += 1;
        }
        if let Some(summary) = learning.training_summary.clone() {
            last_training_summary = Some(summary);
        }
    }

    let baseline = json!({
        "iterations": args.iterations,
        "latencies_ms": {
            "avg": average_u64(&latency_ms),
            "p50": percentile(&latency_ms, 0.5),
            "p95": percentile(&latency_ms, 0.95),
        },
        "quality_score": {
            "avg": average(&quality),
        },
        "rouge_l": {
            "avg": average(&rouges),
        },
        "learning_loop": {
            "final_buffer_count": final_buffer_count,
            "trainings_triggered": trainings_triggered,
            "last_training_summary": last_training_summary,
        },
    });

    if let Some(parent) = args.baseline_file.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&args.baseline_file, serde_json::to_vec_pretty(&baseline)?)?;
    writeln!(
        log_file,
        "[system2] baseline written to {:?}",
        args.baseline_file
    )?;

    println!(
        "[system2] completed {} iterations | avg quality {:.2} | avg rouge {:.3} | buffer {}",
        args.iterations,
        baseline
            .get("quality_score")
            .and_then(|v| v.get("avg"))
            .and_then(Value::as_f64)
            .unwrap_or_default(),
        baseline
            .get("rouge_l")
            .and_then(|v| v.get("avg"))
            .and_then(Value::as_f64)
            .unwrap_or_default(),
        final_buffer_count
    );

    Ok(())
}

fn resolve_endpoint(override_endpoint: Option<String>) -> Result<reqwest::Url> {
    let raw = override_endpoint
        .or_else(|| std::env::var("NIODOO_VLLM_ENDPOINT").ok())
        .unwrap_or_else(|| "http://127.0.0.1:8000/v1/completions".to_string());
    reqwest::Url::parse(&raw).with_context(|| format!("invalid endpoint URL: {}", raw))
}

fn default_tasks() -> Vec<String> {
    vec![
        "Explain hyperfocus detection.".to_string(),
        "How do we avoid cooldown drift in ERAG?".to_string(),
        "What payloads should we log after a breakthrough?".to_string(),
        "Summarize the memory cadence during a Discover quadrant.".to_string(),
        "When should beta_meta snapshots be persisted?".to_string(),
        "Outline how Reconnaissance quadrant memories should be triaged before storage."
            .to_string(),
        "Describe the signals that indicate the Master quadrant is approaching saturation."
            .to_string(),
        "Suggest safeguards to prevent hallucinated payloads from entering ERAG.".to_string(),
        "Explain how the Consciousness Compass influences memory replay cadence.".to_string(),
        "Identify which telemetry proves a Discover-to-Persist transition succeeded.".to_string(),
        "Draft a checklist for handing off Discover quadrant findings to the Persist quadrant owner."
            .to_string(),
        "List three failure modes that would trigger a Compass quadrant rollback.".to_string(),
        "Explain how torus projections help reconcile conflicting memories.".to_string(),
        "Propose monitoring hooks to detect Reconnaissance quadrant blind spots.".to_string(),
        "Define the minimum evidence required to promote a memory into the Master quadrant."
            .to_string(),
    ]
}

async fn generate_completion(
    client: &reqwest::Client,
    endpoint: &reqwest::Url,
    prompt: &str,
) -> Result<CompletionResponse> {
    let payload = CompletionRequest {
        model: "ibm-granite/granite-3b-code-instruct",
        prompt,
        max_tokens: 256,
        temperature: 0.1,
    };

    let response = client
        .post(endpoint.clone())
        .json(&payload)
        .send()
        .await
        .context("request to Granite failed")?;

    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_else(|_| "<no body>".into());
        return Err(anyhow!("Granite request failed: {}\n{}", status, body));
    }

    let completion = response
        .json::<CompletionResponse>()
        .await
        .context("failed to decode completion response")?;

    Ok(completion)
}

fn run_curator(prompt: &str, response: &str) -> Result<CuratorOutput> {
    let output = Command::new("python3")
        .arg("src/curator.py")
        .arg(prompt)
        .arg(response)
        .output()
        .context("failed to launch curator scorer")?;

    if !output.status.success() {
        return Err(anyhow!(
            "curator scoring failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    let result = serde_json::from_slice::<CuratorOutput>(&output.stdout)
        .context("failed to parse curator output")?;

    Ok(result)
}

fn run_learning_loop(config_path: &str, sample: &Value) -> Result<LearningLoopOutput> {
    let output = Command::new("python3")
        .arg("src/learning_loop.py")
        .arg("--config")
        .arg(config_path)
        .arg("process-sample")
        .arg("--sample")
        .arg(sample.to_string())
        .output()
        .context("failed to launch learning loop controller")?;

    if !output.status.success() {
        return Err(anyhow!(
            "learning loop controller failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    let result = serde_json::from_slice::<LearningLoopOutput>(&output.stdout)
        .context("failed to parse learning loop output")?;

    Ok(result)
}

fn average_u64(values: &[u64]) -> f32 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().map(|v| *v as f32).sum::<f32>() / values.len() as f32
    }
}

fn average(values: &[f32]) -> f32 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f32>() / values.len() as f32
    }
}

fn percentile(samples: &[u64], quantile: f32) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let idx = ((sorted.len() - 1) as f32 * quantile).round() as usize;
    sorted[idx] as f32
}

#[derive(Debug, Deserialize)]
struct CompletionResponse {
    id: Option<String>,
    model: Option<String>,
    choices: Vec<CompletionChoice>,
    usage: Option<Usage>,
}

#[derive(Debug, Deserialize)]
struct CompletionChoice {
    text: String,
}

#[derive(Debug, Deserialize)]
struct Usage {
    prompt_tokens: Option<u32>,
    completion_tokens: Option<u32>,
    total_tokens: Option<u32>,
}

#[derive(serde::Serialize)]
struct CompletionRequest<'a> {
    model: &'a str,
    prompt: &'a str,
    max_tokens: u32,
    temperature: f32,
}
