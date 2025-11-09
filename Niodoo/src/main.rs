use std::io::{self, Read};
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use niodoo_cli::context::augment_prompt_with_memory;
use niodoo_cli::erag::EragService;
use reqwest::Url;
use serde::{Deserialize, Serialize};

#[derive(Parser, Debug)]
#[command(author, version, about = "Niodoo Granite CLI (System 0/1)")]
struct Cli {
    /// Enable ERAG memory retrieval before generation
    #[arg(long)]
    with_memory: bool,

    /// Path to ERAG TOML configuration
    #[arg(long, default_value = "config/erag.toml")]
    erag_config: String,

    /// Compass quadrant filter (e.g. Panic, Persist, Discover, Master)
    #[arg(long)]
    compass: Option<String>,
    /// Prompt to send to the Granite model (reads stdin if omitted)
    #[arg(long)]
    prompt: Option<String>,

    /// Model identifier to use
    #[arg(long, default_value = "ibm-granite/granite-3b-code-instruct")]
    model: String,

    /// Maximum tokens to generate
    #[arg(long, default_value_t = 256)]
    max_tokens: u32,

    /// Temperature for sampling
    #[arg(long, default_value_t = 0.7)]
    temperature: f32,

    /// Output mode (text or json)
    #[arg(long, default_value = "text")]
    output: OutputMode,

    /// Override the vLLM endpoint
    #[arg(long)]
    endpoint: Option<String>,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum OutputMode {
    Text,
    Json,
}

#[derive(Serialize)]
struct CompletionRequest<'a> {
    model: &'a str,
    prompt: &'a str,
    max_tokens: u32,
    temperature: f32,
}

#[derive(Serialize, Deserialize)]
struct CompletionResponse {
    id: Option<String>,
    model: Option<String>,
    choices: Vec<CompletionChoice>,
    usage: Option<Usage>,
}

#[derive(Serialize, Deserialize)]
struct CompletionChoice {
    text: String,
}

#[derive(Serialize, Deserialize)]
struct Usage {
    prompt_tokens: Option<u32>,
    completion_tokens: Option<u32>,
    total_tokens: Option<u32>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Cli::parse();
    let source_prompt = obtain_prompt(args.prompt)?;
    let endpoint = resolve_endpoint(args.endpoint)?;

    let erag_service = if args.with_memory {
        Some(EragService::initialise(&args.erag_config).await?)
    } else {
        None
    };

    let (prompt, _memories) = if let Some(service) = &erag_service {
        augment_prompt_with_memory(service, &source_prompt, args.compass.as_deref()).await?
    } else {
        (source_prompt.clone(), Vec::new())
    };

    if args.with_memory {
        eprintln!(
            "[ERAG] Augmented prompt:
{}",
            prompt
        );
    }

    let payload = CompletionRequest {
        model: &args.model,
        prompt: &prompt,
        max_tokens: args.max_tokens,
        temperature: args.temperature,
    };

    let client = reqwest::Client::builder()
        .build()
        .context("failed to construct HTTP client")?;

    let start = Instant::now();
    let response = client
        .post(endpoint.clone())
        .json(&payload)
        .send()
        .await
        .with_context(|| format!("request to {} failed", endpoint))?;

    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_else(|_| "<no body>".into());
        anyhow::bail!("request failed: {}\n{}", status, body);
    }

    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    let completion: CompletionResponse = response
        .json()
        .await
        .context("failed to decode completion response")?;

    match args.output {
        OutputMode::Text => emit_text(&completion, elapsed_ms)?,
        OutputMode::Json => emit_json(&completion, elapsed_ms)?,
    }

    Ok(())
}

fn obtain_prompt(arg_prompt: Option<String>) -> Result<String> {
    if let Some(prompt) = arg_prompt {
        return Ok(prompt);
    }

    let mut buffer = String::new();
    io::stdin()
        .read_to_string(&mut buffer)
        .context("failed to read prompt from stdin")?;
    let prompt = buffer.trim().to_owned();
    if prompt.is_empty() {
        anyhow::bail!("no prompt provided via --prompt or stdin");
    }
    Ok(prompt)
}

fn resolve_endpoint(override_endpoint: Option<String>) -> Result<Url> {
    let raw = override_endpoint
        .or_else(|| std::env::var("NIODOO_VLLM_ENDPOINT").ok())
        .unwrap_or_else(|| "http://127.0.0.1:8000/v1/completions".to_string());
    Url::parse(&raw).with_context(|| format!("invalid endpoint URL: {}", raw))
}

fn emit_text(completion: &CompletionResponse, elapsed_ms: f64) -> Result<()> {
    if let Some(choice) = completion.choices.first() {
        println!("{}", choice.text.trim_end());
    } else {
        println!("<no text>");
    }
    eprintln!(
        "[niodoo-cli] latency_ms={:.2} prompt_tokens={:?} completion_tokens={:?} total_tokens={:?}",
        elapsed_ms,
        completion.usage.as_ref().and_then(|u| u.prompt_tokens),
        completion.usage.as_ref().and_then(|u| u.completion_tokens),
        completion.usage.as_ref().and_then(|u| u.total_tokens),
    );
    Ok(())
}

fn emit_json(completion: &CompletionResponse, elapsed_ms: f64) -> Result<()> {
    #[derive(Serialize)]
    struct Envelope<'a> {
        latency_ms: f64,
        completion: &'a CompletionResponse,
    }

    let envelope = Envelope {
        latency_ms: elapsed_ms,
        completion,
    };
    serde_json::to_writer_pretty(io::stdout(), &envelope)
        .context("failed to serialize JSON output")?;
    println!();
    Ok(())
}
