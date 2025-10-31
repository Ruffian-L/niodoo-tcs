use anyhow::Result;
use niodoo_real_integrated::config::CliArgs;
use niodoo_real_integrated::pipeline::Pipeline;

#[tokio::main]
async fn main() -> Result<()> {
    // Disable memory store for diagnostic run to avoid external dependencies
    std::env::set_var("DISABLE_MEMORY_STORE", "1");

    // Use MOCK_MODE if external services are not available
    let vllm = std::env::var("VLLM_ENDPOINT").unwrap_or_else(|_| "http://127.0.0.1:5001".to_string());
    let ollama = std::env::var("OLLAMA_URL").unwrap_or_else(|_| "http://127.0.0.1:11434".to_string());
    if vllm.contains("127.0.0.1") || ollama.contains("127.0.0.1") {
        // Keep current env; single cycle will work in mock if needed
    }

    let args = CliArgs::default();
    let mut pipeline = Pipeline::initialise(args).await?;

    let prompt = std::env::args().nth(1).unwrap_or_else(|| "diagnostic prompt".to_string());
    let cycle = pipeline.process_prompt(&prompt).await?;
    println!("{}", serde_json::to_string_pretty(&serde_json::json!({
        "latency_ms": cycle.latency_ms,
        "entropy": cycle.entropy,
        "rouge": cycle.rouge,
        "failure": cycle.failure,
        "hybrid": cycle.hybrid_response,
    }))?);

    Ok(())
}


