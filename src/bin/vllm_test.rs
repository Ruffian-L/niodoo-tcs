use niodoo_consciousness::vllm_bridge::VLLMBridge;
use std::env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let vllm_host = env::var("VLLM_HOST").unwrap_or_else(|_| "localhost".to_string());
    let vllm_port = env::var("VLLM_PORT").unwrap_or_else(|_| "8000".to_string());
    let vllm_url = format!("http://{}:{}", vllm_host, vllm_port);

    println!("🔍 Connecting to vLLM at: {}", vllm_url);

    let bridge = VLLMBridge::connect(&vllm_url, None)?;
    println!("✅ Connected!");

    let health = bridge.health().await?;
    println!("🏥 Health check: {}", health);

    if health {
        let response = bridge
            .generate(
                "Hello, how are you?",
                50,
                0.7,
                0.9,
            )
            .await?;
        println!("🤖 Response: {}", response);
    }

    Ok(())
}
