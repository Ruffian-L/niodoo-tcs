/// Test binary for API clients
/// This tests the Claude and GPT clients with actual API calls if keys are available
use niodoo_real_integrated::api_clients::{ClaudeClient, GptClient};
use std::env;
use tracing::{error, info};

#[tokio::main]
async fn main() {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    // Check for API keys
    let claude_api_key = env::var("ANTHROPIC_API_KEY").ok();
    let gpt_api_key = env::var("OPENAI_API_KEY").ok();

    println!("\n=== API Clients Test ===\n");
    println!("Claude API Key Present: {}", claude_api_key.is_some());
    println!("GPT API Key Present: {}\n", gpt_api_key.is_some());

    // Test Claude client
    if let Some(key) = claude_api_key {
        println!("Testing Claude Client...");
        match ClaudeClient::new(key, "claude-sonnet-4-5-20250514", 5) {
            Ok(client) => {
                info!("✓ Claude client created successfully");
                println!("  - Endpoint: {}", client.endpoint());
                println!("  - Model: {}", client.model());

                // Try to make an actual API call
                let test_prompt = "Say 'API client test successful' in one sentence";
                match client.complete(test_prompt).await {
                    Ok(response) => {
                        println!("  ✓ Claude API Response: {}", response);
                    }
                    Err(e) => {
                        eprintln!("  ✗ Claude API Error: {}", e);
                    }
                }
            }
            Err(e) => {
                error!("✗ Failed to create Claude client: {}", e);
            }
        }
    } else {
        println!("⚠ Skipping Claude test - ANTHROPIC_API_KEY not found in environment");
    }

    println!();

    // Test GPT client
    if let Some(key) = gpt_api_key {
        println!("Testing GPT Client...");
        match GptClient::new(key, "gpt-4o", 5) {
            Ok(client) => {
                info!("✓ GPT client created successfully");
                println!("  - Endpoint: {}", client.endpoint());
                println!("  - Model: {}", client.model());

                // Try to make an actual API call
                let test_prompt = "Say 'API client test successful' in one sentence";
                match client.complete(test_prompt).await {
                    Ok(response) => {
                        println!("  ✓ GPT API Response: {}", response);
                    }
                    Err(e) => {
                        eprintln!("  ✗ GPT API Error: {}", e);
                    }
                }
            }
            Err(e) => {
                error!("✗ Failed to create GPT client: {}", e);
            }
        }
    } else {
        println!("⚠ Skipping GPT test - OPENAI_API_KEY not found in environment");
    }

    println!("\n=== Test Complete ===\n");
}
