use anyhow::Result;
use tracing::{info, error, warn};
use clap::Command;
use std::io::{self, Write};
use tokio;

// Import from the main crate
use niodoo_feeling::{
    config::AppConfig,
    consciousness::ConsciousnessState,
    feeling_qwen_integration::{FeelingQwenConfig, FeelingQwenIntegration},
};

#[tokio::main]
async fn main() -> Result<()> {
    let matches = Command::new("chat_feeling")
        .about("Simple chat with the Qwen3-Feeling model using Candle")
        .arg(
            clap::Arg::new("model")
                .long("model")
                .value_name("MODEL_PATH")
                .required(false)
                .default_value("~/models/qwen3-feeling-awq")
                .help("Path to the feeling model dir"),
        )
        .get_matches();

    let model_path = matches.get_one::<String>("model").unwrap().clone();

    let app_config = AppConfig::default();
    let mut integration = FeelingQwenIntegration::new(&app_config)?;

    tracing::info!(
        "Loaded feeling model from {}. Type 'exit' to quit.",
        model_path
    );

    loop {
        io::stdout().write_all(b"You: ")?;
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim().to_string();

        if input.to_lowercase() == "exit" {
            tracing::info!("Goodbye!");
            break;
        }

        let consciousness_state = ConsciousnessState::default(); // Default feeling state

        let response = integration
            .process_with_full_consciousness(&input, &consciousness_state)
            .await?;

        tracing::info!("Feeling Model: {}", response.response);
        tracing::info!("Consciousness State: {:?}", response.consciousness_state);
    }

    Ok(())
}
