use std::env;

use anyhow::{Context, Result};
use tcs_ml::{Brain, MotorBrain};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    let mut args = env::args().skip(1);
    let model_path = args
        .next()
        .context("expected model path as first argument")?;
    let prompt = args.collect::<Vec<String>>().join(" ");
    let prompt = if prompt.is_empty() {
        "Hello, Qwen!".to_string()
    } else {
        prompt
    };

    let mut brain = MotorBrain::new()?;
    brain
        .load_model(&model_path)
        .await
        .with_context(|| format!("failed to load model at {}", model_path))?;

    let response = brain
        .process(&prompt)
        .await
        .with_context(|| "failed to run motor brain inference".to_string())?;

    println!("{}", response);
    Ok(())
}
