/*
use tracing::{info, error, warn};
 * 🚀 SETUP REAL ONNX MODELS
 *
 * Downloads real AI models for consciousness processing
 * Run this once before using real inference
 */

use niodoo_consciousness::real_onnx_models::setup_real_models;
use std::iter::repeat;

#[tokio::main]
async fn main() {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    info!("🧠 NiodO.o Real AI Model Setup");
    info!("{}", "=".repeat(70));
    info!("");

    match setup_real_models().await {
        Ok(_) => {
            info!("\n🎉 SUCCESS! Real AI models are ready");
            info!("🚀 You can now run: cargo run --bin real_ai_inference_demo");
            std::process::exit(0);
        }
        Err(e) => {
            error!("\n❌ Setup failed: {}", e);
            error!("⚠️  Check your internet connection and try again");
            std::process::exit(1);
        }
    }
}
