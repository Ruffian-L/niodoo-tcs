use anyhow::Result;
use tcs_pipeline::{TCSConfig, TCSOrchestrator};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    println!("🎯 TCS Integration Test with Qwen2.5 Model");
    println!("{}", "=".repeat(50));

    // Set up the ONNX runtime library path
    std::env::set_var(
        "ORT_DYLIB_PATH",
        "/home/ruffian/Desktop/Niodoo-Final/third_party/onnxruntime-linux-x64-1.18.1/lib/libonnxruntime.so",
    );

    // Configure the orchestrator
    let config = TCSConfig::default();
    let mut orchestrator = TCSOrchestrator::with_config(100, config)?;

    // Load the Qwen2.5 model
    let model_path = "/home/ruffian/Desktop/Niodoo-Final/models/qwen2.5-coder-0.5b-instruct-onnx/onnx/model_fp16.onnx";
    
    // Test prompts to demonstrate different capabilities
    let test_prompts = vec![
        "Write a short Rust function that adds two numbers.",
        "Explain consciousness in simple terms.",
        "How do neural networks work?",
        "Create a creative story about AI.",
        "Help me debug this code.",
    ];

    for (i, prompt) in test_prompts.iter().enumerate() {
        println!("\n🧠 Test {} - Processing: \"{}\"", i + 1, prompt);
        println!("{}", "-".repeat(40));

        match orchestrator.process(prompt).await {
            Ok(events) => {
                println!("✅ Generated {} topological events:", events.len());
                for (j, event) in events.iter().enumerate() {
                    println!("  Event {}: {:?}", j + 1, event);
                }
            }
            Err(e) => {
                println!("❌ Error processing prompt: {}", e);
            }
        }
    }

    println!("\n🎉 Integration test completed!");
    Ok(())
}