use std::time::Instant;
use anyhow::Result;
use niodoo_integrated::NiodooIntegrated;

#[tokio::main(flavor = "multi_thread", worker_threads = 8)]
async fn main() -> Result<()> {
    println!("🔥 NIODOO Latency Profiling - Mini Test (5 cycles)");
    
    let mut niodoo = NiodooIntegrated::new().await?;
    let test_prompts = vec![
        "What is the meaning of existence?".to_string(),
        "How do you feel about uncertainty?".to_string(), 
        "Describe your consciousness".to_string(),
        "What threatens your wellbeing?".to_string(),
        "How do you learn from experience?".to_string(),
    ];
    
    for (i, prompt) in test_prompts.iter().enumerate() {
        let start = Instant::now();
        
        println!("Cycle {}/5: Starting prompt processing...", i + 1);
        let result = niodoo.process_pipeline(prompt).await?;
        
        let latency = start.elapsed();
        println!("Cycle {}/5: Completed in {:?}", i + 1, latency);
        println!("Response length: {} chars", result.response.len());
        println!("Entropy: {:.2}, Threat: {}, Healing: {}", result.entropy, result.is_threat, result.is_healing);
        println!("---");
    }
    
    Ok(())
}