use std::time::Instant;
use anyhow::Result;
use niodoo_integrated::{QwenEmbedder, GenerationEngine, TokenizedResult};

#[tokio::main(flavor = "multi_thread", worker_threads = 8)]
async fn main() -> Result<()> {
    println!("🚀 NIODOO Optimized Performance Test - Before vs After");
    
    let embedder = QwenEmbedder::new()?;
    let generator = GenerationEngine::new()?;
    
    let test_prompts = vec![
        "What is consciousness?",
        "How do you process emotions?",
        "What drives your decision-making?",
    ];
    
    for (i, prompt) in test_prompts.iter().enumerate() {
        println!("\n🔍 Test {}/{}: {}", i + 1, test_prompts.len(), prompt);
        
        let cycle_start = Instant::now();
        
        // 1. Embedding (fast)
        let step_start = Instant::now();
        let embedding = embedder.embed(prompt).await?;
        println!("  ✅ Embedding: {} dims in {:?}", embedding.len(), step_start.elapsed());
        
        // 2. Generation (optimized)
        let step_start = Instant::now();
        let mock_tokenized = TokenizedResult {
            tokens: prompt.split_whitespace().map(|s| s.to_string()).collect(),
            promotions: vec!["Test promotion".to_string()],
            mirage_applied: true,
        };
        let generation = generator.generate(&mock_tokenized).await?;
        println!("  ✅ Generation: {} chars in {:?}", generation.text.len(), step_start.elapsed());
        
        let total_time = cycle_start.elapsed();
        println!("  🏁 Total cycle: {:?}", total_time);
        
        if total_time.as_secs() < 10 {
            println!("  🎯 OPTIMIZED: Sub-10s cycle achieved!");
        }
        
        println!("  💬 Response: {}", &generation.text[..std::cmp::min(100, generation.text.len())]);
    }
    
    Ok(())
}