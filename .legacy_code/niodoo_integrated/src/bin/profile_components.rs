use std::time::Instant;
use anyhow::Result;
use niodoo_integrated::{QwenEmbedder, GenerationEngine, ERAGSystem, TokenizerEngine, TokenizedResult};

#[tokio::main(flavor = "multi_thread", worker_threads = 8)]
async fn main() -> Result<()> {
    println!("🔥 NIODOO ML Pipeline Profiling - Component Test");
    
    // Test each component individually to isolate the bottleneck
    let test_prompt = "What is the meaning of existence?";
    
    // 1. Test Embedding
    println!("\n🔍 Testing QwenEmbedder...");
    let start = Instant::now();
    let embedder = QwenEmbedder::new()?;
    let embedding = embedder.embed(test_prompt).await?;
    println!("✅ Embedding: {} dims in {:?}", embedding.len(), start.elapsed());
    
    // 2. Test ERAG (this might use mock mode if Qdrant not set up)
    println!("\n🔍 Testing ERAG System...");
    let start = Instant::now();
    let erag = ERAGSystem::new(0.8)?;
    // Skip ERAG for now as it needs compass_result
    println!("✅ ERAG init in {:?}", start.elapsed());
    
    // 3. Test Tokenizer
    println!("\n🔍 Testing Tokenizer...");
    let start = Instant::now();
    let tokenizer = TokenizerEngine::new(0.1)?;
    // Skip tokenizer for now as it needs erag_result
    println!("✅ Tokenizer init in {:?}", start.elapsed());
    
    // 4. Test vLLM Generation (likely the bottleneck)
    println!("\n🔍 Testing vLLM Generation (THE SUSPECTED BOTTLENECK)...");
    let start = Instant::now();
    let generator = GenerationEngine::new()?;
    
    // Create a mock tokenized input for direct generation test
    let mock_tokenized = TokenizedResult {
        tokens: vec!["What".to_string(), "is".to_string(), "existence".to_string()],
        promotions: vec!["Testing promotion".to_string()],
        mirage_applied: true,
    };
    
    let generation = generator.generate(&mock_tokenized).await?;
    let vllm_time = start.elapsed();
    println!("✅ vLLM Generation: {} chars in {:?}", generation.text.len(), vllm_time);
    
    if vllm_time.as_secs() > 10 {
        println!("🎯 BOTTLENECK FOUND: vLLM generation took {}s - this is our 26s culprit!", vllm_time.as_secs());
        println!("💡 vLLM Response: {}", &generation.text[..std::cmp::min(200, generation.text.len())]);
    }
    
    Ok(())
}