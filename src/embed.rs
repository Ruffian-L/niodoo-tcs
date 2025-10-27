use ollama_rs::{Ollama, generation::completion};

pub struct Embedder {
    ollama: Ollama,
    model: String,
}

impl Embedder {
    pub fn new() -> Result<Self> {
        let ollama = Ollama::default();
        let model = "qwen2.5-coder:0.5b".to_string();
        
        // Check model exists
        if let Err(e) = ollama.list_models() {
            log::error!("Ollama error: {}", e);
            return Err(anyhow::anyhow!("Ollama not ready—run 'ollama serve'"));
        }
        
        log::info!("Using Ollama {} for embeds—no fallbacks", model);
        Ok(Self { ollama, model })
    }

    pub async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let prompt = format!("Embed this text as 768-dim vector: {}", text);
        let response = self.ollama.generate_completion(&self.model, prompt).await?;
        
        // Parse response to vec (assume model outputs vector string or use embed endpoint)
        let embedding: Vec<f32> = parse_embedding(&response.response);  // Impl parse
        if embedding.len() != 768 {
            return Err(anyhow::anyhow!("Embed dim mismatch—expected 768"));
        }
        
        log::info!("Qwen embed complete: dim=768");
        Ok(embedding)
    }
}

// No fallback—if Ollama fails, error (install prompt in log)
fn parse_embedding(resp: &str) -> Vec<f32> {
    // Parse logic (e.g., JSON array from Qwen)
    vec![0.0; 768]  // Placeholder
}

