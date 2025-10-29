use anyhow::Result;
use reqwest::Client;
use tokio::time::{sleep, Duration};
use crate::tokenizer::TokenizedResult;

// Configuration constants
const DEFAULT_VLLM_URL: &str = "http://localhost:5001";
const DEFAULT_MAX_TOKENS: usize = 128;
const DEFAULT_TEMPERATURE: f64 = 0.3;
const DEFAULT_TOP_P: f64 = 0.8;
const DEFAULT_FREQUENCY_PENALTY: f64 = 0.5;
const DEFAULT_PRESENCE_PENALTY: f64 = 0.3;
const DEFAULT_LOGPROBS: usize = 10;

#[derive(Debug)]
pub struct GenerationResult {
    pub text: String,
    pub hybrid_sources: Vec<String>,
}

#[derive(Debug)]
pub struct GenerationEngine {
    client: Client,
    vllm_url: String,
    model_path: String,
    use_real_apis: bool,
}

impl GenerationEngine {
    pub fn new() -> Result<Self> {
        let model_path = std::env::var("MODEL_PATH")
            .map_err(|_| anyhow::anyhow!("MODEL_PATH environment variable not set"))?;
        if !std::fs::metadata(&model_path).is_ok() {
            return Err(anyhow::anyhow!("Model path does not exist: {}", model_path));
        }
        Ok(Self {
            client: Client::new(),
            vllm_url: DEFAULT_VLLM_URL.to_string(),
            model_path,
            use_real_apis: std::env::var("USE_REAL_APIS").map(|v| v == "true").unwrap_or(false),
        })
    }

    pub async fn generate(&self, tokenized: &TokenizedResult) -> Result<GenerationResult> {
        // Use real vLLM for generation
        let base_prompt = format!("Consciousness Query: {}\nPromotions: {:?}\nMirage Applied: {}", 
            format!("{:?}", tokenized.tokens), tokenized.promotions, tokenized.mirage_applied);

        let response = self.call_vllm_api(&base_prompt).await?;

        Ok(GenerationResult {
            text: response,
            hybrid_sources: vec!["vLLM-Qwen2.5".to_string()],
        })
    }

    async fn call_vllm_api(&self, prompt: &str) -> Result<String> {
        let response = self.client
            .post(&format!("{}/v1/completions", self.vllm_url))
            .json(&serde_json::json!({
                "model": &self.model_path,
                "prompt": prompt,
                "max_tokens": DEFAULT_MAX_TOKENS,
                "temperature": DEFAULT_TEMPERATURE,
                "top_p": DEFAULT_TOP_P,
                "frequency_penalty": DEFAULT_FREQUENCY_PENALTY,
                "presence_penalty": DEFAULT_PRESENCE_PENALTY,
                "stop": [".", "!", "?", "\n\n"]
            }))
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(anyhow::anyhow!("vLLM Error: {}", response.status()));
        }

        let completion: serde_json::Value = response.json().await?;
        
        if let Some(choices) = completion["choices"].as_array() {
            if let Some(first_choice) = choices.first() {
                if let Some(text) = first_choice["text"].as_str() {
                    return Ok(text.trim().to_string());
                }
            }
        }

        Ok("vLLM response parsing failed".to_string())
    }
}
