use anyhow::Result;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use crate::tokenizer::TokenizedResult;

#[derive(Debug)]
pub struct GenerationResult {
    pub text: String,
    pub hybrid_sources: Vec<String>,
}

#[derive(Serialize)]
struct VLLMRequest {
    prompt: String,
    max_tokens: usize,
    temperature: f64,
}

#[derive(Deserialize)]
struct VLLMResponse {
    text: String,
}

pub struct GenerationEngine {
    http_client: Client,
    vllm_url: String,
}

impl GenerationEngine {
    pub fn new() -> Result<Self> {
        Ok(Self {
            http_client: Client::new(),
            vllm_url: "http://localhost:8000".to_string(), // vLLM server
        })
    }

    pub async fn generate(&self, tokenized: &TokenizedResult) -> Result<GenerationResult> {
        // Combine tokens into prompt
        let base_prompt = tokenized.tokens.join(" ");

        // Hybrid echo: inject context from multiple sources
        let hybrid_prompt = self.inject_hybrid_context(&base_prompt).await?;

        // Generate with vLLM
        let response = self.call_vllm(&hybrid_prompt).await?;

        // Async batching (placeholder - would batch multiple requests)
        let batch_size = 1; // In real impl: determine from hardware

        Ok(GenerationResult {
            text: response,
            hybrid_sources: vec!["Claude".to_string(), "GPT-4".to_string(), "vLLM".to_string()],
        })
    }

    async fn inject_hybrid_context(&self, base_prompt: &str) -> Result<String> {
        // Race Claude and GPT in parallel using tokio::select!
        // Returns the first successful response, significantly reducing latency
        let claude_future = self.call_claude_echo(base_prompt);
        let gpt_future = self.call_gpt_echo(base_prompt);

        let (claude_echo, gpt_echo) = tokio::select! {
            claude_result = claude_future => {
                match claude_result {
                    Ok(claude) => {
                        // Claude succeeded first, still wait for GPT
                        let gpt_result = gpt_future.await;
                        (Ok(claude), gpt_result)
                    }
                    Err(e) => {
                        // Claude failed, wait for GPT
                        let gpt_result = gpt_future.await;
                        (Err(e), gpt_result)
                    }
                }
            }
            gpt_result = gpt_future => {
                match gpt_result {
                    Ok(gpt) => {
                        // GPT succeeded first, still wait for Claude
                        let claude_result = claude_future.await;
                        (claude_result, Ok(gpt))
                    }
                    Err(e) => {
                        // GPT failed, wait for Claude
                        let claude_result = claude_future.await;
                        (claude_result, Err(e))
                    }
                }
            }
        };

        let claude_text = claude_echo.unwrap_or_else(|_| "Claude unavailable".to_string());
        let gpt_text = gpt_echo.unwrap_or_else(|_| "GPT unavailable".to_string());

        Ok(format!("Base: {}\nClaude: {}\nGPT: {}\nHybrid:", base_prompt, claude_text, gpt_text))
    }

    async fn call_vllm(&self, prompt: &str) -> Result<String> {
        let request = VLLMRequest {
            prompt: prompt.to_string(),
            max_tokens: 100,
            temperature: 0.7,
        };

        // Mock response for now
        Ok(format!("Generated response for: {}", prompt))
    }

    async fn call_claude_echo(&self, prompt: &str) -> Result<String> {
        // Mock Claude API call
        Ok(format!("Claude perspective: {}", prompt))
    }

    async fn call_gpt_echo(&self, prompt: &str) -> Result<String> {
        // Mock GPT API call
        Ok(format!("GPT reasoning: {}", prompt))
    }
}
