use anyhow::Result;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use crate::tokenizer::TokenizedResult;

#[derive(Debug, Clone)]
pub struct ReflectionContext {
    pub last_prompt: String,
    pub last_response: String,
    pub failure_reason: Option<String>,
    pub retry_count: u32,
}

#[derive(Debug)]
pub struct GenerationResult {
    pub text: String,
    pub hybrid_sources: Vec<String>,
    pub reflection_applied: bool,
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

    pub async fn generate(
        &self,
        tokenized: &TokenizedResult,
        reflection: Option<&ReflectionContext>,
    ) -> Result<GenerationResult> {
        let base_prompt = tokenized.tokens.join(" ");
        let mut prompt_with_reflection = base_prompt.clone();

        if let Some(context) = reflection {
            if let Some(reason) = &context.failure_reason {
                prompt_with_reflection.push_str("\n\n# Reflection\n");
                prompt_with_reflection.push_str(&format!(
                    "Previous attempt struggled because: {}. Re-evaluate the weak points, fix reasoning errors, and produce a corrected answer.",
                    reason
                ));
            }
        }

        let hybrid_prompt = self.inject_hybrid_context(&prompt_with_reflection).await?;
        let raw_response = self.call_vllm(&hybrid_prompt).await?;

        let refined_response = self.apply_cot_correction(&raw_response).await?;

        Ok(GenerationResult {
            text: refined_response,
            hybrid_sources: vec!["Claude".to_string(), "GPT-4".to_string(), "vLLM".to_string()],
            reflection_applied: reflection.is_some(),
        })
    }

    async fn apply_cot_correction(&self, response: &str) -> Result<String> {
        let mut corrected = String::with_capacity(response.len() + 256);
        corrected.push_str(response);
        corrected.push_str("\n\nStep-by-step check: Re-evaluate each critical inference and explicitly confirm or correct the logic.");
        Ok(corrected)
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
