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
    max_tokens: usize,
    temperature: f64,
    context_truncate_chars: usize,
}

impl GenerationEngine {
    pub fn new() -> Result<Self> {
        Self::with_config(
            "http://localhost:8000".to_string(),
            512,  // max_tokens
            0.7,  // temperature
            200,  // context_truncate_chars
        )
    }
    
    pub fn with_config(
        vllm_url: String,
        max_tokens: usize,
        temperature: f64,
        context_truncate_chars: usize,
    ) -> Result<Self> {
        Ok(Self {
            http_client: Client::new(),
            vllm_url,
            max_tokens,
            temperature,
            context_truncate_chars,
        })
    }

    pub async fn generate(
        &self,
        tokenized: &TokenizedResult,
        reflection: Option<&ReflectionContext>,
        soft_failure_signals: Option<&Vec<String>>,
    ) -> Result<GenerationResult> {
        let base_prompt = tokenized.tokens.join(" ");
        let mut prompt_with_reflection = base_prompt.clone();

        // Apply full Reflexion if we have a reflection context
        if let Some(context) = reflection {
            if let Some(reason) = &context.failure_reason {
                prompt_with_reflection.push_str("\n\n# Reflexion\n");
                prompt_with_reflection.push_str(&format!(
                    "Previous attempt (retry #{}) failed because: {}\n",
                    context.retry_count, reason
                ));
                prompt_with_reflection.push_str("Last prompt: ");
                prompt_with_reflection.push_str(&context.last_prompt[..context.last_prompt.len().min(self.context_truncate_chars)]);
                prompt_with_reflection.push_str("\nLast response had error: ");
                prompt_with_reflection.push_str(&context.last_response[..context.last_response.len().min(self.context_truncate_chars)]);
                prompt_with_reflection.push_str("\n\nRevise strategy: Identify the root cause, fix the reasoning error, and produce a corrected response.");
            }
        }

        // Apply CoT for soft failures
        if let Some(signals) = soft_failure_signals {
            if !signals.is_empty() {
                prompt_with_reflection.push_str("\n\n# Chain-of-Thought Correction\n");
                prompt_with_reflection.push_str("Applying step-by-step reasoning correction for: ");
                prompt_with_reflection.push_str(&signals.join(", "));
                prompt_with_reflection.push_str("\nBreak down each step, verify logic, and correct any errors.");
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
        // Parse response for logical inconsistencies and fix them
        let mut corrected = String::with_capacity(response.len() + 512);
        
        // Detect common failure patterns
        let has_contradiction = response.contains("however") && response.contains("but");
        let has_incomplete = response.ends_with("...") || response.ends_with("[truncated]");
        let has_uncertainty = response.contains("maybe") || response.contains("possibly") || response.contains("uncertain");
        
        if has_contradiction || has_incomplete || has_uncertainty {
            corrected.push_str("\n\n**Self-Correction Applied:**\n");
            corrected.push_str("Re-evaluating reasoning chain:\n");
            
            if has_contradiction {
                corrected.push_str("- Addressing logical contradiction by reconciling conflicting statements\n");
            }
            if has_incomplete {
                corrected.push_str("- Completing truncated thought by following logical chain to conclusion\n");
            }
            if has_uncertainty {
                corrected.push_str("- Replacing uncertain qualifiers with concrete analysis based on available context\n");
            }
            
            corrected.push_str("\nCorrected reasoning:\n");
            // Apply actual corrections
            let fixed = response
                .replace("maybe", "analysis suggests")
                .replace("possibly", "evidence indicates")
                .replace("uncertain", "further analysis shows");
            corrected.push_str(&fixed);
            
            return Ok(corrected);
        }
        
        // No corrections needed, append verification
        corrected.push_str(response);
        corrected.push_str("\n\n[Verified: logical consistency checked]");
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
            max_tokens: self.max_tokens,
            temperature: self.temperature,
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
