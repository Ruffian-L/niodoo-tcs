//! Curator: Memory guardian and knowledge distiller
//! Adapted from curator_executor for niodoo_real_integrated integration

use anyhow::{anyhow, ensure, Context, Result};
use reqwest::Client;
use serde::Deserialize;
use serde_json::json;
use std::time::{Duration, Instant};
use tracing::{error, info, warn};

use crate::config::CuratorConfig;
use crate::data::Experience;

#[derive(Deserialize)]
struct CuratorJson {
    learned: bool,
    refined: String,
    reason: String,
}

/// Curated response with curator judgement
#[derive(Debug, Clone)]
pub struct CuratedResponse {
    pub refined_response: String,
    pub learned: bool,
    pub reason: String,
    pub processing_time_ms: f64,
}

/// The Curator interacts with a lightweight Qwen coder via Ollama
pub struct Curator {
    client: Client,
    config: CuratorConfig,
    mock_mode: bool,
}

impl Curator {
    pub fn new(config: CuratorConfig) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .no_proxy()
            .build()
            .context("failed to build curator HTTP client")?;

        if config.mock_mode {
            info!("Curator mock mode enabled; responses will bypass Ollama");
        }

        let mock_mode = config.mock_mode;
        Ok(Self {
            client,
            config,
            mock_mode,
        })
    }

    pub async fn curate(
        &self,
        experience: &Experience,
        knot: f64,
        entropy: f64,
    ) -> Result<CuratedResponse> {
        let start = Instant::now();

        if self.mock_mode {
            return Ok(self.mock_curated(experience, start.elapsed().as_secs_f64() * 1000.0));
        }
        let prompt = format!(
            "As code reviewer, validate/refine for quality>0.8, untangle knot {knot}, balance entropy {entropy}: Response '{response}'. Output JSON: {{\"learned\": true/false, \"refined\": \"text\", \"reason\": \"brief\"}}",
            response = experience.output
        );

        let request = json!({
            "model": self.config.model_name,
            "prompt": prompt,
            "stream": false,
            "options": {
                "temperature": self.config.temperature,
                "top_p": 0.9
            }
        });

        let url = format!(
            "{}/api/generate",
            self.config.vllm_endpoint.trim_end_matches('/')
        );
        let response = self
            .client
            .post(url)
            .json(&request)
            .send()
            .await
            .context("Ollama request for curator failed")?;

        if !response.status().is_success() {
            error!(
                status = %response.status(),
                "Ollama down—run 'ollama serve && ollama pull qwen2:0.5b'"
            );
            ensure!(
                self.mock_mode,
                "Ollama unavailable; fix with 'ollama serve && ollama pull qwen2:0.5b'"
            );
            return Ok(self.mock_curated(experience, start.elapsed().as_secs_f64() * 1000.0));
        }

        let payload = response
            .json::<serde_json::Value>()
            .await
            .context("invalid JSON from Ollama curator")?;
        let raw = payload
            .get("response")
            .and_then(|value| value.as_str())
            .ok_or_else(|| anyhow!("Curator did not return response field"))?;

        let parsed: CuratorJson = match serde_json::from_str(raw) {
            Ok(value) => value,
            Err(error) => {
                warn!(?error, "Curator JSON parse failed, returning mock response");
                ensure!(self.mock_mode, "Curator JSON parse failed: {error}");
                return Ok(self.mock_curated(experience, start.elapsed().as_secs_f64() * 1000.0));
            }
        };

        info!(
            learned = parsed.learned,
            reason = %parsed.reason,
            "Qwen curator refined: learned={learned}, quality~0.8, reason={reason}",
            learned = parsed.learned,
            reason = parsed.reason,
        );

        Ok(CuratedResponse {
            refined_response: parsed.refined,
            learned: parsed.learned,
            reason: parsed.reason,
            processing_time_ms: start.elapsed().as_secs_f64() * 1000.0,
        })
    }

    /// Refine a response using Qwen JSON call (returns refined text and learning flag)
    pub async fn refine(
        &self,
        response: &str,
        rouge: f64,
        knot: f64,
        entropy: f64,
    ) -> Result<(String, bool)> {
        let quality = rouge * 0.6 + (1.0 / (knot + 1.0)) * 0.4;

        if quality >= 0.8 {
            // Using a default value for curator_threshold
            info!("Curator skipped: quality={} >= {}", quality, 0.8);
            return Ok((response.to_string(), false));
        }

        if self.mock_mode {
            return Ok((response.to_string(), false));
        }

        info!(
            "Curator Qwen call: quality={}, knot={}, entropy={}",
            quality, knot, entropy
        );

        let prompt = format!(
            "As code reviewer, validate/refine for quality>0.8, untangle knot {}, balance entropy {}: Response '{}'. Output JSON: {{\"learned\": true/false, \"refined\": \"text\", \"reason\": \"brief\"}}",
            knot, entropy, response
        );

        let ollama_url = format!(
            "{}/api/generate",
            self.config.ollama_endpoint.trim_end_matches('/')
        );
        let body = json!({
            "model": "qwen2", // Using a default value for curator_model
            "prompt": prompt,
            "stream": false,
            "options": {
                "temperature": 0.7, // Using a default value for curator_temp
                "top_p": 0.9
            }
        });

        let resp = self.client.post(ollama_url).json(&body).send().await?;

        if !resp.status().is_success() {
            error!("Ollama down—run 'ollama serve && ollama pull qwen2:0.5b'");
            ensure!(self.mock_mode, "Ollama unavailable");
            return Ok((response.to_string(), false));
        }

        let result: serde_json::Value = resp.json().await?;
        // Ollama returns {"response": "text here"}
        let response_text = result["response"].as_str().unwrap_or("");

        // Try to parse as JSON first, fallback to treating as plain text
        let (learned, refined, reason) =
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(response_text) {
                (
                    parsed["learned"].as_bool().unwrap_or(false),
                    parsed["refined"].as_str().unwrap_or(response).to_string(),
                    parsed["reason"].as_str().unwrap_or("No reason").to_string(),
                )
            } else {
                // Fallback: treat as plain text refinement
                (
                    false,
                    response_text.to_string(),
                    "Plain text response".to_string(),
                )
            };

        info!(
            "Qwen curator refined: learned={}, refined len={}, reason={}",
            learned,
            refined.len(),
            reason
        );
        Ok((refined, learned))
    }

    /// Refine a response (simpler version for compatibility)
    pub async fn refine_response(&self, _input: &str, output: &str) -> Result<String> {
        let (refined, _learned) = self.refine(output, 0.5, 0.5, 0.5).await?;
        Ok(refined)
    }
}

impl Curator {
    fn mock_curated(&self, experience: &Experience, elapsed_ms: f64) -> CuratedResponse {
        CuratedResponse {
            refined_response: experience.output.clone(),
            learned: false,
            reason: "curator mock mode".to_string(),
            processing_time_ms: elapsed_ms,
        }
    }
}
