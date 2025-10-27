//! Curator: Memory guardian and knowledge distiller
//! Adapted from curator_executor for niodoo_real_integrated integration

use anyhow::{anyhow, Context, Result};
use reqwest::Client;
use serde::Deserialize;
use serde_json::json;
use std::time::{Duration, Instant};
use tracing::{error, info};

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
}

impl Curator {
    pub fn new(config: CuratorConfig) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .no_proxy()
            .build()
            .context("failed to build curator HTTP client")?;

        Ok(Self { client, config })
    }

    pub async fn curate(
        &self,
        experience: &Experience,
        knot: f64,
        entropy: f64,
    ) -> Result<CuratedResponse> {
        let start = Instant::now();
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

        let url = format!("{}/api/generate", self.config.vllm_endpoint.trim_end_matches('/'));
        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .context("Ollama request for curator failed")?;

        if !response.status().is_success() {
            error!(
                status = %response.status(),
                "Ollama down—run 'ollama serve && ollama pull qwen2:0.5b'"
            );
            return Err(anyhow!(
                "Ollama unavailable; fix with 'ollama serve && ollama pull qwen2:0.5b'"
            ));
        }

        let payload = response
            .json::<serde_json::Value>()
            .await
            .context("invalid JSON from Ollama curator")?;
        let raw = payload
            .get("response")
            .and_then(|value| value.as_str())
            .ok_or_else(|| anyhow!("Curator did not return response field"))?;

        let parsed: CuratorJson = serde_json::from_str(raw).context("curator JSON parse failed")?;

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
}
