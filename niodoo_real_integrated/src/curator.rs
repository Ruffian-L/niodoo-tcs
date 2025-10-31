//! Curator: Memory guardian and knowledge distiller
//! Adapted from curator_executor for niodoo_real_integrated integration
//! Supports both Ollama (CPU) and vLLM (GPU) backends

use anyhow::{Context, Result};
use reqwest::Client;
use serde::Deserialize;
use serde_json::json;
use std::time::{Duration, Instant};
use tokio::time::timeout;
use tracing::{error, info, warn};

use crate::config::{CuratorBackend, CuratorConfig};
use crate::consonance::ConsonanceMetrics;
use crate::data::Experience;

#[derive(Deserialize)]
struct CuratorJson {
    learned: bool,
    refined: String,
    reason: String,
}

/// Curated response with curator judgement and truth attractor scoring
#[derive(Debug, Clone)]
pub struct CuratedResponse {
    pub refined_response: String,
    pub learned: bool,
    pub reason: String,
    pub processing_time_ms: f64,
    pub consonance_score: f64,  // Truth attractor score (0.0-1.0)
}

#[derive(Debug, Clone)]
pub struct CuratorRefineResult {
    pub refined: String,
    pub learned: bool,
    pub reason: String,
}

/// The Curator interacts with a lightweight Qwen model via Ollama or vLLM
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
            info!("Curator mock mode enabled; responses will bypass backend");
        } else {
            match config.curator_backend {
                CuratorBackend::Vllm => {
                    info!(
                        endpoint = %config.vllm_endpoint,
                        model = %config.model_name,
                        "Curator using vLLM backend (GPU-accelerated)"
                    );
                }
                CuratorBackend::Ollama => {
                    info!(
                        endpoint = %config.ollama_endpoint,
                        model = %config.model_name,
                        "Curator using Ollama backend (CPU)"
                    );
                }
            }
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
        self.curate_with_consonance(experience, knot, entropy, None).await
    }

    /// Curate with consonance metrics for truth attractor scoring
    pub async fn curate_with_consonance(
        &self,
        experience: &Experience,
        knot: f64,
        entropy: f64,
        consonance: Option<&ConsonanceMetrics>,
    ) -> Result<CuratedResponse> {
        let start = Instant::now();

        if self.mock_mode {
            return Ok(self.mock_curated(experience, start.elapsed().as_secs_f64() * 1000.0));
        }

        // Include consonance context in prompt if available
        let consonance_context = if let Some(cons) = consonance {
            format!(" Context consonance: {:.2} ({}), confidence: {:.2}.", 
                cons.score, 
                if cons.score > 0.7 { "high resonance" } else if cons.score > 0.5 { "moderate" } else { "low resonance" },
                cons.confidence)
        } else {
            String::new()
        };

        let prompt = format!(
            "As code reviewer, validate/refine for quality>0.8, untangle knot {knot}, balance entropy {entropy}: Response '{response}'.{consonance_context} Output JSON: {{\"learned\": true/false, \"refined\": \"text\", \"reason\": \"brief\"}}",
            response = experience.output,
            consonance_context = consonance_context
        );

        // Use backend-specific refinement
        let result = match self.config.curator_backend {
            CuratorBackend::Vllm => self.refine_with_vllm(&prompt).await,
            CuratorBackend::Ollama => self.refine_with_ollama(&prompt).await,
        };

        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

        match result {
            Ok(refined) => {
                // Compute truth attractor score (consonance-based)
                let consonance_score = self.compute_truth_attractor_score(&refined, consonance);
                
                Ok(CuratedResponse {
                    refined_response: refined.refined,
                    learned: refined.learned,
                    reason: refined.reason,
                    processing_time_ms: elapsed_ms,
                    consonance_score,
                })
            }
            Err(e) => {
                error!(error = %e, "Curator refinement failed; using original response");
                let consonance_score = consonance.map(|c| c.score).unwrap_or(0.5);
                
                Ok(CuratedResponse {
                    refined_response: experience.output.clone(),
                    learned: false,
                    reason: format!("Curator error: {}", e),
                    processing_time_ms: elapsed_ms,
                    consonance_score,
                })
            }
        }
    }

    /// Compute truth attractor score from curator response and consonance
    /// High consonance + learned=true = "This resonates, lean into it"
    /// Low consonance = "Something's wrong, investigate"
    fn compute_truth_attractor_score(
        &self,
        refined: &CuratorRefineResult,
        consonance: Option<&ConsonanceMetrics>,
    ) -> f64 {
        // Base score from curator's learned flag
        let base_score = if refined.learned {
            0.7 // High base score if curator learned something
        } else {
            0.4 // Lower base score if not learned
        };

        // Boost or reduce based on consonance
        if let Some(cons) = consonance {
            // High consonance → boost score (truth attractor)
            // Low consonance → reduce score (dissonance detector)
            let consonance_factor = cons.score;
            
            // Combine base score with consonance
            // If consonance is high (>0.7), boost significantly
            // If consonance is low (<0.5), reduce
            if consonance_factor > 0.7 {
                // High resonance: lean into it
                base_score + (consonance_factor - 0.7) * 0.5 // Up to +0.15 boost
            } else if consonance_factor < 0.5 {
                // Low resonance: something's wrong
                base_score * consonance_factor // Reduce proportionally
            } else {
                // Moderate resonance
                base_score * 0.95 + consonance_factor * 0.05 // Slight blend
            }
        } else {
            base_score
        }
        .clamp(0.0, 1.0)
    }

    async fn refine_with_vllm(&self, prompt: &str) -> Result<CuratorRefineResult> {
        let url = format!("{}/v1/completions", self.config.vllm_endpoint);
        let payload = json!({
            "model": self.config.model_name,
            "prompt": prompt,
            "temperature": 0.3,
            "max_tokens": 512,
            "stop": ["\n\n"],
        });

        let response = timeout(
            Duration::from_secs(self.config.timeout_secs),
            self.client.post(&url).json(&payload).send(),
        )
        .await
        .context("vLLM curator request timeout")?
        .context("vLLM curator request failed")?;

        if !response.status().is_success() {
            anyhow::bail!("vLLM curator returned status: {}", response.status());
        }

        let json: serde_json::Value = response.json().await.context("failed to parse vLLM response")?;
        let text = json["choices"][0]["text"]
            .as_str()
            .context("vLLM response missing text")?;

        self.parse_curator_response(text)
    }

    async fn refine_with_ollama(&self, prompt: &str) -> Result<CuratorRefineResult> {
        let url = format!("{}/api/generate", self.config.ollama_endpoint);
        let payload = json!({
            "model": self.config.model_name,
            "prompt": prompt,
            "stream": false,
            "options": {
                "temperature": 0.3,
            }
        });

        let response = timeout(
            Duration::from_secs(self.config.timeout_secs),
            self.client.post(&url).json(&payload).send(),
        )
        .await
        .context("Ollama curator request timeout")?
        .context("Ollama curator request failed")?;

        if !response.status().is_success() {
            anyhow::bail!("Ollama curator returned status: {}", response.status());
        }

        let json: serde_json::Value = response.json().await.context("failed to parse Ollama response")?;
        let text = json["response"]
            .as_str()
            .context("Ollama response missing text")?;

        self.parse_curator_response(text)
    }

    fn parse_curator_response(&self, text: &str) -> Result<CuratorRefineResult> {
        // Try to extract JSON from response
        let json_start = text.find('{').unwrap_or(0);
        let json_text = &text[json_start..];
        let json_end = json_text.rfind('}').map(|i| i + 1).unwrap_or(json_text.len());
        let json_str = &json_text[..json_end];

        match serde_json::from_str::<CuratorJson>(json_str) {
            Ok(parsed) => Ok(CuratorRefineResult {
                refined: parsed.refined,
                learned: parsed.learned,
                reason: parsed.reason,
            }),
            Err(_) => {
                // Fallback: treat entire response as refined text
                warn!("Curator response not valid JSON; treating as refined text");
                Ok(CuratorRefineResult {
                    refined: text.trim().to_string(),
                    learned: false,
                    reason: "JSON parse failed".to_string(),
                })
            }
        }
    }

    fn mock_curated(&self, experience: &Experience, elapsed_ms: f64) -> CuratedResponse {
        CuratedResponse {
            refined_response: experience.output.clone(),
            learned: false,
            reason: "Mock mode: bypassing curator".to_string(),
            processing_time_ms: elapsed_ms,
            consonance_score: 0.5, // Neutral score in mock mode
        }
    }
}




