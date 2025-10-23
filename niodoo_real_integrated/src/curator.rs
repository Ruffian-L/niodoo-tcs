//! Curator: Memory guardian and knowledge distiller
//! Adapted from curator_executor for niodoo_real_integrated integration

use anyhow::{anyhow, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

use crate::config::CuratorConfig;
use crate::curator_parser::{CascadingParser, ResponseParser};
use crate::data::Experience;

/// Curated response with quality assessment
#[derive(Debug, Clone)]
pub struct CuratedResponse {
    pub refined_response: String,
    pub quality_score: f32,
    pub should_store: bool,
    pub reasoning: String,
    pub processing_time_ms: f64,
}

/// Distilled training example from experience clusters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistilledExample {
    pub instruction: String,
    pub output: String,
    pub quality_score: f32,
    pub cluster_size: usize,
}

/// The Curator: Memory guardian and knowledge distiller
pub struct Curator {
    client: Client,
    config: CuratorConfig,
}

impl Curator {
    /// Initialize the Curator with vLLM connection
    pub fn new(config: CuratorConfig) -> Result<Self> {
        info!("Initializing Curator with vLLM endpoint: {}", config.vllm_endpoint);

        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .no_proxy() // Disable proxy for local Ollama/vLLM connections
            .build()?;

        info!("Curator initialized successfully");

        Ok(Self { client, config })
    }

    /// Embed text into a vector representation using the vLLM embeddings endpoint
    pub async fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        let request = json!({
            "model": self.config.model_name,
            "input": text
        });

        let response = self.client
            .post(format!("{}/v1/embeddings", self.config.vllm_endpoint))
            .json(&request)
            .send()
            .await?
            .json::<Value>()
            .await?;

        let embedding_array = response["data"]
            .as_array()
            .and_then(|data| data.first())
            .and_then(|item| item.get("embedding"))
            .and_then(|embedding| embedding.as_array())
            .ok_or_else(|| anyhow!("No embedding in response"))?;

        let embedding: Vec<f32> = embedding_array
            .iter()
            .map(|value| value.as_f64().unwrap_or(0.0) as f32)
            .collect();

        if embedding.is_empty() {
            return Err(anyhow!("Empty embedding returned from vLLM"));
        }

        if self.config.embedding_dim > 0 && embedding.len() != self.config.embedding_dim {
            warn!(
                "Embedding size {} does not match configured dimension {}",
                embedding.len(),
                self.config.embedding_dim
            );
        }

        Ok(embedding)
    }

    /// Call the vLLM model for text generation
    pub async fn call_model(&self, prompt: &str) -> Result<String> {
        // Check if using Ollama (not vLLM)
        let is_ollama = self.config.vllm_endpoint.contains("11434");
        
        if is_ollama {
            // Ollama API format
            let request = json!({
                "model": self.config.model_name,
                "prompt": prompt,
                "stream": false
            });

            let response = self.client
                .post(&format!("{}/api/generate", self.config.vllm_endpoint))
                .json(&request)
                .send()
                .await?
                .json::<Value>()
                .await?;

            debug!("Curator Ollama response: {:?}", response);
            
            let content = response["response"]
                .as_str()
                .ok_or_else(|| {
                    warn!("Ollama response format: {:?}", response);
                    anyhow!("Invalid Ollama response format")
                })?;

            Ok(content.to_string())
        } else {
            // vLLM OpenAI-compatible API format
            let request = json!({
                "model": self.config.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens
            });

            let response = self.client
                .post(&format!("{}/v1/chat/completions", self.config.vllm_endpoint))
                .json(&request)
                .send()
                .await?
                .json::<Value>()
                .await?;

            debug!("Curator vLLM response: {:?}", response);
            
            let content = response["choices"][0]["message"]["content"]
                .as_str()
                .ok_or_else(|| anyhow!("Invalid response format"))?;

            Ok(content.to_string())
        }
    }

    /// Assess quality of a response using mini Qwen with cascading parse strategy
    pub async fn assess_quality(
        &self,
        prompt: &str,
        response: &str,
        pad_state_entropy: f64,
        compass_quadrant: &str,
    ) -> Result<f32> {
        // Build quality assessment prompt using template
        let assessment_prompt = self.config.assessment_prompt_template
            .replace("{}", &format!("Score this response (0.0-1.0) for emotional breakthrough potential.\nConsider: breakthrough→high score, stagnation→low score, LearningWill advance→boost score.\n\nPrompt: {}\nResponse: {}\nEntropy: {:.3}, Quadrant: {}\n\nOUTPUT FORMAT: Respond with ONLY a single number (e.g., '0.85'). No text, no explanation, no JSON, just the number.:", prompt, response, pad_state_entropy, compass_quadrant));

        match self.call_model(&assessment_prompt).await {
            Ok(result) => {
                // Use cascading parser with heuristic fallback
                let parser = CascadingParser::new(self.config.parse_mode)
                    .with_heuristic_fallback(response.to_string(), pad_state_entropy);
                
                match parser.parse(&result) {
                    Ok(score) => {
                        debug!("Curator quality assessment (mode: {:?}): {:.3}", self.config.parse_mode, score);
                        Ok(score)
                    }
                    Err(e) => {
                        warn!("All parsing strategies failed: {}, using direct heuristic", e);
                        // Last resort: create heuristic parser with config values
                        let heuristic = crate::curator_parser::HeuristicParser::new(
                            response.to_string(),
                            pad_state_entropy
                        )
                        .with_config(
                            self.config.heuristic_max_length,
                            self.config.heuristic_optimal_entropy_low,
                            self.config.heuristic_optimal_entropy_high,
                            self.config.heuristic_optimal_entropy_score,
                            self.config.heuristic_suboptimal_entropy_score,
                            self.config.heuristic_length_weight,
                        );
                        heuristic.parse_score("")
                    }
                }
            }
            Err(e) => {
                warn!("Curator model call failed: {}, using heuristic fallback", e);
                // Fallback: create heuristic parser with config values
                let heuristic = crate::curator_parser::HeuristicParser::new(
                    response.to_string(),
                    pad_state_entropy
                )
                .with_config(
                    self.config.heuristic_max_length,
                    self.config.heuristic_optimal_entropy_low,
                    self.config.heuristic_optimal_entropy_high,
                    self.config.heuristic_optimal_entropy_score,
                    self.config.heuristic_suboptimal_entropy_score,
                    self.config.heuristic_length_weight,
                );
                heuristic.parse_score("")
            }
        }
    }

    /// Refine a low-quality response
    pub async fn refine_response(&self, prompt: &str, response: &str) -> Result<String> {
        let refinement_prompt = format!(
            "Refine this response to be more accurate, helpful, coherent, and emotionally aligned with the system state:\n\n\
            Original Prompt: {}\n\
            Original Response: {}\n\n\
            Provide an improved response:",
            prompt, response
        );

        match self.call_model(&refinement_prompt).await {
            Ok(refined) => {
                info!("Curator refined response");
                Ok(refined)
            }
            Err(e) => {
                warn!("Curator refinement failed: {}, using original", e);
                Ok(response.to_string())
            }
        }
    }

    /// Curate a response: assess quality and optionally refine
    pub async fn curate_response(
        &self,
        experience: Experience,
    ) -> Result<Experience> {
        let start = Instant::now();
        let prompt = &experience.input;
        let mut response = experience.output.clone();

        // Assess quality
        let compass_quadrant_str = match experience.compass_quadrant { // Assume added to Experience
            crate::compass::CompassQuadrant::Panic => "Panic",
            crate::compass::CompassQuadrant::Persist => "Persist",
            crate::compass::CompassQuadrant::Discover => "Discover",
            crate::compass::CompassQuadrant::Master => "Master",
        };
        
        let quality_score = self.assess_quality(prompt, &response, experience.pad_entropy, compass_quadrant_str).await?;

        // Determine if we should store
        let should_store = quality_score >= self.config.quality_threshold;

        // Refine if low quality but above absolute minimum
        if !should_store && quality_score >= self.config.minimum_threshold {
            info!("Response below quality threshold ({:.3}), attempting refinement", quality_score);
            response = self.refine_response(prompt, &response).await?;
        }

        let processing_time_ms = start.elapsed().as_secs_f64() * 1000.0;

        // Update experience
        let mut curated = experience;
        curated.quality_score = Some(quality_score); // Assume added field
        curated.should_store = Some(should_store);
        curated.refined_output = Some(response); // Assume added field, or update output
        curated.processing_time_ms = Some(processing_time_ms);

        Ok(curated)
    }

    /// Stub for Phase 3: Knowledge distillation from experience clusters
    pub async fn distill_knowledge(
        &self,
        _experiences: &[Experience],
        _num_clusters: usize,
    ) -> Result<Vec<DistilledExample>> {
        // TODO: Phase 3 - Implement clustering and distillation
        // For now, return empty (stub)
        warn!("Knowledge distillation not yet implemented (Phase 3)");
        Ok(Vec::new())
    }

    /// Cosine similarity between two vectors
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a > 0.0 && norm_b > 0.0 {
            dot_product / (norm_a * norm_b)
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::CuratorConfig;
    use crate::data::Experience;
    use crate::compass::CompassOutcome;
    use crate::torus::PadGhostState;
    use crate::config::RuntimeConfig;
    use crate::curator_parser::{ParserMode, CascadingParser};

    // Simple mock for testing
    struct MockCurator {
        mock_scores: Vec<f32>,
        mock_refined: Vec<String>,
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert_eq!(Curator::cosine_similarity(&a, &b), 1.0);

        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert_eq!(Curator::cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_curator_parse_modes() {
        // Test JSON parser mode
        let json_response = r#"{"score": 0.85}"#;
        let parser = CascadingParser::new(ParserMode::Json)
            .with_heuristic_fallback("test".to_string(), 1.8);
        let score = parser.parse(json_response).unwrap();
        assert_eq!(score, 0.85);

        // Test regex parser mode with embedded text
        let text_response = "The quality score is 0.75 for this response";
        let parser = CascadingParser::new(ParserMode::Regex)
            .with_heuristic_fallback("test".to_string(), 1.8);
        let score = parser.parse(text_response).unwrap();
        assert_eq!(score, 0.75);

        // Test regex parser mode with clean number
        let clean_response = "0.92";
        let parser = CascadingParser::new(ParserMode::Regex)
            .with_heuristic_fallback("test".to_string(), 1.8);
        let score = parser.parse(clean_response).unwrap();
        assert_eq!(score, 0.92);

        // Test heuristic fallback when all parsers fail
        let garbage_response = "This is not a valid score at all";
        let parser = CascadingParser::new(ParserMode::Regex)
            .with_heuristic_fallback("test response with moderate length".to_string(), 1.8);
        let score = parser.parse(garbage_response).unwrap();
        // Should fall back to heuristic (length_score * 0.4 + entropy_score * 0.6)
        assert!(score > 0.0 && score <= 1.0);
    }

    #[test]
    fn test_parser_mode_env_parsing() {
        // Test default mode
        std::env::remove_var("CURATOR_PARSE_MODE");
        let mode = ParserMode::from_env();
        assert_eq!(mode, ParserMode::Regex);

        // Test JSON mode
        std::env::set_var("CURATOR_PARSE_MODE", "json");
        let mode = ParserMode::from_env();
        assert_eq!(mode, ParserMode::Json);

        // Test heuristic mode
        std::env::set_var("CURATOR_PARSE_MODE", "heuristic");
        let mode = ParserMode::from_env();
        assert_eq!(mode, ParserMode::Heuristic);

        // Cleanup
        std::env::remove_var("CURATOR_PARSE_MODE");
    }

    #[test]
    fn test_cascading_parser_fallback() {
        // JSON response but regex mode - should cascade to JSON parser
        let json_response = r#"{"score": 0.88}"#;
        let parser = CascadingParser::new(ParserMode::Regex)
            .with_heuristic_fallback("test".to_string(), 1.8);
        let score = parser.parse(json_response).unwrap();
        assert_eq!(score, 0.88);

        // Text response but JSON mode - should cascade to regex parser
        let text_response = "Score: 0.66";
        let parser = CascadingParser::new(ParserMode::Json)
            .with_heuristic_fallback("test".to_string(), 1.8);
        let score = parser.parse(text_response).unwrap();
        assert_eq!(score, 0.66);
    }
}

