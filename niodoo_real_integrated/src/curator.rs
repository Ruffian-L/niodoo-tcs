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
        let start = Instant::now();

        if self.mock_mode {
            return Ok(self.mock_curated(experience, start.elapsed().as_secs_f64() * 1000.0));
        }

        let prompt = format!(
            "As code reviewer, validate/refine for quality>0.8, untangle knot {knot}, balance entropy {entropy}: Response '{response}'. Output JSON: {{\"learned\": true/false, \"refined\": \"text\", \"reason\": \"brief\"}}",
            response = experience.output
        );

        // Use backend-specific refinement
        let refine_result = match self.config.curator_backend {
            CuratorBackend::Vllm => {
                self.refine_with_vllm(&prompt, &experience.output).await?
            }
            CuratorBackend::Ollama => {
                self.refine_with_ollama(&prompt, &experience.output).await?
            }
        };

        Ok(CuratedResponse {
            refined_response: refine_result.refined,
            learned: refine_result.learned,
            reason: refine_result.reason,
            processing_time_ms: start.elapsed().as_secs_f64() * 1000.0,
        })
    }

    /// Refine a response using backend (Ollama or vLLM)
    pub async fn refine(
        &self,
        response: &str,
        rouge: f64,
        knot: f64,
        entropy: f64,
    ) -> Result<CuratorRefineResult> {
        let quality = rouge * 0.6 + (1.0 / (knot + 1.0)) * 0.4;

        if quality >= 0.8 {
            info!("Curator skipped: quality={} >= {}", quality, 0.8);
            return Ok(CuratorRefineResult {
                refined: response.to_string(),
                learned: false,
                reason: "quality_ok".to_string(),
            });
        }

        if self.mock_mode {
            return Ok(CuratorRefineResult {
                refined: response.to_string(),
                learned: false,
                reason: "curator mock mode".to_string(),
            });
        }

        info!(
            "Curator Qwen call: quality={}, knot={}, entropy={}",
            quality, knot, entropy
        );

        let prompt = format!(
            "As code reviewer, validate/refine for quality>0.8, untangle knot {}, balance entropy {}: Response '{}'. Output JSON: {{\"learned\": true/false, \"refined\": \"text\", \"reason\": \"brief\"}}",
            knot, entropy, response
        );

        match self.config.curator_backend {
            CuratorBackend::Vllm => {
                self.refine_with_vllm(&prompt, response).await
            }
            CuratorBackend::Ollama => {
                self.refine_with_ollama(&prompt, response).await
            }
        }
    }

    /// Refine using vLLM API (GPU-accelerated, more reliable)
    async fn refine_with_vllm(&self, prompt: &str, fallback_response: &str) -> Result<CuratorRefineResult> {
        let endpoint = if self.config.vllm_endpoint.contains("/v1/chat/completions") {
            self.config.vllm_endpoint.clone()
        } else {
            format!("{}/v1/chat/completions", self.config.vllm_endpoint.trim_end_matches('/'))
        };

        let messages = vec![
            json!({
                "role": "system",
                "content": "You are a code reviewer. Analyze responses and output JSON only: {\"learned\": true/false, \"refined\": \"text\", \"reason\": \"brief\"}"
            }),
            json!({
                "role": "user",
                "content": prompt
            }),
        ];

        let payload = json!({
            "model": self.config.model_name,
            "messages": messages,
            "temperature": self.config.temperature,
            "top_p": 0.9,
            "max_tokens": self.config.max_tokens,
        });

        let resp = match timeout(
            Duration::from_secs(self.config.timeout_secs),
            self.client.post(&endpoint).json(&payload).send(),
        )
        .await
        {
            Ok(Ok(r)) => r,
            Ok(Err(e)) => {
                warn!(%e, "vLLM curator request failed, returning unmodified response");
                return Ok(CuratorRefineResult {
                    refined: fallback_response.to_string(),
                    learned: false,
                    reason: format!("vllm_request_failed:{e}"),
                });
            }
            Err(_) => {
                warn!("vLLM curator request timed out, returning unmodified response");
                return Ok(CuratorRefineResult {
                    refined: fallback_response.to_string(),
                    learned: false,
                    reason: "vllm_timeout".to_string(),
                });
            }
        };

        if !resp.status().is_success() {
            warn!("vLLM curator request failed with status {}, returning unmodified", resp.status());
            return Ok(CuratorRefineResult {
                refined: fallback_response.to_string(),
                learned: false,
                reason: format!("vllm_status_{}", resp.status().as_u16()),
            });
        }

        let result: serde_json::Value = timeout(
            Duration::from_secs(self.config.timeout_secs),
            resp.json(),
        )
        .await
        .context("vLLM curator JSON parsing timed out")??;

        let response_text = result["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("");

        // Try to parse as JSON first, fallback to treating as plain text
        let result = if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(response_text) {
            CuratorRefineResult {
                learned: parsed["learned"].as_bool().unwrap_or(false),
                refined: parsed["refined"].as_str().unwrap_or(fallback_response).to_string(),
                reason: parsed["reason"].as_str().unwrap_or("No reason").to_string(),
            }
        } else {
            // Fallback: treat as plain text refinement
            CuratorRefineResult {
                refined: response_text.to_string(),
                learned: false,
                reason: "plain_text_response".to_string(),
            }
        };

        info!(
            "vLLM curator refined: learned={}, refined len={}, reason={}",
            result.learned,
            result.refined.len(),
            result.reason
        );
        Ok(result)
    }

    /// Refine using Ollama API (CPU, fallback)
    async fn refine_with_ollama(&self, prompt: &str, fallback_response: &str) -> Result<CuratorRefineResult> {
        let ollama_url = format!(
            "{}/api/generate",
            self.config.ollama_endpoint.trim_end_matches('/')
        );
        let body = json!({
            "model": self.config.model_name,
            "prompt": prompt,
            "stream": false,
            "options": {
                "temperature": self.config.temperature,
                "top_p": 0.9
            }
        });

        let resp = match timeout(
            Duration::from_secs(self.config.timeout_secs),
            self.client.post(ollama_url).json(&body).send(),
        )
        .await
        {
            Ok(Ok(r)) => r,
            Ok(Err(e)) => {
                warn!(%e, "Ollama request failed, returning unmodified response");
                return Ok(CuratorRefineResult {
                    refined: fallback_response.to_string(),
                    learned: false,
                    reason: format!("ollama_request_failed:{e}"),
                });
            }
            Err(_) => {
                warn!("Ollama request timed out, returning unmodified response");
                return Ok(CuratorRefineResult {
                    refined: fallback_response.to_string(),
                    learned: false,
                    reason: "ollama_timeout".to_string(),
                });
            }
        };

        if !resp.status().is_success() {
            error!("Ollama down—run 'ollama serve && ollama pull qwen2:0.5b'");
            warn!("Ollama unavailable; returning unmodified response (learned=false)");
            return Ok(CuratorRefineResult {
                refined: fallback_response.to_string(),
                learned: false,
                reason: "ollama_unavailable".to_string(),
            });
        }

        let result: serde_json::Value = timeout(
            Duration::from_secs(self.config.timeout_secs),
            resp.json(),
        )
        .await
        .context("Ollama curator JSON parsing timed out")??;
        
        // Ollama returns {"response": "text here"}
        let response_text = result["response"].as_str().unwrap_or("");

        // Try to parse as JSON first, fallback to treating as plain text
        let result = if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(response_text) {
            CuratorRefineResult {
                learned: parsed["learned"].as_bool().unwrap_or(false),
                refined: parsed["refined"].as_str().unwrap_or(fallback_response).to_string(),
                reason: parsed["reason"].as_str().unwrap_or("No reason").to_string(),
            }
        } else {
            // Fallback: treat as plain text refinement
            CuratorRefineResult {
                refined: response_text.to_string(),
                learned: false,
                reason: "plain_text_response".to_string(),
            }
        };

        info!(
            "Ollama curator refined: learned={}, refined len={}, reason={}",
            result.learned,
            result.refined.len(),
            result.reason
        );
        Ok(result)
    }

    /// Refine a response (simpler version for compatibility)
    pub async fn refine_response(&self, _input: &str, output: &str) -> Result<String> {
        let result = self.refine(output, 0.5, 0.5, 0.5).await?;
        Ok(result.refined)
    }

    fn mock_curated(&self, experience: &Experience, elapsed_ms: f64) -> CuratedResponse {
        CuratedResponse {
            refined_response: experience.output.clone(),
            learned: false,
            reason: "curator mock mode".to_string(),
            processing_time_ms: elapsed_ms,
        }
    }

    /// Distill knowledge from memory clusters - creates generalized training examples
    pub async fn distill_knowledge(
        &self,
        erag: &crate::erag::EragClient,
        num_clusters: usize,
    ) -> Result<Vec<DistilledExample>> {
        if self.mock_mode {
            return Ok(Vec::new());
        }

        // Get all experiences from ERAG (sample of recent)
        // Use search with dummy query to get all points
        let dummy_query = "memory_distillation_sample";
        let hits = erag.search(dummy_query, 1000, None).await?;

        if hits.is_empty() {
            return Ok(Vec::new());
        }

        // Convert SearchHit to Experience-like structures
        let experiences: Vec<MemoryExperience> = hits
            .iter()
            .filter_map(|hit| {
                let payload = &hit.payload;
                let input = payload.get("input").and_then(|v| v.as_str())?.to_string();
                let output = payload.get("output").and_then(|v| v.as_str())?.to_string();
                let quality = payload.get("quality_score")
                    .and_then(|v| v.as_f64())
                    .map(|f| f as f32)
                    .unwrap_or(0.5);

                Some(MemoryExperience {
                    input,
                    output,
                    quality_score: quality,
                })
            })
            .collect();

        if experiences.is_empty() {
            return Ok(Vec::new());
        }

        // Cluster experiences by similarity in background task to avoid blocking
        let threshold = self.config.clustering_threshold;
        let clusters = tokio::task::spawn_blocking(move || {
            Self::cluster_experiences_static(&experiences, threshold)
        }).await??;

        // Generate distilled examples from clusters
        let mut distilled_examples = Vec::new();
        for cluster in clusters.into_iter().take(num_clusters) {
            if let Some(example) = self.distill_cluster(&cluster)? {
                distilled_examples.push(example);
            }
        }

        Ok(distilled_examples)
    }

    /// Cluster experiences using agglomerative clustering
    fn cluster_experiences_static(experiences: &[MemoryExperience], threshold: f32) -> Result<Vec<Vec<MemoryExperience>>> {
        let mut clusters: Vec<Vec<MemoryExperience>> = experiences.iter().cloned().map(|e| vec![e]).collect();

        // Simple clustering: merge clusters with high similarity
        loop {
            let mut merged = false;

            for i in 0..clusters.len() {
                for j in (i + 1)..clusters.len() {
                    if Self::cluster_similarity_static(&clusters[i], &clusters[j]) > threshold {
                        // Merge clusters
                        let cluster_j = clusters.swap_remove(j);
                        clusters[i].extend(cluster_j);
                        merged = true;
                        break;
                    }
                }
                if merged {
                    break;
                }
            }

            if !merged {
                break;
            }
        }

        Ok(clusters)
    }

    /// Calculate similarity between two clusters (average quality-weighted similarity)
    fn cluster_similarity_static(cluster_a: &[MemoryExperience], cluster_b: &[MemoryExperience]) -> f32 {
        // Simple similarity: based on output similarity (cosine similarity of text)
        // For now, use quality-weighted average
        let mut total_sim = 0.0;
        let mut count = 0;

        for exp_a in cluster_a {
            for exp_b in cluster_b {
                // Simple text similarity (can be improved with embeddings)
                let sim = Self::text_similarity(&exp_a.output, &exp_b.output);
                let weight = (exp_a.quality_score + exp_b.quality_score) / 2.0;
                total_sim += sim * weight;
                count += 1;
            }
        }

        if count > 0 {
            total_sim / count as f32
        } else {
            0.0
        }
    }

    /// Simple text similarity (Jaccard similarity of words)
    fn text_similarity(a: &str, b: &str) -> f32 {
        let words_a: std::collections::HashSet<String> = a.to_lowercase().split_whitespace().map(|s| s.to_string()).collect();
        let words_b: std::collections::HashSet<String> = b.to_lowercase().split_whitespace().map(|s| s.to_string()).collect();

        let intersection = words_a.intersection(&words_b).count();
        let union = words_a.union(&words_b).count();

        if union == 0 {
            0.0
        } else {
            intersection as f32 / union as f32
        }
    }

    /// Distill a cluster into a single training example
    fn distill_cluster(&self, cluster: &[MemoryExperience]) -> Result<Option<DistilledExample>> {
        if cluster.is_empty() {
            return Ok(None);
        }

        // Find the most successful experience in the cluster
        let best_experience = cluster.iter()
            .max_by(|a, b| a.quality_score.partial_cmp(&b.quality_score).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();

        // Generate a generalized instruction from the cluster
        let instruction = self.generate_instruction_from_cluster(cluster)?;

        // Use the best experience's output as the target
        let output = best_experience.output.clone();

        Ok(Some(DistilledExample {
            instruction,
            output,
            quality_score: best_experience.quality_score,
            cluster_size: cluster.len(),
        }))
    }

    /// Generate a generalized instruction from a cluster
    fn generate_instruction_from_cluster(&self, cluster: &[MemoryExperience]) -> Result<String> {
        // Simple generalization: take the first experience's input as template
        if let Some(first) = cluster.first() {
            Ok(format!("Generalized task from {} similar experiences:\n{}", cluster.len(), first.input.clone()))
        } else {
            Ok("Generalized task".to_string())
        }
    }
}

/// Memory experience for distillation
#[derive(Debug, Clone)]
struct MemoryExperience {
    input: String,
    output: String,
    quality_score: f32,
}

/// A distilled training example
#[derive(Debug, Clone)]
pub struct DistilledExample {
    pub instruction: String,
    pub output: String,
    pub quality_score: f32,
    pub cluster_size: usize,
}
