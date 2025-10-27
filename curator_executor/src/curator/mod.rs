use anyhow::Result;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::time::Duration;
use crate::memory_core::{Experience, MemoryCore};

/// Configuration for the Curator
#[derive(Debug, Clone)]
pub struct CuratorConfig {
    pub vllm_endpoint: String,
    pub model_name: String,
    pub embedding_dim: usize,
    pub max_context_length: usize,
    pub distillation_batch_size: usize,
    pub clustering_threshold: f32,
}

impl Default for CuratorConfig {
    fn default() -> Self {
        Self {
            vllm_endpoint: std::env::var("VLLM_ENDPOINT").unwrap_or_else(|_| "http://localhost:5001".to_string()),
            model_name: "Qwen2.5-0.5B-Instruct".to_string(),
            embedding_dim: 768,
            max_context_length: 2048,
            distillation_batch_size: 32,
            clustering_threshold: 0.8,
        }
    }
}

/// The Curator: Memory guardian and knowledge distiller
pub struct Curator {
    client: Client,
    config: CuratorConfig,
}

impl Curator {
    /// Initialize the Curator with vLLM connection
    pub fn new(config: CuratorConfig) -> Result<Self> {
        println!("Initializing Curator with vLLM endpoint: {}", config.vllm_endpoint);

        let client = Client::builder().timeout(Duration::from_secs(10)).build()?;

        println!("Curator initialized successfully");

        Ok(Self {
            client,
            config,
        })
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
            .ok_or_else(|| anyhow::anyhow!("No embedding in response"))?;

        let embedding: Vec<f32> = embedding_array
            .iter()
            .map(|value| value.as_f64().unwrap_or(0.0) as f32)
            .collect();

        if embedding.is_empty() {
            return Err(anyhow::anyhow!("Empty embedding returned from vLLM"));
        }

        if self.config.embedding_dim > 0 && embedding.len() != self.config.embedding_dim {
            tracing::warn!(
                "Embedding size {} does not match configured dimension {}",
                embedding.len(),
                self.config.embedding_dim
            );
        }

        Ok(embedding)
    }

    /// Call the vLLM model for text generation
    pub async fn call_model(&self, prompt: &str) -> Result<String> {
        let request = json!({
            "model": self.config.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 1024
        });

        let response = self.client
            .post(&format!("{}/v1/chat/completions", self.config.vllm_endpoint))
            .json(&request)
            .send()
            .await?
            .json::<Value>()
            .await?;

        let content = response["choices"][0]["message"]["content"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Invalid response format"))?;

        Ok(content.to_string())
    }

    /// Process and store an experience
    pub async fn process_experience(
        &mut self,
        experience: Experience,
        memory: &MemoryCore,
    ) -> Result<()> {
        // Embed the experience content
        let content = format!("Input: {}\nOutput: {}\nContext: {}",
                             experience.input, experience.output, experience.context);
        let mut embedded_exp = experience;
        embedded_exp.embedding = Some(self.embed_text(&content).await?);

        // Store in memory
        memory.store_experience(&embedded_exp).await?;

        Ok(())
    }

    /// Perform knowledge distillation from experience clusters
    pub async fn distill_knowledge(
        &mut self,
        memory: &MemoryCore,
        num_clusters: usize,
    ) -> Result<Vec<DistilledExample>> {
        // Get all experiences (in batches for memory efficiency)
        let all_experiences = self.get_all_experiences(memory).await?;

        if all_experiences.is_empty() {
            return Ok(Vec::new());
        }

        // Cluster experiences by similarity in background task to avoid blocking
        let threshold = self.config.clustering_threshold;
        let clusters = tokio::task::spawn_blocking(move || {
            Self::cluster_experiences_static(&all_experiences, threshold)
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

    /// Retrieve all experiences from memory
    async fn get_all_experiences(&self, memory: &MemoryCore) -> Result<Vec<Experience>> {
        // This is a simplified version - in practice you'd scroll through all points
        // For now, we'll get a sample of recent experiences
        let dummy_query = vec![0.0; self.config.embedding_dim];
        let similar = memory.search_similar(&dummy_query, 1000).await?;

        Ok(similar)
    }

    /// Cluster experiences using simple agglomerative clustering
    fn cluster_experiences(&self, experiences: &[Experience]) -> Result<Vec<Vec<Experience>>> {
        Self::cluster_experiences_static(experiences, self.config.clustering_threshold)
    }

    /// Static version for background execution
    fn cluster_experiences_static(experiences: &[Experience], threshold: f32) -> Result<Vec<Vec<Experience>>> {
        let mut clusters: Vec<Vec<Experience>> = experiences.iter().cloned().map(|e| vec![e]).collect();

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

    /// Calculate similarity between two clusters (average pairwise similarity)
    fn cluster_similarity(&self, cluster_a: &[Experience], cluster_b: &[Experience]) -> f32 {
        Self::cluster_similarity_static(cluster_a, cluster_b)
    }

    /// Static version for background execution
    fn cluster_similarity_static(cluster_a: &[Experience], cluster_b: &[Experience]) -> f32 {
        let mut total_sim = 0.0;
        let mut count = 0;

        for exp_a in cluster_a {
            for exp_b in cluster_b {
                if let (Some(emb_a), Some(emb_b)) = (&exp_a.embedding, &exp_b.embedding) {
                    let sim = cosine_similarity(emb_a, emb_b);
                    total_sim += sim;
                    count += 1;
                }
            }
        }

        if count > 0 {
            total_sim / count as f32
        } else {
            0.0
        }
    }

    /// Distill a cluster into a single training example
    fn distill_cluster(&mut self, cluster: &[Experience]) -> Result<Option<DistilledExample>> {
        if cluster.is_empty() {
            return Ok(None);
        }

        // Find the most successful experience in the cluster
        let best_experience = cluster.iter()
            .max_by(|a, b| a.success_score.partial_cmp(&b.success_score).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();

        // Generate a generalized instruction from the cluster
        let instruction = self.generate_instruction_from_cluster(cluster)?;

        // Use the best experience's output as the target
        let output = best_experience.output.clone();

        Ok(Some(DistilledExample {
            instruction,
            output,
            quality_score: best_experience.success_score,
            cluster_size: cluster.len(),
        }))
    }

    /// Generate a generalized instruction from a cluster of experiences
    fn generate_instruction_from_cluster(&mut self, cluster: &[Experience]) -> Result<String> {
        // Simple approach: use the most common task type and generalize the input
        let mut task_counts = HashMap::new();
        for exp in cluster {
            *task_counts.entry(exp.task_type.clone()).or_insert(0) += 1;
        }

        let most_common_task = task_counts.into_iter()
            .max_by_key(|&(_, count)| count)
            .map(|(task, _)| task)
            .unwrap_or_else(|| "general_task".to_string());

        // Create a generalized instruction
        let generalized_input = self.generalize_input(cluster)?;

        Ok(format!("Task: {}\nInput: {}", most_common_task, generalized_input))
    }

    /// Generalize input patterns from a cluster
    fn generalize_input(&self, cluster: &[Experience]) -> Result<String> {
        // Simple generalization: take the first experience's input as template
        // In a more sophisticated implementation, this would use pattern mining
        if let Some(first) = cluster.first() {
            Ok(first.input.clone())
        } else {
            Ok("".to_string())
        }
    }

    /// Curate memory: remove low-quality or outdated experiences
    pub async fn curate_memory(&self, memory: &MemoryCore) -> Result<()> {
        // Perform memory compaction
        memory.compact_memory(0.8).await?;
        Ok(())
    }
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

/// A distilled training example
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistilledExample {
    pub instruction: String,
    pub output: String,
    pub quality_score: f32,
    pub cluster_size: usize,
}