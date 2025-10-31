use anyhow::{anyhow, Result};
use chrono::Utc;
use rand::{thread_rng, Rng};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map as JsonMap, Value as JsonValue};
use std::time::Duration;
use tracing::{info, instrument, warn};

use crate::compass::{CascadeStage, CompassOutcome};
use crate::torus::PadGhostState;
use crate::weighted_episodic_mem::{
    WeightedMemoryMetadata, calculate_fitness_score, age_in_days, update_retrieval_stats,
    initialize_memory_metadata, DEFAULT_FITNESS_WEIGHTS, TemporalDecayConfig,
};
use chrono::DateTime;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EmotionalVector {
    pub joy: f32,
    pub sadness: f32,
    pub anger: f32,
    pub fear: f32,
    pub surprise: f32,
}

impl EmotionalVector {
    pub fn from_pad(state: &PadGhostState) -> Self {
        let mut rng = thread_rng();
        let joy = (state.pad[0] + rng.gen_range(-0.4..0.4)).clamp(-1.0, 1.0);
        let arousal = (state.pad[1] + rng.gen_range(-0.4..0.4)).clamp(-1.0, 1.0);
        let surprise = (state.pad[2] + rng.gen_range(-0.3..0.3)).clamp(-1.0, 1.0);

        Self {
            joy: joy as f32,
            sadness: (-joy).max(0.0) as f32,
            anger: arousal as f32,
            fear: (-arousal).max(0.0) as f32,
            surprise: surprise as f32,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EragMemory {
    pub input: String,
    pub output: String,
    pub emotional_vector: EmotionalVector,
    pub erag_context: Vec<String>,
    pub entropy_before: f64,
    pub entropy_after: f64,
    pub timestamp: String,
    pub compass_state: Option<String>,
    pub cascade_stage: Option<CascadeStage>, // Cascade stage metadata
    /// Weighted episodic memory metadata (optional for backward compatibility)
    #[serde(default)]
    pub weighted_metadata: Option<WeightedMemoryMetadata>,
}

pub struct EragClient {
    client: Client,
    base_url: String,
    collection: String,
    vector_dim: usize,
    pub similarity_threshold: f32,
    /// Fitness weights for weighted episodic memory [temporal, pad, beta1, retrieval, consonance]
    pub fitness_weights: [f32; 5],
    /// Temporal decay configuration
    pub temporal_config: TemporalDecayConfig,
}

#[derive(Debug, Clone)]
pub struct CollapseResult {
    pub top_hits: Vec<EragMemory>,
    pub aggregated_context: String,
    pub average_similarity: f32,
    pub curator_quality: Option<f64>, // Add missing field
}

impl EragClient {
    pub async fn new(
        url: &str,
        collection: &str,
        vector_dim: usize,
        similarity_threshold: f32,
    ) -> Result<Self> {
        let base_url = url.trim_end_matches('/').to_string();
        let client = Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .map_err(|err| anyhow!("failed to build qdrant http client: {err}"))?;
        Ok(Self {
            client,
            base_url,
            collection: collection.to_string(),
            vector_dim,
            similarity_threshold,
            fitness_weights: DEFAULT_FITNESS_WEIGHTS,
            temporal_config: TemporalDecayConfig::default(),
        })
    }

    #[instrument(skip_all, fields(dim = vector.len()))]
    pub async fn collapse(&self, vector: &[f32]) -> Result<CollapseResult> {
        self.collapse_with_cascade_preference(vector, None).await
    }

    /// Collapse with limit (for backward compatibility)
    pub async fn collapse_with_limit(&self, vector: &[f32], limit: usize) -> Result<CollapseResult> {
        self.collapse_with_limit_and_cascade(vector, limit, None).await
    }

    /// Collapse with cascade stage preference - prefers memories from same cascade stage
    #[instrument(skip_all, fields(dim = vector.len()))]
    pub async fn collapse_with_cascade_preference(
        &self,
        vector: &[f32],
        preferred_stage: Option<CascadeStage>,
    ) -> Result<CollapseResult> {
        self.collapse_with_limit_and_cascade(vector, 3, preferred_stage).await
    }

    /// Collapse with limit and cascade stage preference
    pub async fn collapse_with_limit_and_cascade(
        &self,
        vector: &[f32],
        limit: usize,
        preferred_stage: Option<CascadeStage>,
    ) -> Result<CollapseResult> {
        anyhow::ensure!(
            vector.len() == self.vector_dim,
            "embedding dimension mismatch: expected {}, got {}",
            self.vector_dim,
            vector.len()
        );

        let request = SearchRequest {
            vector: vector.to_vec(),
            limit: limit as u64,
            score_threshold: Some(self.similarity_threshold),
            with_payload: true,
            with_vectors: false,
        };

        let url = format!(
            "{}/collections/{}/points/search",
            self.base_url, self.collection
        );
        let response = self.client.post(url).json(&request).send().await;
        let mut memories = Vec::new();
        let mut sims = Vec::new();
        match response {
            Ok(resp) => {
                if resp.status().is_success() {
                    match resp.json::<SearchResponse>().await {
                        Ok(parsed) => {
                            for hit in parsed.result {
                                memories.push(deserialize_memory(&hit.payload));
                                sims.push(hit.score);
                            }
                        }
                        Err(err) => {
                            warn!(%err, "failed to decode qdrant search response");
                        }
                    }
                } else {
                    warn!(status = %resp.status(), "qdrant search returned error status");
                }
            }
            Err(err) => {
                warn!(%err, "qdrant search failed - proceeding without hits");
            }
        }

        if memories.is_empty() {
            sims.push(0.0);
        }

        // Update retrieval stats and calculate fitness scores for weighted retrieval
        for memory in &mut memories {
            if let Some(ref mut metadata) = memory.weighted_metadata {
                update_retrieval_stats(metadata);
            }
        }

        // If we have a preferred cascade stage, boost scores for matching memories
        if let Some(preferred) = preferred_stage {
            for (mem, sim) in memories.iter_mut().zip(sims.iter_mut()) {
                if let Some(stage) = mem.cascade_stage {
                    if stage == preferred {
                        // Boost similarity score by 20% for cascade-aligned memories
                        *sim = (*sim * 1.2).min(1.0);
                    }
                }
            }
            
            // Re-sort memories by boosted similarity scores
            let mut memory_sim_pairs: Vec<_> = memories.into_iter().zip(sims.iter().copied()).collect();
            memory_sim_pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            memories = memory_sim_pairs.iter().map(|(m, _)| (*m).clone()).collect();
            sims = memory_sim_pairs.iter().map(|(_, s)| *s).collect();
        }

        // Apply weighted fitness scoring if metadata is available
        if memories.iter().any(|m| m.weighted_metadata.is_some()) {
            let mut memory_fitness_pairs: Vec<_> = memories
                .into_iter()
                .map(|mem| {
                    let fitness = if let Some(ref metadata) = mem.weighted_metadata {
                        metadata.fitness_score
                    } else {
                        // Fallback: use similarity score as fitness
                        sims.get(0).copied().unwrap_or(0.0)
                    };
                    (mem, fitness)
                })
                .collect();
            
            // Sort by fitness score (higher is better)
            memory_fitness_pairs.sort_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            
            memories = memory_fitness_pairs.iter().map(|(m, _)| (*m).clone()).collect();
            
            // Update sims to match fitness scores for consistency
            sims = memory_fitness_pairs
                .iter()
                .map(|(_, f)| *f)
                .collect();
        }

        let average_similarity = if sims.is_empty() {
            0.0
        } else {
            sims.iter().copied().sum::<f32>() / sims.len() as f32
        };

        let mut aggregated_context = memories
            .iter()
            .flat_map(|m| m.erag_context.clone())
            .collect::<Vec<_>>()
            .join("\n");

        if aggregated_context.len() > 100 {
            aggregated_context.truncate(100);
        }

        Ok(CollapseResult {
            top_hits: memories,
            aggregated_context,
            average_similarity,
            curator_quality: None,
        })
    }

    pub async fn upsert_memory(
        &self,
        vector: &[f32],
        pad_state: &PadGhostState,
        compass: &CompassOutcome,
        prompt: &str,
        response: &str,
        context: &[String],
        entropy_before: f64,
    ) -> Result<()> {
        self.upsert_memory_with_cascade(
            vector,
            pad_state,
            compass,
            prompt,
            response,
            context,
            entropy_before,
            compass.cascade_stage,
        )
        .await
    }

    /// Upsert memory with explicit cascade stage
    pub async fn upsert_memory_with_cascade(
        &self,
        vector: &[f32],
        pad_state: &PadGhostState,
        compass: &CompassOutcome,
        prompt: &str,
        response: &str,
        context: &[String],
        entropy_before: f64,
        cascade_stage: Option<CascadeStage>,
    ) -> Result<()> {
        // Initialize weighted metadata for new memory
        let weighted_metadata = Some(initialize_memory_metadata(pad_state, 0.0));
        
        let memory = EragMemory {
            input: prompt.to_string(),
            output: response.to_string(),
            emotional_vector: EmotionalVector::from_pad(pad_state),
            erag_context: context.to_vec(),
            entropy_before,
            entropy_after: pad_state.entropy,
            timestamp: Utc::now().to_rfc3339(),
            compass_state: Some(format!("{:?}", compass.quadrant)),
            cascade_stage,
            weighted_metadata,
        };

        let payload = encode_payload(&memory);
        let request_body = json!({
            "points": [
                {
                    "id": uuid::Uuid::new_v4().to_string(),
                    "vector": vector,
                    "payload": payload,
                }
            ]
        });

        let url = format!("{}/collections/{}/points", self.base_url, self.collection);
        let response = self.client.put(url).json(&request_body).send().await;
        match response {
            Ok(resp) if resp.status().is_success() => {
                info!(collection = %self.collection, "stored ERAG memory");
                Ok(())
            }
            Ok(resp) => Err(anyhow!(
                "failed to upsert erag memory: http {}",
                resp.status()
            )),
            Err(err) => Err(anyhow!("failed to upsert erag memory: {err}")),
        }
    }

    /// Consolidate memories based on cascade stages
    /// When Recognition→Satisfaction cascade completes, consolidate into "truth attractor" memories
    /// When dissonance detected, flag for review/pruning
    pub async fn consolidate_by_cascade(
        &self,
        recognition_to_satisfaction_memories: &[EragMemory],
    ) -> Result<Vec<EragMemory>> {
        // Consolidate memories from Recognition→Satisfaction cascades
        // These represent "truth attractor" moments - high consonance breakthroughs
        let mut consolidated = Vec::new();
        
        for memory in recognition_to_satisfaction_memories {
            // Only consolidate memories from Recognition or Satisfaction stages
            if let Some(stage) = memory.cascade_stage {
                if matches!(stage, CascadeStage::Recognition | CascadeStage::Satisfaction) {
                    // Check if entropy improved (breakthrough indicator)
                    if memory.entropy_after > memory.entropy_before {
                        consolidated.push(memory.clone());
                    }
                }
            }
        }
        
        // Sort by entropy improvement (best breakthroughs first)
        consolidated.sort_by(|a, b| {
            let a_improvement = a.entropy_after - a.entropy_before;
            let b_improvement = b.entropy_after - b.entropy_before;
            b_improvement.partial_cmp(&a_improvement).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        Ok(consolidated)
    }
}

#[derive(Debug, Serialize)]
struct SearchRequest {
    vector: Vec<f32>,
    limit: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    score_threshold: Option<f32>,
    with_payload: bool,
    with_vectors: bool,
}

#[derive(Debug, Deserialize)]
struct SearchResponse {
    #[serde(default)]
    result: Vec<SearchHit>,
}

#[derive(Debug, Deserialize)]
struct SearchHit {
    score: f32,
    #[serde(default)]
    payload: JsonMap<String, JsonValue>,
}

fn encode_payload(memory: &EragMemory) -> JsonMap<String, JsonValue> {
    let mut payload = JsonMap::new();
    payload.insert("input".to_string(), JsonValue::String(memory.input.clone()));
    payload.insert(
        "output".to_string(),
        JsonValue::String(memory.output.clone()),
    );
    payload.insert(
        "entropy_before".to_string(),
        JsonValue::from(memory.entropy_before),
    );
    payload.insert(
        "entropy_after".to_string(),
        JsonValue::from(memory.entropy_after),
    );
    payload.insert(
        "timestamp".to_string(),
        JsonValue::String(memory.timestamp.clone()),
    );
    if let Some(ref state) = memory.compass_state {
        payload.insert(
            "compass_state".to_string(),
            JsonValue::String(state.clone()),
        );
    }
    
    // Store cascade stage as string
    if let Some(stage) = memory.cascade_stage {
        payload.insert(
            "cascade_stage".to_string(),
            JsonValue::String(stage.name().to_string()),
        );
    }

    let emotions = &memory.emotional_vector;
    payload.insert("joy".to_string(), JsonValue::from(emotions.joy as f64));
    payload.insert(
        "sadness".to_string(),
        JsonValue::from(emotions.sadness as f64),
    );
    payload.insert("anger".to_string(), JsonValue::from(emotions.anger as f64));
    payload.insert("fear".to_string(), JsonValue::from(emotions.fear as f64));
    payload.insert(
        "surprise".to_string(),
        JsonValue::from(emotions.surprise as f64),
    );

    payload.insert(
        "erag_context".to_string(),
        JsonValue::Array(
            memory
                .erag_context
                .iter()
                .cloned()
                .map(JsonValue::String)
                .collect(),
        ),
    );

    // Store weighted memory metadata if available
    if let Some(ref metadata) = memory.weighted_metadata {
        payload.insert(
            "fitness_score".to_string(),
            JsonValue::from(metadata.fitness_score as f64),
        );
        payload.insert(
            "retrieval_count".to_string(),
            JsonValue::from(metadata.retrieval_count as u64),
        );
        payload.insert(
            "last_accessed".to_string(),
            JsonValue::String(metadata.last_accessed.to_rfc3339()),
        );
        payload.insert(
            "consolidation_level".to_string(),
            JsonValue::from(metadata.consolidation_level as f64),
        );
        payload.insert(
            "beta_1_connectivity".to_string(),
            JsonValue::from(metadata.beta_1_connectivity as f64),
        );
        payload.insert(
            "consonance_score".to_string(),
            JsonValue::from(metadata.consonance_score as f64),
        );
        if let Some(comm_id) = metadata.community_id {
            payload.insert(
                "community_id".to_string(),
                JsonValue::from(comm_id as u64),
            );
        }
    }

    payload
}

fn deserialize_memory(payload: &JsonMap<String, JsonValue>) -> EragMemory {
    let context = payload
        .get("erag_context")
        .and_then(|value| value.as_array())
        .map(|values| {
            values
                .iter()
                .filter_map(|val| val.as_str().map(|s| s.to_string()))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    // Deserialize weighted metadata if available
    let weighted_metadata = if payload.contains_key("fitness_score") {
        Some(WeightedMemoryMetadata {
            fitness_score: extract_number(payload, "fitness_score") as f32,
            retrieval_count: extract_number(payload, "retrieval_count") as u32,
            last_accessed: payload
                .get("last_accessed")
                .and_then(|v| v.as_str())
                .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(Utc::now),
            consolidation_level: extract_number(payload, "consolidation_level") as f32,
            beta_1_connectivity: extract_number(payload, "beta_1_connectivity") as f32,
            consonance_score: extract_number(payload, "consonance_score") as f32,
            community_id: payload
                .get("community_id")
                .and_then(|v| v.as_u64())
                .map(|id| id as u32),
        })
    } else {
        None
    };

    EragMemory {
        input: extract_string(payload, "input"),
        output: extract_string(payload, "output"),
        emotional_vector: EmotionalVector {
            joy: extract_number(payload, "joy") as f32,
            sadness: extract_number(payload, "sadness") as f32,
            anger: extract_number(payload, "anger") as f32,
            fear: extract_number(payload, "fear") as f32,
            surprise: extract_number(payload, "surprise") as f32,
        },
        erag_context: context,
        entropy_before: extract_number(payload, "entropy_before"),
        entropy_after: extract_number(payload, "entropy_after"),
        timestamp: extract_string(payload, "timestamp"),
        compass_state: payload
            .get("compass_state")
            .and_then(|value| value.as_str().map(|s| s.to_string())),
        cascade_stage: payload
            .get("cascade_stage")
            .and_then(|value| value.as_str())
            .and_then(|s| match s {
                "Recognition" => Some(CascadeStage::Recognition),
                "Satisfaction" => Some(CascadeStage::Satisfaction),
                "Calm" => Some(CascadeStage::Calm),
                "Motivation" => Some(CascadeStage::Motivation),
                _ => None,
            }),
        weighted_metadata,
    }
}

fn extract_string(payload: &JsonMap<String, JsonValue>, key: &str) -> String {
    payload
        .get(key)
        .and_then(|value| value.as_str().map(|s| s.to_string()))
        .unwrap_or_default()
}

fn extract_number(payload: &JsonMap<String, JsonValue>, key: &str) -> f64 {
    payload
        .get(key)
        .and_then(|value| {
            if let Some(v) = value.as_f64() {
                Some(v)
            } else if let Some(v) = value.as_i64() {
                Some(v as f64)
            } else if let Some(v) = value.as_u64() {
                Some(v as f64)
            } else {
                None
            }
        })
        .unwrap_or_default()
}

impl EragClient {
    /// Query low-reward experience tuples
    pub async fn query_low_reward_tuples(&self, threshold: f64, limit: usize) -> Result<Vec<crate::data::Experience>> {
        // Stub implementation - returns empty vector
        Ok(Vec::new())
    }

    /// Query replay batch
    pub async fn query_replay_batch(&self, _query: &str, _metrics: &[f32], batch_size: usize) -> Result<Vec<crate::data::Experience>> {
        // Stub implementation - returns empty vector
        Ok(Vec::new())
    }

    /// Query old DQN tuples
    pub async fn query_old_dqn_tuples(&self, _epoch: usize, limit: usize) -> Result<Vec<crate::data::Experience>> {
        // Stub implementation - returns empty vector
        Ok(Vec::new())
    }

    /// Query tough knots
    pub async fn query_tough_knots(&self, limit: usize) -> Result<Vec<EragMemory>> {
        // Stub implementation - returns empty vector
        Ok(Vec::new())
    }

    /// Store failure case
    pub async fn store_failure(
        &self,
        input: &str,
        output: &str,
        metrics: &dyn std::any::Any,
        details: Option<String>,
        failure_type: &str,
        retry_count: u32,
    ) -> Result<()> {
        // Store failure memory with weighted metadata
        // For now, just log - full implementation would store to Qdrant
        tracing::warn!(
            input = %input,
            output = %output,
            failure_type = %failure_type,
            retry_count = retry_count,
            details = ?details,
            "Storing failure case"
        );
        Ok(())
    }

    /// Check collection info
    pub async fn check_collection_info(&self) -> Result<()> {
        // Stub implementation
        Ok(())
    }
}

impl EragClient {
    /// Calculate fitness score for a memory
    pub fn calculate_memory_fitness(
        &self,
        memory: &EragMemory,
        pad_state: &PadGhostState,
    ) -> f32 {
        let metadata = memory.weighted_metadata.as_ref().unwrap_or_else(|| {
            // Use default metadata if missing
            static DEFAULT: std::sync::OnceLock<WeightedMemoryMetadata> = std::sync::OnceLock::new();
            DEFAULT.get_or_init(WeightedMemoryMetadata::default)
        });

        let timestamp = DateTime::parse_from_rfc3339(&memory.timestamp)
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or_else(|_| Utc::now());
        let age_days = age_in_days(&timestamp);

        calculate_fitness_score(
            age_days,
            pad_state,
            metadata.retrieval_count,
            metadata.beta_1_connectivity,
            metadata.consonance_score,
            metadata.consolidation_level,
            &self.fitness_weights,
            &self.temporal_config,
        )
    }

    /// Batch calculate fitness scores for multiple memories
    pub fn batch_calculate_fitness(
        &self,
        memories: &[EragMemory],
        pad_state: &PadGhostState,
    ) -> Vec<f32> {
        memories
            .iter()
            .map(|mem| self.calculate_memory_fitness(mem, pad_state))
            .collect()
    }

    /// Update fitness score for a memory
    pub fn update_memory_fitness(&self, memory: &mut EragMemory, pad_state: &PadGhostState) {
        let fitness = self.calculate_memory_fitness(memory, pad_state);
        if let Some(ref mut metadata) = memory.weighted_metadata {
            metadata.fitness_score = fitness;
        } else {
            // Create metadata if missing
            let mut metadata = initialize_memory_metadata(pad_state, 0.0);
            metadata.fitness_score = fitness;
            memory.weighted_metadata = Some(metadata);
        }
    }
}

impl Clone for EragClient {
    fn clone(&self) -> Self {
        Self {
            client: self.client.clone(),
            base_url: self.base_url.clone(),
            collection: self.collection.clone(),
            vector_dim: self.vector_dim,
            similarity_threshold: self.similarity_threshold,
            fitness_weights: self.fitness_weights,
            temporal_config: self.temporal_config,
        }
    }
}
