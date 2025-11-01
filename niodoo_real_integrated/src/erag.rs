use anyhow::{anyhow, Result};
use chrono::Utc;
use qdrant_client::{
    client::{Payload, QdrantClient, QdrantClientConfig},
    qdrant::{
        self, quantization_config_diff, value::Kind as QdrantValueKind, CreateCollection, Distance,
        PointStruct, QuantizationConfig, QuantizationConfigDiff,
        QuantizationType as QdrantQuantizationType, ScalarQuantization, SearchPoints,
        UpdateCollection, VectorParams, VectorsConfig,
    },
};
use rand::{thread_rng, Rng};
use serde::{Deserialize, Serialize};
use serde_json::{Map as JsonMap, Value as JsonValue};
use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::time::{interval, Duration};
use tracing::{info, instrument, warn};
use uuid::Uuid;

use crate::circuit_breaker::{CircuitBreaker, CircuitBreakerConfig};

use crate::compass::{CascadeStage, CompassOutcome};
use crate::torus::PadGhostState;
use crate::weighted_episodic_mem::{
    age_in_days, calculate_fitness_score, initialize_memory_metadata, update_retrieval_stats,
    TemporalDecayConfig, WeightedMemoryMetadata, DEFAULT_FITNESS_WEIGHTS,
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
    client: Arc<QdrantClient>,
    collection: String,
    vector_dim: usize,
    pub similarity_threshold: f32,
    /// Fitness weights for weighted episodic memory [temporal, pad, beta1, retrieval, consonance, resource_penalty]
    pub fitness_weights: [f32; 6],
    /// Temporal decay configuration
    pub temporal_config: TemporalDecayConfig,
    /// Optional resource budget for resource-aware fitness calculation
    pub resource_budget: Option<std::sync::Arc<crate::resource_budget::GlobalResourceBudget>>,
    /// Circuit breaker for Qdrant requests
    circuit_breaker: Arc<CircuitBreaker>,
    /// Batch queue for optimized upserts (Phase 1.2)
    batch_queue: Arc<Mutex<VecDeque<PointStruct>>>,
    /// Batch size configuration
    batch_size: usize,
    /// Batch flush interval in milliseconds
    batch_flush_ms: u64,
    /// Whether batching is enabled
    optimized_erag: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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
        Self::new_with_config(
            url,
            collection,
            vector_dim,
            similarity_threshold,
            false,
            128,
            300,
        )
        .await
    }

    /// Create EragClient with optimization configuration (Phase 1.2)
    pub async fn new_with_config(
        url: &str,
        collection: &str,
        vector_dim: usize,
        similarity_threshold: f32,
        optimized_erag: bool,
        batch_size: usize,
        batch_flush_ms: u64,
    ) -> Result<Self> {
        Self::new_with_config_and_quantization(
            url,
            collection,
            vector_dim,
            similarity_threshold,
            optimized_erag,
            batch_size,
            batch_flush_ms,
            None,
        )
        .await
    }

    /// Create EragClient with optimization configuration including quantization (Phase 1.3)
    pub async fn new_with_config_and_quantization(
        url: &str,
        collection: &str,
        vector_dim: usize,
        similarity_threshold: f32,
        optimized_erag: bool,
        batch_size: usize,
        batch_flush_ms: u64,
        quantization: Option<crate::config::QuantizationType>,
    ) -> Result<Self> {
        // Normalise URL for qdrant-client. We expect to talk to the gRPC endpoint on port 6334
        // using HTTP/2, which the SDK represents as an `http://` URI. Accept a variety of inputs
        // (raw host, http://:6333 REST URL, or legacy grpc:// scheme) and rewrite them so we
        // always land on the gRPC port.
        let mut normalized_url = if url.starts_with("grpc://") {
            let fallback = url.replace("grpc://", "http://");
            warn!(original = %url, normalized = %fallback, "Qdrant URL used grpc schema; rewriting to http for SDK compatibility");
            fallback
        } else if url.starts_with("http://") || url.starts_with("https://") {
            url.to_string()
        } else {
            format!("http://{}", url)
        };

        if normalized_url.contains(":6333") {
            normalized_url = normalized_url.replace(":6333", ":6334");
            info!(original = %url, rewritten = %normalized_url, "Adjusted Qdrant URL to gRPC port 6334");
        } else if !normalized_url.contains(":") {
            normalized_url = format!("{}:6334", normalized_url.trim_end_matches('/'));
            info!(original = %url, rewritten = %normalized_url, "Appended gRPC port 6334 to Qdrant URL");
        }

        let config = QdrantClientConfig::from_url(&normalized_url);
        let client = QdrantClient::new(Some(config))
            .map_err(|err| anyhow!("failed to build qdrant client: {}", err))?;

        ensure_collection(&client, collection, vector_dim, quantization).await?;

        // Apply quantization to existing collection if specified
        if let Some(quant_type) = quantization {
            if let Err(e) = update_collection_quantization(&client, collection, quant_type).await {
                warn!(%e, "Failed to update collection quantization; continuing without quantization");
            } else {
                info!(collection = %collection, quantization = ?quant_type, "Enabled quantization for Qdrant collection");
            }
        }

        info!(url = %normalized_url, original_url = %url, collection = %collection, "EragClient initialized for ERAG memory store");

        let circuit_breaker = Arc::new(CircuitBreaker::new(
            "qdrant",
            CircuitBreakerConfig::default(),
        ));

        let batch_queue = Arc::new(Mutex::new(VecDeque::new()));

        let client = Self {
            client: Arc::new(client),
            collection: collection.to_string(),
            vector_dim,
            similarity_threshold,
            fitness_weights: DEFAULT_FITNESS_WEIGHTS,
            temporal_config: TemporalDecayConfig::default(),
            resource_budget: None,
            circuit_breaker: circuit_breaker.clone(),
            batch_queue: batch_queue.clone(),
            batch_size,
            batch_flush_ms,
            optimized_erag,
        };

        // Start background flush task if batching is enabled
        if optimized_erag {
            let client_clone = client.clone();
            let collection_clone = collection.to_string();
            let batch_queue_clone = batch_queue.clone();
            let circuit_breaker_clone = circuit_breaker.clone();
            let batch_size_clone = batch_size;
            let flush_interval = Duration::from_millis(batch_flush_ms);

            tokio::spawn(async move {
                let mut interval_timer = interval(flush_interval);
                loop {
                    interval_timer.tick().await;
                    if let Err(e) = Self::flush_batch_internal(
                        &client_clone.client,
                        &collection_clone,
                        &batch_queue_clone,
                        &circuit_breaker_clone,
                        batch_size_clone,
                    )
                    .await
                    {
                        warn!(%e, "Failed to flush ERAG batch queue");
                    }
                }
            });
        }

        Ok(client)
    }

    #[instrument(skip_all, fields(dim = vector.len()))]
    pub async fn collapse(&self, vector: &[f32]) -> Result<CollapseResult> {
        self.collapse_with_cascade_preference(vector, None).await
    }

    /// Collapse with limit (for backward compatibility)
    pub async fn collapse_with_limit(
        &self,
        vector: &[f32],
        limit: usize,
    ) -> Result<CollapseResult> {
        self.collapse_with_limit_and_cascade(vector, limit, None)
            .await
    }

    /// Collapse with cascade stage preference - prefers memories from same cascade stage
    #[instrument(skip_all, fields(dim = vector.len()))]
    pub async fn collapse_with_cascade_preference(
        &self,
        vector: &[f32],
        preferred_stage: Option<CascadeStage>,
    ) -> Result<CollapseResult> {
        self.collapse_with_limit_and_cascade(vector, 3, preferred_stage)
            .await
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

        // Use gRPC search via qdrant-client
        let client = &self.client;
        let collection = self.collection.clone();
        let threshold = self.similarity_threshold;

        let response_result = self
            .circuit_breaker
            .call(|| {
                let vector_clone = vector.to_vec();
                let collection_clone = collection.clone();
                async move {
                    let search_points = SearchPoints {
                        collection_name: collection_clone,
                        vector: vector_clone,
                        limit: limit as u64,
                        score_threshold: Some(threshold),
                        with_payload: Some(true.into()),
                        with_vectors: Some(false.into()),
                        ..Default::default()
                    };

                    client
                        .search_points(&search_points)
                        .await
                        .map_err(|e| anyhow!("Qdrant gRPC search failed: {}", e))
                }
            })
            .await;

        let mut memories = Vec::new();
        let mut sims = Vec::new();
        match response_result {
            Ok(search_result) => {
                for hit in search_result.result {
                    let payload_json: JsonMap<String, JsonValue> = hit
                        .payload
                        .into_iter()
                        .map(|(k, v)| {
                            // Convert Qdrant Value to JsonValue
                            let json_val = match v.kind {
                                Some(qdrant_client::qdrant::value::Kind::BoolValue(b)) => {
                                    JsonValue::Bool(b)
                                }
                                Some(qdrant_client::qdrant::value::Kind::IntegerValue(i)) => {
                                    JsonValue::Number(serde_json::Number::from(i))
                                }
                                Some(qdrant_client::qdrant::value::Kind::DoubleValue(d)) => {
                                    JsonValue::Number(
                                        serde_json::Number::from_f64(d)
                                            .unwrap_or(serde_json::Number::from(0)),
                                    )
                                }
                                Some(qdrant_client::qdrant::value::Kind::StringValue(s)) => {
                                    JsonValue::String(s)
                                }
                                _ => JsonValue::Null,
                            };
                            (k, json_val)
                        })
                        .collect();
                    memories.push(deserialize_memory(&payload_json));
                    sims.push(hit.score);
                }
            }
            Err(err) => {
                warn!(%err, "qdrant gRPC search failed - proceeding without hits");
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
            let mut memory_sim_pairs: Vec<_> =
                memories.into_iter().zip(sims.iter().copied()).collect();
            memory_sim_pairs
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
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
            memory_fitness_pairs
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            memories = memory_fitness_pairs
                .iter()
                .map(|(m, _)| (*m).clone())
                .collect();

            // Update sims to match fitness scores for consistency
            sims = memory_fitness_pairs.iter().map(|(_, f)| *f).collect();
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

        // Convert payload to Qdrant Payload format
        let mut qdrant_payload = Payload::new();
        for (k, v) in payload {
            match v {
                JsonValue::String(s) => qdrant_payload.insert(k, s),
                JsonValue::Number(n) => {
                    if let Some(i) = n.as_i64() {
                        qdrant_payload.insert(k, i);
                    } else if let Some(f) = n.as_f64() {
                        qdrant_payload.insert(k, f);
                    } else {
                        qdrant_payload.insert(k, n.to_string());
                    }
                }
                JsonValue::Bool(b) => qdrant_payload.insert(k, b),
                JsonValue::Array(arr) => {
                    // Try to extract strings from array
                    let strs: Vec<String> = arr
                        .iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect();
                    if !strs.is_empty() {
                        qdrant_payload.insert(k, strs);
                    } else {
                        // Fallback: convert to string array
                        qdrant_payload
                            .insert(k, arr.iter().map(|v| v.to_string()).collect::<Vec<_>>());
                    }
                }
                _ => qdrant_payload.insert(k, v.to_string()),
            }
        }

        let point = PointStruct::new(
            uuid::Uuid::new_v4().to_string(),
            vector.to_vec(),
            qdrant_payload,
        );

        // Phase 1.2: Batch upserts if optimization is enabled
        if self.optimized_erag {
            // Queue point for batch processing
            let mut queue = self.batch_queue.lock().await;
            queue.push_back(point);
            let queue_size = queue.len();

            // Record queued points metric (Phase 1.5)
            crate::metrics::erag_batch_metrics().record_queued_points(queue_size);

            // Flush if batch size reached
            if queue_size >= self.batch_size {
                drop(queue); // Release lock before async operation
                self.flush_batch().await?;
            }

            Ok(())
        } else {
            // Legacy: immediate upsert
            crate::metrics::erag_batch_metrics().record_immediate_upsert();
            match self
                .client
                .upsert_points(self.collection.clone(), None, vec![point], None)
                .await
            {
                Ok(_) => {
                    info!(collection = %self.collection, "stored ERAG memory via gRPC");
                    Ok(())
                }
                Err(err) => Err(anyhow!("failed to upsert erag memory via gRPC: {}", err)),
            }
        }
    }

    /// Flush batched points to Qdrant
    pub async fn flush_batch(&self) -> Result<()> {
        Self::flush_batch_internal(
            &self.client,
            &self.collection,
            &self.batch_queue,
            &self.circuit_breaker,
            self.batch_size,
        )
        .await
    }

    /// Internal batch flush implementation
    async fn flush_batch_internal(
        client: &Arc<QdrantClient>,
        collection: &str,
        batch_queue: &Arc<Mutex<VecDeque<PointStruct>>>,
        circuit_breaker: &Arc<CircuitBreaker>,
        batch_size: usize,
    ) -> Result<()> {
        let start = std::time::Instant::now();
        let mut queue = batch_queue.lock().await;
        if queue.is_empty() {
            return Ok(());
        }

        // Take up to batch_size points
        let mut points_to_flush = Vec::new();
        for _ in 0..batch_size.min(queue.len()) {
            if let Some(point) = queue.pop_front() {
                points_to_flush.push(point);
            } else {
                break;
            }
        }
        let batch_count = points_to_flush.len();
        drop(queue);

        if points_to_flush.is_empty() {
            return Ok(());
        }

        // Upsert with circuit breaker protection
        let client_clone = client.clone();
        let collection_clone = collection.to_string();
        let points_clone = points_to_flush.clone();

        let result = circuit_breaker
            .call(|| async move {
                client_clone
                    .upsert_points(collection_clone, None, points_clone, None)
                    .await
                    .map_err(|e| anyhow!("Qdrant gRPC batch upsert failed: {}", e))
            })
            .await;

        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

        match result {
            Ok(_) => {
                // Record successful batch flush metrics (Phase 1.5)
                crate::metrics::erag_batch_metrics().record_batch_flush(batch_count, latency_ms);
                info!(
                    collection = %collection,
                    count = batch_count,
                    latency_ms = latency_ms,
                    "flushed ERAG batch to Qdrant via gRPC"
                );
                Ok(())
            }
            Err(e) => {
                // Record failure metric
                crate::metrics::erag_batch_metrics().record_batch_flush_failure();
                Err(e)
            }
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
                if matches!(
                    stage,
                    CascadeStage::Recognition | CascadeStage::Satisfaction
                ) {
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
            b_improvement
                .partial_cmp(&a_improvement)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(consolidated)
    }
}

async fn ensure_collection(
    client: &QdrantClient,
    collection: &str,
    vector_dim: usize,
    quantization: Option<crate::config::QuantizationType>,
) -> Result<()> {
    let collections = client
        .list_collections()
        .await
        .map_err(|err| anyhow!("failed to list Qdrant collections: {err}"))?;

    let exists = collections.collections.iter().any(|c| c.name == collection);

    if !exists {
        let mut vectors_config = VectorsConfig {
            config: Some(qdrant::vectors_config::Config::Params(VectorParams {
                size: vector_dim as u64,
                distance: Distance::Cosine.into(),
                ..Default::default()
            })),
        };

        // Phase 1.3: Add quantization config if specified
        if let Some(quant_type) = quantization {
            let quantization_config = match quant_type {
                crate::config::QuantizationType::ScalarPQ4 => Some(QuantizationConfig {
                    quantization: Some(qdrant::quantization_config::Quantization::Scalar(
                        ScalarQuantization {
                            r#type: QdrantQuantizationType::Int8.into(),
                            quantile: Some(0.99),
                            always_ram: Some(true),
                        },
                    )),
                }),
                crate::config::QuantizationType::None => None,
            };

            if let Some(quant_config) = quantization_config {
                // Note: Quantization is typically set via UpdateCollection after creation
                // For now, we'll create the collection and update it separately
            }
        }

        let create_request = CreateCollection {
            collection_name: collection.to_string(),
            vectors_config: Some(vectors_config),
            ..Default::default()
        };

        client
            .create_collection(&create_request)
            .await
            .map_err(|err| anyhow!("failed to create Qdrant collection '{collection}': {err}"))?;

        info!(
            collection,
            vector_dim, "Created Qdrant collection for ERAG memories"
        );
    }

    Ok(())
}

/// Update collection quantization configuration (Phase 1.3)
pub async fn update_collection_quantization(
    client: &QdrantClient,
    collection: &str,
    quantization: crate::config::QuantizationType,
) -> Result<()> {
    match quantization {
        crate::config::QuantizationType::ScalarPQ4 => {
            // Create QuantizationConfigDiff for update_collection API
            // Note: QuantizationConfigDiff uses quantization_config_diff::Quantization enum
            let quantization_config_diff = QuantizationConfigDiff {
                quantization: Some(quantization_config_diff::Quantization::Scalar(
                    ScalarQuantization {
                        r#type: QdrantQuantizationType::Int8.into(),
                        quantile: Some(0.99),
                        always_ram: Some(true),
                    },
                )),
            };

            // Use new API signature: update_collection takes 7 arguments
            client
                .update_collection(
                    collection,
                    None,                            // optimizers_config
                    None,                            // params
                    None,                            // sparse_vector_config
                    None,                            // hnsw_config
                    None,                            // vectors_config
                    Some(&quantization_config_diff), // quantization_config
                )
                .await
                .map_err(|err| anyhow!("failed to update collection quantization: {}", err))?;

            info!(
                collection = %collection,
                quantization = "ScalarPQ4",
                "Updated Qdrant collection quantization"
            );
        }
        crate::config::QuantizationType::None => {
            // No quantization - nothing to do
        }
    }

    Ok(())
}

#[derive(Debug, Clone, Serialize)]
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
            payload.insert("community_id".to_string(), JsonValue::from(comm_id as u64));
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
    pub async fn query_low_reward_tuples(
        &self,
        threshold: f64,
        limit: usize,
    ) -> Result<Vec<crate::data::Experience>> {
        let max_fetch = (limit.max(1) * 4).min(512);
        let search_points = self
            .search_points_with_vector(vec![0.0; self.vector_dim], max_fetch, None, true)
            .await?;

        let mut experiences: Vec<_> = search_points
            .into_iter()
            .filter_map(|point| scored_point_to_experience(&point, self.vector_dim))
            .filter(|exp| exp.reward < threshold)
            .collect();

        experiences.sort_by(|a, b| {
            a.reward
                .partial_cmp(&b.reward)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        experiences.truncate(limit);
        Ok(experiences)
    }

    /// Query replay batch
    pub async fn query_replay_batch(
        &self,
        _query: &str,
        metrics: &[f32],
        batch_size: usize,
    ) -> Result<Vec<crate::data::Experience>> {
        let mut query_vector = vec![0.0f32; self.vector_dim];
        if !metrics.is_empty() {
            let count = metrics.len().min(self.vector_dim);
            query_vector[..count].copy_from_slice(&metrics[..count]);
        }

        let max_fetch = (batch_size.max(1) * 4).min(512);
        let search_points = self
            .search_points_with_vector(query_vector, max_fetch, None, true)
            .await?;

        let mut experiences: Vec<_> = search_points
            .into_iter()
            .filter_map(|point| scored_point_to_experience(&point, self.vector_dim))
            .collect();

        experiences.sort_by(|a, b| {
            b.reward
                .partial_cmp(&a.reward)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        experiences.truncate(batch_size);
        Ok(experiences)
    }

    /// Query old DQN tuples
    pub async fn query_old_dqn_tuples(
        &self,
        _epoch: usize,
        limit: usize,
    ) -> Result<Vec<crate::data::Experience>> {
        let search_points = self
            .search_points_with_vector(vec![0.0; self.vector_dim], limit.max(1) * 4, None, true)
            .await?;

        let mut experiences: Vec<_> = search_points
            .into_iter()
            .filter_map(|point| scored_point_to_experience(&point, self.vector_dim))
            .collect();

        experiences.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
        experiences.truncate(limit);
        Ok(experiences)
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

    /// Check collection info and index health (Phase 1.4)
    pub async fn check_collection_info(&self) -> Result<()> {
        let collection_info = self
            .client
            .collection_info(&self.collection)
            .await
            .map_err(|err| anyhow!("failed to get collection info: {}", err))?;

        // Extract collection info from response
        let points_count = collection_info
            .result
            .as_ref()
            .and_then(|r| r.points_count)
            .unwrap_or(0);
        let indexed_vectors_count = collection_info
            .result
            .as_ref()
            .and_then(|r| r.indexed_vectors_count)
            .unwrap_or(0);

        info!(
            collection = %self.collection,
            points_count = points_count,
            indexed_vectors_count = indexed_vectors_count,
            "Collection info retrieved"
        );

        // Check if index needs rebuilding
        let indexed_ratio = if points_count > 0 {
            indexed_vectors_count as f64 / points_count as f64
        } else {
            1.0
        };

        if indexed_ratio < 0.95 {
            warn!(
                collection = %self.collection,
                indexed_ratio = indexed_ratio,
                "Collection index may need rebuilding (indexed ratio < 95%)"
            );
        }

        Ok(())
    }

    /// Trigger HNSW index rebuild for collection (Phase 1.4)
    pub async fn rebuild_index(&self) -> Result<()> {
        // Use UpdateCollection to trigger index rebuild
        // Note: Qdrant will rebuild index automatically on next optimization cycle
        // This method serves as a hook for manual rebuild triggers
        self.circuit_breaker
            .call(|| async move {
                self.client
                    .update_collection(
                        &self.collection,
                        None, // optimizers_config
                        None, // params
                        None, // sparse_vector_config
                        None, // hnsw_config
                        None, // vectors_config
                        None, // quantization_config
                    )
                    .await
                    .map_err(|e| anyhow!("failed to trigger index rebuild: {}", e))
            })
            .await?;

        info!(
            collection = %self.collection,
            "Triggered HNSW index rebuild"
        );

        Ok(())
    }

    /// Check if index health is good and trigger rebuild if needed (Phase 1.4)
    pub async fn ensure_index_health(&self) -> Result<bool> {
        let collection_info = self
            .client
            .collection_info(&self.collection)
            .await
            .map_err(|err| anyhow!("failed to get collection info: {}", err))?;

        let points_count = collection_info
            .result
            .as_ref()
            .and_then(|r| r.points_count)
            .unwrap_or(0);
        let indexed_vectors_count = collection_info
            .result
            .as_ref()
            .and_then(|r| r.indexed_vectors_count)
            .unwrap_or(0);

        let indexed_ratio = if points_count > 0 {
            indexed_vectors_count as f64 / points_count as f64
        } else {
            1.0
        };

        // If indexed ratio is below threshold, trigger rebuild
        if indexed_ratio < 0.90 {
            warn!(
                collection = %self.collection,
                indexed_ratio = indexed_ratio,
                "Index health below threshold, triggering rebuild"
            );
            self.rebuild_index().await?;
            Ok(false) // Index is being rebuilt
        } else {
            Ok(true) // Index is healthy
        }
    }
}

impl EragClient {
    /// Calculate fitness score for a memory
    pub fn calculate_memory_fitness(&self, memory: &EragMemory, pad_state: &PadGhostState) -> f32 {
        let metadata = memory.weighted_metadata.as_ref().unwrap_or_else(|| {
            // Use default metadata if missing
            static DEFAULT: std::sync::OnceLock<WeightedMemoryMetadata> =
                std::sync::OnceLock::new();
            DEFAULT.get_or_init(WeightedMemoryMetadata::default)
        });

        let timestamp = DateTime::parse_from_rfc3339(&memory.timestamp)
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or_else(|_| Utc::now());
        let age_days = age_in_days(&timestamp);

        // Get resource availability if resource budget is available
        let resource_availability = self
            .resource_budget
            .as_ref()
            .map(|budget| budget.get_resource_availability());

        calculate_fitness_score(
            age_days,
            pad_state,
            metadata.retrieval_count,
            metadata.beta_1_connectivity,
            metadata.consonance_score,
            metadata.consolidation_level,
            &self.fitness_weights,
            &self.temporal_config,
            resource_availability.as_ref(),
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
            collection: self.collection.clone(),
            vector_dim: self.vector_dim,
            similarity_threshold: self.similarity_threshold,
            fitness_weights: self.fitness_weights,
            temporal_config: self.temporal_config.clone(),
            resource_budget: self.resource_budget.clone(),
            circuit_breaker: self.circuit_breaker.clone(),
            batch_queue: self.batch_queue.clone(),
            batch_size: self.batch_size,
            batch_flush_ms: self.batch_flush_ms,
            optimized_erag: self.optimized_erag,
        }
    }
}

impl EragClient {
    async fn search_points_with_vector(
        &self,
        vector: Vec<f32>,
        limit: usize,
        filter: Option<qdrant::Filter>,
        include_vectors: bool,
    ) -> Result<Vec<qdrant::ScoredPoint>> {
        let search_points = SearchPoints {
            collection_name: self.collection.clone(),
            vector,
            limit: limit as u64,
            filter,
            with_payload: Some(true.into()),
            with_vectors: Some(include_vectors.into()),
            ..Default::default()
        };

        let result = self
            .client
            .search_points(&search_points)
            .await
            .map_err(|err| anyhow!("failed to query Qdrant points: {}", err))?;

        Ok(result.result)
    }
}

fn scored_point_to_experience(
    point: &qdrant::ScoredPoint,
    vector_dim: usize,
) -> Option<crate::data::Experience> {
    let payload = &point.payload;

    let input = payload
        .get("input")
        .and_then(value_as_string)
        .unwrap_or_default();
    let output = payload
        .get("output")
        .and_then(value_as_string)
        .unwrap_or_default();

    let entropy_before = payload
        .get("entropy_before")
        .and_then(value_as_f64)
        .unwrap_or(0.0);
    let entropy_after = payload
        .get("entropy_after")
        .and_then(value_as_f64)
        .unwrap_or(entropy_before);
    let fitness_score = payload
        .get("fitness_score")
        .and_then(value_as_f64)
        .unwrap_or(0.0);

    let timestamp = payload
        .get("timestamp")
        .and_then(value_as_string)
        .and_then(|ts| DateTime::parse_from_rfc3339(&ts).ok())
        .map(|dt| dt.with_timezone(&Utc))
        .unwrap_or_else(Utc::now);

    let compass_state = payload
        .get("compass_state")
        .and_then(value_as_string)
        .unwrap_or_default();

    let context = payload
        .get("erag_context")
        .and_then(value_as_string_vec)
        .unwrap_or_default();

    let mut state_vec = vec![0.0f32; vector_dim];
    if let Some(vectors) = point.vectors.as_ref() {
        if let Some(qdrant::vectors_output::VectorsOptions::Vector(vec_data)) =
            vectors.vectors_options.as_ref()
        {
            if !vec_data.data.is_empty() {
                state_vec = vec_data.data.clone();
            }
        }
    }

    if state_vec.len() < vector_dim {
        state_vec.resize(vector_dim, 0.0);
    } else if state_vec.len() > vector_dim {
        state_vec.truncate(vector_dim);
    }

    let reward = if fitness_score.abs() > f64::EPSILON {
        fitness_score
    } else {
        entropy_before - entropy_after
    };

    Some(crate::data::Experience {
        id: Uuid::new_v4(),
        timestamp,
        input,
        output,
        context,
        task_type: payload
            .get("task_type")
            .and_then(value_as_string)
            .unwrap_or_else(|| "hybrid_generation".to_string()),
        success_score: fitness_score as f32,
        state: state_vec.clone(),
        action: compass_state_to_action(&compass_state),
        reward,
        next_state: state_vec,
        done: false,
        replay: None,
    })
}

fn compass_state_to_action(state: &str) -> usize {
    match state {
        "Persist" => 1,
        "Discover" => 2,
        "Master" => 3,
        _ => 0,
    }
}

fn value_as_string(value: &qdrant::Value) -> Option<String> {
    match &value.kind {
        Some(QdrantValueKind::StringValue(s)) => Some(s.clone()),
        Some(QdrantValueKind::IntegerValue(i)) => Some(i.to_string()),
        Some(QdrantValueKind::DoubleValue(d)) => Some(d.to_string()),
        Some(QdrantValueKind::BoolValue(b)) => Some(b.to_string()),
        _ => None,
    }
}

fn value_as_string_vec(value: &qdrant::Value) -> Option<Vec<String>> {
    match &value.kind {
        Some(QdrantValueKind::ListValue(list)) => {
            let items = list
                .values
                .iter()
                .filter_map(value_as_string)
                .collect::<Vec<_>>();
            if items.is_empty() {
                None
            } else {
                Some(items)
            }
        }
        _ => None,
    }
}

fn value_as_f64(value: &qdrant::Value) -> Option<f64> {
    match &value.kind {
        Some(QdrantValueKind::DoubleValue(v)) => Some(*v),
        Some(QdrantValueKind::IntegerValue(i)) => Some(*i as f64),
        Some(QdrantValueKind::StringValue(s)) => s.parse::<f64>().ok(),
        _ => None,
    }
}
