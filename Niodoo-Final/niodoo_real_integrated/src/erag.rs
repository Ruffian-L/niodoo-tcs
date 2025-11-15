use anyhow::{anyhow, Result};
use chrono::Utc;
use parking_lot::RwLock;
use qdrant_client::{
    client::{Payload, QdrantClient, QdrantClientConfig},
    qdrant::{
        self, quantization_config_diff, value::Kind as QdrantValueKind, CreateCollection, Distance,
        PointStruct, QuantizationConfig, QuantizationConfigDiff,
        QuantizationType as QdrantQuantizationType, ScalarQuantization, SearchPoints, VectorParams,
        VectorsConfig,
    },
};
use rand::{thread_rng, Rng};
use serde::{Deserialize, Serialize};
use serde_json::{Map as JsonMap, Value as JsonValue};
use std::collections::{hash_map::DefaultHasher, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::time::{interval, Duration};
use tracing::{debug, info, instrument, warn};
use uuid::Uuid;

use crate::circuit_breaker::{CircuitBreaker, CircuitBreakerConfig};

use crate::compass::{CascadeStage, CompassOutcome};
use crate::config::{env_value, RuntimeConfig};
use crate::torus::PadGhostState;
use crate::weighted_episodic_mem::{
    age_in_days, calculate_fitness_score, initialize_memory_metadata, update_retrieval_stats,
    TemporalDecayConfig, WeightedMemoryMetadata, DEFAULT_FITNESS_WEIGHTS,
};
use chrono::DateTime;

/// Golden Memory payload for Memory Gate
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GoldenMemoryPayload {
    pub prompt: String,
    pub response: String,
    pub quality_score: u8,
    pub betti_numbers: Vec<u32>,
    pub pad_state: [f64; 7],
    pub entropy: f64,
    pub compass_state: String,
    pub priority: f64,
    pub knot_complexity: f32,
    pub spectral_gap: f32,
    pub persistence_entropy: f64,
    pub timestamp: String,
}

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
    rest_client: Option<reqwest::Client>, // REST API fallback for cloud Qdrant
    rest_url: Option<String>,             // REST API base URL
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
    /// Phase 4.3: GPU fitness calculator (optional)
    pub gpu_fitness_calculator: Option<Arc<crate::gpu_fitness::GPUMemoryFitnessCalculator>>,
    /// Config for ERAG parameters (optional, falls back to defaults if None)
    config: Option<Arc<RwLock<RuntimeConfig>>>,
}

fn is_cloud_qdrant_endpoint(url: &str) -> bool {
    let lower = url.to_ascii_lowercase();
    lower.contains("cloud.qdrant.io") || lower.contains("qdrant.tech")
}

fn extract_host(url: &str) -> Option<String> {
    let without_scheme = if let Some(idx) = url.find("://") {
        &url[idx + 3..]
    } else {
        url
    };
    let host_segment = without_scheme.split('/').next().unwrap_or("");
    if host_segment.is_empty() {
        return None;
    }
    let host = host_segment
        .split('@')
        .last()
        .unwrap_or(host_segment)
        .split(':')
        .next()
        .unwrap_or(host_segment)
        .trim();
    if host.is_empty() {
        None
    } else {
        Some(host.to_string())
    }
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
            None, // No GPU calculator by default
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
        gpu_calculator: Option<Arc<crate::gpu_fitness::GPUMemoryFitnessCalculator>>,
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
            gpu_calculator,
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
        gpu_calculator: Option<Arc<crate::gpu_fitness::GPUMemoryFitnessCalculator>>,
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

        // For Qdrant Cloud we must use TLS gRPC on port 6334 even if the provided URL
        // points at the REST endpoint. The REST URL (6333) is still recorded for health checks
        // or future fallback, but all primary traffic stays on the supported gRPC port.
        let is_cloud = is_cloud_qdrant_endpoint(&normalized_url);
        let mut use_grpc = env_value("QDRANT_USE_GRPC")
            .map(|v| !matches!(v.to_lowercase().as_str(), "false" | "0" | "no"))
            .unwrap_or(true);

        let mut rest_fallback_target: Option<String> = None;

        if is_cloud {
            use_grpc = true; // Always talk to cloud over gRPC/TLS
            if let Some(host) = extract_host(&normalized_url) {
                normalized_url = format!("https://{}:6334", host);
                rest_fallback_target = Some(format!("https://{}:6333", host));
                info!(
                    original = %url,
                    grpc_endpoint = %normalized_url,
                    rest_endpoint = %rest_fallback_target.as_deref().unwrap_or(""),
                    "Detected Qdrant Cloud endpoint; forcing TLS gRPC with REST mirror"
                );
            } else {
                warn!(original = %url, "Unable to parse Qdrant cloud hostname; leaving URL unchanged");
            }
        }

        if use_grpc {
            if !is_cloud {
                if normalized_url.contains(":6333") {
                    normalized_url = normalized_url.replace(":6333", ":6334");
                    info!(original = %url, rewritten = %normalized_url, "Adjusted Qdrant URL to gRPC port 6334");
                } else if !normalized_url.contains(":") {
                    normalized_url = format!("{}:6334", normalized_url.trim_end_matches('/'));
                    info!(original = %url, rewritten = %normalized_url, "Appended gRPC port 6334 to Qdrant URL");
                }
            }
        } else if is_cloud {
            warn!(original = %url, "REST-only mode disabled for Qdrant Cloud; continuing with gRPC");
        } else {
            info!("Using REST API (QDRANT_USE_GRPC=false) - keeping original port");
            rest_fallback_target = Some(url.to_string());
        }

        // Get API key from environment if available (checks .env files via config module)
        let api_key = env_value("QDRANT_API_KEY");

        let mut config = QdrantClientConfig::from_url(&normalized_url);
        if let Some(ref key) = api_key {
            config = config.with_api_key(key.as_str());
            info!("Qdrant client configured with API key");
        } else {
            warn!("QDRANT_API_KEY not set - using unauthenticated connection");
        }

        let client = QdrantClient::new(Some(config))
            .map_err(|err| anyhow!("failed to build qdrant client: {}", err))?;

        // Initialize REST client for cloud Qdrant fallback (uses reqwest like working System2_loop)
        let (rest_client, rest_url) = if let Some(rest_target) = rest_fallback_target.clone() {
            let mut rest_client_builder = reqwest::Client::builder();
            if let Some(ref key) = api_key {
                rest_client_builder = rest_client_builder.default_headers({
                    let mut headers = reqwest::header::HeaderMap::new();
                    headers.insert(
                        reqwest::header::HeaderName::from_static("api-key"),
                        reqwest::header::HeaderValue::from_str(key.as_str())
                            .map_err(|e| anyhow!("invalid QDRANT_API_KEY: {}", e))?,
                    );
                    headers
                });
            }
            let rest_client = rest_client_builder
                .build()
                .map_err(|e| anyhow!("failed to build REST client: {}", e))?;
            info!(rest_endpoint = %rest_target, "REST API client initialized for Qdrant fallback");
            (Some(rest_client), Some(rest_target))
        } else {
            (None, None)
        };

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
            rest_client,
            rest_url,
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
            gpu_fitness_calculator: gpu_calculator, // Phase 4.3: Store GPU calculator
            config: None,
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

    pub fn set_config(&mut self, config: Arc<RwLock<RuntimeConfig>>) {
        self.config = Some(config);
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
        let mut threshold = self.similarity_threshold;

        // Try with initial threshold, fallback to lower threshold if not enough results
        let response_result = self
            .circuit_breaker
            .call(|| {
                let vector_clone = vector.to_vec();
                let collection_clone = collection.clone();
                let threshold_clone = threshold;
                async move {
                    let search_points = SearchPoints {
                        collection_name: collection_clone,
                        vector: vector_clone,
                        limit: (limit * 2) as u64, // Request more to account for deduplication
                        score_threshold: Some(threshold_clone),
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

        // If we got fewer results than requested, try with a lower threshold
        let mut final_result = response_result;
        if let Ok(ref search_result) = final_result {
            if search_result.result.len() < limit && threshold > 0.0 {
                let lower_threshold = (threshold * 0.5).max(0.0);
                info!(
                    initial_results = search_result.result.len(),
                    requested_limit = limit,
                    retrying_with_lower_threshold = lower_threshold,
                    "ERAG retrying search with lower similarity threshold"
                );
                threshold = lower_threshold;
                final_result = self
                    .circuit_breaker
                    .call(|| {
                        let vector_clone = vector.to_vec();
                        let collection_clone = collection.clone();
                        let threshold_clone = threshold;
                        async move {
                            let search_points = SearchPoints {
                                collection_name: collection_clone,
                                vector: vector_clone,
                                limit: (limit * 2) as u64,
                                score_threshold: Some(threshold_clone),
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
            }
        }

        // FIX: Use a priority queue to maintain best diverse memories
        // Continue iterating through oversampled hits to find better diverse candidates
        let mut candidate_memories: Vec<(EragMemory, f64, f64)> = Vec::new(); // (memory, score, diversity_score)
        let mut seen_content_hashes = std::collections::HashSet::<u64>::new();
        let mut memories = Vec::new();
        let mut sims = Vec::new();

        match final_result {
            Ok(search_result) => {
                info!(
                    requested_limit = limit,
                    similarity_threshold = threshold,
                    qdrant_results = search_result.result.len(),
                    "ERAG Qdrant search completed"
                );

                // FIX: Continue iterating through ALL hits, not just until limit
                for (idx, hit) in search_result.result.iter().enumerate() {
                    let payload_json: JsonMap<String, JsonValue> = hit
                        .payload
                        .clone()
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
                    let memory = deserialize_memory(&payload_json);

                    // Deduplicate by content hash (input + output) to detect truly duplicate memories
                    let mut hasher = DefaultHasher::new();
                    memory.input.hash(&mut hasher);
                    memory.output.hash(&mut hasher);
                    memory.erag_context.hash(&mut hasher);
                    let content_hash = hasher.finish();

                    if seen_content_hashes.contains(&content_hash) {
                        debug!(
                            memory_idx = idx,
                            score = hit.score,
                            "ERAG skipping duplicate memory (same content hash)"
                        );
                        continue;
                    }
                    seen_content_hashes.insert(content_hash);

                    // FIX: Compute semantic diversity score using Jaccard and Rouge-L
                    let mut diversity_score = 1.0; // Default: fully diverse
                    if !candidate_memories.is_empty() {
                        let mut min_jaccard: f64 = 1.0;
                        let mut min_rouge: f64 = 1.0;
                        for (existing_mem, _, _) in &candidate_memories {
                            let jaccard =
                                crate::util::jaccard_similarity(&memory.input, &existing_mem.input);
                            let rouge = crate::util::rouge_l(&memory.input, &existing_mem.input);
                            min_jaccard = min_jaccard.min(jaccard as f64);
                            min_rouge = min_rouge.min(rouge as f64);
                        }
                        // Diversity score: lower similarity = higher diversity
                        diversity_score = (1.0 - min_jaccard) * 0.5 + (1.0 - min_rouge) * 0.5;
                    }

                    // Store candidate with combined score (similarity + diversity)
                    let combined_score = hit.score as f64 * 0.7 + diversity_score * 0.3;
                    candidate_memories.push((memory, hit.score as f64, combined_score));
                }

                // FIX: Sort by combined score and take top limit, maintaining diversity
                candidate_memories
                    .sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

                // Select diverse subset: take top candidates ensuring diversity threshold
                let candidate_pool_size = candidate_memories.len();
                let mut selected_memories: Vec<EragMemory> = Vec::new();
                let mut selected_sims = Vec::new();
                let semantic_threshold = 0.85; // Jaccard/Rouge threshold for semantic duplicates

                for (memory, score, _) in candidate_memories.into_iter() {
                    if selected_memories.len() >= limit {
                        break;
                    }

                    // Check semantic similarity against already selected memories
                    let mut is_semantic_duplicate = false;
                    for existing_mem in &selected_memories {
                        let jaccard =
                            crate::util::jaccard_similarity(&memory.input, &existing_mem.input);
                        let rouge = crate::util::rouge_l(&memory.input, &existing_mem.input);
                        if jaccard > semantic_threshold || rouge > semantic_threshold {
                            is_semantic_duplicate = true;
                            debug!(
                                jaccard = jaccard,
                                rouge = rouge,
                                "ERAG skipping semantic duplicate"
                            );
                            break;
                        }
                    }

                    if !is_semantic_duplicate {
                        selected_memories.push(memory);
                        selected_sims.push(score);
                    }
                }

                // FIX: Compute diversity telemetry
                let memory_texts: Vec<String> = selected_memories
                    .iter()
                    .map(|m| format!("{} {}", m.input, m.output))
                    .collect();
                let mut pairwise_jaccards = Vec::new();
                for i in 0..memory_texts.len() {
                    for j in (i + 1)..memory_texts.len() {
                        let jaccard =
                            crate::util::jaccard_similarity(&memory_texts[i], &memory_texts[j]);
                        pairwise_jaccards.push(jaccard);
                    }
                }
                let avg_jaccard = if pairwise_jaccards.is_empty() {
                    0.0
                } else {
                    pairwise_jaccards.iter().sum::<f64>() / pairwise_jaccards.len() as f64
                };
                let diversity_entropy_val = if pairwise_jaccards.is_empty() {
                    0.0
                } else {
                    // Convert to entropy-like metric
                    let probs: Vec<f64> = pairwise_jaccards
                        .iter()
                        .map(|&j| (1.0 - j).max(0.0))
                        .collect();
                    crate::util::shannon_entropy(&probs)
                };

                let unique_count_after_dedupe = seen_content_hashes.len();
                info!(
                    total_memories_retrieved = selected_memories.len(),
                    unique_memories = selected_memories.len(),
                    candidate_pool_size = candidate_pool_size,
                    unique_after_dedupe = unique_count_after_dedupe,
                    duplicates_skipped_during_retrieval = search_result
                        .result
                        .len()
                        .saturating_sub(candidate_pool_size),
                    avg_jaccard_overlap = avg_jaccard,
                    diversity_entropy = diversity_entropy_val,
                    "ERAG memory retrieval complete with diversity filtering"
                );

                // Early warning if duplicates were detected during retrieval
                if unique_count_after_dedupe < candidate_pool_size {
                    warn!(
                        unique_memories = unique_count_after_dedupe,
                        candidate_memories = candidate_pool_size,
                        "ERAG detected duplicate memories during retrieval - dedupe hash executed but some duplicates may have slipped through"
                    );
                }

                memories = selected_memories;
                sims = selected_sims;
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

        // Compute trust metrics from trajectory analysis if we have enough memories
        if memories.len() >= 10 {
            use crate::behavior_trajectory::{TrajectoryAnalyzer, TrustMetrics};
            use crate::topology_memory::TopologyMemoryAnalyzer;

            let trajectory_analyzer = TrajectoryAnalyzer::default();
            let trajectory = trajectory_analyzer.collect_trajectory(&memories);

            if trajectory.len() >= 10 {
                let topology_analyzer = TopologyMemoryAnalyzer::default();
                let point_cloud = trajectory_analyzer.to_point_cloud(&trajectory);

                // Compute H1 and H2 persistence
                if let (Ok(h1_barcodes), Ok(h2_barcodes)) = (
                    topology_analyzer.compute_h1_persistence(&point_cloud),
                    topology_analyzer.compute_h2_persistence(&point_cloud),
                ) {
                    let trust_metrics = trajectory_analyzer.compute_trust_metrics(
                        &trajectory,
                        &h1_barcodes,
                        &h2_barcodes,
                    );

                    // Update memory metadata with trust metrics
                    for memory in &mut memories {
                        if let Some(ref mut metadata) = memory.weighted_metadata {
                            metadata.h1_trust_score = trust_metrics.h1_trust_score;
                            metadata.h2_anomaly_score = trust_metrics.h2_anomaly_score;
                            metadata.persistence_entropy = trust_metrics.persistence_entropy;

                            // Update beta_1_connectivity with H1 trust score
                            metadata.beta_1_connectivity = trust_metrics.h1_trust_score;
                        }
                    }
                }
            }
        }

        // If we have a preferred cascade stage, boost scores for matching memories
        if let Some(preferred) = preferred_stage {
            for (mem, sim) in memories.iter_mut().zip(sims.iter_mut()) {
                if let Some(stage) = mem.cascade_stage {
                    if stage == preferred {
                        // Boost similarity score by configurable multiplier for cascade-aligned memories
                        let boost_multiplier = self
                            .config
                            .as_ref()
                            .map(|c| c.read().erag_similarity_boost_multiplier)
                            .unwrap_or(1.2);
                        let boost_max = self
                            .config
                            .as_ref()
                            .map(|c| c.read().erag_similarity_boost_max)
                            .unwrap_or(1.0);
                        *sim = (*sim * boost_multiplier).min(boost_max);
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
                        sims.get(0).copied().unwrap_or(0.0) as f32
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
            sims = memory_fitness_pairs
                .iter()
                .map(|(_, f)| *f as f64)
                .collect();
        }

        let average_similarity = if sims.is_empty() {
            0.0
        } else {
            sims.iter().copied().sum::<f64>() as f32 / sims.len() as f32
        };

        // Log final memory count before returning
        info!(
            final_memory_count = memories.len(),
            requested_limit = limit,
            average_similarity = average_similarity,
            "ERAG collapse complete"
        );

        // Final dedupe check: verify no duplicates were introduced after initial dedupe
        let mut final_content_hashes = std::collections::HashSet::new();
        let mut duplicate_indices = Vec::new();
        for (idx, memory) in memories.iter().enumerate() {
            let mut hasher = DefaultHasher::new();
            memory.input.hash(&mut hasher);
            memory.output.hash(&mut hasher);
            memory.erag_context.hash(&mut hasher);
            let content_hash = hasher.finish();

            if final_content_hashes.contains(&content_hash) {
                duplicate_indices.push(idx);
                debug!(
                    memory_idx = idx,
                    content_hash = content_hash,
                    input_preview = memory.input.chars().take(30).collect::<String>(),
                    "ERAG final check: duplicate memory detected"
                );
            } else {
                final_content_hashes.insert(content_hash);
            }
        }

        if !duplicate_indices.is_empty() {
            warn!(
                unique_memories = final_content_hashes.len(),
                total_memories = memories.len(),
                duplicate_count = duplicate_indices.len(),
                duplicate_indices = ?duplicate_indices,
                "ERAG final check: detected duplicate memories after processing - dedupe hash may not have executed correctly in all code paths"
            );
        }

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

    /// Count memories with specific Betti number signature (for novelty detection)
    pub async fn count_memories_with_betti_signature(
        &self,
        betti_b0: i32,
        betti_b1: i32,
    ) -> Result<u64> {
        // Search Golden_Memory collection for matching Betti signatures
        let search_request = SearchPoints {
            collection_name: "Golden_Memory".to_string(),
            vector: vec![0.0; self.vector_dim],
            limit: 100,
            score_threshold: Some(0.0),
            with_payload: Some(true.into()),
            ..Default::default()
        };

        match self.client.search_points(&search_request).await {
            Ok(result) => {
                let mut count = 0u64;
                for point in result.result {
                    if let Some(betti_array) = point.payload.get("betti_numbers") {
                        if let Some(qdrant_client::qdrant::value::Kind::ListValue(list)) =
                            &betti_array.kind
                        {
                            if list.values.len() >= 2 {
                                let b0_match = list.values[0]
                                    .kind
                                    .as_ref()
                                    .and_then(|k| {
                                        if let qdrant_client::qdrant::value::Kind::IntegerValue(v) =
                                            k
                                        {
                                            Some(*v)
                                        } else {
                                            None
                                        }
                                    })
                                    .map(|v| v == betti_b0 as i64)
                                    .unwrap_or(false);
                                let b1_match = list.values[1]
                                    .kind
                                    .as_ref()
                                    .and_then(|k| {
                                        if let qdrant_client::qdrant::value::Kind::IntegerValue(v) =
                                            k
                                        {
                                            Some(*v)
                                        } else {
                                            None
                                        }
                                    })
                                    .map(|v| v == betti_b1 as i64)
                                    .unwrap_or(false);
                                if b0_match && b1_match {
                                    count += 1;
                                }
                            }
                        }
                    }
                }
                Ok(count)
            }
            Err(_) => Ok(0), // Collection doesn't exist yet, so novelty = true
        }
    }

    /// Initialize Golden_Memory collection with 2560D vectors
    pub async fn ensure_golden_memory_collection(&self) -> Result<()> {
        let collection_name = "Golden_Memory";

        let collections = self.client.list_collections().await?;
        let exists = collections
            .collections
            .iter()
            .any(|c| c.name == collection_name);

        if exists {
            info!("Golden_Memory collection already exists");
            return Ok(());
        }

        let collection_config = qdrant_client::qdrant::CreateCollection {
            collection_name: collection_name.to_string(),
            vectors_config: Some(VectorsConfig {
                config: Some(qdrant::vectors_config::Config::Params(VectorParams {
                    size: self.vector_dim as u64,
                    distance: Distance::Cosine.into(),
                    ..Default::default()
                })),
            }),
            ..Default::default()
        };

        self.client.create_collection(&collection_config).await?;
        info!(
            "✅ Golden_Memory collection initialized with {}D vectors",
            self.vector_dim
        );
        Ok(())
    }

    /// Upsert memory to Golden_Memory collection
    pub async fn upsert_golden_memory(
        &self,
        embedding: &[f32],
        payload: &GoldenMemoryPayload,
    ) -> Result<()> {
        use qdrant_client::qdrant::PointStruct;
        use uuid::Uuid;

        self.ensure_golden_memory_collection().await?;

        let mut qdrant_payload = Payload::new();
        qdrant_payload.insert("prompt", payload.prompt.clone());
        qdrant_payload.insert("response", payload.response.clone());
        qdrant_payload.insert("quality_score", payload.quality_score as i64);
        qdrant_payload.insert(
            "betti_numbers",
            payload
                .betti_numbers
                .iter()
                .map(|&x| x as i64)
                .collect::<Vec<_>>(),
        );
        qdrant_payload.insert("pad_state", payload.pad_state.to_vec());
        qdrant_payload.insert("entropy", payload.entropy);
        qdrant_payload.insert("compass_state", payload.compass_state.clone());
        qdrant_payload.insert("priority", payload.priority);
        qdrant_payload.insert("knot_complexity", payload.knot_complexity as f64);
        qdrant_payload.insert("spectral_gap", payload.spectral_gap as f64);
        qdrant_payload.insert("persistence_entropy", payload.persistence_entropy);
        qdrant_payload.insert("timestamp", payload.timestamp.clone());

        let point = PointStruct::new(
            Uuid::new_v4().to_string(),
            embedding.to_vec(),
            qdrant_payload,
        );

        self.circuit_breaker
            .call(|| async {
                self.client
                    .upsert_points("Golden_Memory", None, vec![point.clone()], None)
                    .await
            })
            .await?;

        info!(
            "✨ Golden Memory stored: quality={}/10, priority={:.1}",
            payload.quality_score, payload.priority
        );
        Ok(())
    }

    /// Search Golden_Memory collection for high-quality retrieval
    pub async fn search_golden_memory(
        &self,
        query_embedding: &[f32],
        limit: usize,
        min_priority: f32,
    ) -> Result<Vec<EragMemory>> {
        let search_request = SearchPoints {
            collection_name: "Golden_Memory".to_string(),
            vector: query_embedding.to_vec(),
            limit: limit as u64,
            score_threshold: Some(0.3),
            with_payload: Some(true.into()),
            ..Default::default()
        };

        let search_result = self.client.search_points(&search_request).await?;

        let mut memories = Vec::new();
        for point in search_result.result {
            let prompt = point
                .payload
                .get("prompt")
                .and_then(value_as_string)
                .unwrap_or_default();
            let response = point
                .payload
                .get("response")
                .and_then(value_as_string)
                .unwrap_or_default();
            let entropy = point
                .payload
                .get("entropy")
                .and_then(value_as_f64)
                .unwrap_or(0.0);
            let timestamp = point
                .payload
                .get("timestamp")
                .and_then(value_as_string)
                .unwrap_or_default();
            let compass_state = point.payload.get("compass_state").and_then(value_as_string);

            // Extract PAD state for emotional vector
            let pad_array = if let Some(pad_val) = point.payload.get("pad_state") {
                if let Some(qdrant_client::qdrant::value::Kind::ListValue(list)) = &pad_val.kind {
                    let mut pad = [0.0f64; 7];
                    for (i, val) in list.values.iter().take(7).enumerate() {
                        pad[i] = value_as_f64(val).unwrap_or(0.0);
                    }
                    pad
                } else {
                    [0.0; 7]
                }
            } else {
                [0.0; 7]
            };

            let pad_state = PadGhostState {
                pad: pad_array,
                entropy,
                mu: [0.0; 7],
                sigma: [0.0; 7],
            };

            let memory = EragMemory {
                input: prompt,
                output: response,
                emotional_vector: EmotionalVector::from_pad(&pad_state),
                erag_context: vec!["Golden Memory".to_string()],
                entropy_before: 0.0,
                entropy_after: entropy,
                timestamp,
                compass_state,
                cascade_stage: None,
                weighted_metadata: None,
            };
            memories.push(memory);
        }

        if !memories.is_empty() {
            info!("🌟 Retrieved {} Golden Memories", memories.len());
        }

        Ok(memories)
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
    let collections = match client.list_collections().await {
        Ok(c) => c,
        Err(e) => {
            // gRPC failed - try REST API fallback
            warn!(
                "gRPC list_collections failed ({}), assuming collection exists",
                e
            );
            // Don't fail - just assume collection exists and continue
            return Ok(());
        }
    };

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
                .map(|n| n as u32),
            h1_trust_score: extract_number(payload, "h1_trust_score") as f32,
            h2_anomaly_score: extract_number(payload, "h2_anomaly_score") as f32,
            persistence_entropy: extract_number(payload, "persistence_entropy"),
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

    /// Query tough knots - memories with high knot complexity or low curator quality
    /// Used for anti-forgetting training to focus on difficult cases
    ///
    /// # Parameters
    /// - `limit`: Maximum number of tough memories to return
    /// - `multiplier`: Fetch multiplier (fetch limit * multiplier samples to filter from)
    /// - `max_fetch`: Maximum fetch size cap (prevents excessive queries)
    /// - `knot_threshold`: Knot complexity threshold (memories with complexity > threshold are considered tough)
    /// - `quality_threshold`: Curator quality threshold (memories with quality < threshold are considered tough)
    /// - `knot_multiplier`: Multiplier for knot complexity in toughness score calculation
    pub async fn query_tough_knots(
        &self,
        limit: usize,
        multiplier: usize,
        max_fetch: usize,
        knot_threshold: f64,
        quality_threshold: f64,
        knot_multiplier: f64,
    ) -> Result<Vec<EragMemory>> {
        // Validate parameters
        if limit == 0 {
            return Ok(Vec::new());
        }
        if multiplier == 0 {
            return Err(anyhow!("query_tough_knots: multiplier must be > 0"));
        }
        if max_fetch == 0 {
            return Err(anyhow!("query_tough_knots: max_fetch must be > 0"));
        }
        if !knot_threshold.is_finite() || knot_threshold < 0.0 {
            return Err(anyhow!(
                "query_tough_knots: knot_threshold must be finite and >= 0.0"
            ));
        }
        if !quality_threshold.is_finite() || quality_threshold < 0.0 || quality_threshold > 1.0 {
            return Err(anyhow!(
                "query_tough_knots: quality_threshold must be finite and in [0.0, 1.0]"
            ));
        }
        if !knot_multiplier.is_finite() || knot_multiplier < 0.0 {
            return Err(anyhow!(
                "query_tough_knots: knot_multiplier must be finite and >= 0.0"
            ));
        }

        // Fetch a larger sample to filter from
        let max_fetch_calc = (limit.max(1) * multiplier).min(max_fetch);
        let search_points = self
            .search_points_with_vector(vec![0.0; self.vector_dim], max_fetch_calc, None, true)
            .await?;

        let mut tough_memories: Vec<(EragMemory, f64)> = search_points
            .into_iter()
            .filter_map(|point| {
                // Convert payload to JsonMap for deserialization
                let payload_json: JsonMap<String, JsonValue> = point
                    .payload
                    .iter()
                    .map(|(k, v)| {
                        let json_val = match v.kind.as_ref() {
                            Some(QdrantValueKind::BoolValue(b)) => JsonValue::Bool(*b),
                            Some(QdrantValueKind::IntegerValue(i)) => {
                                JsonValue::Number(serde_json::Number::from(*i))
                            }
                            Some(QdrantValueKind::DoubleValue(d)) => JsonValue::Number(
                                serde_json::Number::from_f64(*d)
                                    .unwrap_or(serde_json::Number::from(0)),
                            ),
                            Some(QdrantValueKind::StringValue(s)) => JsonValue::String(s.clone()),
                            _ => JsonValue::Null,
                        };
                        (k.clone(), json_val)
                    })
                    .collect();

                let memory = deserialize_memory(&payload_json);

                // Extract knot complexity and curator quality from payload
                let knot_complexity = point
                    .payload
                    .get("knot_complexity")
                    .and_then(value_as_f64)
                    .unwrap_or(0.0);

                let curator_quality = point
                    .payload
                    .get("curator_quality")
                    .and_then(value_as_f64)
                    .unwrap_or_else(|| {
                        // Fallback: use fitness score if available (lower = tougher)
                        if let Some(ref meta) = memory.weighted_metadata {
                            meta.fitness_score as f64
                        } else {
                            1.0 // Default to assuming good quality if unknown
                        }
                    });

                // Calculate toughness score: high knot complexity OR low curator quality
                let toughness_score =
                    if knot_complexity > knot_threshold || curator_quality < quality_threshold {
                        // Prioritize memories with both high knot complexity AND low quality
                        knot_complexity * knot_multiplier + (1.0 - curator_quality)
                    } else {
                        0.0
                    };

                if toughness_score > 0.0 {
                    Some((memory, toughness_score))
                } else {
                    None
                }
            })
            .collect();

        // Sort by toughness score (highest first)
        tough_memories
            .sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        // Return top limit memories
        let result: Vec<EragMemory> = tough_memories
            .into_iter()
            .take(limit)
            .map(|(memory, _)| memory)
            .collect();

        Ok(result)
    }

    /// Store failure case
    /// Stores failure metadata to Qdrant for analysis and anti-forgetting training
    /// Uses zero vector for embedding since failures don't have proper context/embedding
    pub async fn store_failure(
        &self,
        input: &str,
        output: &str,
        details: Option<String>,
        failure_type: &str,
        retry_count: u32,
    ) -> Result<()> {
        // Create a zero vector for failures (they don't have proper embeddings)
        // This ensures failures don't interfere with similarity search
        let failure_vector = vec![0.0f32; self.vector_dim];

        // Create minimal PAD state for failure (neutral state)
        let pad_state = PadGhostState {
            pad: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            mu: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            sigma: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            entropy: 0.0,
        };

        // Create minimal compass state for failure
        let compass = CompassOutcome {
            quadrant: crate::compass::CompassQuadrant::Panic, // Failures map to Panic quadrant
            intrinsic_reward: 0.0,
            is_threat: true,
            is_healing: false,
            mcts_branches: vec![],
            cascade_stage: None,
            ucb1_score: None,
        };

        // Create failure memory with metadata
        let failure_context = if let Some(details) = details {
            vec![
                format!("Failure type: {}", failure_type),
                format!("Retry count: {}", retry_count),
                details,
            ]
        } else {
            vec![
                format!("Failure type: {}", failure_type),
                format!("Retry count: {}", retry_count),
            ]
        };

        let memory = EragMemory {
            input: input.to_string(),
            output: output.to_string(),
            emotional_vector: EmotionalVector::from_pad(&pad_state),
            erag_context: failure_context,
            entropy_before: 0.0,
            entropy_after: 0.0,
            timestamp: Utc::now().to_rfc3339(),
            compass_state: Some("Failure".to_string()),
            cascade_stage: None,
            weighted_metadata: Some(initialize_memory_metadata(&pad_state, 0.0)),
        };

        let mut payload = encode_payload(&memory);
        // Add failure-specific metadata
        payload.insert(
            "failure_type".to_string(),
            JsonValue::String(failure_type.to_string()),
        );
        payload.insert(
            "retry_count".to_string(),
            JsonValue::from(retry_count as u64),
        );
        payload.insert("is_failure".to_string(), JsonValue::Bool(true));

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
                    let strs: Vec<String> = arr
                        .iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect();
                    if !strs.is_empty() {
                        qdrant_payload.insert(k, strs);
                    } else {
                        qdrant_payload
                            .insert(k, arr.iter().map(|v| v.to_string()).collect::<Vec<_>>());
                    }
                }
                _ => qdrant_payload.insert(k, v.to_string()),
            }
        }

        let point = PointStruct::new(
            uuid::Uuid::new_v4().to_string(),
            failure_vector,
            qdrant_payload,
        );

        // Store failure immediately (don't batch failures)
        match self
            .circuit_breaker
            .call(|| async {
                self.client
                    .upsert_points(self.collection.clone(), None, vec![point], None)
                    .await
                    .map_err(|e| anyhow!("failed to store failure case: {}", e))
            })
            .await
        {
            Ok(_) => {
                info!(
                    failure_type = %failure_type,
                    retry_count = retry_count,
                    "Stored failure case to Qdrant"
                );
                Ok(())
            }
            Err(e) => {
                warn!(
                    error = %e,
                    failure_type = %failure_type,
                    "Failed to store failure case to Qdrant"
                );
                Err(e)
            }
        }
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

        if points_count < self.batch_size as u64 {
            debug!(
                collection = %self.collection,
                points = points_count,
                batch_size = self.batch_size,
                "Skipping index health rebuild until collection exceeds a full batch"
            );
            return Ok(true);
        }

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

    /// Phase 4.3: Refresh weighted memory state to keep metrics healthy
    pub async fn refresh_weighted_memory(&self) -> Result<()> {
        let healthy = self.ensure_index_health().await?;
        if healthy {
            info!(collection = %self.collection, "Weighted memory index is healthy");
        } else {
            info!(collection = %self.collection, "Weighted memory index was rebuilt during refresh");
        }
        Ok(())
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
            Some(metadata.persistence_entropy),
            Some(metadata.h1_trust_score),
            Some(metadata.h2_anomaly_score),
        )
    }

    /// Batch calculate fitness scores for multiple memories
    /// Phase 4.3: Uses GPU acceleration if available and enabled
    pub fn batch_calculate_fitness(
        &self,
        memories: &[EragMemory],
        pad_state: &PadGhostState,
    ) -> Vec<f32> {
        // Phase 4.3: Use GPU fitness calculator if available
        if let Some(ref gpu_calc) = self.gpu_fitness_calculator {
            self.batch_calculate_fitness_gpu(memories, pad_state, gpu_calc)
        } else {
            // Fallback to CPU-based calculation
            memories
                .iter()
                .map(|mem| self.calculate_memory_fitness(mem, pad_state))
                .collect()
        }
    }

    /// Phase 4.3: GPU-accelerated batch fitness calculation
    fn batch_calculate_fitness_gpu(
        &self,
        memories: &[EragMemory],
        pad_state: &PadGhostState,
        gpu_calculator: &crate::gpu_fitness::GPUMemoryFitnessCalculator,
    ) -> Vec<f32> {
        // Extract fitness components from memories
        let pad_states: Vec<_> = memories
            .iter()
            .map(|mem| {
                // Extract pad_state from memory (would need to store in EragMemory)
                // For now, use the provided pad_state for all memories
                pad_state.clone()
            })
            .collect();

        let ages: Vec<f64> = memories
            .iter()
            .map(|mem| {
                let timestamp = DateTime::parse_from_rfc3339(&mem.timestamp)
                    .map(|dt| dt.with_timezone(&Utc))
                    .unwrap_or_else(|_| Utc::now());
                age_in_days(&timestamp)
            })
            .collect();

        let retrieval_counts: Vec<u32> = memories
            .iter()
            .map(|mem| {
                mem.weighted_metadata
                    .as_ref()
                    .map(|m| m.retrieval_count)
                    .unwrap_or(0)
            })
            .collect();

        let beta1_scores: Vec<f32> = memories
            .iter()
            .map(|mem| {
                mem.weighted_metadata
                    .as_ref()
                    .map(|m| m.beta_1_connectivity)
                    .unwrap_or(0.0)
            })
            .collect();

        let consonance_scores: Vec<f32> = memories
            .iter()
            .map(|mem| {
                mem.weighted_metadata
                    .as_ref()
                    .map(|m| m.consonance_score)
                    .unwrap_or(0.0)
            })
            .collect();

        let consolidation_levels: Vec<f32> = memories
            .iter()
            .map(|mem| {
                mem.weighted_metadata
                    .as_ref()
                    .map(|m| m.consolidation_level)
                    .unwrap_or(0.0)
            })
            .collect();

        // Use GPU calculator for batch processing
        gpu_calculator.batch_fitness_from_arrays(
            &pad_states,
            &ages,
            &retrieval_counts,
            &beta1_scores,
            &consonance_scores,
            &consolidation_levels,
            &self.fitness_weights,
            &self.temporal_config,
        )
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
            rest_client: self.rest_client.clone(),
            rest_url: self.rest_url.clone(),
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
            gpu_fitness_calculator: self.gpu_fitness_calculator.clone(), // Phase 4.3: Clone GPU calculator
            config: self.config.clone(),
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
