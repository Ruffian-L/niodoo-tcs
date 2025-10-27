use anyhow::{anyhow, Result};
use chrono::Utc;
use rand::{thread_rng, Rng};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map as JsonMap, Value as JsonValue};
use std::sync::Arc;
use std::time::Duration;
use tracing::{info, instrument, warn};

use crate::compass::CompassOutcome;
use crate::embedding::QwenStatefulEmbedder;
use crate::learning::{DqnAction, DqnState, ReplayTuple};
use crate::metrics::PipelineMetrics;
use crate::torus::PadGhostState;
use qdrant_client::Qdrant;
use uuid::Uuid;

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
    pub quality_score: Option<f32>,
    pub topology_betti: Option<[usize; 3]>,
    pub topology_knot_complexity: Option<f32>,
    // Continual learning fields
    pub solution_path: Option<String>,
    pub conversation_history: Vec<String>,
    pub iteration_count: u32,
}

#[derive(Clone)]
pub struct EragClient {
    client: Client,
    base_url: String,
    collection: String,
    vector_dim: usize,
    pub similarity_threshold: f32,
    embedder: Arc<QwenStatefulEmbedder>,
}

#[derive(Debug, Clone)]
pub struct CollapseResult {
    pub top_hits: Vec<EragMemory>,
    pub aggregated_context: String,
    pub average_similarity: f32,
    pub curator_quality: Option<f32>,
    pub failure_type: Option<String>,
    pub failure_details: Option<String>,
}

impl EragClient {
    pub async fn new(
        url: &str,
        collection: &str,
        vector_dim: usize,
        similarity_threshold: f32,
        embedder: Arc<QwenStatefulEmbedder>,
    ) -> Result<Self> {
        // Priority: env var > config > default
        let base_url = std::env::var("QDRANT_URL")
            .unwrap_or_else(|_| url.to_string())
            .trim_end_matches('/')
            .to_string();
        let client = Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .map_err(|err| anyhow!("failed to build qdrant http client: {err}"))?;

        let _qdrant = Qdrant::from_url(&base_url); // Unused, for init if needed

        // Create collections via HTTP
        let create_body = json!({
            "vectors": {
                "size": vector_dim,
                "distance": "Cosine"
            }
        });

        // Experiences
        let create_url = format!("{}/collections/{}", base_url, collection);
        let _ = client.put(&create_url).json(&create_body).send().await?;

        // Failures
        let failures_url = format!("{}/collections/failures", base_url);
        let _ = client.put(&failures_url).json(&create_body).send().await?;

        Ok(Self {
            client,
            base_url: base_url.clone(),
            collection: collection.to_string(),
            vector_dim,
            similarity_threshold,
            embedder,
        })
    }

    #[instrument(skip_all, fields(dim = vector.len()))]
    pub async fn collapse(&self, vector: &[f32]) -> Result<CollapseResult> {
        self.collapse_with_limit(vector, 3).await
    }

    pub async fn collapse_with_limit(&self, vector: &[f32], limit: usize) -> Result<CollapseResult> {
        anyhow::ensure!(
            vector.len() == self.vector_dim,
            "embedding dimension mismatch: expected {}, got {}",
            self.vector_dim,
            vector.len()
        );

        // Build search request manually
        let request_json = json!({
            "vector": vector.to_vec(),
            "limit": limit.max(1).min(50),
            "score_threshold": self.similarity_threshold,
            "with_payload": true,
            "with_vectors": false
        });

        let request_dump = request_json.to_string();

        let url = format!(
            "{}/collections/{}/points/search",
            self.base_url, self.collection
        );
        let response = self.client.post(url).json(&request_json).send().await;
        let mut memories = Vec::new();
        let mut sims = Vec::new();
        match response {
            Ok(resp) => {
                if resp.status().is_success() {
                    match resp.json::<SearchResponse>().await {
                        Ok(parsed) => {
                            for hit in parsed.result {
                                if hit.payload.is_empty() {
                                    continue;
                                }
                                let memory = deserialize_memory(&hit.payload);
                                sims.push(hit.score);
                                memories.push(memory);
                            }
                        }
                        Err(err) => {
                            info!(%err, "failed to decode qdrant search response");
                        }
                    }
                } else {
                    let status = resp.status();
                    let body = resp
                        .text()
                        .await
                        .unwrap_or_else(|_| "<no body>".to_string());
                    warn!(
                        %status,
                        body = %body,
                        request = %request_dump,
                        "qdrant search returned error status"
                    );
                }
            }
            Err(err) => {
                warn!(
                    %err,
                    request = %request_dump,
                    "qdrant search request errored"
                );
            }
        }

        if memories.is_empty() {
            sims.push(0.0);
        }

        // Sort memories by quality score if available (quality-weighted retrieval)
        memories.sort_by(|a, b| {
            let quality_a = a.quality_score.unwrap_or(0.5);
            let quality_b = b.quality_score.unwrap_or(0.5);
            quality_b
                .partial_cmp(&quality_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let average_similarity = if sims.is_empty() {
            0.0
        } else {
            sims.iter().copied().sum::<f32>() / sims.len() as f32
        };

        // Collect context strings and join
        let mut aggregated_context: String = memories
            .iter()
            .flat_map(|m| m.erag_context.iter())
            .cloned()
            .collect::<Vec<_>>()
            .join("\n");

        // Check for higher quality previous solutions
        let better_solution = memories
            .iter()
            .filter(|m| m.quality_score.unwrap_or(0.0) > 0.8)
            .find(|m| m.solution_path.is_some());

        if let Some(better) = better_solution {
            aggregated_context.push_str(&format!(
                "\n[Previous optimal solution (quality {:.2}): {}]",
                better.quality_score.unwrap_or(0.0),
                better.solution_path.as_ref().unwrap_or(&"N/A".to_string())
            ));

            if better.iteration_count > 0 {
                aggregated_context.push_str(&format!(
                    "\n[Note: This problem was solved optimally in {} iterations previously]",
                    better.iteration_count
                ));
            }
        }

        if aggregated_context.is_empty() {
            aggregated_context = "No relevant ERAG context retrieved.".to_string();
        } else if aggregated_context.len() > 1000 {
            aggregated_context.truncate(1000);
        }

        let curator_quality = memories
            .iter()
            .filter_map(|m| m.quality_score)
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        Ok(CollapseResult {
            top_hits: memories,
            aggregated_context,
            average_similarity,
            curator_quality,
            failure_type: None,
            failure_details: None,
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
        quality_score: Option<f32>,
        topology: Option<&crate::tcs_analysis::TopologicalSignature>,
        solution_path: Option<String>,
        iteration_count: u32,
    ) -> Result<()> {
        let memory = EragMemory {
            input: prompt.to_string(),
            output: response.to_string(),
            emotional_vector: EmotionalVector::from_pad(pad_state),
            erag_context: context.to_vec(),
            entropy_before,
            entropy_after: pad_state.entropy,
            timestamp: Utc::now().to_rfc3339(),
            compass_state: Some(format!("{:?}", compass.quadrant)),
            quality_score,
            topology_betti: topology.map(|t| t.betti_numbers),
            topology_knot_complexity: topology.map(|t| t.knot_complexity as f32),
            solution_path,
            conversation_history: vec![prompt.to_string(), response.to_string()],
            iteration_count,
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

    pub async fn store_failure(
        &self,
        prompt: &str,
        output: &str,
        _metrics: &PipelineMetrics,
        reflection: Option<String>,
        failure_type: &str,
        retry_count: u32,
    ) -> Result<()> {
        let payload = json!({
            "type": "failure_episode",
            "prompt": prompt,
            "output": output,
            "metrics": {
                "rouge": 0.0,
                "entropy_delta": 0.0,
                "ucb1": 0.0,
                "curator": 0.0,
                "latency_ms": 0.0,
            },
            "reflection": reflection,
            "failure_type": failure_type,
            "retry_count": retry_count,
        });

        let embedding = self.embedder.embed(prompt).await?;
        let point = json!({
            "id": uuid::Uuid::new_v4().to_string(),
            "vector": embedding,
            "payload": payload
        });
        let url = format!("{}/collections/failures/points", self.base_url);
        let resp = self
            .client
            .put(&url)
            .json(&json!({"points": [point]}))
            .send()
            .await?;
        if resp.status().is_success() {
            Ok(())
        } else {
            Err(anyhow!(
                "Failed to store failure episode: {}",
                resp.status()
            ))
        }
    }

    pub async fn search(
        &self,
        query: &str,
        k: usize,
        filter: Option<JsonValue>,
    ) -> Result<Vec<SearchHit>> {
        let embedding = self.embedder.embed(query).await?;

        // Build search request manually
        let mut request_json = json!({
            "vector": embedding,
            "limit": k,
            "with_payload": true,
            "with_vectors": false
        });

        if let Some(f) = filter {
            request_json["filter"] = f;
        }

        let resp = self
            .client
            .post(&format!(
                "{}/collections/{}/points/search",
                self.base_url, self.collection
            ))
            .json(&request_json)
            .send()
            .await?;

        #[derive(Deserialize)]
        struct SearchResponse {
            result: Vec<SearchHit>,
        }

        let search_resp: SearchResponse = resp.json().await?;
        Ok(search_resp.result)
    }

    pub async fn store_replay_tuple(&self, tuple: &ReplayTuple) -> Result<()> {
        let content = format!(
            "DQN: state={:?} action={} reward={} next={:?}",
            tuple.state, tuple.action.param, tuple.reward, tuple.next_state
        );
        let embedding = self.embedder.embed(&content).await?;
        let payload = json!({
            "type": "dqn_tuple",
            "tuple": {
                "state": tuple.state.metrics,
                "action_param": tuple.action.param,
                "action_delta": tuple.action.delta,
                "reward": tuple.reward,
                "next_state": tuple.next_state.metrics,
            }
        });
        let point = json!({
            "id": Uuid::new_v4().to_string(),
            "vector": embedding,
            "payload": payload
        });
        let url = format!("{}/collections/{}/points", self.base_url, self.collection);
        let resp = self
            .client
            .put(&url)
            .json(&json!({"points": [point]}))
            .send()
            .await?;
        if resp.status().is_success() {
            Ok(())
        } else {
            Err(anyhow!("Failed to store replay tuple: {}", resp.status()))
        }
    }

    pub async fn query_replay_batch(
        &self,
        query: &str,
        _query_metrics: &[f64],
        k: usize,
    ) -> Result<Vec<ReplayTuple>> {
        let hits = self.search(query, k, None).await?;
        let mut tuples = Vec::new();
        for hit in hits {
            let payload = &hit.payload;

            if let Some(tp) = payload.get("tuple").and_then(|t| t.as_object()) {
                let state = tp["state"]
                    .as_array()
                    .map(|arr| arr.iter().map(|v| v.as_f64().unwrap_or(0.0)).collect())
                    .unwrap_or_default();
                let action_param = tp["action_param"].as_str().unwrap_or("").to_string();
                let action_delta = tp["action_delta"].as_f64().unwrap_or(0.0);
                let reward = tp["reward"].as_f64().unwrap_or(0.0);
                let next_state = tp["next_state"]
                    .as_array()
                    .map(|arr| arr.iter().map(|v| v.as_f64().unwrap_or(0.0)).collect())
                    .unwrap_or_default();
                tuples.push(ReplayTuple {
                    state: DqnState { metrics: state },
                    action: DqnAction {
                        param: action_param,
                        delta: action_delta,
                    },
                    reward,
                    next_state: DqnState {
                        metrics: next_state,
                    },
                });
            }
        }
        Ok(tuples)
    }

    pub async fn query_low_reward_tuples(
        &self,
        min_reward: f64,
        k: usize,
    ) -> Result<Vec<DqnTuple>> {
        // Use HTTP API directly since Filter API has changed
        let filter_json = json!({
            "must": [
                {
                    "key": "type",
                    "match": {"value": "dqn_tuple"}
                },
                {
                    "key": "tuple.reward",
                    "range": {"lt": min_reward}
                }
            ]
        });

        let request_json = json!({
            "filter": filter_json,
            "limit": k,
            "with_payload": true,
            "with_vectors": false
        });

        let url = format!(
            "{}/collections/{}/points/scroll",
            self.base_url, self.collection
        );
        let resp = self.client.post(&url).json(&request_json).send().await?;

        #[derive(Deserialize)]
        struct ScrollResponse {
            result: Vec<ScrollPoint>,
        }

        let scroll_resp: ScrollResponse = resp.json().await?;
        let mut tuples = Vec::new();
        for point in scroll_resp.result {
            let payload = point.payload.unwrap_or_default();
            if let Some(tp) = payload.get("tuple").and_then(|t| t.as_object()) {
                let state = tp["state"]
                    .as_array()
                    .map(|arr| arr.iter().map(|v| v.as_f64().unwrap_or(0.0)).collect())
                    .unwrap_or_default();
                let action_param = tp["action_param"].as_str().unwrap_or("").to_string();
                let action_delta = tp["action_delta"].as_f64().unwrap_or(0.0);
                let reward = tp["reward"].as_f64().unwrap_or(0.0);
                let next_state = tp["next_state"]
                    .as_array()
                    .map(|arr| arr.iter().map(|v| v.as_f64().unwrap_or(0.0)).collect())
                    .unwrap_or_default();
                tuples.push(DqnTuple {
                    state,
                    action_param,
                    action_delta,
                    reward,
                    next_state,
                });
            }
        }
        // Sort by reward asc for low rewards
        tuples.sort_by(|a, b| {
            a.reward
                .partial_cmp(&b.reward)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(tuples)
    }

    pub async fn query_old_dqn_tuples(&self, batch_id: u32, num: usize) -> Result<Vec<DqnTuple>> {
        // Query older DQN tuples for experience replay (anti-forgetting)
        // Use batch_id as a seed for deterministic sampling
        let offset = (batch_id as u64 * 100) % 1000;
        
        // Use HTTP API for scrolling through tuples
        let request_json = json!({
            "limit": num,
            "offset": offset.to_string(),
            "with_payload": true,
            "with_vectors": false
        });
        
        let url = format!("{}/collections/{}/points/scroll", self.base_url, self.collection);
        let resp = self.client.post(&url).json(&request_json).send().await?;
        
        #[derive(Deserialize)]
        struct ScrollResponse {
            result: Vec<ScrollPoint>,
        }
        
        let scroll_resp: ScrollResponse = resp.json().await?;
        let mut tuples = Vec::new();
        
        for point in scroll_resp.result {
            let payload = point.payload.unwrap_or_default();
            if let Some(tp) = payload.get("tuple").and_then(|t| t.as_object()) {
                let state = tp["state"]
                    .as_array()
                    .map(|arr| arr.iter().map(|v| v.as_f64().unwrap_or(0.0)).collect())
                    .unwrap_or_default();
                let action_param = tp["action_param"].as_str().unwrap_or("").to_string();
                let action_delta = tp["action_delta"].as_f64().unwrap_or(0.0);
                let reward = tp["reward"].as_f64().unwrap_or(0.0);
                let next_state = tp["next_state"]
                    .as_array()
                    .map(|arr| arr.iter().map(|v| v.as_f64().unwrap_or(0.0)).collect())
                    .unwrap_or_default();
                
                tuples.push(DqnTuple {
                    state,
                    action_param,
                    action_delta,
                    reward,
                    next_state,
                });
            }
        }
        
        Ok(tuples)
    }

    pub async fn query_tough_knots(&self, num: usize) -> Result<Vec<EragMemory>> {
        // Query memories with high topology_knot_complexity > 0.4
        // Use filter-based search via HTTP API
        let filter_json = json!({
            "must": [
                {
                    "key": "topology_knot_complexity",
                    "range": {"gt": 0.4}
                }
            ]
        });
        
        let request_json = json!({
            "vector": vec![0.0f32; self.vector_dim], // Dummy vector for filter-only search
            "limit": num,
            "filter": filter_json,
            "with_payload": true,
            "with_vectors": false
        });
        
        let url = format!("{}/collections/{}/points/search", self.base_url, self.collection);
        let resp = self.client.post(&url).json(&request_json).send().await?;
        
        #[derive(Deserialize)]
        struct SearchResponse {
            result: Vec<SearchHit>,
        }
        
        let search_resp: SearchResponse = resp.json().await?;
        let mut memories = Vec::new();
        
        for hit in search_resp.result {
            if !hit.payload.is_empty() {
                let memory = deserialize_memory(&hit.payload);
                memories.push(memory);
            }
        }
        
        Ok(memories)
    }
}

#[allow(dead_code)]
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
pub struct SearchHit {
    score: f32,
    #[serde(default)]
    payload: JsonMap<String, JsonValue>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DqnTuple {
    pub state: Vec<f64>,
    pub action_param: String,
    pub action_delta: f64,
    pub reward: f64,
    pub next_state: Vec<f64>,
}

#[derive(Debug, Clone, Default)]
pub struct SimpleMetrics {
    pub rouge_score: f64,
    pub entropy_delta: f64,
    pub ucb1_score: f64,
    pub curator_quality: f64,
    pub latency_ms: f64,
}

impl From<&JsonMap<String, JsonValue>> for SimpleMetrics {
    fn from(p: &JsonMap<String, JsonValue>) -> Self {
        Self {
            rouge_score: extract_number(p, "rouge"),
            entropy_delta: extract_number(p, "entropy_delta"),
            ucb1_score: extract_number(p, "ucb1"),
            curator_quality: extract_number(p, "curator"),
            latency_ms: extract_number(p, "latency_ms"),
        }
    }
}

#[derive(Deserialize)]
pub struct ScrollResponse {
    pub result: Vec<ScrollPoint>,
    #[serde(default)]
    pub next_page_offset: Option<u64>,
}

#[derive(Deserialize)]
pub struct ScrollPoint {
    pub id: String,
    pub payload: Option<JsonMap<String, JsonValue>>,
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

    // Store quality score and topology data
    if let Some(quality) = memory.quality_score {
        payload.insert("quality_score".to_string(), JsonValue::from(quality as f64));
    }
    if let Some(betti) = memory.topology_betti {
        payload.insert(
            "topology_betti_0".to_string(),
            JsonValue::from(betti[0] as f64),
        );
        payload.insert(
            "topology_betti_1".to_string(),
            JsonValue::from(betti[1] as f64),
        );
        payload.insert(
            "topology_betti_2".to_string(),
            JsonValue::from(betti[2] as f64),
        );
    }
    if let Some(knot_complexity) = memory.topology_knot_complexity {
        payload.insert(
            "topology_knot_complexity".to_string(),
            JsonValue::from(knot_complexity as f64),
        );
    }

    // Add continual learning fields
    if let Some(ref solution_path) = memory.solution_path {
        payload.insert(
            "solution_path".to_string(),
            JsonValue::String(solution_path.clone()),
        );
    }
    payload.insert(
        "iteration_count".to_string(),
        JsonValue::from(memory.iteration_count as f64),
    );
    payload.insert(
        "conversation_history".to_string(),
        JsonValue::Array(
            memory
                .conversation_history
                .iter()
                .cloned()
                .map(JsonValue::String)
                .collect(),
        ),
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

    let quality_score = payload
        .get("quality_score")
        .and_then(|v| v.as_f64())
        .map(|f| f as f32);
    let topology_betti = if payload.contains_key("topology_betti_0") {
        Some([
            extract_number(payload, "topology_betti_0") as usize,
            extract_number(payload, "topology_betti_1") as usize,
            extract_number(payload, "topology_betti_2") as usize,
        ])
    } else {
        None
    };
    let topology_knot_complexity = payload
        .get("topology_knot_complexity")
        .and_then(|v| v.as_f64())
        .map(|f| f as f32);

    let solution_path = payload
        .get("solution_path")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    let conversation_history = payload
        .get("conversation_history")
        .and_then(|value| value.as_array())
        .map(|values| {
            values
                .iter()
                .filter_map(|val| val.as_str().map(|s| s.to_string()))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    let iteration_count = extract_number(payload, "iteration_count") as u32;

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
        quality_score,
        topology_betti,
        topology_knot_complexity,
        solution_path,
        conversation_history,
        iteration_count,
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
