use anyhow::{anyhow, Result};
use chrono::Utc;
use rand::{thread_rng, Rng};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map as JsonMap, Value as JsonValue};
use std::time::Duration;
use tracing::{info, instrument, warn};

use crate::compass::CompassOutcome;
use crate::torus::PadGhostState;

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

pub struct EragClient {
    client: Client,
    base_url: String,
    collection: String,
    vector_dim: usize,
    pub similarity_threshold: f32,
}

#[derive(Debug, Clone)]
pub struct CollapseResult {
    pub top_hits: Vec<EragMemory>,
    pub aggregated_context: String,
    pub average_similarity: f32,
}

impl EragClient {
    pub async fn new(
        url: &str,
        collection: &str,
        vector_dim: usize,
        similarity_threshold: f32,
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
        Ok(Self {
            client,
            base_url,
            collection: collection.to_string(),
            vector_dim,
            similarity_threshold,
        })
    }

    #[instrument(skip_all, fields(dim = vector.len()))]
    pub async fn collapse(&self, vector: &[f32]) -> Result<CollapseResult> {
        anyhow::ensure!(
            vector.len() == self.vector_dim,
            "embedding dimension mismatch: expected {}, got {}",
            self.vector_dim,
            vector.len()
        );

        let request = SearchRequest {
            vector: vector.to_vec(),
            limit: 3,
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

        let mut aggregated_context = memories
            .iter()
            .flat_map(|m| m.erag_context.clone())
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

            // Add warning if current approach seems suboptimal
            if better.iteration_count > 0 {
                aggregated_context.push_str(&format!(
                    "\n[Note: This problem was solved optimally in {} iterations previously]",
                    better.iteration_count
                ));
            }
        }

        if aggregated_context.len() > 1000 {
            // Increased from 100 to accommodate corrections
            aggregated_context.truncate(1000);
        }

        Ok(CollapseResult {
            top_hits: memories,
            aggregated_context,
            average_similarity,
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
            topology_knot_complexity: topology.map(|t| t.knot_complexity),
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

    pub async fn store_failure(&self, prompt: &str, output: &str, metrics: &crate::metrics::Metrics, reflection: Option<String>) -> Result<()> {
        let embedding = self.embedder.embed(prompt).await?;
        let doc = Document {
            content: format!("Failed: prompt={prompt}, output={output}, metrics={metrics:?}, reflection={reflection:?}"),
            embedding,
        };
        self.qdrant.upsert_documents(vec![doc]).await?;
        Ok(())\n    }
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
