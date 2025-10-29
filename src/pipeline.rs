use super::prelude::*;
use crate::compass::CompassOutcome;
use crate::learning::LearningLoop;
use crate::tcs::TCSAnalyzer;
use anyhow::{Context, Result};
use prost::Message as ProstMessage;
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};
use tracing::{info, warn, error};

pub mod proto {
    include!(concat!(env!("OUT_DIR"), "/niodoo.rs"));
}

#[derive(Clone)]
pub struct Pipeline {
    pub embedding_client: Arc<dyn EmbeddingClient + Send + Sync>,
    pub tcs_analyzer: Arc<TCSAnalyzer>,
    pub curator: Arc<dyn Curator + Send + Sync>,
    pub qdrant_client: Arc<dyn VectorStore + Send + Sync>,
    pub learning_loop: Arc<LearningLoop>,
    pub config: Arc<RuntimeConfig>,
    pub retry_context: Arc<AsyncMutex<RetryContext>>,
}

impl Pipeline {
    pub fn new(
        embedding_client: Arc<dyn EmbeddingClient + Send + Sync>,
        tcs_analyzer: Arc<TCSAnalyzer>,
        curator: Arc<dyn Curator + Send + Sync>,
        qdrant_client: Arc<dyn VectorStore + Send + Sync>,
        learning_loop: Arc<LearningLoop>,
        config: Arc<RuntimeConfig>,
    ) -> Self {
        Self {
            embedding_client,
            tcs_analyzer,
            curator,
            qdrant_client,
            learning_loop,
            config,
            retry_context: Arc::new(AsyncMutex::new(RetryContext::default())),
        }
    }

    pub async fn process(&self, prompt: &str) -> Result<PipelineResponse> {
        let start = std::time::Instant::now();
        info!("Processing prompt: {}", prompt);

        // Embed
        let embedding = self.embedding_client.embed(prompt).await
            .context("Embedding failed")?;
        info!("Embedding completed in {:?}, dim={}", start.elapsed(), embedding.len());

        // Analyze TCS Topology
        let topo_state = self.tcs_analyzer.analyze(&embedding)
            .context("TCS analysis failed")?;
        let topo_proto = proto::TopologyState {
            entropy: topo_state.entropy as f32,
            iit_phi: topo_state.iit_phi as f32,
            knots: topo_state.knots.iter().map(|&k| k as f32).collect(),
            betti_numbers: topo_state.betti.iter().map(|&b| b as i32).collect(),
            spectral_gap: topo_state.spectral_gap as f32,
            persistent_entropy: topo_state.persistent_entropy as f32,
        };

        // Retrieve from Qdrant (using Protobuf for query if needed)
        let docs = self.qdrant_client.retrieve(&embedding, 5).await
            .context("Retrieval failed")?;

        // Generate baseline and hybrid
        let baseline = self.generate_baseline(prompt, &docs).await?;
        let hybrid = self.generate_hybrid(prompt, &docs, &topo_proto).await?;

        // Compass and refine
        let compass = self.evaluate_compass(&topo_proto, &baseline, &hybrid).await?;
        let refined = if compass.needs_refinement() {
            self.curator.refine(&hybrid, &compass).await?
        } else {
            hybrid
        };

        // Learn and update state
        let learning_reward = self.learning_loop.update(&prompt, &refined, &compass).await?;
        let cons_state = proto::ConsciousnessState {
            topology: Some(topo_proto),
            pad_ghost: Some(proto::PadGhostState {
                pad: vec![0.99, 0.99, 0.99, 0.99, 0.96, 0.99, 0.99], // Example from logs
                mu: vec![0.0025, -0.0006, 0.0037, 0.0006, 0.0017, 0.0013, -0.0074],
                sigma: vec![0.123, 0.094, 0.095, 0.099, 0.090, 0.098, 0.095],
                raw_stds: vec![0.044, 0.034, 0.034, 0.036, 0.033, 0.035, 0.034],
            }),
            quadrant: compass.quadrant.to_string(),
            threat: compass.threat,
            healing: compass.healing,
            rouge_score: self.compute_rouge(&baseline, &refined) as f32,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };

        let latency = start.elapsed().as_millis() as f64;
        let response = PipelineResponse {
            prompt: prompt.to_string(),
            baseline_response: baseline,
            hybrid_response: refined,
            state: Some(cons_state),
            latency_ms: latency as f32,
        };

        // Serialize state for logging/Qdrant (binary efficient)
        let state_bytes = response.state.as_ref().unwrap().encode_to_vec();
        info!("Serialized state size: {} bytes (vs JSON ~2x larger)", state_bytes.len());
        self.log_metrics(&response, &state_bytes).await?;

        Ok(response)
    }

    fn serialize_state(&self, state: &proto::ConsciousnessState) -> Vec<u8> {
        state.encode_to_vec()
    }

    fn deserialize_state(&self, bytes: &[u8]) -> Result<proto::ConsciousnessState> {
        proto::ConsciousnessState::decode(bytes).map_err(|e| anyhow::anyhow!("Decode failed: {}", e))
    }

    async fn log_metrics(&self, response: &PipelineResponse, state_bytes: &[u8]) -> Result<()> {
        // Upsert to Qdrant with binary payload
        self.qdrant_client.upsert_binary(
            &response.prompt,
            state_bytes,
            &self.embedding_client.embed(&response.prompt).await?,
        ).await?;
        info!("Metrics logged: ROUGE={}, latency={}ms", response.state.as_ref().unwrap().rouge_score, response.latency_ms);
        Ok(())
    }

    // Placeholder for other methods (generate_baseline, etc.) - assume they exist
    async fn generate_baseline(&self, _prompt: &str, _docs: &[Document]) -> Result<String> {
        Ok("Baseline response".to_string())
    }

    async fn generate_hybrid(&self, _prompt: &str, _docs: &[Document], _topo: &proto::TopologyState) -> Result<String> {
        Ok("Hybrid response".to_string())
    }

    async fn evaluate_compass(&self, _topo: &proto::TopologyState, _baseline: &str, _hybrid: &str) -> Result<CompassOutcome> {
        Ok(CompassOutcome::default())
    }

    fn compute_rouge(&self, _a: &str, _b: &str) -> f64 {
        0.244 // From logs example
    }
}

#[derive(Default)]
pub struct RetryContext {
    attempts: u32,
}

#[async_trait]
impl AsyncMutex<RetryContext> for Arc<Mutex<RetryContext>> {
    // Simplified - actual impl would use tokio::sync::Mutex
}

#[derive(Default)]
pub struct RuntimeConfig {
    max_retries: u32,
}

// Other imports and structs as needed
use crate::rag::Document;
use async_trait::async_trait;
