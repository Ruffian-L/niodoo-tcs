pub mod embeddings;
pub mod generation;
pub mod ingestion;
pub mod local_embeddings;
pub mod privacy;
pub mod retrieval;
pub mod storage;

use super::config::ConsciousnessConfig;
use super::dual_mobius_gaussian::{gaussian_process, GaussianMemorySphere};
use anyhow::Result;
use candle_core::Device;
pub use privacy::{EmbeddingPrivacyShield, PrivacyConfig};
// use std::f64::consts::FRAC_1_SQRT_PI; // Removed unstable feature

pub use embeddings::EmbeddingGenerator;
pub use generation::RagGeneration;
pub use ingestion::IngestionEngine;
pub use retrieval::{QueryCharacteristics, RetrievalConfig, RetrievalEngine};
pub use storage::MemoryStorage;

// Real RAG pipeline trait
pub trait RagPipeline {
    fn process_query(
        &mut self,
        query: &str,
        context: &crate::consciousness::ConsciousnessState,
    ) -> Result<String>;
    fn load_documents(&mut self, documents: Vec<Document>) -> Result<()>;
    fn search_similar(&self, query: &str, k: usize) -> Result<Vec<(Document, f32)>>;
}

// Document structure for RAG
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Document {
    pub id: String,
    pub content: String,
    pub metadata: std::collections::HashMap<String, String>,
    pub embedding: Option<Vec<f32>>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub entities: Vec<String>,
    pub chunk_id: Option<u64>,
    pub source_type: Option<String>,
    pub resonance_hint: Option<f32>,
    pub token_count: usize,
}

impl RetrievalEngine {
    pub fn search_similar_mobius(&mut self, query: &str, k: usize) -> Result<Vec<(Document, f32)>> {
        // Retrieve documents from storage (returns local_embeddings::Document)
        use local_embeddings::Document as LocalDocument;
        let local_documents = self
            .storage()
            .get_all_documents()
            .map_err(|e| anyhow::anyhow!("Failed to retrieve documents from storage: {}", e))?;
        let spheres: Vec<GaussianMemorySphere> = local_documents
            .iter()
            .filter_map(|record: &LocalDocument| {
                let embedding = &record.embedding;
                if embedding.is_empty() {
                    return None;
                }
                let embedding_len = embedding.len();
                let variance = 1.0 / embedding_len as f64; // Derived variance from dimension
                let mut covariance = vec![vec![0.0f64; embedding_len]; embedding_len];
                for i in 0..embedding_len {
                    covariance[i][i] = variance;
                }
                let mean: Vec<f64> = embedding.iter().map(|&x| x as f64).collect();
                let cov_matrix: Vec<Vec<f64>> = covariance; // Already f64
                GaussianMemorySphere::new(
                    mean,
                    cov_matrix, // Fixed: use built covariance instead of vec![vec![1.0]]
                    &Device::Cpu,
                )
                .ok()
            })
            .collect();
        let similarities =
            gaussian_process(&spheres, &Device::Cpu, &ConsciousnessConfig::default())
                .map_err(|e| anyhow::anyhow!("GP error: {}", e))?;

        // Convert local_documents to rag::Document and pair with scores
        let mut scored_docs: Vec<(Document, f32)> = local_documents
            .iter()
            .zip(similarities.iter().map(|s| *s as f32))
            .map(|(local_doc, score)| {
                let rag_doc = Document {
                    id: local_doc.id.clone(),
                    content: local_doc.content.clone(),
                    metadata: local_doc.metadata.clone(),
                    embedding: Some(local_doc.embedding.clone()),
                    created_at: chrono::Utc::now(),
                    entities: Vec::new(),
                    chunk_id: None,
                    source_type: None,
                    resonance_hint: None,
                    token_count: local_doc.content.split_whitespace().count(),
                };
                (rag_doc, score)
            })
            .collect();
        // Ethical nurturing: adjust threshold dynamically
        for (doc, score) in &scored_docs {
            if *score < self.retrieval_config().base_threshold {
                // Add ()
                let meta_msg = format!(
                    "Why suppress low-similarity doc? Nurturing as LearningWill: {:?}",
                    doc
                );
                tracing::info!("{}", meta_msg);
            }
        }
        scored_docs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        Ok(scored_docs.into_iter().take(k).collect::<Vec<_>>()) // k from parameter
    }
}
