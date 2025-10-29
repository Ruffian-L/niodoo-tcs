//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

pub mod embeddings;
pub mod generation;
pub mod ingestion;
pub mod local_embeddings;
pub mod privacy;
pub mod retrieval;
pub mod storage;

use anyhow::Result;
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
        let mut temp_state = crate::consciousness::ConsciousnessState::default();
        let mut results = self.try_retrieve(query, &temp_state)?;
        if results.len() > k {
            results.truncate(k);
        }
        Ok(results)
    }
}
