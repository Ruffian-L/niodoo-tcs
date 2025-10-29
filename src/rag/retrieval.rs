//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

use std::collections::HashMap;

use anyhow::Result;
use chrono::Utc;
use tracing::{info, warn};

use super::local_embeddings::{
    Document as LocalDocument, LocalEmbeddingConfig, LocalEmbeddingGenerator,
};
use crate::consciousness::ConsciousnessState;

#[derive(Clone, Debug)]
pub struct RetrievalConfig {
    pub base_threshold: f32,
    pub token_adjustment_factor: f32,
    pub max_results: usize,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            base_threshold: 0.42,
            token_adjustment_factor: 0.007,
            max_results: 5,
        }
    }
}

#[derive(Clone, Debug)]
pub struct QueryCharacteristics {
    pub token_count: usize,
    pub is_long_query: bool,
}

impl QueryCharacteristics {
    pub fn new(query: &str) -> Self {
        let token_count = estimate_token_count(query);
        Self {
            token_count,
            is_long_query: token_count > 12,
        }
    }

    pub fn calculate_optimal_threshold(&self, config: &RetrievalConfig) -> f32 {
        if self.is_long_query {
            let penalty = (self.token_count as f32 * config.token_adjustment_factor).min(0.35);
            (config.base_threshold - penalty).max(0.05)
        } else {
            config.base_threshold
        }
    }
}

#[derive(Clone)]
struct DocumentRecord {
    document: LocalDocument,
}

enum EmbeddingBackend {
    Local {
        generator: LocalEmbeddingGenerator,
        embedding_dim: usize,
    },
}

impl EmbeddingBackend {
    fn initialize() -> Result<Self> {
        let embedding_dim = std::env::var("NIODOO_EMBEDDINGS_DIM")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(384);

        let mut config = LocalEmbeddingConfig::default();
        config.embedding_dim = embedding_dim;
        config.cache_size = std::env::var("NIODOO_EMBEDDINGS_CACHE")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(1_000);

        let generator = LocalEmbeddingGenerator::new(config)?;
        Ok(Self::Local {
            generator,
            embedding_dim,
        })
    }

    fn embedding_dim(&self) -> usize {
        match self {
            Self::Local { embedding_dim, .. } => *embedding_dim,
        }
    }

    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        match self {
            Self::Local {
                generator,
                embedding_dim,
            } => {
                let tensor = generator.generate_embedding(text)?;
                let flattened = tensor.flatten_all()?;
                let mut vector = flattened.to_vec1::<f32>()?;
                finalize_embedding(&mut vector, *embedding_dim);
                Ok(vector)
            }
        }
    }
}

pub struct RetrievalEngine {
    config: RetrievalConfig,
    backend: EmbeddingBackend,
    documents: Vec<DocumentRecord>,
}

impl Default for RetrievalEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl RetrievalEngine {
    pub fn new() -> Self {
        Self::with_config(RetrievalConfig::default())
    }

    pub fn with_config(config: RetrievalConfig) -> Self {
        let backend = EmbeddingBackend::initialize().unwrap_or_else(|err| {
            warn!("⚠️  Falling back to deterministic local embeddings: {err}");
            let mut fallback_config = LocalEmbeddingConfig::default();
            fallback_config.embedding_dim = 384;
            let generator = LocalEmbeddingGenerator::new(fallback_config)
                .expect("local embedding backend must initialize");
            EmbeddingBackend::Local {
                generator,
                embedding_dim: 384,
            }
        });

        Self {
            config,
            backend,
            documents: Vec::new(),
        }
    }

    pub fn config(&self) -> &RetrievalConfig {
        &self.config
    }

    pub fn retrieval_config(&self) -> &RetrievalConfig {
        &self.config
    }

    pub fn set_retrieval_config(&mut self, config: RetrievalConfig) {
        self.config = config;
    }

    pub fn try_add_document(&mut self, mut document: LocalDocument) -> Result<()> {
        if document.embedding.is_empty() {
            document.embedding = self.backend.embed(&document.content)?;
        } else {
            finalize_embedding(&mut document.embedding, self.backend.embedding_dim());
        }

        if document.metadata.is_empty() {
            document.metadata = HashMap::new();
        }

        self.documents.push(DocumentRecord { document });
        Ok(())
    }

    pub fn add_document(&mut self, document: LocalDocument) {
        if let Err(err) = self.try_add_document(document) {
            warn!("⚠️  Failed to add document to retrieval engine: {err}");
        }
    }

    pub fn try_retrieve(
        &self,
        query: &str,
        _state: &ConsciousnessState,
    ) -> Result<Vec<(super::Document, f32)>> {
        let mut query_embedding = self.backend.embed(query)?;
        finalize_embedding(&mut query_embedding, self.backend.embedding_dim());
        let token_count = estimate_token_count(query);
        Ok(self.prioritize_and_retrieve(
            &query_embedding,
            query,
            token_count,
            &self.documents,
            &self.config,
            self.config.max_results,
        ))
    }

    pub fn retrieve(&self, query: &str, state: &ConsciousnessState) -> Vec<(super::Document, f32)> {
        match self.try_retrieve(query, state) {
            Ok(results) => results,
            Err(err) => {
                warn!("⚠️  Retrieval failed: {}", err);
                Vec::new()
            }
        }
    }

    pub fn prioritize_and_retrieve(
        &self,
        query_embedding: &[f32],
        query_text: &str,
        token_count: usize,
        documents: &[DocumentRecord],
        config: &RetrievalConfig,
        max_results: usize,
    ) -> Vec<(super::Document, f32)> {
        let characteristics = QueryCharacteristics {
            token_count,
            is_long_query: token_count > 12,
        };
        let threshold = characteristics.calculate_optimal_threshold(config);

        let mut scored_docs: Vec<(super::Document, f32)> = documents
            .iter()
            .map(|record| {
                let similarity = cosine_similarity(query_embedding, &record.document.embedding);
                (
                    convert_to_rag_document(&record.document, Some(similarity), query_text),
                    similarity,
                )
            })
            .filter(|(_, score)| *score >= threshold)
            .collect();

        scored_docs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored_docs.truncate(max_results.min(config.max_results));
        scored_docs
    }
}

fn convert_to_rag_document(
    document: &LocalDocument,
    similarity: Option<f32>,
    query_text: &str,
) -> super::Document {
    let mut metadata = document.metadata.clone();
    metadata
        .entry("query".to_string())
        .or_insert_with(|| query_text.to_string());
    if let Some(sim) = similarity {
        metadata.insert("similarity".to_string(), format!("{sim:.4}"));
    }

    super::Document {
        id: document.id.clone(),
        content: document.content.clone(),
        metadata,
        embedding: Some(document.embedding.clone()),
        created_at: Utc::now(),
        entities: Vec::new(),
        chunk_id: None,
        source_type: None,
        resonance_hint: None,
        token_count: estimate_token_count(&document.content),
    }
}

fn finalize_embedding(vector: &mut Vec<f32>, expected_dim: usize) {
    if vector.len() > expected_dim {
        vector.truncate(expected_dim);
    } else if vector.len() < expected_dim {
        vector.resize(expected_dim, 0.0);
    }
    normalize(vector);
}

fn normalize(vector: &mut [f32]) {
    let norm = vector
        .iter()
        .map(|v| (*v as f64) * (*v as f64))
        .sum::<f64>()
        .sqrt();
    if norm > 0.0 {
        let inv = 1.0 / norm as f32;
        for value in vector.iter_mut() {
            *value *= inv;
        }
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

pub fn estimate_token_count(text: &str) -> usize {
    text.split_whitespace().count().max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ensure_local_backend() {
        std::env::set_var("NIODOO_EMBEDDINGS_MOCK", "1");
    }

    #[test]
    fn test_query_characteristics() {
        ensure_local_backend();
        let short = QueryCharacteristics::new("AI");
        assert!(!short.is_long_query);
        assert_eq!(short.token_count, 1);

        let long =
            QueryCharacteristics::new("This is a long query about AI research into consciousness");
        assert!(long.is_long_query);
        assert!(long.token_count > 1);
    }

    #[test]
    fn test_threshold_calculation() {
        ensure_local_backend();
        let config = RetrievalConfig {
            base_threshold: 0.5,
            token_adjustment_factor: 0.01,
            ..Default::default()
        };
        let short = QueryCharacteristics {
            token_count: 2,
            is_long_query: false,
        };
        assert_eq!(short.calculate_optimal_threshold(&config), 0.5);

        let long = QueryCharacteristics {
            token_count: 20,
            is_long_query: true,
        };
        assert!((long.calculate_optimal_threshold(&config) - 0.3).abs() < f32::EPSILON);
    }

    #[test]
    fn test_retrieval_flow() {
        ensure_local_backend();
        let mut engine = RetrievalEngine::new();

        for idx in 0..4 {
            let mut metadata = HashMap::new();
            metadata.insert("topic".to_string(), "ai".to_string());
            engine.add_document(LocalDocument {
                id: format!("doc-{idx}"),
                content: format!("Artificial intelligence insight #{idx}"),
                embedding: Vec::new(),
                metadata,
            });
        }

        let mut state = ConsciousnessState::default();
        let results = engine.retrieve("artificial intelligence", &mut state);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_cosine_similarity_bounds() {
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0];
        assert_eq!(cosine_similarity(&a, &b), 1.0);

        let c = vec![1.0, 0.0];
        let d = vec![0.0, 1.0];
        assert_eq!(cosine_similarity(&c, &d), 0.0);

        let empty: Vec<f32> = Vec::new();
        assert_eq!(cosine_similarity(&empty, &empty), 0.0);
    }
}
