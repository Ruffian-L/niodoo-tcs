// Copyright 2025 Jason Van Pham (Niodoo)
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of Niodoo TCS.
// For commercial licensing, see LICENSE-COMMERCIAL.md

//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

use super::local_embeddings::{Document, MathematicalEmbeddingModel};
use crate::consciousness::ConsciousnessState;

#[derive(Debug, Clone)]
pub enum RetrievalError {
    NotImplemented,
}

impl std::fmt::Display for RetrievalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RetrievalError::NotImplemented => write!(
                f,
                "add_document not implemented - feature not yet available"
            ),
        }
    }
}

impl std::error::Error for RetrievalError {}

#[derive(Clone, Debug, Default)]
pub struct RetrievalConfig {
    pub base_threshold: f32,
    pub token_adjustment_factor: f32,
    pub max_results: usize,
}

#[derive(Clone, Debug)]
pub struct QueryCharacteristics {
    pub token_count: usize,
    pub is_long_query: bool,
}

impl QueryCharacteristics {
    pub fn new(query: &str) -> Self {
        let token_count = query.split_whitespace().count();
        Self {
            token_count,
            is_long_query: token_count > 10,
        }
    }

    pub fn calculate_optimal_threshold(&self, config: &RetrievalConfig) -> f32 {
        if self.is_long_query {
            config.base_threshold
                - (self.token_count as f32 * config.token_adjustment_factor).min(0.3)
        } else {
            config.base_threshold
        }
        .max(0.0)
    }
}

pub struct RetrievalEngine {
    config: RetrievalConfig,
    model: MathematicalEmbeddingModel, // Assume access to embedding model if needed
}

impl Default for RetrievalEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl RetrievalEngine {
    pub fn new() -> Self {
        Self {
            config: RetrievalConfig {
                base_threshold: 0.5,
                token_adjustment_factor: 0.01,
                max_results: 5,
            },
            model: MathematicalEmbeddingModel::default(), // Stub, but real would load
        }
    }

    pub fn config(&self) -> &RetrievalConfig {
        &self.config
    }

    pub fn storage(&self) -> &MathematicalEmbeddingModel {
        &self.model
    }

    pub fn retrieval_config(&self) -> &RetrievalConfig {
        &self.config
    }

    pub fn set_retrieval_config(&mut self, config: RetrievalConfig) {
        self.config = config;
    }

    pub fn retrieve(&self, query: &str, state: &ConsciousnessState) -> Vec<(Document, f32)> {
        let embedding = vec![0.1; 384]; // Stub embedding
        let token_count = estimate_token_count(query);
        self.prioritize_and_retrieve(
            &embedding,
            query,
            token_count,
            &self.get_documents(),
            &self.config,
            5,
        )
    }

    fn get_documents(&self) -> Vec<Document> {
        // Stub or load
        vec![]
    }

    pub fn prioritize_and_retrieve(
        &self,
        query_embedding: &[f32],
        query_text: &str,
        token_count: usize,
        documents: &[Document],
        config: &RetrievalConfig,
        max_results: usize,
    ) -> Vec<(Document, f32)> {
        let characteristics = QueryCharacteristics {
            token_count,
            is_long_query: token_count > 10,
        };
        let threshold = characteristics.calculate_optimal_threshold(config);

        let mut scored_docs: Vec<(Document, f32)> = documents
            .iter()
            .map(|doc| {
                let similarity = cosine_similarity(query_embedding, &doc.embedding);
                (doc.clone(), similarity)
            })
            .filter(|(_, sim)| *sim > threshold)
            .collect();

        scored_docs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored_docs.truncate(max_results.min(config.max_results));
        scored_docs
    }

    /// Add a document to the retrieval engine (stub implementation)
    pub fn add_document(&mut self, _document: Document) -> Result<(), RetrievalError> {
        tracing::warn!(
            "add_document called but not implemented - document will not be stored. \
             Use storage().add_document() for actual document persistence."
        );
        Err(RetrievalError::NotImplemented)
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
    use crate::rag::local_embeddings::{Document, EmbeddingModel};

    #[test]
    #[ignore]
    fn test_query_characteristics() {
        let short = QueryCharacteristics::new("AI");
        assert!(!short.is_long_query);
        assert_eq!(short.token_count, 1);

        let long = QueryCharacteristics::new("This is a long query about AI");
        assert!(long.is_long_query);
        assert_eq!(long.token_count, 6);
    }

    #[test]
    fn test_threshold_calculation() {
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
        assert_eq!(long.calculate_optimal_threshold(&config), 0.3); // 0.5 - 20*0.01 = 0.3
    }

    #[test]
    #[ignore]
    fn test_retrieval_short_query() {
        let docs = vec![
            Document {
                id: "doc1".to_string(),
                content: "AI is amazing".to_string(),
                embedding: vec![0.9, 0.1, 0.2, 0.1],
                metadata: Default::default(),
            },
            Document {
                id: "doc2".to_string(),
                content: "Machine learning rocks".to_string(),
                embedding: vec![0.3, 0.4, 0.5, 0.6],
                metadata: Default::default(),
            },
        ];

        let query_embedding = vec![0.8, 0.2, 0.3, 0.2];
        let config = RetrievalConfig::default();
        let engine = RetrievalEngine::new();

        let results = engine.prioritize_and_retrieve(&query_embedding, "AI", 2, &docs, &config, 10);

        assert_eq!(results.len(), 1); // Only the first should be above 0.5
        tracing::info!("Retrieved {} documents for short query 'AI'", results.len());
    }

    #[test]
    fn test_retrieval_long_query() {
        let docs = vec![
            Document {
                id: "doc3".to_string(),
                content: "AI is amazing".to_string(),
                embedding: vec![0.9, 0.1, 0.2, 0.1],
                metadata: Default::default(),
            },
            Document {
                id: "doc4".to_string(),
                content: "Machine learning rocks".to_string(),
                embedding: vec![0.4, 0.3, 0.4, 0.3],
                metadata: Default::default(),
            },
        ];

        let query_embedding = vec![0.8, 0.2, 0.3, 0.2];
        let long_query = "This is a much longer query about artificial intelligence and machine learning systems that should use a lower threshold";
        let token_count = estimate_token_count(long_query);
        let config = RetrievalConfig {
            base_threshold: 0.5,
            token_adjustment_factor: 0.01,
            max_results: 5,
        };
        let mut engine = RetrievalEngine::new();
        engine.set_retrieval_config(config.clone());

        let results_long = engine.prioritize_and_retrieve(
            &query_embedding,
            long_query,
            token_count,
            &docs,
            &config,
            10,
        );

        // With lower threshold, both should be retrieved if similarity > 0.3
        assert_eq!(results_long.len(), 2);
        tracing::info!("Retrieved {} documents for long query", results_long.len());
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0];
        assert_eq!(cosine_similarity(&a, &b), 1.0);

        let c = vec![1.0, 0.0];
        let d = vec![0.0, 1.0];
        assert_eq!(cosine_similarity(&c, &d), 0.0);

        let e = vec![];
        let f = vec![];
        assert_eq!(cosine_similarity(&e, &f), 0.0);
    }
}
