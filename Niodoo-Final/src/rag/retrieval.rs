//! Retrieval engine orchestrating local embedding search.

use super::local_embeddings::{
    Document as LocalDocument, LocalEmbeddingConfig, LocalEmbeddingGenerator,
};
use super::Document;
use anyhow::Result;
use std::cmp::Ordering;
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct RetrievalConfig {
    pub base_threshold: f32,
    pub token_adjustment_factor: f32,
    pub max_results: usize,
    pub diversity_penalty: f32,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            base_threshold: 0.45,
            token_adjustment_factor: 0.012,
            max_results: 5,
            diversity_penalty: 0.05,
        }
    }
}

#[derive(Debug, Clone)]
pub struct QueryCharacteristics {
    pub token_count: usize,
    pub is_long_query: bool,
}

impl QueryCharacteristics {
    pub fn new(query: &str) -> Self {
        let token_count = estimate_token_count(query);
        Self {
            token_count,
            is_long_query: token_count > 18,
        }
    }

    pub fn calculate_optimal_threshold(&self, config: &RetrievalConfig) -> f32 {
        let adjustment = if self.is_long_query {
            (self.token_count as f32 * config.token_adjustment_factor).min(0.35)
        } else {
            0.0
        };
        (config.base_threshold - adjustment).clamp(0.05, 0.9)
    }
}

pub struct RetrievalStorage<'a> {
    documents: &'a [LocalDocument],
}

impl<'a> RetrievalStorage<'a> {
    pub fn get_all_documents(&self) -> Result<Vec<LocalDocument>, String> {
        Ok(self.documents.to_vec())
    }
}

pub struct RetrievalEngine {
    config: RetrievalConfig,
    embedder: LocalEmbeddingGenerator,
    documents: Vec<LocalDocument>,
    document_norms: Vec<f32>,
}

impl Default for RetrievalEngine {
    fn default() -> Self {
        Self::new().expect("retrieval engine initialisation should not fail")
    }
}

impl RetrievalEngine {
    pub fn new() -> Result<Self> {
        let embedder = LocalEmbeddingGenerator::new(LocalEmbeddingConfig::default())?;
        Ok(Self {
            config: RetrievalConfig::default(),
            embedder,
            documents: Vec::new(),
            document_norms: Vec::new(),
        })
    }

    pub fn add_document(&mut self, mut document: LocalDocument) -> Result<()> {
        if document.embedding.is_empty() {
            document.embedding = self.embedder.generate_embedding(&document.content)?;
        }
        let norm = vector_norm(&document.embedding);
        self.documents.push(document);
        self.document_norms.push(norm);
        Ok(())
    }

    pub fn storage(&self) -> RetrievalStorage<'_> {
        RetrievalStorage {
            documents: &self.documents,
        }
    }

    pub fn config(&self) -> &RetrievalConfig {
        &self.config
    }

    pub fn set_retrieval_config(&mut self, config: RetrievalConfig) {
        self.config = config;
    }

    pub fn retrieve(&self, query: &str) -> Result<Vec<(LocalDocument, f32)>> {
        let query_embedding = self.embedder.generate_embedding(query)?;
        self.prioritize_and_retrieve(
            &query_embedding,
            query,
            estimate_token_count(query),
            &self.documents,
            &self.config,
            self.config.max_results,
        )
    }

    pub fn prioritize_and_retrieve(
        &self,
        query_embedding: &[f32],
        query_text: &str,
        token_count: usize,
        documents: &[LocalDocument],
        config: &RetrievalConfig,
        max_results: usize,
    ) -> Result<Vec<(LocalDocument, f32)>> {
        let characteristics = QueryCharacteristics::new(query_text);
        let threshold = characteristics.calculate_optimal_threshold(config);
        let query_norm = vector_norm(query_embedding);

        let mut scored: Vec<(LocalDocument, f32)> = documents
            .iter()
            .enumerate()
            .map(|(idx, document)| {
                let similarity = cosine_similarity(
                    query_embedding,
                    query_norm,
                    &document.embedding,
                    self.document_norms[idx],
                );
                let penalty = if token_count > 24 {
                    config.diversity_penalty * (token_count as f32 / 100.0)
                } else {
                    0.0
                };
                (document.clone(), similarity - penalty)
            })
            .filter(|(_, score)| *score >= threshold)
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        scored.truncate(max_results);
        Ok(scored)
    }

    pub fn search_similar(&self, query: &str, k: usize) -> Result<Vec<(Document, f32)>> {
        let results = self.retrieve(query)?;
        let converted: Vec<(Document, f32)> = results
            .into_iter()
            .take(k)
            .map(|(local_doc, score)| {
                (
                    Document {
                        id: local_doc.id,
                        content: local_doc.content,
                        metadata: local_doc.metadata,
                        embedding: Some(local_doc.embedding),
                        created_at: chrono::Utc::now(),
                        entities: Vec::new(),
                        chunk_id: None,
                        source_type: None,
                        resonance_hint: None,
                        token_count: 0,
                    },
                    score,
                )
            })
            .collect();
        Ok(converted)
    }
}

fn estimate_token_count(text: &str) -> usize {
    text.split_whitespace().count().max(1)
}

fn vector_norm(vector: &[f32]) -> f32 {
    vector.iter().map(|value| value * value).sum::<f32>().sqrt()
}

fn cosine_similarity(query: &[f32], query_norm: f32, doc: &[f32], doc_norm: f32) -> f32 {
    if query.is_empty() || doc.is_empty() || query.len() != doc.len() {
        return 0.0;
    }

    let dot: f32 = query.iter().zip(doc).map(|(a, b)| a * b).sum();
    if query_norm == 0.0 || doc_norm == 0.0 {
        0.0
    } else {
        dot / (query_norm * doc_norm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn retrieval_returns_relevant_documents() {
        let mut engine = RetrievalEngine::new().unwrap();
        engine
            .add_document(LocalDocument {
                id: "doc1".into(),
                content: "Topological empathy practice".into(),
                embedding: Vec::new(),
                metadata: HashMap::new(),
            })
            .unwrap();
        engine
            .add_document(LocalDocument {
                id: "doc2".into(),
                content: "Cooking dinner".into(),
                embedding: Vec::new(),
                metadata: HashMap::new(),
            })
            .unwrap();

        let results = engine.retrieve("topological empathy").unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].0.id, "doc1");
    }
}
