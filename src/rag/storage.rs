//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

use crate::rag::{
    privacy::{EmbeddingPrivacyShield, PrivacyConfig},
    Document,
};
use ndarray::Array1;
use std::collections::HashMap;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DocumentRecord {
    /// Unique hash of the document content for privacy
    pub content_hash: String,

    /// Masked/anonymized document
    pub document: Document,

    /// Potentially anonymized or noised embedding
    pub embedding: Vec<f32>,

    pub created_at: chrono::DateTime<chrono::Utc>,
    pub entities: Vec<String>,
    pub chunk_id: Option<u64>,
    pub resonance_hint: Option<f32>,
    pub token_count: usize,
}

/// Simple cosine similarity calculation
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}

#[derive(Clone)]
pub struct MemoryStorage {
    documents: Vec<DocumentRecord>,
    dim: usize,
    doc_to_index: HashMap<String, usize>,
    privacy_shield: EmbeddingPrivacyShield,
}

impl MemoryStorage {
    pub fn new(dim: usize) -> Self {
        let privacy_config = PrivacyConfig::default();
        let privacy_shield = EmbeddingPrivacyShield::new(privacy_config);

        Self {
            documents: Vec::new(),
            dim,
            doc_to_index: HashMap::new(),
            privacy_shield,
        }
    }

    /// Create storage with custom privacy configuration
    pub fn with_privacy_config(dim: usize, config: PrivacyConfig) -> Self {
        let privacy_shield = EmbeddingPrivacyShield::new(config);

        Self {
            documents: Vec::new(),
            dim,
            doc_to_index: HashMap::new(),
            privacy_shield,
        }
    }

    /// Add document with embedding to storage
    pub fn add_document(
        &mut self,
        mut document: Document,
        embedding: Array1<f32>,
    ) -> Result<usize, String> {
        let embedding_vec: Vec<f32> = embedding.to_vec();
        let doc_id = self.documents.len();

        // Process embedding with privacy shield
        let (content_hash, processed_embedding) = self
            .privacy_shield
            .process_embedding(&document.content, embedding);

        let processed_embedding_vec = processed_embedding.to_vec();
        document.embedding = Some(processed_embedding_vec.clone());

        if document.token_count == 0 {
            document.token_count = document.content.split_whitespace().count();
        }

        let record = DocumentRecord {
            content_hash: content_hash.clone(),
            created_at: document.created_at,
            entities: document.entities.clone(),
            chunk_id: document.chunk_id,
            resonance_hint: document.resonance_hint,
            token_count: document.token_count,
            document,
            embedding: processed_embedding_vec,
        };

        self.doc_to_index.insert(content_hash, doc_id);
        self.documents.push(record);

        Ok(doc_id)
    }

    /// Get all documents for retrieval operations
    pub fn get_documents(&self) -> &Vec<DocumentRecord> {
        &self.documents
    }

    /// Search for similar documents (simple linear search for demo)
    pub fn search_similar(
        &self,
        query_embedding: &Array1<f32>,
        k: usize,
    ) -> Result<Vec<(usize, f32, Document)>, String> {
        let query_vec: Vec<f32> = query_embedding.to_vec();

        let mut similarities = Vec::new();
        for (idx, doc) in self.documents.iter().enumerate() {
            let similarity = cosine_similarity(&query_vec, &doc.embedding);
            similarities.push((idx, similarity, doc.document.clone()));
        }

        // Sort by similarity (descending) and take top k
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        similarities.truncate(k);

        Ok(similarities)
    }

    /// Get document by ID
    pub fn get_document(&self, id: &str) -> Option<&Document> {
        self.doc_to_index
            .get(id)
            .and_then(|&idx| self.documents.get(idx))
            .map(|record| &record.document)
    }

    /// Save documents with embeddings to JSON file for persistence
    pub fn save_to_json(&self, path: &str) -> Result<(), String> {
        use std::fs;
        use std::path::Path;

        // Ensure directory exists
        if let Some(parent) = Path::new(path).parent() {
            fs::create_dir_all(parent).map_err(|e| format!("Failed to create directory: {}", e))?;
        }

        let data = serde_json::to_string_pretty(&self.documents)
            .map_err(|e| format!("Serialization error: {}", e))?;

        fs::write(path, data).map_err(|e| format!("IO error: {}", e))?;

        tracing::info!(
            "💾 Saved {} documents with embeddings to {}",
            self.documents.len(),
            path
        );

        Ok(())
    }

    /// Load documents with embeddings from JSON file
    pub fn load_from_json(&mut self, path: &str) -> Result<(), String> {
        use std::fs;
        use std::path::Path;

        if !Path::new(path).exists() {
            return Err(format!("Storage file not found: {}", path));
        }

        let data = fs::read_to_string(path).map_err(|e| format!("IO error: {}", e))?;
        let loaded: Vec<DocumentRecord> =
            serde_json::from_str(&data).map_err(|e| format!("Deserialization error: {}", e))?;

        self.documents = loaded;

        // Rebuild document index mapping
        self.doc_to_index.clear();

        for (i, record) in self.documents.iter().enumerate() {
            self.doc_to_index.insert(record.document.id.clone(), i);
        }

        tracing::info!("📂 Loaded {} documents from {}", self.documents.len(), path);

        Ok(())
    }

    /// Get number of documents in storage
    pub fn len(&self) -> usize {
        self.documents.len()
    }

    /// Check if storage is empty
    pub fn is_empty(&self) -> bool {
        self.documents.is_empty()
    }

    /// Clear all documents from storage
    pub fn clear(&mut self) {
        self.documents.clear();
        self.doc_to_index.clear();
        tracing::info!("🗑️ Cleared all documents from storage");
    }
}

// Test
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_storage() {
        let mut storage = MemoryStorage::new(3); // 3D spheres
                                                 // TODO: Implement GaussianMemorySphere and add_sphere method
                                                 // let sphere1 = GaussianMemorySphere::new(
                                                 //     vec![0.1, 0.2, 0.3],
                                                 //     vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.0, 0.0, 1.0]],
                                                 // );

        let query = Array1::from_vec(vec![0.1, 0.2, 0.3]);
        let results = storage.search_similar(&query, 1).unwrap();
        assert_eq!(results.len(), 0); // No spheres added yet
        tracing::info!("Storage test completed with {} results", results.len());
    }
}
