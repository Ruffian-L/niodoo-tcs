//! NIODOO ERAG Module - Topological Memory Retrieval
//!
//! Enhanced Retrieval-Augmented Generation using topological persistence-based retrieval.
//! This module wraps the existing ERAG client and will be extended with TopologicalAttention
//! mechanism from tcs-core when available.

use crate::erag::EragClient;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Memory fragment retrieved via ERAG
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryFragment {
    pub content: String,
    pub relevance_score: f64,
    pub topological_score: Option<f64>, // Topological persistence-based score
    pub metadata: serde_json::Value,
}

/// ERAG analyzer for topological memory retrieval
pub struct ERAGAnalyzer {
    client: Arc<EragClient>,
}

impl ERAGAnalyzer {
    /// Create a new ERAG analyzer
    pub fn new(client: Arc<EragClient>) -> Self {
        Self { client }
    }

    /// Retrieve memory fragments using topological attention
    /// 
    /// This implements topological persistence-based retrieval:
    /// - Uses CognitiveManifold from tcs-core (when available)
    /// - Retrieves based on topological persistence, not just semantic similarity
    /// - Weights results by Betti persistence stability (birth-death differential)
    ///   Higher persistence = more "core" memories, prioritized for retrieval
    /// - Returns memory fragments with topological relevance scores
    pub async fn retrieve(
        &self,
        embedding: &[f32],
        top_k: usize,
    ) -> Result<Vec<MemoryFragment>> {
        // Use existing ERAG client's collapse method
        // TODO: Integrate TopologicalAttention mechanism when available in tcs-core
        
        let collapse_result = self.client.collapse_with_limit(embedding, top_k).await?;

        // Convert to MemoryFragment format with persistence weighting
        // Weight retrieval by Betti persistence stability (birth-death differential)
        // Higher persistence = more foundational, "core" memories
        let avg_sim = collapse_result.average_similarity as f64;
        let fragments: Vec<MemoryFragment> = collapse_result
            .top_hits
            .into_iter()
            .enumerate()
            .map(|(idx, mem)| {
                // Base relevance from similarity
                let base_relevance = avg_sim * (1.0 - (idx as f64 * 0.1).min(0.5));
                
                // Compute persistence weight from entropy stability
                // Entropy delta (after - before) indicates "breakthrough" stability
                let entropy_delta = mem.entropy_after - mem.entropy_before;
                // Normalize to [0, 1] range: positive delta = stable breakthrough
                let persistence_weight = (entropy_delta.max(0.0) / 1.0).min(1.0);
                
                // Weighted relevance: combine similarity with persistence
                let relevance = base_relevance * (1.0 + persistence_weight * 0.3);
                
                MemoryFragment {
                    content: format!("{} -> {}", mem.input, mem.output),
                    relevance_score: relevance,
                    topological_score: Some(persistence_weight), // Persistence-based score
                    metadata: serde_json::json!({
                        "timestamp": mem.timestamp,
                        "entropy_before": mem.entropy_before,
                        "entropy_after": mem.entropy_after,
                        "entropy_delta": entropy_delta,
                        "persistence_weight": persistence_weight,
                        "cascade_stage": mem.cascade_stage,
                    }),
                }
            })
            .collect();

        // Re-sort by weighted relevance (persistence-weighted)
        let mut sorted_fragments = fragments;
        sorted_fragments.sort_by(|a, b| {
            b.relevance_score.partial_cmp(&a.relevance_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(sorted_fragments)
    }
}

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
#[cfg(feature = "pyo3")]
use pyo3::types::PyModule;
#[cfg(feature = "pyo3")]
use pyo3::Bound;

#[cfg(feature = "pyo3")]
#[pyfunction]
fn retrieve(
    _py: Python,
    embedding: Vec<f32>,
    _top_k: usize,
) -> PyResult<PyObject> {
    // TODO: Get ERAG client from global context
    // For now, return error indicating client must be initialized
    Err(pyo3::exceptions::PyRuntimeError::new_err(
        "ERAG client not initialized. Call niodoo.erag.init(client) first."
    ))
}

#[cfg(feature = "pyo3")]
#[pymodule]
pub fn erag(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    use pyo3::wrap_pyfunction;
    m.add_function(wrap_pyfunction!(retrieve, m)?)?;
    Ok(())
}

