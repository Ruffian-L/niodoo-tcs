//! Retrieval augmented generation orchestrator.

use super::local_embeddings::Document as LocalDocument;
use super::retrieval::RetrievalEngine;
use super::{Document, RagPipeline};
use crate::consciousness::ConsciousnessState;
use crate::vllm_bridge::VLLMBridge;
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::env;
use tracing::{info, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagGenerationConfig {
    pub max_context_documents: usize,
    pub max_tokens: usize,
    pub temperature: f64,
    pub top_p: f64,
}

impl Default for RagGenerationConfig {
    fn default() -> Self {
        Self {
            max_context_documents: env::var("RAG_MAX_CONTEXT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(4),
            max_tokens: env::var("RAG_MAX_TOKENS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(192),
            temperature: env::var("RAG_TEMPERATURE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.55),
            top_p: env::var("RAG_TOP_P")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.85),
        }
    }
}

pub struct RagGeneration {
    retrieval: RetrievalEngine,
    vllm: Option<VLLMBridge>,
    config: RagGenerationConfig,
}

impl RagGeneration {
    pub fn new(retrieval: RetrievalEngine) -> Result<Self> {
        Self::with_config(retrieval, RagGenerationConfig::default())
    }

    pub fn with_config(retrieval: RetrievalEngine, config: RagGenerationConfig) -> Result<Self> {
        let endpoint = format!(
            "http://{}:{}",
            env::var("VLLM_HOST").unwrap_or_else(|_| "127.0.0.1".to_string()),
            env::var("VLLM_PORT").unwrap_or_else(|_| "8000".to_string())
        );

        let vllm = match VLLMBridge::connect(&endpoint, env::var("VLLM_API_KEY").ok()) {
            Ok(bridge) => {
                info!(%endpoint, "RAG generation using vLLM backend");
                Some(bridge)
            }
            Err(err) => {
                warn!(%err, "vLLM backend unavailable for RAG generation; falling back to local summariser");
                None
            }
        };

        Ok(Self {
            retrieval,
            vllm,
            config,
        })
    }

    fn build_prompt(
        &self,
        query: &str,
        docs: &[(Document, f32)],
        context: &ConsciousnessState,
    ) -> String {
        let mut prompt = format!(
            "You are Niodoo, an empathetic cognition engine. Current emotional mode: {:?} (coherence {:.2}, authenticity {:.2}).\n",
            context.current_emotion,
            context.coherence,
            context.authenticity_metric
        );

        prompt.push_str("Use the retrieved evidence to answer succinctly with care.\n\n");
        prompt.push_str("<query>\n");
        prompt.push_str(query.trim());
        prompt.push_str("\n</query>\n\n<evidence>\n");

        for (idx, (doc, score)) in docs.iter().enumerate() {
            let snippet = doc.content.lines().take(6).collect::<Vec<_>>().join(" ");
            prompt.push_str(&format!(
                "[{} | relevance {:.3}] {}\n",
                idx + 1,
                score,
                truncate(&snippet, 320)
            ));
        }

        prompt.push_str("</evidence>\n\nRespond with a short reflection (<= 3 paragraphs), weave in one follow-up question if appropriate, and close with a grounded action step.\n");
        prompt
    }

    fn summarise_locally(&self, query: &str, docs: &[(Document, f32)]) -> String {
        if docs.is_empty() {
            return format!(
                "I want to help with \"{}\", yet nothing relevant surfaced. Could you share more detail or clarify the angle you're exploring?",
                query.trim()
            );
        }

        let mut bullets = Vec::new();
        for (doc, score) in docs.iter().take(self.config.max_context_documents) {
            let snippet = truncate(&doc.content, 160);
            bullets.push(format!("• ({:.2}) {}", score, snippet));
        }

        let joined = bullets.join("\n");
        format!(
            "Here's what stood out for \"{}\":\n{}\n\nAction: choose one bullet that feels most alive and take a five-minute experiment around it. I'm ready to iterate with you after that step.",
            query.trim(),
            joined
        )
    }

    fn call_vllm(&self, prompt: &str) -> Result<String> {
        let bridge = self
            .vllm
            .as_ref()
            .ok_or_else(|| anyhow!("vLLM backend not configured"))?;
        let fut = bridge.generate(
            prompt,
            self.config.max_tokens,
            self.config.temperature,
            self.config.top_p,
        );

        let response = if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.block_on(fut)?
        } else {
            tokio::runtime::Runtime::new()?.block_on(fut)?
        };

        Ok(response.trim().to_string())
    }
}

impl RagPipeline for RagGeneration {
    fn process_query(&mut self, query: &str, context: &ConsciousnessState) -> Result<String> {
        let retrieved = self
            .retrieval
            .search_similar(query, self.config.max_context_documents)?;
        let prompt = self.build_prompt(query, &retrieved, context);

        if let Ok(response) = self.call_vllm(&prompt) {
            return Ok(response);
        }

        Ok(self.summarise_locally(query, &retrieved))
    }

    fn load_documents(&mut self, documents: Vec<Document>) -> Result<()> {
        for document in documents {
            let local = LocalDocument {
                id: document.id.clone(),
                content: document.content.clone(),
                embedding: document.embedding.clone().unwrap_or_default(),
                metadata: document.metadata.clone(),
            };
            self.retrieval.add_document(local)?;
        }
        Ok(())
    }

    fn search_similar(&self, query: &str, k: usize) -> Result<Vec<(Document, f32)>> {
        self.retrieval.search_similar(query, k)
    }
}

fn truncate(text: &str, max_len: usize) -> String {
    if text.len() <= max_len {
        text.to_string()
    } else {
        let mut truncated = text[..max_len].to_string();
        truncated.push_str("…");
        truncated
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consciousness::ConsciousnessState;

    #[test]
    fn local_fallback_is_used_when_no_vllm() {
        let engine = RetrievalEngine::new().unwrap();
        let mut rag = RagGeneration::with_config(engine, RagGenerationConfig::default()).unwrap();
        let context = ConsciousnessState::default();
        let response = rag.process_query("test prompt", &context).unwrap();
        assert!(response.contains("test prompt"));
    }
}
