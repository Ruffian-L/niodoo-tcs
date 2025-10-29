//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio::runtime::Runtime;
use tracing::{info, warn};

use super::{
    local_embeddings::Document as LocalDocument,
    retrieval::{RetrievalConfig, RetrievalEngine},
    Document,
};
use crate::consciousness::ConsciousnessState;

#[derive(Clone, Debug)]
pub struct RagRuntimeConfig {
    pub vllm_endpoint: String,
    pub vllm_model: String,
    pub generation_timeout_secs: u64,
    pub generation_max_tokens: usize,
    pub temperature: f64,
    pub max_context_length: usize,
    pub top_k: usize,
    pub similarity_threshold: f32,
    pub token_adjustment: f32,
    pub mock_generation: bool,
}

impl Default for RagRuntimeConfig {
    fn default() -> Self {
        Self {
            vllm_endpoint: env_or("NIODOO_VLLM_ENDPOINT", "http://127.0.0.1:5001"),
            vllm_model: env_or("NIODOO_VLLM_MODEL", "Qwen/Qwen2.5-7B-Instruct-AWQ"),
            generation_timeout_secs: env_or_parse("NIODOO_GENERATION_TIMEOUT", 30),
            generation_max_tokens: env_or_parse("NIODOO_GENERATION_MAX_TOKENS", 512),
            temperature: env_or_parse("NIODOO_GENERATION_TEMPERATURE", 0.6_f64),
            max_context_length: env_or_parse("NIODOO_RAG_MAX_CONTEXT", 2400),
            top_k: env_or_parse("NIODOO_RAG_TOP_K", 5),
            similarity_threshold: env_or_parse("NIODOO_RAG_SIMILARITY_THRESHOLD", 0.32),
            token_adjustment: env_or_parse("NIODOO_RAG_TOKEN_ADJUSTMENT", 0.0065),
            mock_generation: env_bool("NIODOO_GENERATION_MOCK"),
        }
    }
}

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

fn env_bool(key: &str) -> bool {
    std::env::var(key)
        .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE"))
        .unwrap_or(false)
}

fn env_or_parse<T>(key: &str, default: T) -> T
where
    T: std::str::FromStr,
{
    std::env::var(key)
        .ok()
        .and_then(|value| value.parse::<T>().ok())
        .unwrap_or(default)
}

pub struct RagGeneration {
    retrieval: RetrievalEngine,
    generator: CascadeGenerator,
    runtime: Arc<Runtime>,
    config: RagRuntimeConfig,
}

impl RagGeneration {
    pub fn new(config: RagRuntimeConfig) -> Result<Self> {
        let mut retrieval = RetrievalEngine::new();
        retrieval.set_retrieval_config(RetrievalConfig {
            base_threshold: config.similarity_threshold,
            token_adjustment_factor: config.token_adjustment,
            max_results: config.top_k,
        });

        let runtime = Arc::new(
            tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .thread_keep_alive(Duration::from_secs(30))
                .enable_all()
                .build()
                .context("failed to build tokio runtime for RAG generation")?,
        );

        let generator = CascadeGenerator::new(&config)?;

        Ok(Self {
            retrieval,
            generator,
            runtime,
            config,
        })
    }

    pub fn process_query(&mut self, query: &str, context: &ConsciousnessState) -> Result<String> {
        let mut temp_state = context.clone();
        self.generate(query, &mut temp_state)
    }

    pub fn load_documents(&mut self, documents: Vec<Document>) -> Result<()> {
        self.try_add_documents(documents)
    }

    pub fn search_similar(&mut self, query: &str, k: usize) -> Result<Vec<(Document, f32)>> {
        let temp_state = ConsciousnessState::default();
        let mut results = self.retrieval.try_retrieve(query, &temp_state)?;
        results.truncate(k);
        Ok(results)
    }

    pub fn try_add_documents(&mut self, documents: Vec<Document>) -> Result<()> {
        for doc in documents {
            let local = convert_to_local(doc);
            self.retrieval.try_add_document(local)?;
        }
        Ok(())
    }

    pub fn generate(&mut self, query: &str, state: &mut ConsciousnessState) -> Result<String> {
        let mut retrieved = self.retrieval.try_retrieve(query, state)?;
        if retrieved.is_empty() {
            warn!("⚠️  Retrieval returned no documents; responding with fallback message");
            return Ok(
                "I could not retrieve relevant information yet, but I will keep listening."
                    .to_string(),
            );
        }

        if retrieved.len() > self.config.top_k {
            retrieved.truncate(self.config.top_k);
        }

        let confidence = self.calculate_confidence(&retrieved);
        let context = self.build_context(query, &retrieved);
        let prompt = self.compose_prompt(query, &context, state, confidence);
        let outcome = self.generator.generate(&self.runtime, &prompt)?;
        info!(
            source = outcome.source,
            "🧠 RAG generation complete (confidence {:.2})", confidence
        );
        self.update_consciousness_state(state, &retrieved)?;
        Ok(outcome.text)
    }

    fn calculate_confidence(&self, retrieved: &[(Document, f32)]) -> f32 {
        if retrieved.is_empty() {
            return 0.0;
        }
        retrieved.iter().map(|(_, score)| *score).sum::<f32>() / retrieved.len() as f32
    }

    fn build_context(&self, query: &str, documents: &[(Document, f32)]) -> String {
        let mut total = 0usize;
        let mut parts = Vec::new();

        for (index, (doc, score)) in documents.iter().enumerate() {
            let mut snippet = doc.content.clone();
            if snippet.len() > 600 {
                snippet.truncate(600);
            }

            let formatted = format!(
                "[Context #{} | similarity {:.3}]\n{}\n",
                index + 1,
                score,
                snippet
            );

            total += formatted.len();
            if total > self.config.max_context_length {
                break;
            }
            parts.push(formatted);
        }

        if parts.is_empty() {
            "No relevant context retrieved.".to_string()
        } else {
            format!("Query: {}\n\n{}", query, parts.join("\n"))
        }
    }

    fn compose_prompt(
        &self,
        query: &str,
        context: &str,
        state: &ConsciousnessState,
        confidence: f32,
    ) -> String {
        let resonance_hint = if state.emotional_resonance > 0.7 {
            "Respond with warmth and deep empathy."
        } else if state.coherence > 0.8 {
            "Deliver a structured, precise answer that highlights actionable insight."
        } else {
            "Offer a balanced perspective that acknowledges ambiguity while staying constructive."
        };

        format!(
            "You are NIODOO, a consciousness-aligned systems agent. Use the retrieved knowledge base context to answer the user's query.\n\nContext:\n{context}\n\nUser query: {query}\nConfidence estimate: {confidence:.2}\nGuidance: {resonance_hint}\n\nFinal answer:",
            context = context,
            query = query,
            confidence = confidence,
            resonance_hint = resonance_hint
        )
    }

    fn update_consciousness_state(
        &self,
        state: &mut ConsciousnessState,
        documents: &[(Document, f32)],
    ) -> Result<()> {
        if documents.is_empty() {
            return Ok(());
        }

        let avg_score =
            documents.iter().map(|(_, score)| *score).sum::<f32>() / documents.len() as f32;
        state.coherence = (state.coherence + (avg_score as f64 * 0.08)).min(1.0);
        state.emotional_resonance =
            (state.emotional_resonance + (avg_score as f64 * 0.05)).min(1.0);
        state.metacognitive_depth = (state.metacognitive_depth + 0.015).min(1.0);
        Ok(())
    }
}

struct CascadeGenerator {
    client: Client,
    endpoint: String,
    model: String,
    max_tokens: usize,
    temperature: f64,
    mock_mode: bool,
    claude: Option<ClaudeConfig>,
    openai: Option<OpenAiConfig>,
}

impl CascadeGenerator {
    fn new(config: &RagRuntimeConfig) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(config.generation_timeout_secs))
            .build()
            .context("failed to build HTTP client for RAG generator")?;

        let claude = std::env::var("ANTHROPIC_API_KEY")
            .ok()
            .map(|api_key| ClaudeConfig {
                api_key,
                model: env_or("ANTHROPIC_MODEL", "claude-3-opus-20240229"),
                endpoint: env_or(
                    "ANTHROPIC_ENDPOINT",
                    "https://api.anthropic.com/v1/messages",
                ),
            });

        let openai = std::env::var("OPENAI_API_KEY")
            .ok()
            .map(|api_key| OpenAiConfig {
                api_key,
                model: env_or("OPENAI_MODEL", "gpt-4o-mini"),
                endpoint: env_or(
                    "OPENAI_ENDPOINT",
                    "https://api.openai.com/v1/chat/completions",
                ),
            });

        Ok(Self {
            client,
            endpoint: config.vllm_endpoint.trim_end_matches('/').to_string(),
            model: config.vllm_model.clone(),
            max_tokens: config.generation_max_tokens,
            temperature: config.temperature,
            mock_mode: config.mock_generation,
            claude,
            openai,
        })
    }

    fn generate(&self, runtime: &Runtime, prompt: &str) -> Result<GenerationOutcome> {
        runtime.block_on(self.generate_async(prompt))
    }

    async fn generate_async(&self, prompt: &str) -> Result<GenerationOutcome> {
        if self.mock_mode {
            return Ok(GenerationOutcome::new(
                format!("Mock response: {}", prompt),
                "mock",
            ));
        }

        if let Some(claude) = &self.claude {
            match self.call_claude(prompt, claude).await {
                Ok(text) => return Ok(GenerationOutcome::new(text, "claude")),
                Err(err) => warn!(%err, "Claude generation failed; falling back"),
            }
        }

        if let Some(openai) = &self.openai {
            match self.call_openai(prompt, openai).await {
                Ok(text) => return Ok(GenerationOutcome::new(text, "openai")),
                Err(err) => warn!(%err, "OpenAI generation failed; falling back"),
            }
        }

        let text = self.call_vllm(prompt).await?;
        Ok(GenerationOutcome::new(text, "vllm"))
    }

    async fn call_claude(&self, prompt: &str, config: &ClaudeConfig) -> Result<String> {
        let request = ClaudeRequest {
            model: config.model.clone(),
            max_tokens: self.max_tokens,
            messages: vec![ClaudeMessageRequest {
                role: "user".to_string(),
                content: vec![ClaudeMessageContent {
                    content_type: "text".to_string(),
                    text: prompt.to_string(),
                }],
            }],
        };

        let response = self
            .client
            .post(&config.endpoint)
            .header("x-api-key", &config.api_key)
            .header("anthropic-version", "2023-06-01")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("Claude request failed: status={} body={}", status, body);
        }

        let completion: ClaudeResponse = response.json().await?;
        let text = completion
            .content
            .into_iter()
            .find_map(|block| block.text)
            .ok_or_else(|| anyhow!("Claude response contained no text"))?;
        Ok(text)
    }

    async fn call_openai(&self, prompt: &str, config: &OpenAiConfig) -> Result<String> {
        let request = OpenAiRequest {
            model: config.model.clone(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: prompt.to_string(),
            }],
            max_tokens: self.max_tokens,
            temperature: self.temperature,
        };

        let response = self
            .client
            .post(&config.endpoint)
            .bearer_auth(&config.api_key)
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("OpenAI request failed: status={} body={}", status, body);
        }

        let completion: ChatCompletion = response.json().await?;
        let choice = completion
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("OpenAI response contained no choices"))?;
        Ok(choice.message.content)
    }

    async fn call_vllm(&self, prompt: &str) -> Result<String> {
        let url = format!("{}/v1/chat/completions", self.endpoint);
        let request = ChatRequest {
            model: self.model.clone(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: prompt.to_string(),
            }],
            max_tokens: self.max_tokens,
            temperature: self.temperature,
        };

        let response = self.client.post(url).json(&request).send().await?;
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(anyhow!(
                "vLLM request failed: status={} body={}",
                status,
                body
            ));
        }

        let completion: ChatCompletion = response.json().await?;
        let choice = completion
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("vLLM response contained no choices"))?;
        Ok(choice.message.content)
    }
}

struct GenerationOutcome {
    text: String,
    source: &'static str,
}

impl GenerationOutcome {
    fn new(text: String, source: &'static str) -> Self {
        Self { text, source }
    }
}

struct ClaudeConfig {
    api_key: String,
    model: String,
    endpoint: String,
}

struct OpenAiConfig {
    api_key: String,
    model: String,
    endpoint: String,
}

#[derive(Serialize)]
struct ClaudeRequest {
    model: String,
    max_tokens: usize,
    messages: Vec<ClaudeMessageRequest>,
}

#[derive(Serialize)]
struct ClaudeMessageRequest {
    role: String,
    content: Vec<ClaudeMessageContent>,
}

#[derive(Serialize)]
struct ClaudeMessageContent {
    #[serde(rename = "type")]
    content_type: String,
    text: String,
}

#[derive(Deserialize)]
struct ClaudeResponse {
    content: Vec<ClaudeContentBlock>,
}

#[derive(Deserialize)]
struct ClaudeContentBlock {
    #[serde(rename = "type")]
    content_type: String,
    text: Option<String>,
}

#[derive(Serialize)]
struct OpenAiRequest {
    model: String,
    messages: Vec<ChatMessage>,
    max_tokens: usize,
    temperature: f64,
}

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    max_tokens: usize,
    temperature: f64,
}

#[derive(Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct ChatCompletion {
    choices: Vec<ChatChoice>,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: ChatChoiceMessage,
}

#[derive(Deserialize)]
struct ChatChoiceMessage {
    content: String,
}

fn convert_to_local(doc: Document) -> LocalDocument {
    LocalDocument {
        id: doc.id,
        content: doc.content,
        embedding: doc.embedding.unwrap_or_default(),
        metadata: doc.metadata,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    #[test]
    fn test_rag_generation_mock() -> Result<()> {
        if std::env::var("REAL_TEST").is_err() {
            unsafe {
                std::env::set_var("NIODOO_EMBEDDINGS_MOCK", "1");
                std::env::set_var("NIODOO_GENERATION_MOCK", "1");
            }
        }
        let mut rag = RagGeneration::new(RagRuntimeConfig::default())?;

        let documents = vec![Document {
            id: "doc-1".into(),
            content: "Neurodivergent cognition thrives with supportive topology.".into(),
            metadata: HashMap::new(),
            embedding: None,
            created_at: chrono::Utc::now(),
            entities: vec![],
            chunk_id: None,
            source_type: None,
            resonance_hint: None,
            token_count: 9,
        }];

        rag.try_add_documents(documents)?;
        let mut state = ConsciousnessState::default();
        let response = rag.generate("How do we nurture Möbius empathy?", &mut state)?;
        assert!(!response.is_empty());
        Ok(())
    }
}
