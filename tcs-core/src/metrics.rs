//! Prometheus metrics for TCS monitoring
//! Separate from core engine logic

use prometheus::{register_counter_vec, register_histogram_vec, register_gauge_vec, Registry};
use once_cell::sync::Lazy;
use std::sync::Arc;

// Metrics registry with ALL metrics for full project monitoring
static REGISTRY: Lazy<Arc<Registry>> = Lazy::new(|| {
    let registry = Registry::new();
    
    // Register all metrics
    let _entropy_gauge = register_gauge_vec!("tcs_entropy", "Current persistence entropy", &["component"]).unwrap();
    let _prompt_counter = register_counter_vec!("tcs_prompts_total", "Prompt status counts", &["type"]).unwrap();
    let _output_hist = register_histogram_vec!("tcs_output_duration_seconds", "Output processing time", &["type"]).unwrap();
    let _output_var = register_histogram_vec!("tcs_output_var", "Output variance histogram", &["component"]).unwrap();
    let _memory_counter = register_counter_vec!("tcs_memories_saved_total", "Memory save counter", &["type"]).unwrap();
    let _memory_gauge = register_gauge_vec!("tcs_memories_size_bytes", "Memory storage size", &["type"]).unwrap();
    let _rag_hist = register_histogram_vec!("tcs_rag_latency_seconds", "RAG retrieval latency", &["success"]).unwrap();
    let _rag_counter = register_counter_vec!("tcs_rag_hits_total", "RAG retrieval hits", &["component"]).unwrap();
    let _rag_sim = register_histogram_vec!("tcs_rag_similarity", "RAG similarity scores", &["component"]).unwrap();
    let _llm_counter = register_counter_vec!("tcs_llm_prompts_total", "LLM prompt counter", &["type"]).unwrap();
    let _learning_entropy_delta = register_gauge_vec!("tcs_learning_entropy_delta", "Entropy delta over epochs", &["component"]).unwrap();

    Arc::new(registry)
});

pub fn init_metrics() {
    Lazy::force(&REGISTRY);
}

pub fn get_registry() -> Arc<Registry> {
    REGISTRY.clone()
}

