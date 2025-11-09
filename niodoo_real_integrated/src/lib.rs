pub mod api_clients;
pub mod benchmark;
pub mod circuit_breaker;
pub mod compass;
pub mod config;
pub mod constants;
pub mod consonance;
pub mod curator;
pub mod curator_parser;
pub mod data;
pub mod embedded_qdrant;
pub mod embedding;
pub mod erag;
pub mod eval;
pub mod generation;
#[cfg(feature = "svc")]
pub mod grpc_inference;
pub mod health;
pub mod hyperfocus;
pub mod learning;
pub mod learning_actor;
pub mod lora_trainer;
pub mod mcts;
pub mod mcts_config;
pub mod metrics;
pub mod rce;
pub mod mock_qdrant;
pub mod mock_vllm;
// Legacy pipeline module - commented out as we're using the new pipeline/ structure
// #[path = "pipeline_legacy.rs"]
// pub mod pipeline;
pub mod sandbox;
#[cfg(feature = "niodoo-core")]
pub mod conversation_log;
pub mod constitutional;
pub mod degradation_tiers;
pub mod code_topology;
pub mod rl_harness;
#[cfg(feature = "niodoo-core")]
pub mod emotional_graph;
pub mod gpu_fitness;
#[cfg(feature = "gpu")]
pub mod gpu_fusion;
#[cfg(feature = "gpu")]
pub mod gpu_memory_pool;
#[cfg(feature = "gpu")]
pub mod gpu_async;
#[cfg(feature = "gpu")]
pub mod gpu_batch;
#[cfg(feature = "gpu")]
pub mod gpu_prefetch;
#[cfg(feature = "gpu")]
pub mod gpu_consonance;
#[cfg(feature = "niodoo-core")]
pub mod graph_exporter;
#[cfg(feature = "niodoo-core")]
pub mod memory_architect;
pub mod memory_consolidation;
pub mod pipeline;
pub mod resource_budget;
pub mod security;
pub mod signals;
pub mod tcs_analysis;
pub mod tcs_lora;
pub mod tcs_predictor;
pub mod temporal_tda;
#[cfg(not(feature = "niodoo-core"))]
pub mod token_manager_stub;
#[cfg(feature = "niodoo-core")]
pub mod token_manager;
#[cfg(not(feature = "niodoo-core"))]
pub use token_manager_stub as token_manager;
pub mod tokenizer;
pub mod topology_crawler;
pub mod topology_memory;
pub mod torus;
pub mod tracing_integration;
pub mod util;
pub mod validation;
pub mod vector_store;
pub mod weight_evolution;
pub mod weighted_episodic_mem;
pub mod ntoken_client;
pub mod cqs_calculator;
pub mod fused_agent;

#[cfg(feature = "pyo3")]
pub mod niodoo_api;

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
#[cfg(feature = "pyo3")]
use pyo3::types::PyModule;
#[cfg(feature = "pyo3")]
use pyo3::Bound;

/// Python extension module for NIODOO
#[cfg(feature = "pyo3")]
#[pymodule]
fn niodoo_real_integrated(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    use pyo3::wrap_pyfunction;
    
    // Register submodules
    let parser_module = PyModule::new_bound(_py, "parser")?;
    niodoo_api::parser::parser(_py, &parser_module)?;
    m.add_submodule(&parser_module)?;
    
    let tcs_module = PyModule::new_bound(_py, "tcs")?;
    niodoo_api::tcs::tcs(_py, &tcs_module)?;
    m.add_submodule(&tcs_module)?;
    
    let erag_module = PyModule::new_bound(_py, "erag")?;
    niodoo_api::erag::erag(_py, &erag_module)?;
    m.add_submodule(&erag_module)?;
    
    let tqft_module = PyModule::new_bound(_py, "tqft")?;
    niodoo_api::tqft::tqft(_py, &tqft_module)?;
    m.add_submodule(&tqft_module)?;
    
    Ok(())
}
