//! Niodoo Real Integrated: Topological Cognitive System (TCS) integration crate.
//!
//! This crate provides a comprehensive integrated runtime that bridges pipelines, embeddings,
//! learning, and generation into a cohesive Topological Cognitive System.
//!
//! ## Overview
//!
//! The TCS architecture combines:
//! - **Topological Data Analysis (TDA)**: Persistent homology, Betti numbers, and knot invariants
//! - **Emotional Reasoning Augmented Generation (ERAG)**: Memory retrieval and context collapse
//! - **Learning Loop**: DQN-based reinforcement learning with LoRA adapters
//! - **Generation**: Hybrid model synthesis with consistency voting
//!
//! ## Key Components
//!
//! - [`Pipeline`]: Main orchestration loop connecting all components
//! - [`TCSAnalyzer`]: Computes topological signatures from emotional states
//! - [`EragClient`]: ERAG memory retrieval and storage via Qdrant
//! - [`LearningLoop`]: Continuous learning with DQN and LoRA fine-tuning
//! - [`GenerationEngine`]: Multi-model generation with Claude/GPT/vLLM fallback
//!
//! ## Usage
//!
//! ```no_run
//! use anyhow::Result;
//!
//! #[tokio::main]
//! async fn main() -> Result<()> {
//!     let mut harness = niodoo_real_integrated::test_support::mock_pipeline("embed").await?;
//!     let cycle = harness
//!         .pipeline_mut()
//!         .process_prompt("Blueprint a Möbius-safe rollout")
//!         .await?;
//!     println!("Response: {}", cycle.hybrid_response);
//!     Ok(())
//! }
//! ```
//!
//! ## Architecture
//!
//! The pipeline follows this flow:
//! 1. **Embedding**: Convert input to vector space via Qwen embeddings
//! 2. **Torus Mapping**: Transform to PAD emotional state space
//! 3. **TCS Analysis**: Compute persistent homology and knot invariants
//! 4. **Compass**: UCB1-based exploration/exploitation
//! 5. **ERAG Collapse**: Retrieve relevant memories from Qdrant
//! 6. **Tokenization**: Dynamic token promotion and vocabulary evolution
//! 7. **Generation**: Hybrid synthesis with consistency voting
//! 8. **Learning**: DQN updates and LoRA fine-tuning
//!
//! ## Benchmarks
//!
//! Run the million cycle test for long-run stability:
//! ```bash
//! cargo run -p niodoo_real_integrated --bin million_cycle_test
//! ```
//!
//! Performance targets:
//! - <200ms per cycle
//! - Deterministic RNG flow enabled via `RNG_SEED`
//! - Parallel processing via rayon
//!
//! ## License
//!
//! MIT License

/// Clients for external/model/service APIs (tokenizers, hubs, inference backends).
pub mod api_clients;

/// Compass-style exploration/exploitation engine and search heuristics.
pub mod compass;

/// Runtime and persistent configuration (CLI/env, model paths, limits).
pub mod config;

/// Curator for dataset/sample selection and scheduling.
pub mod curator;

/// Parsers that turn free-form inputs into structured curator directives.
pub mod curator_parser;

/// Core data structures shared across the integrated runtime.
pub mod data;

/// Embedding utilities (projection, normalization, similarity).
pub mod embedding;

/// ERAG memory integration and retrieval glue.
pub mod erag;

/// Text and action generation orchestrators.
pub mod generation;

/// Learning loops, trainers, and policy updates.
pub mod learning;

/// LoRA training helpers and adapters.
pub mod lora_trainer;

/// Monte Carlo Tree Search utilities.
pub mod mcts;

/// MCTS configuration and tuning parameters.
pub mod mcts_config;

/// Metrics collection and reporting.
pub mod metrics;

/// Pipeline composition from input → analysis → generation.
pub mod pipeline;

// pub mod resilience;
pub mod federated;

/// Evaluation harnesses and scoring.
pub mod eval;

/// TCS analysis utilities for state inspection and PAD-like projections.
pub mod tcs_analysis;

/// Predictors over TCS states for next-step guidance.
pub mod tcs_predictor;

/// Token budget management and accounting.
pub mod token_manager;

/// Topology crawlers and structure discovery.
pub mod topology_crawler;

/// Torus/state-space utilities and transforms.
pub mod torus;

/// Miscellaneous helpers and shared utilities.
pub mod util;

/// Vector store implementation for binary protobuf storage
#[cfg(feature = "qdrant")]
pub mod vector_store;

pub mod test_support;

#[cfg(test)]
mod tests;
