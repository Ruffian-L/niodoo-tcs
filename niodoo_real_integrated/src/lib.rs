#![allow(missing_docs)]
//! Niodoo Real Integrated: Topological Cognitive System (TCS) integration crate.
//! Bridges pipelines, embeddings, learning, and generation into a cohesive runtime.

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

/// Metrics collection and reporting.
pub mod metrics;

/// Pipeline composition from input → analysis → generation.
pub mod pipeline;

// pub mod resilience;
// pub mod federated;

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

pub mod test_support;

#[cfg(test)]
mod tests;
