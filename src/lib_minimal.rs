//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

//! # Niodoo-Feeling: Minimal Core Consciousness Framework
//!
//! Minimal build configuration for BUILD MASTER to get core consciousness framework compiling
//! This bypasses problematic dependencies while maintaining core functionality

// Core system modules - MINIMAL SET
pub mod core;
pub mod error;
pub mod config;
pub mod consciousness;
pub mod memory;
pub mod brain;
pub mod brains;
pub mod empathy;
pub mod optimization;
pub mod events;
pub mod profiling;
pub mod kv_cache;

// Advanced modules - MINIMAL SET
pub mod oscillatory;
pub mod personal_memory;
pub mod philosophy;
pub mod phase6_config;
pub mod phase6_integration;
pub mod phase7_consciousness_psychology;
pub mod phase7;
pub mod soul_resonance;

// Integration and orchestration modules - MINIMAL SET
pub mod consciousness_pipeline_orchestrator;

// AI inference modules - MINIMAL SET (no external dependencies)
pub mod ai_inference;
pub mod real_ai_inference;

// Mathematical and topological modules - MINIMAL SET
pub mod dual_mobius_gaussian;
pub mod sparse_gaussian_processes;
pub mod topology;
pub mod dynamics;
pub mod advanced_empathy;
pub mod bert_emotion;
pub mod real_model;
pub mod echomemoria_real_inference;

// Visualization and Qt integration - MINIMAL SET
pub mod viz_qt_bridge;

// RAG modules - MINIMAL SET
pub mod rag;

// Consciousness engine modules - MINIMAL SET
pub mod consciousness_engine;

// Main consciousness processing
pub mod main_consciousness;

// Error types
pub use error::*;

// Configuration
pub use crate::config::*;

// Core consciousness functionality
pub use consciousness::*;
pub use memory::*;
pub use brain::*;
pub use brains::*;
pub use empathy::*;

// Re-export key types for external use
pub use consciousness::ConsciousnessEngine;
pub use memory::MemoryConsolidator;
pub use brain::BrainEngine;
pub use empathy::EmpathyEngine;
