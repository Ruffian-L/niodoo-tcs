#![allow(
// Copyright 2025 Jason Van Pham (Niodoo)
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of Niodoo TCS.
// For commercial licensing, see LICENSE-COMMERCIAL.md

    dead_code,
    unused_imports,
    unused_variables,
    unused_assignments,
    unused_must_use
)]

//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

//! Niodoo Core: Consciousness engine and ERAG memory system

// ============================================================================
// MODULE DECLARATIONS
// ============================================================================

// Core consciousness modules
pub mod consciousness;
pub mod consciousness_compass;
pub mod consciousness_constants;
pub mod consciousness_state_inversion;
pub mod real_mobius_consciousness;

// Configuration and errors
pub mod config;
pub mod error;

// Memory systems
pub mod advanced_memory_retrieval;
pub mod ai_memory_adapter;
pub mod dream_state_processor;
pub mod enhanced_memory_management;
pub mod groks_memory_retriever;
pub mod kv_cache;
pub mod memory;
pub mod memory_management;
pub mod memory_optimization_engine;
pub mod memory_sync_master;
pub mod optimized_memory_management;
pub mod personal_memory;
pub mod real_memory_bridge;
pub mod transformer_memory;
pub mod true_nonorientable_memory;

// Möbius topology and mathematics
pub mod dual_mobius_gaussian;
pub mod mobius_consciousness_identifiers;
pub mod mobius_flip_integration;
pub mod mobius_gaussian_framework;
pub mod sparse_gaussian_processes;
pub mod topology;
pub mod topology_engine;

// RAG and embeddings
pub mod rag;
pub mod rag_integration;

// Qwen integration
pub mod qwen_curator;
pub mod qwen_integration;

// Token promotion and processing
pub mod token_promotion;

// Events and bridges
pub mod events;
pub mod vllm_bridge;

// Phase systems
pub mod engines;
pub mod phase6_config;
pub mod phase6_integration;
pub mod phase7_consciousness_psychology;

// ============================================================================
// PUBLIC RE-EXPORTS - PHASE 6
// ============================================================================

pub use phase6_config::{
    GpuMetrics, IoMetrics, LatencyMetrics, LearningMetrics, LoggingLearningAnalytics,
    LoggingPerformanceMetrics, MemoryStats, PerformanceSnapshot, Phase6Config, SystemMetrics,
};

// Training and export
pub mod training_data_export;

// ============================================================================
// PUBLIC RE-EXPORTS - ERROR TYPES
// ============================================================================

pub use error::{
    BrainError, CandleFeelingError, CandleFeelingResult, CircuitBreaker, ConsciousnessResult,
    ErrorRecovery, MemoryError, NiodoError, QtBridgeError,
};

// ============================================================================
// PUBLIC RE-EXPORTS - CONFIGURATION
// ============================================================================

pub use config::{
    configure_claude_mcp, AppConfig, ConfigError, ConfigResult, ConsciousnessConfig, McpConfig,
    McpServer, MemoryConfig, ModelConfig,
};

// ============================================================================
// PUBLIC RE-EXPORTS - CONSCIOUSNESS
// ============================================================================

pub use consciousness::{
    ConsciousnessState, EmotionType, EmotionalState as ConsciousnessEmotionalState,
    EmotionalUrgency, ReasoningMode,
};

pub use consciousness_compass::{
    BreakthroughMoment, CompassState, CompassStats, CompassTracker, ConfidenceLevel,
    StrategicAction, StuckState,
};

pub use real_mobius_consciousness::{
    ConsciousnessResult as MobiusConsciousnessResult, EmotionalState as MobiusEmotionalState,
    GoldenSlipperConfig, GoldenSlipperTransformer, KTwistedTorus, MemoryAccessPattern,
    MemoryLayer as MobiusMemoryLayer, MobiusConsciousnessProcessor, MobiusMemorySystem,
    MobiusStrip,
};

// ============================================================================
// PUBLIC RE-EXPORTS - MEMORY SYSTEMS
// ============================================================================

pub use memory::{
    EmotionalTransformParams, EmotionalVector, GuessingMemorySystem, MemoryEntry, MemoryLayer,
    MemorySystemState, MobiusMemorySystem as SixLayerMobiusMemorySystem, SphereId,
    StabilityMetrics, TraversalDirection,
};

// ============================================================================
// PUBLIC RE-EXPORTS - EVENTS
// ============================================================================

pub use events::ConsciousnessEvent;

// ============================================================================
// PUBLIC RE-EXPORTS - QWEN INTEGRATION
// ============================================================================

pub use qwen_integration::{
    QwenConfig, QwenIntegrator, QwenModelInterface, ValidationComparison, ValidationResult,
};

pub use qwen_curator::QloraCurator;

// ============================================================================
// PUBLIC RE-EXPORTS - RAG SYSTEM
// ============================================================================

pub use rag::{
    Document, EmbeddingGenerator, EmbeddingPrivacyShield, IngestionEngine, MemoryStorage,
    PrivacyConfig, RagGeneration, RagPipeline, RetrievalEngine,
};

// ============================================================================
// PUBLIC RE-EXPORTS - TOPOLOGY
// ============================================================================

// Note: topology module currently has no public exports at mod.rs level
// Individual submodules can be accessed via topology::{mobius_graph, etc.}

// ============================================================================
// PUBLIC RE-EXPORTS - TOKEN PROMOTION
// ============================================================================

pub use token_promotion::{
    ConsensusEngine, DynamicTokenizer, PatternDiscoveryEngine, PromotedToken, TokenCandidate,
    TokenPromotionEngine,
};

// ============================================================================
// PUBLIC RE-EXPORTS - KV CACHE
// ============================================================================

pub use kv_cache::{EmbeddingCache, KVCacheStats, LayerKVCache, QwenKVCache, SimilarityCache};
