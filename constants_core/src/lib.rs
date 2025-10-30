// Copyright 2025 Jason Van Pham (Niodoo)
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of Niodoo TCS.
// For commercial licensing, see LICENSE-COMMERCIAL.md

// Constants Core - Modular Constants System
// No hardcoding, no magic numbers, no bullshit

pub mod consciousness;
pub mod consciousness_engine;
pub mod gaussian;
pub mod mathematical;
pub mod memory;
pub mod model;
pub mod monitoring;
pub mod network;
pub mod processing;
pub mod topology;
pub mod visualization;

// Re-export commonly used constants
pub use consciousness::*;
pub use gaussian::*;
pub use mathematical::*;
pub use memory::*;
pub use model::*;
pub use monitoring::*;
pub use network::*;
pub use processing::*;
pub use topology::*;
pub use visualization::*;

// Re-export consciousness engine constants (avoiding conflicts with consciousness module)
pub use consciousness_engine::{
    ACTIVITY_ENHANCEMENT_MULTIPLIER, ACTIVITY_THRESHOLD, BRAIN_PROCESSING_TIMEOUT_SECS,
    BRAIN_RESULT_TIMEOUT_SECS, COMPLEXITY_FACTOR_DIVISOR, DEFAULT_ACTIVITY_LEVEL,
    DEFAULT_BROADCAST_CHANNEL_CAPACITY, DEFAULT_MAX_FILES_RETAINED, DEFAULT_MAX_FILE_SIZE_MB,
    DEFAULT_PROCESSING_TIME_SECS, DEFAULT_QWEN_MAX_TOKENS, DEFAULT_ROTATION_INTERVAL_HOURS,
    DEFAULT_TEST_QWEN_MAX_TOKENS, DEPTH_SCORE_DIVISOR, EMOTIONAL_FACTOR_MULTIPLIER,
    HIGH_COMPLEXITY_THRESHOLD, HIGH_CONFIDENCE_THRESHOLD, HIGH_EMOTIONAL_WEIGHT_THRESHOLD,
    INPUT_PREVIEW_LENGTH, KEYWORDS_COUNT, LOW_EMOTIONAL_WEIGHT_THRESHOLD, MAX_COMPLEXITY_FACTOR,
    MAX_SEMANTIC_DEPTH, MAX_STRUCTURAL_COMPLEXITY, MAX_TEMPERATURE, MAX_VOCABULARY_DEPTH,
    MEMORY_STORE_MAX_SIZE, MEMORY_SUMMARIES_COUNT, MIN_KEYWORD_LENGTH, MIN_RESPONSE_LENGTH,
    MODERATE_COMPLEXITY_THRESHOLD, QUESTION_COUNT_WEIGHT, RECENT_MEMORIES_COUNT,
    RESPONSE_PREVIEW_LENGTH, SENTENCE_COUNT_WEIGHT, STATE_LOCK_TIMEOUT_SECS,
    TELEMETRY_BUS_SHUTDOWN_TIMEOUT_SECS, TEMPERATURE_RESPONSE_LENGTH_DIVISOR, TEST_INPUT_COUNT,
    UNIFIED_FIELD_PROCESSOR_SEED, WORD_LENGTH_DIVISOR,
};

// Re-export consciousness memory consolidation constants and config
pub use consciousness::{
    ConsciousnessConfig, CONSOLIDATION_THRESHOLD_BASE, CONSOLIDATION_THRESHOLD_MAX,
    MEMORY_COMPRESSION_SCALING_DIVISOR, MEMORY_CONSOLIDATION_SCALING_DIVISOR,
    MEMORY_GROUP_SIZE_SCALING_DIVISOR, MEMORY_MIN_GROUP_SIZE_BASE,
    MEMORY_SIMILARITY_SCALING_DIVISOR, MERGE_THRESHOLD_BASE, MERGE_THRESHOLD_MAX,
};
