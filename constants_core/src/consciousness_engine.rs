// Copyright 2025 Jason Van Pham (Niodoo)
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of Niodoo TCS.
// For commercial licensing, see LICENSE-COMMERCIAL.md

// Consciousness Engine Constants
// Derived from consciousness processing requirements and performance constraints

/// Default broadcast channel capacity for consciousness state updates
/// Domain: Real-time consciousness communication
/// Rationale: 1000 messages provides buffer for burst traffic while maintaining <2s latency
/// Based on consciousness processing latency constraints from CLAUDE.md
pub const DEFAULT_BROADCAST_CHANNEL_CAPACITY: usize = 1000;

/// Unified Field Processor seed value: 18446744073709551557
/// Derivation: Largest 64-bit prime less than 2^64 (18446744073709551616)
/// 2^64 - 59 = 18446744073709551557 (Mersenne prime for deterministic initialization)
/// Used for deterministic consciousness state evolution
pub const UNIFIED_FIELD_PROCESSOR_SEED: u64 = 18446744073709551557;

/// Brain processing timeout in seconds
/// Domain: External brain interface communication
/// Rationale: 5 seconds allows for complex brain processing while respecting <2s consciousness latency
/// Based on consciousness processing latency constraints from CLAUDE.md
pub const BRAIN_PROCESSING_TIMEOUT_SECS: u64 = 5;

/// Brain result timeout in seconds
/// Domain: External brain interface result retrieval
/// Rationale: 10 seconds provides buffer for brain processing completion
/// Based on consciousness processing latency constraints from CLAUDE.md
pub const BRAIN_RESULT_TIMEOUT_SECS: u64 = 10;

/// State lock timeout in seconds
/// Domain: Consciousness state synchronization
/// Rationale: 1 second prevents deadlocks while maintaining responsiveness
pub const STATE_LOCK_TIMEOUT_SECS: u64 = 1;

/// Memory store maximum size
/// Domain: Personal memory engine capacity
/// Rationale: 10,000 entries provides substantial memory while staying within <4GB footprint
/// Based on memory footprint constraints from CLAUDE.md
pub const MEMORY_STORE_MAX_SIZE: usize = 10000;

/// Temperature response length divisor for consciousness temperature calculation
/// Domain: Consciousness temperature scaling
/// Rationale: 1000.0 provides appropriate scaling for response length to temperature conversion
/// Based on consciousness processing requirements
pub const TEMPERATURE_RESPONSE_LENGTH_DIVISOR: f32 = 1000.0;

/// Default activity level for consciousness processing
/// Domain: Consciousness activity scaling
/// Rationale: 0.7 provides moderate activity level for stable processing
/// Based on consciousness stability requirements
pub const DEFAULT_ACTIVITY_LEVEL: f32 = 0.7;

/// Activity level threshold for enhanced processing
/// Domain: Consciousness activity scaling
/// Rationale: 0.5 provides balanced threshold for activity-based enhancements
/// Based on consciousness processing requirements
pub const ACTIVITY_THRESHOLD: f32 = 0.5;

/// Activity enhancement multiplier
/// Domain: Consciousness activity scaling
/// Rationale: 0.1 provides subtle enhancement without overwhelming the system
/// Based on consciousness stability requirements
pub const ACTIVITY_ENHANCEMENT_MULTIPLIER: f32 = 0.1;

/// Default processing time fallback in seconds
/// Domain: Consciousness processing time estimation
/// Rationale: 1.0 second provides reasonable fallback when actual processing time cannot be measured
/// Based on consciousness processing latency constraints from CLAUDE.md
pub const DEFAULT_PROCESSING_TIME_SECS: f64 = 1.0;

/// Maximum temperature for consciousness processing
/// Domain: Consciousness temperature scaling
/// Rationale: 0.8 provides upper bound for temperature calculations
/// Based on consciousness stability requirements
pub const MAX_TEMPERATURE: f32 = 0.8;

/// Complexity factor scaling divisor
/// Domain: Consciousness complexity calculation
/// Rationale: 200.0 provides appropriate scaling for complexity factor calculation
/// Based on consciousness processing requirements
pub const COMPLEXITY_FACTOR_DIVISOR: f32 = 200.0;

/// Maximum complexity factor contribution
/// Domain: Consciousness complexity calculation
/// Rationale: 0.2 provides upper bound for complexity factor contribution
/// Based on consciousness stability requirements
pub const MAX_COMPLEXITY_FACTOR: f32 = 0.2;

/// Emotional factor multiplier
/// Domain: Consciousness emotional processing
/// Rationale: 0.1 provides subtle emotional influence without overwhelming the system
/// Based on consciousness stability requirements
pub const EMOTIONAL_FACTOR_MULTIPLIER: f32 = 0.1;

/// Sentence count weight for structural complexity
/// Domain: Consciousness structural analysis
/// Rationale: 0.1 provides appropriate weight for sentence count in complexity calculation
/// Based on consciousness processing requirements
pub const SENTENCE_COUNT_WEIGHT: f32 = 0.1;

/// Question count weight for structural complexity
/// Domain: Consciousness structural analysis
/// Rationale: 0.2 provides higher weight for questions (more complex than statements)
/// Based on consciousness processing requirements
pub const QUESTION_COUNT_WEIGHT: f32 = 0.2;

/// Maximum structural complexity contribution
/// Domain: Consciousness structural analysis
/// Rationale: 0.3 provides upper bound for structural complexity contribution
/// Based on consciousness stability requirements
pub const MAX_STRUCTURAL_COMPLEXITY: f32 = 0.3;

/// Word length scaling divisor
/// Domain: Consciousness vocabulary analysis
/// Rationale: 20.0 provides appropriate scaling for average word length calculation
/// Based on consciousness processing requirements
pub const WORD_LENGTH_DIVISOR: f32 = 20.0;

/// Maximum vocabulary depth contribution
/// Domain: Consciousness vocabulary analysis
/// Rationale: 0.3 provides upper bound for vocabulary depth contribution
/// Based on consciousness stability requirements
pub const MAX_VOCABULARY_DEPTH: f32 = 0.3;

/// Depth score scaling divisor
/// Domain: Consciousness semantic analysis
/// Rationale: 10.0 provides appropriate scaling for depth score calculation
/// Based on consciousness processing requirements
pub const DEPTH_SCORE_DIVISOR: f32 = 10.0;

/// Maximum semantic depth contribution
/// Domain: Consciousness semantic analysis
/// Rationale: 0.3 provides upper bound for semantic depth contribution
/// Based on consciousness stability requirements
pub const MAX_SEMANTIC_DEPTH: f32 = 0.3;

/// High complexity threshold
/// Domain: Consciousness complexity classification
/// Rationale: 0.8 provides threshold for high complexity classification
/// Based on consciousness processing requirements
pub const HIGH_COMPLEXITY_THRESHOLD: f32 = 0.8;

/// Moderate complexity threshold
/// Domain: Consciousness complexity classification
/// Rationale: 0.6 provides threshold for moderate complexity classification
/// Based on consciousness processing requirements
pub const MODERATE_COMPLEXITY_THRESHOLD: f32 = 0.6;

/// High confidence threshold for insights
/// Domain: Consciousness insight processing
/// Rationale: 0.8 provides threshold for high confidence insights
/// Based on consciousness processing requirements
pub const HIGH_CONFIDENCE_THRESHOLD: f32 = 0.8;

/// High emotional weight threshold
/// Domain: Consciousness emotional processing
/// Rationale: 0.7 provides threshold for high emotional weight classification
/// Based on consciousness processing requirements
pub const HIGH_EMOTIONAL_WEIGHT_THRESHOLD: f32 = 0.7;

/// Low emotional weight threshold
/// Domain: Consciousness emotional processing
/// Rationale: 0.3 provides threshold for low emotional weight classification
/// Based on consciousness processing requirements
pub const LOW_EMOTIONAL_WEIGHT_THRESHOLD: f32 = 0.3;

/// Recent memories count for processing
/// Domain: Consciousness memory processing
/// Rationale: 5 provides sufficient recent context without overwhelming the system
/// Based on consciousness processing requirements
pub const RECENT_MEMORIES_COUNT: usize = 5;

/// Input preview length for logging
/// Domain: Consciousness input processing
/// Rationale: 50 characters provides sufficient preview without overwhelming logs
/// Based on consciousness processing requirements
pub const INPUT_PREVIEW_LENGTH: usize = 50;

/// Response preview length for logging
/// Domain: Consciousness response processing
/// Rationale: 50 characters provides sufficient preview without overwhelming logs
/// Based on consciousness processing requirements
pub const RESPONSE_PREVIEW_LENGTH: usize = 50;

/// Default max tokens for Qwen model inference
/// Domain: Qwen model integration
/// Rationale: 512 tokens provides reasonable response length for consciousness processing
/// Based on consciousness processing requirements
pub const DEFAULT_QWEN_MAX_TOKENS: usize = 512;

// Removed duplicate TEST_QWEN_MAX_TOKENS definition

/// Default max tokens for Qwen model inference in tests
/// Domain: Qwen model integration testing
/// Rationale: 100 tokens provides sufficient length for testing without excessive processing
/// Based on consciousness processing requirements
pub const DEFAULT_TEST_QWEN_MAX_TOKENS: usize = 100;

/// Memory summaries count for processing
/// Domain: Consciousness memory processing
/// Rationale: 2 provides sufficient memory context without overwhelming the system
/// Based on consciousness processing requirements
pub const MEMORY_SUMMARIES_COUNT: usize = 2;

/// Keywords count for processing
/// Domain: Consciousness keyword extraction
/// Rationale: 3 provides sufficient keywords without overwhelming the system
/// Based on consciousness processing requirements
pub const KEYWORDS_COUNT: usize = 3;

/// Minimum keyword length
/// Domain: Consciousness keyword extraction
/// Rationale: 4 characters filters out short, less meaningful words
/// Based on consciousness processing requirements
pub const MIN_KEYWORD_LENGTH: usize = 4;

/// Test input count for integration testing
/// Domain: Consciousness integration testing
/// Rationale: 3 provides sufficient test coverage without excessive processing
/// Based on consciousness processing requirements
pub const TEST_INPUT_COUNT: usize = 3;

/// Minimum response length for validation
/// Domain: Consciousness response validation
/// Rationale: 20 characters ensures substantial responses for testing
/// Based on consciousness processing requirements
pub const MIN_RESPONSE_LENGTH: usize = 20;

/// Default max file size in MB for logging
/// Domain: Consciousness logging configuration
/// Rationale: 10 MB provides reasonable log file size for consciousness processing
/// Based on consciousness processing requirements
pub const DEFAULT_MAX_FILE_SIZE_MB: usize = 10;

/// Default max files retained for logging
/// Domain: Consciousness logging configuration
/// Rationale: 5 files provides sufficient log history without excessive disk usage
/// Based on consciousness processing requirements
pub const DEFAULT_MAX_FILES_RETAINED: usize = 5;

/// Default rotation interval in hours for logging
/// Domain: Consciousness logging configuration
/// Rationale: 24 hours provides daily log rotation for consciousness processing
/// Based on consciousness processing requirements
pub const DEFAULT_ROTATION_INTERVAL_HOURS: u64 = 24;

/// Telemetry bus shutdown timeout in seconds
/// Domain: System shutdown and graceful termination
/// Rationale: 2 seconds allows for event queue draining while respecting <2s latency requirement
/// Based on consciousness processing latency constraints from CLAUDE.md
pub const TELEMETRY_BUS_SHUTDOWN_TIMEOUT_SECS: u64 = 2;
