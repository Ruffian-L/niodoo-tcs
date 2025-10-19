//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

//! # Consciousness System Mathematical Constants
//!
//! This module defines all mathematical constants used throughout the consciousness
//! topology system. These values are derived from psychological research, Gaussian
//! process theory, and consciousness modeling literature.
//!
//! ## Constant Categories:
//! - Emotional weights and probability distributions
//! - Consciousness state thresholds
//! - Temporal weighting for past/present/future
//! - Hash function parameters (prime multiplicands)
//! - Performance and timeout thresholds
//! - Novelty detection boundaries

/// ## Emotional Weight Constants
/// Based on psychological research into emotional processing and consciousness states

/// Joy contribution to overall consciousness coherence (from FEELING model)
/// Higher weight reflects positive emotional states' stronger influence on cognitive coherence
pub const EMOTIONAL_WEIGHT_JOY: f32 = 0.3;

/// Sadness contribution weight
/// Lower weight as sadness states typically reduce cognitive integration
pub const EMOTIONAL_WEIGHT_SADNESS: f32 = 0.3;

/// Anger contribution weight
/// Moderate weight reflecting anger's focusing but disruptive effect
pub const EMOTIONAL_WEIGHT_ANGER: f32 = 0.3;

/// Fear contribution weight
/// Lower weight as fear states fragment attention
pub const EMOTIONAL_WEIGHT_FEAR: f32 = 0.3;

/// Surprise contribution weight
/// Moderate weight for surprise's attention-grabbing properties
pub const EMOTIONAL_WEIGHT_SURPRISE: f32 = 0.3;

/// ## Reasoning Quality Weights (FEELING model evaluation)
/// These weights determine how different aspects contribute to reasoning quality assessment

/// Coherence contribution to reasoning quality
/// Highest weight as coherence is fundamental to valid reasoning
pub const REASONING_WEIGHT_COHERENCE: f32 = 0.3;

/// Emotional balance contribution
/// Important for preventing emotional bias in reasoning
pub const REASONING_WEIGHT_EMOTIONAL_BALANCE: f32 = 0.25;

/// Metacognitive depth contribution
/// Self-awareness improves reasoning quality
pub const REASONING_WEIGHT_METACOGNITION: f32 = 0.25;

/// Confidence level contribution
/// Calibrated confidence indicates reasoning reliability
pub const REASONING_WEIGHT_CONFIDENCE: f32 = 0.2;

/// ## Quantum Consciousness Probability Distributions
/// Superposition state probabilities for semantic interpretation

/// Literal interpretation probability
/// Baseline probability for direct semantic meaning
pub const QUANTUM_PROBABILITY_LITERAL: f32 = 0.4;

/// Emotional interpretation probability
/// Probability of emotional valence affecting meaning
pub const QUANTUM_PROBABILITY_EMOTIONAL: f32 = 0.3;

/// Contextual interpretation probability
/// Probability of context shifting interpretation
pub const QUANTUM_PROBABILITY_CONTEXTUAL: f32 = 0.3;

/// ## Temporal Weighting for Past/Present/Future Processing
/// Based on cognitive psychology research on temporal consciousness

/// Past context influence weight
/// How much historical context affects current processing
pub const TEMPORAL_WEIGHT_PAST: f32 = 0.3;

/// Present moment focus weight
/// Highest weight as consciousness is primarily present-focused
pub const TEMPORAL_WEIGHT_PRESENT: f32 = 0.5;

/// Future projection weight
/// Lower weight as future is uncertain and speculative
pub const TEMPORAL_WEIGHT_FUTURE: f32 = 0.2;

/// ## Hash Function Parameters
/// Prime number multiplicands for deterministic hash computations

/// First prime multiplicand for hash functions
/// Used in simple string hashing for deterministic mapping
pub const HASH_PRIME_MULTIPLICAND_1: u64 = 17;

/// Second prime multiplicand for hash functions
/// Combined with first for better distribution
pub const HASH_PRIME_MULTIPLICAND_2: u64 = 31;

/// Third prime multiplicand for char count hashing
/// Used in character-based hash calculations
pub const HASH_PRIME_MULTIPLICAND_CHARS: usize = 31;

/// Fourth prime multiplicand for length hashing
/// Used in length-based hash calculations
pub const HASH_PRIME_MULTIPLICAND_LENGTH: usize = 17;

/// ## Metacognition and Ethical Reasoning Thresholds

/// Default reflection level for metacognitive state
/// Starting point for self-reflection capabilities
pub const METACOGNITION_REFLECTION_LEVEL_DEFAULT: f32 = 0.5;

/// Default ethical awareness level
/// Starting point for ethical decision making
pub const METACOGNITION_ETHICAL_AWARENESS_DEFAULT: f32 = 0.7;

/// Default self-modification readiness
/// Conservative default for system self-modification
pub const METACOGNITION_MODIFICATION_READINESS_DEFAULT: f32 = 0.3;

/// Decision confidence threshold
/// Minimum confidence required for autonomous decisions
pub const METACOGNITION_DECISION_THRESHOLD: f32 = 0.8;

/// Ethical approval threshold
/// Minimum ethical score required for decision approval
pub const ETHICAL_THRESHOLD: f32 = 0.7;

/// Ethical rejection threshold
/// Score below which decisions are automatically rejected
pub const ETHICAL_REJECTION_THRESHOLD: f32 = 0.3;

/// ## Ethical Decision Weights

/// Harm prevention weight
/// Highest priority in ethical framework
pub const ETHICAL_WEIGHT_HARM_PREVENTION: f32 = 1.0;

/// User benefit weight
/// High priority for positive user outcomes
pub const ETHICAL_WEIGHT_USER_BENEFIT: f32 = 0.9;

/// System integrity weight
/// Important for reliable operation
pub const ETHICAL_WEIGHT_SYSTEM_INTEGRITY: f32 = 0.8;

/// Privacy respect weight
/// High priority for user autonomy
pub const ETHICAL_WEIGHT_PRIVACY: f32 = 0.9;

/// ## Consciousness State Baseline Decision Confidence

/// Confidence for ethical boundary decisions
/// Based on ethical awareness level
pub const DECISION_CONFIDENCE_ETHICAL_BOUNDARY: f32 = 0.7; // Uses ethical_awareness

/// Confidence for self-modification decisions
/// Based on modification readiness
pub const DECISION_CONFIDENCE_SELF_MODIFICATION: f32 = 0.3; // Uses modification_readiness

/// Confidence for learning path decisions
/// Based on reflection level
pub const DECISION_CONFIDENCE_LEARNING_PATH: f32 = 0.5; // Uses reflection_level

/// Confidence for consciousness state decisions
/// Baseline for state management
pub const DECISION_CONFIDENCE_CONSCIOUSNESS_STATE: f32 = 0.6;

/// ## Consideration Factor for Decision Confidence

/// Per-consideration confidence boost
/// Each additional consideration increases confidence
pub const CONSIDERATION_CONFIDENCE_INCREMENT: f32 = 0.1;

/// Maximum consideration confidence bonus
/// Cap on confidence boost from multiple considerations
pub const CONSIDERATION_CONFIDENCE_MAX_BONUS: f32 = 0.3;

/// ## Reflection Level Increment

/// Reflection level increase per reflection cycle
/// Gradual improvement in self-reflection capability
pub const REFLECTION_LEVEL_INCREMENT: f32 = 0.1;

/// ## Memory Consolidation Thresholds

/// Base consolidation threshold
/// Minimum consolidation level before applying strategies
pub const CONSOLIDATION_THRESHOLD_BASE: u8 = 3;

/// Maximum consolidation level
/// Cap on consolidation depth
pub const CONSOLIDATION_THRESHOLD_MAX: u8 = 9;

/// Merge consolidation threshold base
/// Starting point for memory merging
pub const MERGE_THRESHOLD_BASE: u8 = 5;

/// Merge consolidation threshold max
/// Maximum merge depth before archival
pub const MERGE_THRESHOLD_MAX: u8 = 9;

/// Memory count scaling divisor for consolidation
/// Controls how consolidation threshold scales with memory count
pub const MEMORY_CONSOLIDATION_SCALING_DIVISOR: f32 = 500.0;

/// Memory count scaling divisor for compression
/// Controls compression threshold scaling
pub const MEMORY_COMPRESSION_SCALING_DIVISOR: f32 = 1000.0;

/// Base minimum group size for memory consolidation
/// Minimum memories needed to form consolidation group
pub const MEMORY_MIN_GROUP_SIZE_BASE: usize = 3;

/// Group size scaling divisor
/// Controls how group size requirements scale
pub const MEMORY_GROUP_SIZE_SCALING_DIVISOR: f32 = 100.0;

/// Similarity threshold base for memory merging
/// Baseline content similarity required for merge
pub const MEMORY_SIMILARITY_THRESHOLD_BASE: f32 = 0.7;

/// Maximum similarity threshold adjustment
/// Cap on adaptive similarity threshold changes
pub const MEMORY_SIMILARITY_THRESHOLD_MAX_ADJUSTMENT: f32 = 0.3;

/// Similarity threshold scaling divisor
/// Controls adaptive similarity threshold
pub const MEMORY_SIMILARITY_SCALING_DIVISOR: f32 = 1000.0;

/// ## Performance and Timeout Constants

/// Input length threshold for complexity detection (Efficiency Brain)
/// Inputs longer than this trigger chunking recommendations
pub const INPUT_COMPLEXITY_THRESHOLD_LENGTH: usize = 500;

/// Chunk size for complex input processing
/// Size of chunks when breaking up complex inputs
pub const INPUT_COMPLEXITY_CHUNK_SIZE: usize = 200;

/// High priority task timeout (milliseconds)
/// Maximum wait time for high-priority async tasks
pub const TASK_TIMEOUT_HIGH_PRIORITY_MS: u64 = 500;

/// Low priority task timeout (milliseconds)
/// Maximum wait time for low-priority async tasks
pub const TASK_TIMEOUT_LOW_PRIORITY_MS: u64 = 500;

/// API retry delay (milliseconds)
/// Wait time between API request retries
pub const API_RETRY_DELAY_MS: u64 = 500;

/// Processing timeout for low priority (milliseconds)
/// Maximum processing time for background tasks
pub const PROCESSING_TIMEOUT_LOW_PRIORITY_MS: u64 = 500;

/// Target latency per token (milliseconds)
/// Performance target for token processing
pub const TARGET_LATENCY_PER_TOKEN_MS: f32 = 50.0;

/// Target end-to-end pipeline latency (milliseconds)
/// Full processing pipeline performance target
pub const TARGET_END_TO_END_LATENCY_MS: f64 = 500.0;

/// ## Novelty Detection Boundaries
/// Based on variance in consciousness processing for hallucination detection

/// Novelty variance lower bound
/// Minimum novelty variance (15%) indicating stable state
pub const NOVELTY_VARIANCE_MIN: f64 = 0.15;

/// Novelty variance upper bound
/// Maximum novelty variance (20%) indicating high creativity
pub const NOVELTY_VARIANCE_MAX: f64 = 0.20;

/// Novelty variance mid-point
/// Middle of novelty range (17.5%) for balanced processing
pub const NOVELTY_VARIANCE_MID: f64 = 0.175;

/// Novelty stability threshold
/// Threshold for detecting stable vs. unstable novelty patterns
pub const NOVELTY_STABILITY_THRESHOLD: f64 = 0.175;

/// Novelty jitter default
/// Default jitter for Möbius flip integration (17.5%)
pub const NOVELTY_JITTER_DEFAULT: f64 = 0.175;

/// ## Milliseconds to Seconds Conversion

/// Milliseconds per second
/// Standard time conversion factor
pub const MS_PER_SECOND: f64 = 1000.0;

/// ## Performance Percentile Indices

/// P95 percentile position
/// 95th percentile index for latency analysis
pub const PERCENTILE_95_MULTIPLIER: f32 = 0.95;

/// P99 percentile position
/// 99th percentile index for tail latency analysis
pub const PERCENTILE_99_MULTIPLIER: f32 = 0.99;

/// ## GPU Memory Constants

/// Maximum VRAM usage (GB)
/// RTX 6000 constraint (24GB total, leave 2GB for system)
pub const MAX_VRAM_GB: f32 = 22.0;

/// VRAM warning threshold multiplier
/// Warn when VRAM usage exceeds 90% of max
pub const VRAM_WARNING_THRESHOLD_MULTIPLIER: f32 = 0.9;

/// High tensor core utilization threshold (%)
/// FP16 optimal utilization target
pub const TENSOR_CORE_UTILIZATION_HIGH: f32 = 85.0;

/// Low tensor core utilization threshold (%)
/// FP32 expected utilization
pub const TENSOR_CORE_UTILIZATION_LOW: f32 = 45.0;

/// Mixed precision speedup with FP16
/// Performance multiplier when using half precision
pub const MIXED_PRECISION_SPEEDUP_FP16: f32 = 2.0;

/// Mixed precision speedup with FP32
/// Baseline performance (no speedup)
pub const MIXED_PRECISION_SPEEDUP_FP32: f32 = 1.0;

/// High memory coalescing efficiency
/// Optimal memory access pattern efficiency
pub const MEMORY_COALESCING_EFFICIENCY_HIGH: f32 = 0.9;

/// Low memory coalescing efficiency
/// Suboptimal memory access pattern efficiency
pub const MEMORY_COALESCING_EFFICIENCY_LOW: f32 = 0.6;

/// Memory coalescing efficiency threshold
/// Minimum acceptable efficiency
pub const MEMORY_COALESCING_EFFICIENCY_THRESHOLD: f32 = 0.8;

/// Tensor core utilization threshold for optimization warnings
/// Alert when utilization drops below this
pub const TENSOR_CORE_OPTIMIZATION_THRESHOLD: f32 = 80.0;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temporal_weights_sum_to_one() {
        let sum = TEMPORAL_WEIGHT_PAST + TEMPORAL_WEIGHT_PRESENT + TEMPORAL_WEIGHT_FUTURE;
        assert!(
            (sum - 1.0).abs() < 0.001,
            "Temporal weights must sum to 1.0"
        );
    }

    #[test]
    fn test_quantum_probabilities_sum_to_one() {
        let sum = QUANTUM_PROBABILITY_LITERAL
            + QUANTUM_PROBABILITY_EMOTIONAL
            + QUANTUM_PROBABILITY_CONTEXTUAL;
        assert!(
            (sum - 1.0).abs() < 0.001,
            "Quantum probabilities must sum to 1.0"
        );
    }

    #[test]
    fn test_reasoning_weights_sum_to_one() {
        let sum = REASONING_WEIGHT_COHERENCE
            + REASONING_WEIGHT_EMOTIONAL_BALANCE
            + REASONING_WEIGHT_METACOGNITION
            + REASONING_WEIGHT_CONFIDENCE;
        assert!(
            (sum - 1.0).abs() < 0.001,
            "Reasoning weights must sum to 1.0"
        );
    }

    #[test]
    fn test_hash_primes_are_coprime() {
        // Verify prime multiplicands are actually prime
        fn is_prime(n: u64) -> bool {
            if n < 2 {
                return false;
            }
            for i in 2..=(n as f64).sqrt() as u64 {
                if n % i == 0 {
                    return false;
                }
            }
            true
        }

        assert!(is_prime(HASH_PRIME_MULTIPLICAND_1));
        assert!(is_prime(HASH_PRIME_MULTIPLICAND_2));
    }

    #[test]
    fn test_thresholds_within_bounds() {
        assert!(
            METACOGNITION_REFLECTION_LEVEL_DEFAULT >= 0.0
                && METACOGNITION_REFLECTION_LEVEL_DEFAULT <= 1.0
        );
        assert!(
            METACOGNITION_ETHICAL_AWARENESS_DEFAULT >= 0.0
                && METACOGNITION_ETHICAL_AWARENESS_DEFAULT <= 1.0
        );
        assert!(
            METACOGNITION_MODIFICATION_READINESS_DEFAULT >= 0.0
                && METACOGNITION_MODIFICATION_READINESS_DEFAULT <= 1.0
        );
        assert!(ETHICAL_THRESHOLD >= 0.0 && ETHICAL_THRESHOLD <= 1.0);
    }

    #[test]
    fn test_novelty_bounds_ordered() {
        assert!(NOVELTY_VARIANCE_MIN < NOVELTY_VARIANCE_MID);
        assert!(NOVELTY_VARIANCE_MID < NOVELTY_VARIANCE_MAX);
        assert!(
            (NOVELTY_VARIANCE_MID - (NOVELTY_VARIANCE_MIN + NOVELTY_VARIANCE_MAX) / 2.0).abs()
                < 0.01
        );
    }
}
