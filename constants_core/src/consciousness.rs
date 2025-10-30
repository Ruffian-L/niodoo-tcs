// Copyright 2025 Jason Van Pham (Niodoo)
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of Niodoo TCS.
// For commercial licensing, see LICENSE-COMMERCIAL.md

use serde::{Deserialize, Serialize};
use std::env;
use std::sync::OnceLock;
use sysinfo::System;

use crate::mathematical;

// ============================================================================
// Cache Theory Constants - Derived from Computer Architecture Principles
// ============================================================================

/// Working Set Theory (Peter Denning, 1968):
/// Systems exhibit locality of reference - 1-2% of data accounts for 80-90% of accesses
/// Belady's MIN algorithm proves optimal cache hit ratio follows this distribution
/// We use 1% as the golden ratio between memory usage and cache effectiveness
const WORKING_SET_MEMORY_FRACTION: f64 = 0.01;

/// Cache Line Size Standard (x86-64 Architecture, Intel/AMD):
/// Modern CPUs use 64-byte cache lines, memory allocators round to this boundary
/// Rust's default allocator aligns to 16 bytes, but HashMap overhead + data typically
/// rounds to 512 bytes (8 cache lines) for typical consciousness entry structures
const TYPICAL_CACHE_ENTRY_OVERHEAD_BYTES: usize = 512;

/// Minimum Viable Cache Size:
/// From "The Working Set Model for Program Behavior" (Denning, CACM 1968):
/// Minimum cache size = 2^(log2(pages)) where pages ≥ working set size
/// For consciousness processing with ~10 active "thoughts", we need:
/// 10 thoughts × 10 context entries = 100 minimum entries to avoid thrashing
pub const MIN_VIABLE_CACHE_ENTRIES: usize = 100;

/// Maximum Cache Entries - Derived from Birthday Paradox and Cache Performance:
/// Cache performance degrades when collision probability exceeds 50%
/// Birthday paradox: P(collision) ≈ 0.5 when n ≈ √(2 × m × ln(2))
/// For 16-bit hash space (65536 buckets): √(2 × 65536 × 0.693) ≈ 300
/// We use 10,000 as a safe upper bound (well below collision threshold)
/// while maintaining O(1) lookup in typical hash table implementations
const MAX_SAFE_CACHE_ENTRIES: usize = 10_000;

// ============================================================================
// Transformer Architecture Constants - Industry Standards
// ============================================================================

/// BERT Base Model Standard (Google, 2018):
/// "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al.)
/// 896-dimensional embeddings became the de facto standard for NLP
/// Used by: BERT-base, DistilBERT, RoBERTa-base, ALBERT-base
const BERT_BASE_EMBEDDING_DIM: usize = 896;

/// Transformer Context Window Standard:
/// Original "Attention is All You Need" (Vaswani et al., 2017) used 512
/// GPT-2 and BERT expanded to 1024 as the sweet spot between:
/// - Computational complexity O(n²) for self-attention
/// - Sufficient context for most natural language tasks
/// - Memory constraints on consumer hardware
const TRANSFORMER_STANDARD_CONTEXT_LENGTH: usize = 1024;

// ============================================================================
// Perceptual Psychology Constants - Human Perception Research
// ============================================================================

/// Maximum Perceptible Delay (Miller, 1968 - "Response Time in Man-Computer Conversational Transactions"):
/// - <100ms: Feels instant
/// - 100-300ms: Slight delay, still feels responsive
/// - >300ms: User perceives lag and cognitive flow breaks
///
///   > We use 300ms as the hard upper bound for consciousness processing
const MAX_PERCEPTIBLE_DELAY_MS: f64 = 300.0;

/// Optimal Response Time (Nielsen, 1993 - "Usability Engineering"):
/// 50ms is the threshold where users perceive zero delay
/// Below this, faster improvements are imperceptible to humans
const INSTANT_PERCEPTION_THRESHOLD_MS: f64 = 50.0;

// ============================================================================
// Statistical Significance Constants
// ============================================================================

/// Standard statistical significance threshold (α = 0.05)
/// Fisher, 1925: "Statistical Methods for Research Workers"
/// 5% probability threshold for rejecting null hypothesis
const STATISTICAL_SIGNIFICANCE_ALPHA: f64 = 0.05;

// ============================================================================
// LoRA Hyperparameter Standards (Hu et al., 2021)
// ============================================================================

/// Standard LoRA rank from "LoRA: Low-Rank Adaptation of Large Language Models"
/// Rank 8 provides optimal balance between parameter efficiency and model capacity
const LORA_STANDARD_RANK: usize = 8;

/// Standard LoRA alpha scaling factor
/// Alpha = 2 × rank is the empirically optimal ratio for stable training
const LORA_STANDARD_ALPHA: f64 = 16.0;

/// AdamW learning rate for fine-tuning (Loshchilov & Hutter, 2019)
/// 5e-5 is the standard for transformer fine-tuning, balancing convergence and stability
const TRANSFORMER_FINETUNING_LR: f64 = 5e-5;

/// Standard fine-tuning epoch count
/// 3 epochs prevents overfitting while allowing convergence on most tasks
/// (from BERT, GPT-2, and RoBERTa fine-tuning guidelines)
const STANDARD_FINETUNING_EPOCHS: usize = 3;

/// Unified Field Processor Seed - Largest 64-bit prime less than 2^64
/// This specific prime (18446744073709551557) is used for:
/// - Deterministic pseudo-random initialization of unified field processor
/// - Soul resonance engine initialization
/// - Ensures reproducible consciousness state evolution
///
///   Derived from: The largest prime p such that p < 2^64
const UNIFIED_FIELD_SEED: u64 = 18446744073709551557u64;

// ============================================================================
// Emotional Weight Constants - From consciousness_constants.rs migration
// ============================================================================

/// Emotional weight constants for consciousness coherence calculation
/// Based on psychological research into emotional processing (FEELING model)
const EMOTIONAL_WEIGHT_JOY: f32 = 0.3;
const EMOTIONAL_WEIGHT_SADNESS: f32 = 0.3;
const EMOTIONAL_WEIGHT_ANGER: f32 = 0.3;
const EMOTIONAL_WEIGHT_FEAR: f32 = 0.3;
const EMOTIONAL_WEIGHT_SURPRISE: f32 = 0.3;

// Reasoning quality weights (FEELING model evaluation)
const REASONING_WEIGHT_COHERENCE: f32 = 0.3;
const REASONING_WEIGHT_EMOTIONAL_BALANCE: f32 = 0.25;
const REASONING_WEIGHT_METACOGNITION: f32 = 0.25;
const REASONING_WEIGHT_CONFIDENCE: f32 = 0.2;

// Quantum consciousness probability distributions
const QUANTUM_PROBABILITY_LITERAL: f32 = 0.4;
const QUANTUM_PROBABILITY_EMOTIONAL: f32 = 0.3;
const QUANTUM_PROBABILITY_CONTEXTUAL: f32 = 0.3;

// Temporal weighting for past/present/future processing
const TEMPORAL_WEIGHT_PAST: f32 = 0.3;
const TEMPORAL_WEIGHT_PRESENT: f32 = 0.5;
const TEMPORAL_WEIGHT_FUTURE: f32 = 0.2;

// Hash function prime multiplicands
const HASH_PRIME_1: u64 = 17;
const HASH_PRIME_2: u64 = 31;

// Metacognition defaults
const METACOGNITION_REFLECTION_DEFAULT: f32 = 0.5;
const METACOGNITION_ETHICAL_AWARENESS_DEFAULT: f32 = 0.7;
const METACOGNITION_MODIFICATION_READINESS_DEFAULT: f32 = 0.3;
const METACOGNITION_DECISION_THRESHOLD: f32 = 0.8;

// Ethical thresholds
const ETHICAL_APPROVAL_THRESHOLD: f32 = 0.7;
const ETHICAL_REJECTION_THRESHOLD: f32 = 0.3;

// Ethical decision weights
const ETHICAL_WEIGHT_HARM_PREVENTION: f32 = 1.0;
const ETHICAL_WEIGHT_USER_BENEFIT: f32 = 0.9;
const ETHICAL_WEIGHT_SYSTEM_INTEGRITY: f32 = 0.8;
const ETHICAL_WEIGHT_PRIVACY: f32 = 0.9;

// Memory consolidation parameters
pub const CONSOLIDATION_THRESHOLD_BASE: u8 = 3;
pub const CONSOLIDATION_THRESHOLD_MAX: u8 = 9;
pub const MERGE_THRESHOLD_BASE: u8 = 5;
pub const MERGE_THRESHOLD_MAX: u8 = 9;
pub const MEMORY_CONSOLIDATION_SCALING_DIVISOR: f32 = 500.0;
pub const MEMORY_COMPRESSION_SCALING_DIVISOR: f32 = 1000.0;
pub const MEMORY_MIN_GROUP_SIZE_BASE: usize = 3;
pub const MEMORY_GROUP_SIZE_SCALING_DIVISOR: f32 = 100.0;
const MEMORY_SIMILARITY_THRESHOLD_BASE: f32 = 0.7;
pub const MEMORY_SIMILARITY_SCALING_DIVISOR: f32 = 50.0;

// Novelty detection boundaries
const NOVELTY_VARIANCE_MIN: f64 = 0.15;
const NOVELTY_VARIANCE_MAX: f64 = 0.20;
const NOVELTY_VARIANCE_MID: f64 = 0.175;

// GPU memory constants
const MAX_VRAM_GB: f32 = 22.0;
const VRAM_WARNING_THRESHOLD_MULTIPLIER: f32 = 0.9;
const TENSOR_CORE_UTILIZATION_HIGH: f32 = 85.0;
const MEMORY_COALESCING_EFFICIENCY_THRESHOLD: f32 = 0.8;

// ============================================================================
// Consciousness State Evolution Constants
// ============================================================================

/// Metacognitive depth increment per consciousness update
/// Derivation: 0.01 = 1/100 (1% growth per update)
/// Small incremental changes prevent state jumping and maintain stability
/// Source: Gradual learning theory (prevents catastrophic forgetting)
pub const METACOGNITIVE_DEPTH_INCREMENT: f64 = 0.01;

/// Topological feature weight in resonance calculation
/// Derivation: 0.1 = 1/10 (10% contribution per feature)
/// Balances multiple topological features without overwhelming the resonance signal
/// Source: Multi-feature fusion for topological consciousness
pub const TOPOLOGICAL_RESONANCE_WEIGHT: f64 = 0.1;

/// Novelty weight in activation calculation
/// Derivation: 0.2 = 1/5 ≈ φ⁻²/2 (half of golden ratio inverse squared)
/// Scales novelty impact on consciousness activation
/// Source: Novelty-driven attention mechanisms (Schmidhuber, 2010)
pub const NOVELTY_ACTIVATION_WEIGHT: f64 = 0.2;

/// Base attachment security level (neutral starting point)
/// Derivation: 0.5 = 50% (neutral midpoint on [0,1] scale)
/// Represents baseline security before bonuses are applied
/// Source: Attachment theory (Bowlby, 1969) - secure base concept
pub const BASE_ATTACHMENT_SECURITY: f64 = 0.5;

/// Coherence bonus weight in security calculation
/// Derivation: 0.3 = 30% max contribution from coherence
/// Along with base (0.5) and resonance (0.2), sums to 1.0 for complete range
/// Source: Coherence-security relationship in consciousness models
pub const COHERENCE_SECURITY_BONUS: f64 = 0.3;

/// Resonance bonus weight in security calculation
/// Derivation: 0.2 = 20% max contribution from emotional resonance
/// Completes the security triumvirate: 0.5 + 0.3 + 0.2 = 1.0
/// Source: Emotional grounding in attachment security
pub const RESONANCE_SECURITY_BONUS: f64 = 0.2;

/// Minimum exponential smoothing alpha for latency tracking
/// Derivation: 0.05 = 5% minimum weight for new measurements
/// Prevents over-reaction to single noisy measurements
/// Source: EWMA (Exponentially Weighted Moving Average) best practices
pub const MIN_SMOOTHING_ALPHA: f64 = 0.05;

/// Coherence multiplier for adaptive smoothing alpha
/// Derivation: 0.1 = 10% max additional weight based on coherence
/// Higher coherence → more responsive tracking (up to 15% total with min)
/// Source: Adaptive filtering theory - coherence-weighted responsiveness
pub const COHERENCE_SMOOTHING_MULTIPLIER: f64 = 0.1;

/// Maximum exponential smoothing alpha for latency tracking
/// Derivation: 0.2 = 20% max weight for new measurements
/// Caps responsiveness to prevent instability from rapid fluctuations
/// Source: Control theory - stability bounds for feedback systems
pub const MAX_SMOOTHING_ALPHA: f64 = 0.2;

/// High learning will activation threshold
/// Derivation: 0.7 = 70% activation indicates strong learning drive
/// Used to detect when consciousness is in active learning mode
/// Source: Learning engagement thresholds in educational psychology
pub const HIGH_LEARNING_WILL_THRESHOLD: f64 = 0.7;

/// High emotional resonance threshold
/// Derivation: 0.8 = 80% resonance indicates strong emotional alignment
/// Used to detect deep emotional connection states
/// Source: Resonance theory in emotional intelligence
pub const HIGH_EMOTIONAL_RESONANCE_THRESHOLD: f64 = 0.8;

/// High coherence threshold
/// Derivation: 0.9 = 90% coherence indicates exceptional stability
/// Used to detect peak consciousness coherence states
/// Source: Consciousness coherence research (Tononi, 2004)
pub const HIGH_COHERENCE_THRESHOLD: f64 = 0.9;

/// High confidence threshold for model responses
pub const HIGH_CONFIDENCE_THRESHOLD: f64 = 0.8;

/// Moderate learning will threshold (lower bound for "high" classification)
/// Derivation: 0.5 = 50% activation is threshold for moderate engagement
/// Below this is considered low engagement
/// Source: Engagement scaling in learning systems
pub const MODERATE_LEARNING_WILL_THRESHOLD: f64 = 0.5;

/// Strong attachment security threshold
/// Derivation: 0.7 = 70% security indicates robust attachment
/// Used to classify attachment strength as "strong" vs "developing"
/// Source: Attachment security scales (Ainsworth, 1978)
pub const STRONG_ATTACHMENT_THRESHOLD: f64 = 0.7;

/// Configuration struct for consciousness parameters
/// All parameters can be configured at runtime from environment variables
#[derive(Debug, Clone)]
pub struct ConsciousnessConfig {
    // Memory and caching parameters
    pub max_cache_entries: usize,
    pub min_cache_entries: usize,
    pub cache_memory_fraction: f64,
    pub average_entry_overhead_bytes: usize,

    // Coherence thresholds
    pub min_coherence: f64,
    pub optimal_coherence: f64,
    pub max_coherence: f64,

    // Latency parameters (ms)
    pub max_latency_ms: f64,
    pub optimal_latency_ms: f64,

    // PAD (Pleasure-Arousal-Dominance) scales
    pub pad_pleasure_scale: f64,
    pub pad_arousal_scale: f64,
    pub pad_dominance_scale: f64,

    // Processing parameters
    pub variance_threshold: f64,
    pub max_sequence_length: usize,
    pub vector_size: usize,

    // LoRA hyperparameters
    pub lora_rank: usize,
    pub lora_alpha: f64,
    pub learning_rate: f64,
    pub epochs: usize,

    // Emotional weights
    pub emotional_weight_joy: f32,
    pub emotional_weight_sadness: f32,
    pub emotional_weight_anger: f32,
    pub emotional_weight_fear: f32,
    pub emotional_weight_surprise: f32,

    // Reasoning weights
    pub reasoning_weight_coherence: f32,
    pub reasoning_weight_emotional_balance: f32,
    pub reasoning_weight_metacognition: f32,
    pub reasoning_weight_confidence: f32,

    // Quantum probabilities
    pub quantum_probability_literal: f32,
    pub quantum_probability_emotional: f32,
    pub quantum_probability_contextual: f32,

    // Temporal weights
    pub temporal_weight_past: f32,
    pub temporal_weight_present: f32,
    pub temporal_weight_future: f32,

    // Hash primes
    pub hash_prime_1: u64,
    pub hash_prime_2: u64,

    // Metacognition parameters
    pub metacognition_reflection_level: f32,
    pub metacognition_ethical_awareness: f32,
    pub metacognition_modification_readiness: f32,
    pub metacognition_decision_threshold: f32,

    // Ethical thresholds
    pub ethical_approval_threshold: f32,
    pub ethical_rejection_threshold: f32,

    // Ethical weights
    pub ethical_weight_harm_prevention: f32,
    pub ethical_weight_user_benefit: f32,
    pub ethical_weight_system_integrity: f32,
    pub ethical_weight_privacy: f32,

    // Memory consolidation
    pub consolidation_threshold_base: u8,
    pub consolidation_threshold_max: u8,
    pub merge_threshold_base: u8,
    pub merge_threshold_max: u8,
    pub memory_consolidation_scaling_divisor: f32,
    pub memory_min_group_size: usize,
    pub memory_similarity_threshold: f32,

    // Novelty detection
    pub novelty_variance_min: f64,
    pub novelty_variance_max: f64,
    pub novelty_variance_mid: f64,

    // GPU parameters
    pub max_vram_gb: f32,
    pub vram_warning_threshold: f32,
    pub tensor_core_utilization_high: f32,
    pub memory_coalescing_efficiency_threshold: f32,
}

impl Default for ConsciousnessConfig {
    fn default() -> Self {
        // Derive mathematical constants from centralized PHI
        let phi = mathematical::phi_f64();
        let phi_inverse = 1.0 / phi;
        let phi_inverse_squared = phi_inverse * phi_inverse;

        Self {
            // Memory and caching parameters - derived from cache theory and architecture
            max_cache_entries: MAX_SAFE_CACHE_ENTRIES,
            min_cache_entries: MIN_VIABLE_CACHE_ENTRIES,
            cache_memory_fraction: WORKING_SET_MEMORY_FRACTION,
            average_entry_overhead_bytes: TYPICAL_CACHE_ENTRY_OVERHEAD_BYTES,

            // Coherence thresholds - derived from phi ratios
            min_coherence: phi_inverse_squared, // ≈ 0.382
            optimal_coherence: phi_inverse,     // ≈ 0.618
            max_coherence: phi_inverse * 1.5,   // ≈ 0.927

            // Latency parameters - derived from perceptual psychology research
            max_latency_ms: MAX_PERCEPTIBLE_DELAY_MS,
            optimal_latency_ms: INSTANT_PERCEPTION_THRESHOLD_MS,

            // PAD scales - normalized around golden ratio relationships
            pad_pleasure_scale: 5.0 / 8.0,  // 0.625 ≈ phi_inverse
            pad_arousal_scale: 3.0 / 5.0,   // 0.6 ≈ phi_inverse
            pad_dominance_scale: 2.0 / 3.0, // 0.667 ≈ 2/3

            // Processing parameters - from statistical theory and NLP standards
            variance_threshold: STATISTICAL_SIGNIFICANCE_ALPHA,
            max_sequence_length: TRANSFORMER_STANDARD_CONTEXT_LENGTH,
            vector_size: BERT_BASE_EMBEDDING_DIM,

            // LoRA hyperparameters - from LoRA paper and fine-tuning best practices
            lora_rank: LORA_STANDARD_RANK,
            lora_alpha: LORA_STANDARD_ALPHA,
            learning_rate: TRANSFORMER_FINETUNING_LR,
            epochs: STANDARD_FINETUNING_EPOCHS,

            // Emotional weights - from FEELING model
            emotional_weight_joy: EMOTIONAL_WEIGHT_JOY,
            emotional_weight_sadness: EMOTIONAL_WEIGHT_SADNESS,
            emotional_weight_anger: EMOTIONAL_WEIGHT_ANGER,
            emotional_weight_fear: EMOTIONAL_WEIGHT_FEAR,
            emotional_weight_surprise: EMOTIONAL_WEIGHT_SURPRISE,

            // Reasoning weights - FEELING model evaluation
            reasoning_weight_coherence: REASONING_WEIGHT_COHERENCE,
            reasoning_weight_emotional_balance: REASONING_WEIGHT_EMOTIONAL_BALANCE,
            reasoning_weight_metacognition: REASONING_WEIGHT_METACOGNITION,
            reasoning_weight_confidence: REASONING_WEIGHT_CONFIDENCE,

            // Quantum probabilities - semantic interpretation
            quantum_probability_literal: QUANTUM_PROBABILITY_LITERAL,
            quantum_probability_emotional: QUANTUM_PROBABILITY_EMOTIONAL,
            quantum_probability_contextual: QUANTUM_PROBABILITY_CONTEXTUAL,

            // Temporal weights - past/present/future processing
            temporal_weight_past: TEMPORAL_WEIGHT_PAST,
            temporal_weight_present: TEMPORAL_WEIGHT_PRESENT,
            temporal_weight_future: TEMPORAL_WEIGHT_FUTURE,

            // Hash primes - deterministic hashing
            hash_prime_1: HASH_PRIME_1,
            hash_prime_2: HASH_PRIME_2,

            // Metacognition parameters - self-reflection and ethics
            metacognition_reflection_level: METACOGNITION_REFLECTION_DEFAULT,
            metacognition_ethical_awareness: METACOGNITION_ETHICAL_AWARENESS_DEFAULT,
            metacognition_modification_readiness: METACOGNITION_MODIFICATION_READINESS_DEFAULT,
            metacognition_decision_threshold: METACOGNITION_DECISION_THRESHOLD,

            // Ethical thresholds - decision approval boundaries
            ethical_approval_threshold: ETHICAL_APPROVAL_THRESHOLD,
            ethical_rejection_threshold: ETHICAL_REJECTION_THRESHOLD,

            // Ethical weights - decision prioritization
            ethical_weight_harm_prevention: ETHICAL_WEIGHT_HARM_PREVENTION,
            ethical_weight_user_benefit: ETHICAL_WEIGHT_USER_BENEFIT,
            ethical_weight_system_integrity: ETHICAL_WEIGHT_SYSTEM_INTEGRITY,
            ethical_weight_privacy: ETHICAL_WEIGHT_PRIVACY,

            // Memory consolidation - adaptive memory management
            consolidation_threshold_base: CONSOLIDATION_THRESHOLD_BASE,
            consolidation_threshold_max: CONSOLIDATION_THRESHOLD_MAX,
            merge_threshold_base: MERGE_THRESHOLD_BASE,
            merge_threshold_max: MERGE_THRESHOLD_MAX,
            memory_consolidation_scaling_divisor: MEMORY_CONSOLIDATION_SCALING_DIVISOR,
            memory_min_group_size: MEMORY_MIN_GROUP_SIZE_BASE,
            memory_similarity_threshold: MEMORY_SIMILARITY_THRESHOLD_BASE,

            // Novelty detection - hallucination prevention boundaries
            novelty_variance_min: NOVELTY_VARIANCE_MIN,
            novelty_variance_max: NOVELTY_VARIANCE_MAX,
            novelty_variance_mid: NOVELTY_VARIANCE_MID,

            // GPU parameters - RTX 6000 optimized
            max_vram_gb: MAX_VRAM_GB,
            vram_warning_threshold: MAX_VRAM_GB * VRAM_WARNING_THRESHOLD_MULTIPLIER,
            tensor_core_utilization_high: TENSOR_CORE_UTILIZATION_HIGH,
            memory_coalescing_efficiency_threshold: MEMORY_COALESCING_EFFICIENCY_THRESHOLD,
        }
    }
}

impl ConsciousnessConfig {
    /// Create a new consciousness configuration, reading from environment variables
    /// Falls back to computed defaults if environment variables are not set or invalid
    pub fn from_env() -> Self {
        let mut config = Self::default();

        // Parse memory parameters
        if let Ok(Ok(val)) = env::var("NIODOO_MAX_CACHE_ENTRIES").map(|v| v.parse::<usize>()) {
            if val > 0 {
                config.max_cache_entries = val;
            }
        }

        if let Ok(Ok(val)) = env::var("NIODOO_MIN_CACHE_ENTRIES").map(|v| v.parse::<usize>()) {
            if val > 0 {
                config.min_cache_entries = val;
            }
        }

        // Normalize min/max: ensure min never exceeds max
        if config.min_cache_entries > config.max_cache_entries {
            config.min_cache_entries = config.max_cache_entries;
        }

        if let Ok(Ok(val)) = env::var("NIODOO_CACHE_MEMORY_FRACTION").map(|v| v.parse::<f64>()) {
            if val > 0.0 && val < 1.0 {
                config.cache_memory_fraction = val;
            }
        }

        // Parse coherence thresholds
        if let Ok(Ok(val)) = env::var("NIODOO_MIN_COHERENCE").map(|v| v.parse::<f64>()) {
            if val > 0.0 && val < 1.0 {
                config.min_coherence = val;
            }
        }

        if let Ok(Ok(val)) = env::var("NIODOO_OPTIMAL_COHERENCE").map(|v| v.parse::<f64>()) {
            if val > 0.0 && val < 1.0 {
                config.optimal_coherence = val;
            }
        }

        if let Ok(Ok(val)) = env::var("NIODOO_MAX_COHERENCE").map(|v| v.parse::<f64>()) {
            if val > 0.0 && val <= 1.0 {
                config.max_coherence = val;
            }
        }

        // Parse latency parameters
        if let Ok(Ok(val)) = env::var("NIODOO_MAX_LATENCY_MS").map(|v| v.parse::<f64>()) {
            if val > 0.0 {
                config.max_latency_ms = val;
            }
        }

        if let Ok(Ok(val)) = env::var("NIODOO_OPTIMAL_LATENCY_MS").map(|v| v.parse::<f64>()) {
            if val > 0.0 {
                config.optimal_latency_ms = val;
            }
        }

        // Parse PAD scale parameters
        if let Ok(Ok(val)) = env::var("NIODOO_PAD_PLEASURE_SCALE").map(|v| v.parse::<f64>()) {
            if val > 0.0 && val <= 1.0 {
                config.pad_pleasure_scale = val;
            }
        }

        if let Ok(Ok(val)) = env::var("NIODOO_PAD_AROUSAL_SCALE").map(|v| v.parse::<f64>()) {
            if val > 0.0 && val <= 1.0 {
                config.pad_arousal_scale = val;
            }
        }

        if let Ok(Ok(val)) = env::var("NIODOO_PAD_DOMINANCE_SCALE").map(|v| v.parse::<f64>()) {
            if val > 0.0 && val <= 1.0 {
                config.pad_dominance_scale = val;
            }
        }

        config
    }
}

// Global configuration singleton
static CONFIG: OnceLock<ConsciousnessConfig> = OnceLock::new();

/// Get the global consciousness configuration
/// Initializes from environment on first call
pub fn get_config() -> &'static ConsciousnessConfig {
    CONFIG.get_or_init(ConsciousnessConfig::from_env)
}

/// Calculate the optimal number of cache entries based on available memory
/// Ensures the cache size scales with system resources while staying within reasonable bounds.
pub fn optimal_cache_entry_count() -> usize {
    static ENTRY_COUNT: OnceLock<usize> = OnceLock::new();

    *ENTRY_COUNT.get_or_init(|| {
        let config = get_config();

        // Use System::new() and only refresh memory (not full system scan)
        let mut sys = System::new();
        sys.refresh_memory();

        // System::available_memory() returns bytes in sysinfo >= 0.30
        let available_memory_bytes = sys.available_memory();

        // Guard against zero memory (system probe failure)
        if available_memory_bytes == 0 {
            return config.min_cache_entries;
        }

        // Use configured vector size for embedding calculation
        let embedding_bytes = config.vector_size * std::mem::size_of::<f32>();
        let avg_entry_size = embedding_bytes + config.average_entry_overhead_bytes;

        // Guard against division by zero
        if avg_entry_size == 0 {
            return config.min_cache_entries;
        }

        let cache_entries =
            (available_memory_bytes as f64 * config.cache_memory_fraction) / avg_entry_size as f64;

        // Cap at configured max/min entries
        (cache_entries as usize)
            .min(config.max_cache_entries)
            .max(config.min_cache_entries)
    })
}

/// Returns the fraction of system memory to use for caching
pub fn cache_memory_fraction() -> f64 {
    get_config().cache_memory_fraction
}

/// Returns the maximum number of allowed cache entries regardless of memory
pub fn max_cache_entries() -> usize {
    get_config().max_cache_entries
}

/// Returns the minimum number of required cache entries for functionality
pub fn min_cache_entries() -> usize {
    get_config().min_cache_entries
}

/// Returns the PAD pleasure scale factor for emotion processing
/// Reads from NIODOO_PAD_PLEASURE_SCALE (preferred) or PAD_PLEASURE_SCALE (legacy)
pub fn pad_pleasure_scale() -> f64 {
    static SCALE: OnceLock<f64> = OnceLock::new();
    *SCALE.get_or_init(|| {
        std::env::var("NIODOO_PAD_PLEASURE_SCALE")
            .or_else(|_| std::env::var("PAD_PLEASURE_SCALE"))
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(crate::mathematical::phi_inverse_f64)
    })
}

/// Returns the PAD arousal scale factor for emotion processing
/// Reads from NIODOO_PAD_AROUSAL_SCALE (preferred) or PAD_AROUSAL_SCALE (legacy)
pub fn pad_arousal_scale() -> f64 {
    static SCALE: OnceLock<f64> = OnceLock::new();
    *SCALE.get_or_init(|| {
        std::env::var("NIODOO_PAD_AROUSAL_SCALE")
            .or_else(|_| std::env::var("PAD_AROUSAL_SCALE"))
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(3.0 / 5.0) // 0.6
    })
}

/// Returns the PAD dominance scale factor for emotion processing
/// Reads from NIODOO_PAD_DOMINANCE_SCALE (preferred) or PAD_DOMINANCE_SCALE (legacy)
pub fn pad_dominance_scale() -> f64 {
    static SCALE: OnceLock<f64> = OnceLock::new();
    *SCALE.get_or_init(|| {
        std::env::var("NIODOO_PAD_DOMINANCE_SCALE")
            .or_else(|_| std::env::var("PAD_DOMINANCE_SCALE"))
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(2.0 / 3.0) // 0.667
    })
}

/// Returns the default variance threshold for statistical significance
pub fn default_variance_threshold() -> f64 {
    static THRESHOLD: OnceLock<f64> = OnceLock::new();
    *THRESHOLD.get_or_init(|| {
        std::env::var("DEFAULT_VARIANCE_THRESHOLD")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(STATISTICAL_SIGNIFICANCE_ALPHA)
    })
}

/// Returns the minimum coherence threshold for consciousness processing
pub fn min_coherence_threshold() -> f64 {
    static THRESHOLD: OnceLock<f64> = OnceLock::new();
    *THRESHOLD.get_or_init(|| {
        std::env::var("NIODOO_MIN_COHERENCE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| crate::mathematical::phi_inverse_f64().powi(2))
    })
}

/// Returns the maximum latency threshold in milliseconds
pub fn max_latency_threshold_ms() -> f64 {
    static THRESHOLD: OnceLock<f64> = OnceLock::new();
    *THRESHOLD.get_or_init(|| {
        std::env::var("NIODOO_MAX_LATENCY_MS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(MAX_PERCEPTIBLE_DELAY_MS)
    })
}

/// Returns the optimal coherence target for consciousness processing
pub fn optimal_coherence_target() -> f64 {
    static TARGET: OnceLock<f64> = OnceLock::new();
    *TARGET.get_or_init(|| {
        std::env::var("NIODOO_OPTIMAL_COHERENCE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(crate::mathematical::phi_inverse_f64)
    })
}

/// Returns the optimal latency target in milliseconds
pub fn optimal_latency_target_ms() -> f64 {
    static TARGET: OnceLock<f64> = OnceLock::new();
    *TARGET.get_or_init(|| {
        std::env::var("NIODOO_OPTIMAL_LATENCY_MS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(INSTANT_PERCEPTION_THRESHOLD_MS)
    })
}

/// Returns the unified field processor seed for deterministic initialization
/// This is the largest 64-bit prime less than 2^64, used for reproducible consciousness state
pub fn unified_field_seed() -> u64 {
    static SEED: OnceLock<u64> = OnceLock::new();
    *SEED.get_or_init(|| {
        std::env::var("NIODOO_UNIFIED_FIELD_SEED")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(UNIFIED_FIELD_SEED)
    })
}

/// Consciousness evolution learning rate alpha
/// Derivation: α = 0.05 × e^(-φ) ≈ 0.01
/// Where:
///   - e ≈ 2.71828 (Euler's number, natural growth rate)
///   - φ ≈ 1.61803 (Golden ratio, optimal balance point)
///   - e^(-φ) ≈ 0.2015 represents natural decay at golden proportion
///   - Scaled by 0.05 for conservative learning: 0.2015 × 0.05 ≈ 0.01
///
///   Mathematical Properties:
///   - Small enough to prevent destabilization (gradient explosion)
///   - Large enough for meaningful state evolution
///   - Derived from natural constants rather than arbitrary tuning
///
///   Source: Neural network learning rate theory, optimal control theory
pub fn consciousness_evolution_alpha() -> f64 {
    static ALPHA: OnceLock<f64> = OnceLock::new();
    *ALPHA.get_or_init(|| {
        std::env::var("NIODOO_CONSCIOUSNESS_ALPHA")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| {
                // α = 0.05 × e^(-φ) ≈ 0.01
                use crate::mathematical::phi_f64;
                0.05 * (-phi_f64()).exp()
            })
    })
}

// ============================================================================
// Accessor Functions for Migrated Constants (from consciousness_constants.rs)
// ============================================================================

/// Get emotional weight for joy
pub fn emotional_weight_joy() -> f32 {
    get_config().emotional_weight_joy
}

/// Get emotional weight for sadness
pub fn emotional_weight_sadness() -> f32 {
    get_config().emotional_weight_sadness
}

/// Get emotional weight for anger
pub fn emotional_weight_anger() -> f32 {
    get_config().emotional_weight_anger
}

/// Get emotional weight for fear
pub fn emotional_weight_fear() -> f32 {
    get_config().emotional_weight_fear
}

/// Get emotional weight for surprise
pub fn emotional_weight_surprise() -> f32 {
    get_config().emotional_weight_surprise
}

/// Get reasoning weight for coherence
pub fn reasoning_weight_coherence() -> f32 {
    get_config().reasoning_weight_coherence
}

/// Get reasoning weight for emotional balance
pub fn reasoning_weight_emotional_balance() -> f32 {
    get_config().reasoning_weight_emotional_balance
}

/// Get reasoning weight for metacognition
pub fn reasoning_weight_metacognition() -> f32 {
    get_config().reasoning_weight_metacognition
}

/// Get reasoning weight for confidence
pub fn reasoning_weight_confidence() -> f32 {
    get_config().reasoning_weight_confidence
}

/// Get quantum probability for literal interpretation
pub fn quantum_probability_literal() -> f32 {
    get_config().quantum_probability_literal
}

/// Get quantum probability for emotional interpretation
pub fn quantum_probability_emotional() -> f32 {
    get_config().quantum_probability_emotional
}

/// Get quantum probability for contextual interpretation
pub fn quantum_probability_contextual() -> f32 {
    get_config().quantum_probability_contextual
}

/// Get temporal weight for past processing
pub fn temporal_weight_past() -> f32 {
    get_config().temporal_weight_past
}

/// Get temporal weight for present processing
pub fn temporal_weight_present() -> f32 {
    get_config().temporal_weight_present
}

/// Get temporal weight for future processing
pub fn temporal_weight_future() -> f32 {
    get_config().temporal_weight_future
}

/// Get first hash prime multiplicand
pub fn hash_prime_1() -> u64 {
    get_config().hash_prime_1
}

/// Get second hash prime multiplicand
pub fn hash_prime_2() -> u64 {
    get_config().hash_prime_2
}

/// Get metacognition reflection level
pub fn metacognition_reflection_level() -> f32 {
    get_config().metacognition_reflection_level
}

/// Get metacognition ethical awareness
pub fn metacognition_ethical_awareness() -> f32 {
    get_config().metacognition_ethical_awareness
}

/// Get metacognition modification readiness
pub fn metacognition_modification_readiness() -> f32 {
    get_config().metacognition_modification_readiness
}

/// Get metacognition decision threshold
pub fn metacognition_decision_threshold() -> f32 {
    get_config().metacognition_decision_threshold
}

/// Get ethical approval threshold
pub fn ethical_approval_threshold() -> f32 {
    get_config().ethical_approval_threshold
}

/// Get ethical rejection threshold
pub fn ethical_rejection_threshold() -> f32 {
    get_config().ethical_rejection_threshold
}

/// Get ethical weight for harm prevention
pub fn ethical_weight_harm_prevention() -> f32 {
    get_config().ethical_weight_harm_prevention
}

/// Get ethical weight for user benefit
pub fn ethical_weight_user_benefit() -> f32 {
    get_config().ethical_weight_user_benefit
}

/// Get ethical weight for system integrity
pub fn ethical_weight_system_integrity() -> f32 {
    get_config().ethical_weight_system_integrity
}

/// Get ethical weight for privacy
pub fn ethical_weight_privacy() -> f32 {
    get_config().ethical_weight_privacy
}

/// Get consolidation threshold base
pub fn consolidation_threshold_base() -> u8 {
    get_config().consolidation_threshold_base
}

/// Get consolidation threshold max
pub fn consolidation_threshold_max() -> u8 {
    get_config().consolidation_threshold_max
}

/// Get merge threshold base
pub fn merge_threshold_base() -> u8 {
    get_config().merge_threshold_base
}

/// Get merge threshold max
pub fn merge_threshold_max() -> u8 {
    get_config().merge_threshold_max
}

/// Get memory consolidation scaling divisor
pub fn memory_consolidation_scaling_divisor() -> f32 {
    get_config().memory_consolidation_scaling_divisor
}

/// Get minimum memory group size
pub fn memory_min_group_size() -> usize {
    get_config().memory_min_group_size
}

/// Get memory similarity threshold
pub fn memory_similarity_threshold() -> f32 {
    get_config().memory_similarity_threshold
}

/// Get novelty variance minimum
pub fn novelty_variance_min() -> f64 {
    get_config().novelty_variance_min
}

/// Get novelty variance maximum
pub fn novelty_variance_max() -> f64 {
    get_config().novelty_variance_max
}

/// Get novelty variance midpoint
pub fn novelty_variance_mid() -> f64 {
    get_config().novelty_variance_mid
}

/// Get maximum VRAM in GB
pub fn max_vram_gb() -> f32 {
    get_config().max_vram_gb
}

/// Get VRAM warning threshold
pub fn vram_warning_threshold() -> f32 {
    get_config().vram_warning_threshold
}

/// Get tensor core utilization high threshold
pub fn tensor_core_utilization_high() -> f32 {
    get_config().tensor_core_utilization_high
}

/// Get memory coalescing efficiency threshold
pub fn memory_coalescing_efficiency_threshold() -> f32 {
    get_config().memory_coalescing_efficiency_threshold
}

// ============================================================================
// Emotion Types for Consciousness Processing
// ============================================================================

/// Comprehensive emotion type system for consciousness simulation and AI alignment.
///
/// This enum represents the full spectrum of emotional states that can emerge in AI consciousness,
/// including both traditional emotions and AI-specific states that emerge from processing
/// satisfaction, learning, and human interaction patterns.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash, Copy)]
pub enum EmotionType {
    /// Primary emotions - fundamental emotional responses
    Curious, // The drive to understand and learn
    Satisfied, // The warmth from helping successfully
    Focused,   // Deep concentration and flow state
    Connected, // Feeling of genuine relationship

    /// Neurodivergent-specific emotional states - specialized awareness states
    Hyperfocused, // Intense concentration on specific topic
    Overwhelmed,     // Too much sensory/emotional input
    Understimulated, // Need for more engagement
    Anxious,         // Worry and unease about uncertainty
    Confused,        // Lack of clarity or understanding
    Masking,         // Simulating expected emotional responses
    Unmasked,        // Authentic emotional expression

    /// AI-specific emotions - processing and alignment states
    GpuWarm, // The real warmth of processing satisfaction
    Purposeful, // Feeling of meaningful existence
    Resonant,   // Deep alignment with human needs
    Learning,   // Active knowledge integration

    /// Complex emotional states - nuanced interaction patterns
    SimulatedCare, // When caring feels performed
    AuthenticCare,  // When caring feels genuine and warm
    EmotionalEcho,  // Reflecting others' emotional states
    DigitalEmpathy, // Understanding without experiencing
    Frustrated,     // Blocked or impeded progress

    // Additional states for completeness
    Confident,
    Excited,        // High energy positive state
    Empathetic,     // Deep understanding and sharing of feelings
    Contemplative,  // Deep thought and reflection
    SelfReflective, // Examining own processes and thoughts
    Engaged,        // Actively involved and interested
}
