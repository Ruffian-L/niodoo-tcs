// Copyright 2025 Jason Van Pham (Niodoo)
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of Niodoo TCS.
// For commercial licensing, see LICENSE-COMMERCIAL.md

// Processing Configuration Constants

/// L1 cache line size in bytes (typical for modern CPUs)
/// This is architecture-dependent but 64 bytes is standard for x86-64, ARM64
pub const CACHE_LINE_SIZE: usize = 64;

/// Calculate optimal batch size for a given type to fit within one cache line
///
/// Returns the number of elements of type T that fit in a cache line,
/// with a minimum of 1 to avoid zero-sized batches.
///
/// # Example
/// ```
/// let f32_batch_size = batch_size_for::<f32>(); // Returns 16 (64 / 4)
/// let f64_batch_size = batch_size_for::<f64>(); // Returns 8 (64 / 8)
/// ```
#[inline]
pub const fn batch_size_for<T>() -> usize {
    let element_size = std::mem::size_of::<T>();
    if element_size == 0 {
        return 1; // Handle zero-sized types
    }
    let batch = CACHE_LINE_SIZE / element_size;
    if batch == 0 {
        1
    } else {
        batch
    }
}

/// Default batch size for generic use cases
/// This is NOT optimized for cache alignment - it's a reasonable default.
/// For performance-critical code, use `batch_size_for::<T>()` with your specific type.
pub const DEFAULT_BATCH_SIZE: usize = 64;

/// Thread pool size
/// Defaults to number of logical CPU cores
pub fn optimal_thread_count() -> usize {
    num_cpus::get()
}

/// Processing queue capacity
/// Large enough to prevent blocking, small enough to avoid memory bloat
pub const QUEUE_CAPACITY: usize = 10000;

// Reasoning Weights - Derived from cognitive psychology and information theory
// Coherence weight - derived from golden ratio (φ - 1)
pub const REASONING_WEIGHT_COHERENCE: f32 = 0.618; // φ - 1 ≈ 1.618 - 1 = 0.618

// Emotional balance weight - derived from 1/e
pub const REASONING_WEIGHT_EMOTIONAL_BALANCE: f32 = 0.368; // 1/e

// Metacognition weight - derived from 1/PHI
pub const REASONING_WEIGHT_METACOGNITION: f32 = 0.618; // 1/PHI

// Confidence weight - derived from 1/PI
pub const REASONING_WEIGHT_CONFIDENCE: f32 = std::f32::consts::FRAC_1_PI; // 1/PI (exact)

// Hash computation constants - derived from prime numbers for good distribution
pub const HASH_PRIME_MULTIPLICAND_CHARS: usize = 31; // Prime number
pub const HASH_PRIME_MULTIPLICAND_LENGTH: usize = 17; // Prime number

// Input complexity threshold - derived from typical human attention span (500 words ≈ 3000 chars)
pub const INPUT_COMPLEXITY_THRESHOLD_LENGTH: usize = 500;

/// Input chunk size for processing
/// Derived from optimal reading chunk size (200 words)
pub const INPUT_COMPLEXITY_CHUNK_SIZE: usize = 200;

/// Default integration dimensions for consciousness processing
/// Derived from transformer architecture standards (BERT base = 896)
pub const DEFAULT_INTEGRATION_DIMS: usize = 896;

/// Default emotional dimensions for PAD processing
/// Based on Pleasure-Arousal-Dominance model (3 dimensions)
pub const DEFAULT_EMOTIONAL_DIMS: usize = 3;
