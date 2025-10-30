// Copyright 2025 Jason Van Pham (Niodoo)
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of Niodoo TCS.
// For commercial licensing, see LICENSE-COMMERCIAL.md

//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

//! # ⚠️ DEPRECATED - Constants Migrated to `constants_core`
//!
//! **This module is deprecated and will be removed in a future release.**
//!
//! All consciousness system constants have been migrated to the centralized
//! `constants_core` crate for better organization and configurability.
//!
//! ## Migration Guide
//!
//! ### Old (Deprecated):
//! ```rust,ignore
//! use crate::consciousness_constants::EMOTIONAL_WEIGHT_JOY;
//! ```
//!
//! ### New (Correct):
//! ```rust,ignore
//! use constants_core::consciousness::emotional_weight_joy;
//! let weight = emotional_weight_joy();
//! ```
//!
//! ## Benefits of New System:
//! - ✅ **Environment variable overrides**: All constants can be configured at runtime
//! - ✅ **Centralized management**: Single source of truth in `constants_core`
//! - ✅ **Type safety**: Accessed through functions, not raw constants
//! - ✅ **Documentation**: Better organized with research citations
//! - ✅ **Zero tolerance**: Follows "NO HARDCODING" project rules
//!
//! ## Available Constants in `constants_core::consciousness`:
//!
//! ### Emotional Weights:
//! - `emotional_weight_joy()`
//! - `emotional_weight_sadness()`
//! - `emotional_weight_anger()`
//! - `emotional_weight_fear()`
//! - `emotional_weight_surprise()`
//!
//! ### Reasoning Weights:
//! - `reasoning_weight_coherence()`
//! - `reasoning_weight_emotional_balance()`
//! - `reasoning_weight_metacognition()`
//! - `reasoning_weight_confidence()`
//!
//! ### Quantum Probabilities:
//! - `quantum_probability_literal()`
//! - `quantum_probability_emotional()`
//! - `quantum_probability_contextual()`
//!
//! ### Temporal Weights:
//! - `temporal_weight_past()`
//! - `temporal_weight_present()`
//! - `temporal_weight_future()`
//!
//! ### Hash Primes:
//! - `hash_prime_1()`
//! - `hash_prime_2()`
//!
//! ### Metacognition:
//! - `metacognition_reflection_level()`
//! - `metacognition_ethical_awareness()`
//! - `metacognition_modification_readiness()`
//! - `metacognition_decision_threshold()`
//!
//! ### Ethical Thresholds & Weights:
//! - `ethical_approval_threshold()`
//! - `ethical_rejection_threshold()`
//! - `ethical_weight_harm_prevention()`
//! - `ethical_weight_user_benefit()`
//! - `ethical_weight_system_integrity()`
//! - `ethical_weight_privacy()`
//!
//! ### Memory Consolidation:
//! - `consolidation_threshold_base()`
//! - `consolidation_threshold_max()`
//! - `merge_threshold_base()`
//! - `merge_threshold_max()`
//! - `memory_consolidation_scaling_divisor()`
//! - `memory_min_group_size()`
//! - `memory_similarity_threshold()`
//!
//! ### Novelty Detection:
//! - `novelty_variance_min()`
//! - `novelty_variance_max()`
//! - `novelty_variance_mid()`
//!
//! ### GPU Parameters:
//! - `max_vram_gb()`
//! - `vram_warning_threshold()`
//! - `tensor_core_utilization_high()`
//! - `memory_coalescing_efficiency_threshold()`
//!
//! ---
//!
//! **For full documentation, see: `constants_core/src/consciousness.rs`**

#![deprecated(
    since = "0.7.0",
    note = "Use `constants_core::consciousness` instead. See module docs for migration guide."
)]

// Re-export from constants_core for backward compatibility (temporary)
// Remove these re-exports in next major version
pub use constants_core::consciousness::*;
