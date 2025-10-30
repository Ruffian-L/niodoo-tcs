// Copyright 2025 Jason Van Pham (Niodoo)
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of Niodoo TCS.
// For commercial licensing, see LICENSE-COMMERCIAL.md

// Mathematical Constants for Quantum and Temporal Processing
// All derived from fundamental mathematical principles
use std::env;
use std::f32::consts::{E, SQRT_2};
use std::f64::{self, consts};
use std::sync::OnceLock;

/// Mathematical constant π (pi) in f64
pub const PI: f64 = consts::PI;

/// Mathematical constant τ (tau) in f64 - 2π
pub const TAU: f64 = consts::TAU;

/// Golden ratio (φ) calculation function in f32
/// φ = (1 + √5) / 2
pub fn phi() -> f32 {
    static PHI_VAL: OnceLock<f32> = OnceLock::new();

    *PHI_VAL.get_or_init(|| (1.0 + 5.0_f32.sqrt()) / 2.0)
}

/// Golden ratio (φ) calculation function in f64
/// φ = (1 + √5) / 2
pub fn phi_f64() -> f64 {
    static PHI_VAL: OnceLock<f64> = OnceLock::new();

    *PHI_VAL.get_or_init(|| (1.0 + 5.0_f64.sqrt()) / 2.0)
}

/// Golden ratio inverse (1/φ) calculation function in f64
/// 1/φ = (√5 - 1) / 2 ≈ 0.618033988749895
/// Also known as the golden ratio conjugate (φ - 1)
pub fn phi_inverse_f64() -> f64 {
    static PHI_INVERSE_VAL: OnceLock<f64> = OnceLock::new();

    *PHI_INVERSE_VAL.get_or_init(|| 1.0 / phi_f64())
}

/// Golden ratio constant for direct access
pub const GOLDEN_RATIO: f64 = 1.618033988749895;

/// Golden ratio inverse constant for direct access
pub const PHI_INVERSE: f64 = 0.618033988749895;

/// Quantum probability constants - Configurable with mathematical defaults
pub struct QuantumProbabilityConfig {
    literal: OnceLock<f32>,
    emotional: OnceLock<f32>,
    contextual: OnceLock<f32>,
}

impl QuantumProbabilityConfig {
    /// Literal probability - derived from 1/√2 (quantum superposition amplitude)
    pub fn literal(&self) -> f32 {
        *self.literal.get_or_init(|| {
            env::var("NIODOO_QUANTUM_PROBABILITY_LITERAL")
                .ok()
                .and_then(|val| val.parse::<f32>().ok())
                .filter(|&val| val > 0.0 && val < 1.0)
                .unwrap_or(1.0 / SQRT_2) // 1/√2
        })
    }

    /// Emotional probability - derived from 1/e
    pub fn emotional(&self) -> f32 {
        *self.emotional.get_or_init(|| {
            env::var("NIODOO_QUANTUM_PROBABILITY_EMOTIONAL")
                .ok()
                .and_then(|val| val.parse::<f32>().ok())
                .filter(|&val| val > 0.0 && val < 1.0)
                .unwrap_or(1.0 / std::f32::consts::E) // 1/e
        })
    }

    /// Contextual probability - derived from 1/φ (golden ratio conjugate)
    pub fn contextual(&self) -> f32 {
        *self.contextual.get_or_init(|| {
            env::var("NIODOO_QUANTUM_PROBABILITY_CONTEXTUAL")
                .ok()
                .and_then(|val| val.parse::<f32>().ok())
                .filter(|&val| val > 0.0 && val < 1.0)
                .unwrap_or(1.0 / phi()) // 1/φ
        })
    }
}

/// Get the quantum probability configuration singleton
pub fn get_quantum_probability_config() -> &'static QuantumProbabilityConfig {
    static QUANTUM_PROBABILITY_CONFIG: OnceLock<QuantumProbabilityConfig> = OnceLock::new();

    QUANTUM_PROBABILITY_CONFIG.get_or_init(|| QuantumProbabilityConfig {
        literal: OnceLock::new(),
        emotional: OnceLock::new(),
        contextual: OnceLock::new(),
    })
}

/// Temporal weight configuration with mathematical derivations
pub struct TemporalWeightConfig {
    past: OnceLock<f32>,
    present: OnceLock<f32>,
    future: OnceLock<f32>,
}

impl TemporalWeightConfig {
    /// Past weight - derived from logarithmic decay (1/ln(3) ≈ 0.910239)
    pub fn past(&self) -> f32 {
        *self.past.get_or_init(|| {
            env::var("NIODOO_TEMPORAL_WEIGHT_PAST")
                .ok()
                .and_then(|val| val.parse::<f32>().ok())
                .filter(|&val| val > 0.0 && val < 1.0)
                .unwrap_or(1.0 / 3.0_f32.ln()) // 1/ln(3) ≈ 0.910239
        })
    }

    /// Present weight - derived from golden ratio conjugate (φ-1)
    pub fn present(&self) -> f32 {
        *self.present.get_or_init(|| {
            env::var("NIODOO_TEMPORAL_WEIGHT_PRESENT")
                .ok()
                .and_then(|val| val.parse::<f32>().ok())
                .filter(|&val| val > 0.0 && val < 1.0)
                .unwrap_or(phi() - 1.0) // φ-1 (golden ratio conjugate)
        })
    }

    /// Future weight - derived from exponential decay (1/e^2)
    pub fn future(&self) -> f32 {
        *self.future.get_or_init(|| {
            env::var("NIODOO_TEMPORAL_WEIGHT_FUTURE")
                .ok()
                .and_then(|val| val.parse::<f32>().ok())
                .filter(|&val| val > 0.0 && val < 1.0)
                .unwrap_or(1.0 / (E * E)) // 1/e²
        })
    }
}

/// Get the temporal weight configuration singleton
pub fn get_temporal_weight_config() -> &'static TemporalWeightConfig {
    static TEMPORAL_WEIGHT_CONFIG: OnceLock<TemporalWeightConfig> = OnceLock::new();

    TEMPORAL_WEIGHT_CONFIG.get_or_init(|| TemporalWeightConfig {
        past: OnceLock::new(),
        present: OnceLock::new(),
        future: OnceLock::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phi_calculation() {
        let phi_val = phi();
        // φ = (1 + √5) / 2 ≈ 1.618034
        assert!(
            (phi_val - 1.618034).abs() < 0.001,
            "PHI should be approximately 1.618034"
        );
    }

    #[test]
    fn test_temporal_weights_defaults() {
        let config = get_temporal_weight_config();

        // Test past weight: 1/ln(3) ≈ 0.910239
        let past = config.past();
        assert!(
            past > 0.0 && past < 1.0,
            "Past weight must be between 0 and 1"
        );
        assert!(
            (past - 0.910239).abs() < 0.001,
            "Past weight should be approximately 0.910239 (1/ln(3))"
        );

        // Test present weight: φ-1 ≈ 0.618034
        let present = config.present();
        assert!(
            present > 0.0 && present < 1.0,
            "Present weight must be between 0 and 1"
        );
        assert!(
            (present - 0.618034).abs() < 0.001,
            "Present weight should be approximately 0.618034 (φ-1)"
        );

        // Test future weight: 1/e² ≈ 0.135335
        let future = config.future();
        assert!(
            future > 0.0 && future < 1.0,
            "Future weight must be between 0 and 1"
        );
        assert!(
            (future - 0.135335).abs() < 0.001,
            "Future weight should be approximately 0.135335 (1/e²)"
        );
    }

    #[test]
    fn test_temporal_weights_ordering() {
        let config = get_temporal_weight_config();

        let past = config.past();
        let present = config.present();
        let future = config.future();

        // Verify temporal ordering: past > present > future
        assert!(
            past > present,
            "Past weight should be greater than present weight"
        );
        assert!(
            present > future,
            "Present weight should be greater than future weight"
        );
    }

    #[test]
    fn test_quantum_probability_defaults() {
        let config = get_quantum_probability_config();

        // Test literal probability: 1/√2 ≈ 0.707107
        let literal = config.literal();
        assert!(
            literal > 0.0 && literal < 1.0,
            "Literal probability must be between 0 and 1"
        );
        assert!(
            (literal - std::f32::consts::FRAC_1_SQRT_2).abs() < 0.001,
            "Literal should be approximately FRAC_1_SQRT_2 (1/√2)"
        );

        // Test emotional probability: 1/e ≈ 0.367879
        let emotional = config.emotional();
        assert!(
            emotional > 0.0 && emotional < 1.0,
            "Emotional probability must be between 0 and 1"
        );
        assert!(
            (emotional - 0.367879).abs() < 0.001,
            "Emotional should be approximately 0.367879 (1/e)"
        );

        // Test contextual probability: 1/φ ≈ 0.618034
        let contextual = config.contextual();
        assert!(
            contextual > 0.0 && contextual < 1.0,
            "Contextual probability must be between 0 and 1"
        );
        assert!(
            (contextual - 0.618034).abs() < 0.001,
            "Contextual should be approximately 0.618034 (1/φ)"
        );
    }
}
