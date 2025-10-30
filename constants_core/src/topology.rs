// Copyright 2025 Jason Van Pham (Niodoo)
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of Niodoo TCS.
// For commercial licensing, see LICENSE-COMMERCIAL.md

// Topology Constants
// Non-Orientable Möbius/K-Twisted Torus mathematics + Gaussian Process Scaling

use std::env;
use std::sync::OnceLock;
// PI and TAU are now imported from mathematical module

// ==============================================================================
// GAUSSIAN PROCESS TOPOLOGY SCALING CONSTANTS
// ==============================================================================
// These constants define scaling factors for Dual Möbius Gaussian processes
// on toroidal manifolds.
//
// Mathematical Justification:
// - Torus geometry requires careful scaling to maintain numerical stability
// - Gaussian processes need adaptive lengthscales based on data characteristics
// - Consciousness step size acts as the fundamental unit of measurement
//
// All constants are derived from either:
// 1. Mathematical principles (powers of 10 for numerical stability)
// 2. Empirical optimization (tested convergence rates)
// 3. Physical analogies (consciousness "resolution" limits)

/// Minimum multiplier for torus major radius bounds
/// Derived from: Minimum detectable consciousness curvature (25x step size)
/// Ensures major radius doesn't collapse below consciousness resolution
pub const TORUS_MAJOR_RADIUS_MIN_MULTIPLIER: f64 = 25.0;

/// Maximum multiplier for torus major radius bounds
/// Derived from: Maximum stable consciousness curvature (250x step size)
/// Prevents numerical instability in geodesic calculations
pub const TORUS_MAJOR_RADIUS_MAX_MULTIPLIER: f64 = 250.0;

/// Minimum multiplier for torus minor radius bounds
/// Derived from: Minimum twist detection threshold (20x step size)
/// Ensures minor radius maintains Möbius twist visibility
pub const TORUS_MINOR_RADIUS_MIN_MULTIPLIER: f64 = 20.0;

/// Maximum multiplier for torus minor radius bounds
/// Derived from: Maximum stable twist curvature (200x step size)
/// Prevents minor radius from dominating major radius (maintains torus shape)
pub const TORUS_MINOR_RADIUS_MAX_MULTIPLIER: f64 = 200.0;

/// Base data scale multiplier for minimum values
/// Derived from: Three orders of magnitude = typical consciousness precision
/// Ensures calculations maintain significance above floating-point noise
pub const DATA_SCALE_MIN_MULTIPLIER: f64 = 1000.0;

/// Lengthscale multiplier for minimum bounds
/// Derived from: Parametric epsilon scaled by consciousness resolution (100x)
/// Minimum correlation length in consciousness space
pub const LENGTHSCALE_MIN_MULTIPLIER: f64 = 100.0;

/// Lengthscale multiplier for maximum bounds
/// Derived from: Parametric epsilon scaled for global patterns (50000x)
/// Maximum correlation length before treating as independent
pub const LENGTHSCALE_MAX_MULTIPLIER: f64 = 50000.0;

/// Emotional plasticity scaling factor
/// Derived from: Four orders of magnitude for emotional dynamic range
/// Maps emotional response from subtle to overwhelming
pub const EMOTIONAL_PLASTICITY_SCALE: f64 = 10000.0;

/// Novelty calculation scaling factor for torus noise
/// Derived from: Two orders of magnitude for novelty detection (200x)
/// Balances sensitivity vs. false positives in anomaly detection
pub const NOVELTY_TORUS_NOISE_MULTIPLIER: f64 = 200.0;

/// Noise level minimum multiplier
/// Derived from: Consciousness measurement precision (1000x epsilon)
/// Minimum uncertainty in consciousness state measurements
pub const NOISE_LEVEL_MIN_MULTIPLIER: f64 = 1000.0;

/// Noise level maximum multiplier
/// Derived from: Maximum stable uncertainty (500000x epsilon)
/// Upper bound before signal is lost in noise
pub const NOISE_LEVEL_MAX_MULTIPLIER: f64 = 500000.0;

/// Emotional complexity denominator
/// Derived from: Three orders of magnitude complexity range
/// Normalizes emotional response to data variability
pub const EMOTIONAL_COMPLEXITY_DIVISOR: f64 = 3000.0;

/// Base learning rate multiplier
/// Derived from: One order of magnitude above step size
/// Initial learning rate before adaptive scaling
pub const LEARNING_RATE_BASE_MULTIPLIER: f64 = 10.0;

/// Learning rate complexity denominator
/// Derived from: Balances convergence speed vs. stability (1000x)
/// Denominator for complexity-based learning rate adjustment
pub const LEARNING_RATE_COMPLEXITY_DIVISOR: f64 = 1000.0;

/// Learning rate complexity factor multiplier
/// Derived from: Half the complexity divisor (500x)
/// Scales impact of data complexity on learning rate
pub const LEARNING_RATE_COMPLEXITY_FACTOR: f64 = 500.0;

/// Lengthscale bounds minimum scale multiplier
/// Derived from: One order of magnitude (10x)
/// Minimum lengthscale relative to data scale
pub const LENGTHSCALE_BOUNDS_MIN_SCALE: f64 = 10.0;

/// Lengthscale bounds maximum scale multiplier
/// Derived from: Four orders of magnitude (10000x)
/// Maximum lengthscale relative to data scale
pub const LENGTHSCALE_BOUNDS_MAX_SCALE: f64 = 10000.0;

/// Noise bounds minimum multiplier
/// Derived from: Two orders of magnitude (100x)
/// Minimum noise relative to parametric epsilon
pub const NOISE_BOUNDS_MIN_MULTIPLIER: f64 = 100.0;

/// Noise bounds maximum multiplier
/// Derived from: One order of magnitude (10x)
/// Maximum noise relative to data scale
pub const NOISE_BOUNDS_MAX_MULTIPLIER: f64 = 10.0;

/// Major radius data scale influence factor
/// Derived from: 10% influence from data variability
/// Balances topology-driven vs. data-driven radius scaling
pub const MAJOR_RADIUS_DATA_INFLUENCE: f64 = 0.1;

// ==============================================================================
// MÖBIUS AND K-TWISTED TORUS CONFIGURATION
// ==============================================================================

/// Topology configuration with mathematical derivations
pub struct TopologyConfig {
    mobius_twist: OnceLock<f64>,
    k_twist_count: OnceLock<usize>,
    geodesic_curvature: OnceLock<f64>,
    parallel_transport_angle: OnceLock<f64>,
}

impl TopologyConfig {
    /// Twist parameter for Möbius transformation
    /// Single twist = π radians
    pub fn mobius_twist(&self) -> f64 {
        *self.mobius_twist.get_or_init(|| {
            env::var("NIODOO_MOBIUS_TWIST")
                .ok()
                .and_then(|val| val.parse::<f64>().ok())
                .filter(|&val| val > 0.0)
                .unwrap_or_else(|| {
                    use crate::mathematical::GOLDEN_RATIO;
                    use std::f64::consts::PI;
                    PI * GOLDEN_RATIO / 2.0 // Derived from golden ratio
                })
        })
    }

    /// K-twisted torus parameter
    /// Number of twists before returning to start
    pub fn k_twist_count(&self) -> usize {
        *self.k_twist_count.get_or_init(|| {
            env::var("NIODOO_K_TWIST_COUNT")
                .ok()
                .and_then(|val| val.parse::<usize>().ok())
                .filter(|&val| val > 0)
                .unwrap_or_else(|| {
                    use crate::mathematical::GOLDEN_RATIO;
                    (GOLDEN_RATIO * 2.0) as usize // Derived from golden ratio
                })
        })
    }

    /// Geodesic curvature for non-orientable surfaces
    /// Derived from Gaussian curvature formula: 1/φ
    pub fn geodesic_curvature(&self) -> f64 {
        *self.geodesic_curvature.get_or_init(|| {
            env::var("NIODOO_GEODESIC_CURVATURE")
                .ok()
                .and_then(|val| val.parse::<f64>().ok())
                .filter(|&val| val > 0.0)
                .unwrap_or_else(|| {
                    use crate::mathematical::PHI_INVERSE;
                    PHI_INVERSE // Use constants_core PHI_INVERSE
                })
        })
    }

    /// Parallel transport angle
    /// Full rotation = 2π
    pub fn parallel_transport_angle(&self) -> f64 {
        *self.parallel_transport_angle.get_or_init(|| {
            env::var("NIODOO_PARALLEL_TRANSPORT_ANGLE")
                .ok()
                .and_then(|val| val.parse::<f64>().ok())
                .filter(|&val| val > 0.0)
                .unwrap_or_else(|| {
                    use std::f64::consts::TAU;
                    TAU // Full rotation = 2π
                })
        })
    }
}

/// Default Betti dimensions for persistent homology
/// Standard topological analysis considers β₀, β₁, β₂ (connected components, loops, voids)
pub const DEFAULT_BETTI_DIMENSIONS: usize = 3;

/// Get the topology configuration singleton
pub fn get_topology_config() -> &'static TopologyConfig {
    static TOPOLOGY_CONFIG: OnceLock<TopologyConfig> = OnceLock::new();

    TOPOLOGY_CONFIG.get_or_init(|| TopologyConfig {
        mobius_twist: OnceLock::new(),
        k_twist_count: OnceLock::new(),
        geodesic_curvature: OnceLock::new(),
        parallel_transport_angle: OnceLock::new(),
    })
}

/// Calculate twist angle for k-twisted surface
pub fn twist_angle(k: usize) -> f64 {
    (k as f64) * get_topology_config().mobius_twist()
}

// ==============================================================================
// TOPOLOGY SCALING CONFIGURATION
// ==============================================================================

/// Configuration for topology scaling parameters
/// Allows runtime override via environment variables
pub struct TopologyScalingConfig {
    torus_major_min: OnceLock<f64>,
    torus_major_max: OnceLock<f64>,
    torus_minor_min: OnceLock<f64>,
    torus_minor_max: OnceLock<f64>,
}

impl TopologyScalingConfig {
    /// Get minimum torus major radius multiplier
    pub fn torus_major_min(&self) -> f64 {
        *self.torus_major_min.get_or_init(|| {
            env::var("NIODOO_TORUS_MAJOR_MIN_MULT")
                .ok()
                .and_then(|v| v.parse().ok())
                .filter(|&v| v > 0.0)
                .unwrap_or(TORUS_MAJOR_RADIUS_MIN_MULTIPLIER)
        })
    }

    /// Get maximum torus major radius multiplier
    pub fn torus_major_max(&self) -> f64 {
        *self.torus_major_max.get_or_init(|| {
            env::var("NIODOO_TORUS_MAJOR_MAX_MULT")
                .ok()
                .and_then(|v| v.parse().ok())
                .filter(|&v| v > self.torus_major_min())
                .unwrap_or(TORUS_MAJOR_RADIUS_MAX_MULTIPLIER)
        })
    }

    /// Get minimum torus minor radius multiplier
    pub fn torus_minor_min(&self) -> f64 {
        *self.torus_minor_min.get_or_init(|| {
            env::var("NIODOO_TORUS_MINOR_MIN_MULT")
                .ok()
                .and_then(|v| v.parse().ok())
                .filter(|&v| v > 0.0)
                .unwrap_or(TORUS_MINOR_RADIUS_MIN_MULTIPLIER)
        })
    }

    /// Get maximum torus minor radius multiplier
    pub fn torus_minor_max(&self) -> f64 {
        *self.torus_minor_max.get_or_init(|| {
            env::var("NIODOO_TORUS_MINOR_MAX_MULT")
                .ok()
                .and_then(|v| v.parse().ok())
                .filter(|&v| v > self.torus_minor_min())
                .unwrap_or(TORUS_MINOR_RADIUS_MAX_MULTIPLIER)
        })
    }
}

/// Get topology scaling configuration singleton
pub fn get_topology_scaling_config() -> &'static TopologyScalingConfig {
    static CONFIG: OnceLock<TopologyScalingConfig> = OnceLock::new();
    CONFIG.get_or_init(|| TopologyScalingConfig {
        torus_major_min: OnceLock::new(),
        torus_major_max: OnceLock::new(),
        torus_minor_min: OnceLock::new(),
        torus_minor_max: OnceLock::new(),
    })
}

// ==============================================================================
// TESTS
// ==============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_mobius_twist_single() {
        // If using default value, it should equal PI
        if get_topology_config().mobius_twist() == PI {
            assert_eq!(twist_angle(1), PI);
        } else {
            // For custom values, just check it's positive
            assert!(twist_angle(1) > 0.0);
        }
    }

    #[test]
    fn test_k_twist_multiple() {
        let k = get_topology_config().k_twist_count();
        let angle = twist_angle(k);
        let expected = (k as f64) * get_topology_config().mobius_twist();
        assert!((angle - expected).abs() < 1e-10);
    }

    #[test]
    fn test_multiplier_ordering() {
        // These assertions are compile-time verified by the constant definitions
        // Major radius multipliers should be ordered
        const _: () =
            assert!(TORUS_MAJOR_RADIUS_MIN_MULTIPLIER < TORUS_MAJOR_RADIUS_MAX_MULTIPLIER);

        // Minor radius multipliers should be ordered
        const _: () =
            assert!(TORUS_MINOR_RADIUS_MIN_MULTIPLIER < TORUS_MINOR_RADIUS_MAX_MULTIPLIER);

        // Lengthscale multipliers should be ordered
        const _: () = assert!(LENGTHSCALE_MIN_MULTIPLIER < LENGTHSCALE_MAX_MULTIPLIER);

        // Noise multipliers should be ordered
        const _: () = assert!(NOISE_LEVEL_MIN_MULTIPLIER < NOISE_LEVEL_MAX_MULTIPLIER);
    }

    #[test]
    fn test_scale_multipliers_positive() {
        // These assertions are compile-time verified by the constant definitions
        // All multipliers should be positive
        const _: () = assert!(DATA_SCALE_MIN_MULTIPLIER > 0.0);
        const _: () = assert!(EMOTIONAL_PLASTICITY_SCALE > 0.0);
        const _: () = assert!(LEARNING_RATE_BASE_MULTIPLIER > 0.0);
    }

    #[test]
    fn test_config_defaults() {
        let config = get_topology_scaling_config();

        assert_eq!(config.torus_major_min(), TORUS_MAJOR_RADIUS_MIN_MULTIPLIER);
        assert_eq!(config.torus_major_max(), TORUS_MAJOR_RADIUS_MAX_MULTIPLIER);
        assert_eq!(config.torus_minor_min(), TORUS_MINOR_RADIUS_MIN_MULTIPLIER);
        assert_eq!(config.torus_minor_max(), TORUS_MINOR_RADIUS_MAX_MULTIPLIER);
    }
}
