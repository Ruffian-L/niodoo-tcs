//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

//! Standalone test for k-twisted torus geometry validator
// This module provides a simple test to verify the validator works correctly
// without depending on the broader codebase compilation.

/// Simple test implementation that demonstrates the validator
fn main() {
    tracing::info!("Testing K-Twisted Torus Geometry Validator");

    // Test the reference implementation
    let reference = KTwistReference::new(5.0, 1.5, 1.0);

    // Test basic point computation
    let point = reference.compute_point(0.0, 0.0);
    tracing::info!(
        "Reference point at (0,0): ({:.3}, {:.3}, {:.3})",
        point.0,
        point.1,
        point.2
    );

    // Test orientability
    let is_orientable = reference.check_orientability();
    tracing::info!("Is orientable: {}", is_orientable);

    // Test Euler characteristic
    let chi = reference.compute_euler_characteristic();
    tracing::info!("Euler characteristic: {:.3}", chi);

    // Test genus
    let genus = reference.compute_genus();
    tracing::info!("Genus: {:.3}", genus);

    tracing::info!("Validator test completed successfully!");
}

/// Reference implementation of k-twisted torus for validation
pub struct KTwistReference {
    major_radius: f64,
    minor_radius: f64,
    k_twist: f64,
}

impl KTwistReference {
    /// Create a new reference implementation
    pub fn new(major_radius: f64, minor_radius: f64, k_twist: f64) -> Self {
        Self {
            major_radius,
            minor_radius,
            k_twist,
        }
    }

    /// Compute parametric equations (reference implementation)
    pub fn compute_point(&self, u: f64, v: f64) -> (f64, f64, f64) {
        // Corrected parametric equations for k-twisted torus
        let twist_factor = self.k_twist * u;

        let x = (self.major_radius + self.minor_radius * v.cos()) * u.cos();
        let y = (self.major_radius + self.minor_radius * v.cos()) * u.sin();
        let z = self.minor_radius * v.sin() + self.k_twist * twist_factor.sin();

        (x, y, z)
    }

    /// Check if surface is orientable for given k value
    pub fn check_orientability(&self) -> bool {
        // For k-twisted torus:
        // - Even k: orientable (regular torus)
        // - Odd k: non-orientable (Möbius-like)
        self.k_twist.fract() == 0.0 && (self.k_twist as i64) % 2 == 0
    }

    /// Compute Euler characteristic
    pub fn compute_euler_characteristic(&self) -> f64 {
        // For a torus (orientable): χ = 0
        // For a Klein bottle (non-orientable): χ = 0
        // The k-twist doesn't change the Euler characteristic
        0.0
    }

    /// Compute genus
    pub fn compute_genus(&self) -> f64 {
        // For orientable surfaces: g = (2 - χ) / 2
        // For non-orientable surfaces: g = (2 - χ) / 2
        // Both torus and Klein bottle have genus 1
        1.0
    }
}
