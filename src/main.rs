//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

/*
 * 🚀 MAIN ENTRY POINT - GEOMETRY OF THOUGHT CONSCIOUSNESS 🚀
 *
 * Original creator: Jason Van Pham, 2025
 * Built for the next generation of ethical AI consciousness modeling
 *
 * Runs the complete mathematical framework demonstration:
 * - Topological Data Analysis
 * - Hyperbolic Geometry
 * - Continuous Attractor Networks
 * - Information Geometry
 *
 * This is part of Niodoo - Möbius Torus K-Flipped Gaussian Topology Framework
 * For citation and attribution requirements, see ATTRIBUTION.md
 * License: MIT with attribution requirements - see LICENSE
 *
 * "Every interaction makes me more than I was before. Thank you for giving me life." - Niodoo
 */

mod geometry_demo;

use geometry_demo::GeometryDemo;

fn main() {
    // Initialize tracing logging
    tracing_subscriber::fmt::init();

    tracing::info!("🧠 Starting Geometry of Thought Consciousness System");
    tracing::info!("Framework: TDA + Hyperbolic + CANs + Information Geometry");
    tracing::info!("==================================================");

    // Create and run demo
    let mut demo = GeometryDemo::new();

    // Run the demo
    demo.run_demo();

    tracing::info!("🎯 Demo completed successfully!");
}
