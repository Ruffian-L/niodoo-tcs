//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

//! Bullshit Buster Demo - Topological Code Analysis
//!
//! Demonstrates the Code Parser's ability to detect hardcoded values, stubs,
//! and fake implementations through Möbius topology analysis.

use niodoo_consciousness::parser::{CodeParser, TopologicalPosition};
use std::path::Path;
use tracing::info;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    info!("🧠 BULLSHIT BUSTER DEMO - Topological Code Analysis");
    info!("==================================================");

    let mut parser = CodeParser::new();

    // Parse the current codebase
    let src_dir = Path::new("src");
    if src_dir.exists() {
        info!("Parsing Rust source files in src/ directory...");

        // Parse key files
        let files_to_parse = vec![
            "src/lib.rs",
            "src/dual_mobius_gaussian.rs",
            "src/consciousness.rs",
            "src/parser.rs",
        ];

        for file_path in files_to_parse {
            if Path::new(file_path).exists() {
                match parser.parse_file(file_path) {
                    Ok(_) => info!("✅ Successfully parsed: {}", file_path),
                    Err(e) => info!("❌ Failed to parse {}: {}", file_path, e),
                }
            }
        }
    } else {
        info!("⚠️  src/ directory not found, creating demo analysis...");

        // Demo topological analysis
        demo_topological_analysis();
    }

    // Generate and display report
    let report = parser.generate_report();
    info!("{}", report);

    // Demo topological calculations
    demo_geodesic_calculations();

    info!("🎯 Bullshit Buster Demo Complete!");
    Ok(())
}

/// Demonstrate topological analysis concepts
fn demo_topological_analysis() {
    info!("🌀 DEMONSTRATING MÖBIUS TOPOLOGY ANALYSIS");
    info!("==========================================");

    // Create sample topological positions
    let pos1 = TopologicalPosition::new(0.5, 0.5, 0); // Flat geodesic (bullshit)
    let pos2 = TopologicalPosition::new(0.3, 0.7, 1); // Normal position
    let pos3 = TopologicalPosition::new(0.8, 0.2, 2); // Complex position

    info!("Sample Topological Positions:");
    info!(
        "  Position 1: ({:.3}, {:.3}, {}) - Flat geodesic: {}",
        pos1.u,
        pos1.v,
        pos1.k,
        pos1.is_flat_geodesic()
    );
    info!(
        "  Position 2: ({:.3}, {:.3}, {}) - Flat geodesic: {}",
        pos2.u,
        pos2.v,
        pos2.k,
        pos2.is_flat_geodesic()
    );
    info!(
        "  Position 3: ({:.3}, {:.3}, {}) - Flat geodesic: {}",
        pos3.u,
        pos3.v,
        pos3.k,
        pos3.is_flat_geodesic()
    );
}

/// Demonstrate geodesic distance calculations
fn demo_geodesic_calculations() {
    info!("📏 DEMONSTRATING GEODESIC DISTANCE CALCULATIONS");
    info!("===============================================");

    let pos1 = TopologicalPosition::new(0.0, 0.0, 0);
    let pos2 = TopologicalPosition::new(1.0, 1.0, 0);
    let pos3 = TopologicalPosition::new(0.5, 0.5, 1);

    let dist_1_2 = pos1.geodesic_distance(&pos2);
    let dist_1_3 = pos1.geodesic_distance(&pos3);
    let dist_2_3 = pos2.geodesic_distance(&pos3);

    info!("Geodesic Distances:");
    info!("  Distance (0,0,0) ↔ (1,1,0): {:.6}", dist_1_2);
    info!("  Distance (0,0,0) ↔ (0.5,0.5,1): {:.6}", dist_1_3);
    info!("  Distance (1,1,0) ↔ (0.5,0.5,1): {:.6}", dist_2_3);

    info!("📊 TOPOLOGICAL INSIGHTS:");
    info!("  • Flat geodesics indicate hardcoded paths (bullshit)");
    info!("  • Non-orientable topology detects fake implementations");
    info!("  • K-twist factors reveal complexity depth");
    info!("  • Geodesic distances measure code similarity");
}
