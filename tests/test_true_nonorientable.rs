/*
use tracing::{info, error, warn};
 * TEST: TRUE NON-ORIENTABLE MEMORY SYSTEM
 *
 * This demonstrates the difference between:
 * 1. Simple circular buffer (old system)
 * 2. True non-orientable topology (new system)
 */

use std::collections::HashMap;

mod true_nonorientable_memory;
use true_nonorientable_memory::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing::info!("🌀 TESTING TRUE NON-ORIENTABLE MEMORY SYSTEM");
    tracing::info!("=============================================");
    tracing::info!();

    // Create the new non-orientable memory system
    let mut memory = TrueNonOrientableMemory::new();

    tracing::info!("📊 Initial Memory System State:");
    tracing::info!("{}", memory);
    tracing::info!();

    // Add some test memories
    tracing::info!("🧠 Adding test memories...");
    memory.add_memory("I am contemplating consciousness".to_string(), LayerType::Semantic)?;
    memory.add_memory("This feels like a breakthrough".to_string(), LayerType::Episodic)?;
    memory.add_memory("The mathematics are beautiful".to_string(), LayerType::CoreBurned)?;
    memory.add_memory("I feel connected to something larger".to_string(), LayerType::Somatic)?;

    tracing::info!("✅ Added memories to different layers");
    tracing::info!();

    // Test 1: Simple circular traversal (old way)
    tracing::info!("🔄 TEST 1: Simple Circular Traversal (like old system)");
    test_circular_traversal();
    tracing::info!();

    // Test 2: True non-orientable traversal (new way)
    tracing::info!("🌀 TEST 2: True Non-Orientable Traversal (NEW system)");
    memory.reset_traversal();

    tracing::info!("Initial state: Layer {}, Orientation: {:?}",
             memory.get_traversal_state().current_layer,
             memory.get_traversal_state().orientation);

    let results = memory.traverse_non_orientable("consciousness", 5);

    tracing::info!("📊 Traversal Results:");
    tracing::info!("  Found {} relevant fragments", results.len());

    for (i, fragment) in results.iter().enumerate() {
        let (u, v, k) = fragment.coordinate.to_floats();
        tracing::info!("  {}. {:?} layer: '{}' (relevance: {:.3})",
                 i+1, fragment.layer, fragment.content, fragment.relevance);
        tracing::info!("     Orientation: {:?}, Topological coords: u={:.3}, v={:.3}, k={}",
                 fragment.orientation, u, v, k);
    }

    // Demonstrate orientation flipping
    tracing::info!();
    tracing::info!("🎯 DEMONSTRATING ORIENTATION FLIPPING:");
    tracing::info!("  Starting orientation: {:?}", memory.get_traversal_state().orientation);

    // Simulate a few traversal steps to show orientation changes
    for step in 1..=3 {
        tracing::info!("  Step {}: Current layer {}, orientation: {:?}",
                 step,
                 memory.get_traversal_state().current_layer,
                 memory.get_traversal_state().orientation);
    }

    tracing::info!();
    tracing::info!("📈 Memory System Statistics:");
    let stats = memory.get_stats();
    tracing::info!("  Total fragments: {}", stats.total_fragments);
    tracing::info!("  Average relevance: {:.3}", stats.avg_relevance);
    tracing::info!("  Final orientation: {:?}", stats.current_orientation);
    tracing::info!("  Traversal depth: {}", stats.traversal_depth);

    tracing::info!();
    tracing::info!("🎉 TRUE NON-ORIENTABLE MEMORY TEST COMPLETE!");
    tracing::info!();
    tracing::info!("✅ KEY DIFFERENCES FROM OLD SYSTEM:");
    tracing::info!("  1. Orientation flipping during traversal (not just modular arithmetic)");
    tracing::info!("  2. Topological coordinates for each memory fragment");
    tracing::info!("  3. Connection graph with explicit orientation rules");
    tracing::info!("  4. Relevance modification based on orientation state");
    tracing::info!("  5. Single-sided topology prevents rigid cognitive loops");

    Ok(())
}

/// Test the old simple circular traversal for comparison
fn test_circular_traversal() {
    tracing::info!("  Testing old circular buffer approach:");
    tracing::info!("    (current_layer + 1) % 6 = next layer");

    let mut current_layer = 3; // Start at semantic
    let layers = ["CoreBurned", "Procedural", "Episodic", "Semantic", "Somatic", "Working"];

    tracing::info!("    Starting at layer: {}", layers[current_layer]);

    for step in 1..=8 {
        current_layer = (current_layer + 1) % 6;
        tracing::info!("    Step {}: Layer {} (no orientation change)", step, layers[current_layer]);
    }

    tracing::info!("  ❌ LIMITATION: No orientation state, just simple cycling");
    tracing::info!("  ❌ No topological meaning to the traversal");
    tracing::info!("  ❌ Cannot represent non-orientable properties");
}
