//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham
//! Consciousness Stack Probe - Minimal harness to isolate stack overflow
//!
//! This binary does the bare minimum to test ConsciousnessState initialization:
//! 1. Load ConsciousnessConfig::default()
//! 2. Instantiate ConsciousnessState::new_with_config(&config)
//! 3. Print key fields to verify initialization completed
//!
//! Usage:
//!   cargo run --bin consciousness_stack_probe
//!   RUST_BACKTRACE=full cargo run --bin consciousness_stack_probe

use niodoo_consciousness::config::system_config::ConsciousnessConfig;
use niodoo_consciousness::consciousness::ConsciousnessState;

fn main() {
    println!("🧠 Consciousness Stack Probe - Isolating initialization path");
    println!("============================================================\n");

    // Step 1: Load default config
    println!("Step 1: Loading ConsciousnessConfig::default()...");
    let config = ConsciousnessConfig::default();
    println!("✓ Config loaded successfully");
    println!("  - Enabled: {}", config.enabled);
    println!("  - Emotion sensitivity: {}", config.emotion_sensitivity);
    println!("  - Memory threshold: {}\n", config.memory_threshold);

    // Step 2: Instantiate ConsciousnessState
    println!("Step 2: Creating ConsciousnessState::new_with_config(&config)...");
    let state = ConsciousnessState::new_with_config(&config);
    println!("✓ ConsciousnessState created successfully!\n");

    // Step 3: Print key fields to verify initialization
    println!("Step 3: Verifying initialization...");
    println!("  - Current emotion: {:?}", state.current_emotion);
    println!("  - Reasoning mode: {:?}", state.current_reasoning_mode);
    println!(
        "  - Emotional state primary: {:?}",
        state.emotional_state.primary_emotion
    );
    println!("  - Authenticity metric: {:.3}", state.authenticity_metric);
    println!("  - Cycle count: {}", state.cycle_count);
    println!("  - Timestamp: {:.3}", state.timestamp);

    println!("\n✅ All checks passed! ConsciousnessState initialization is healthy.");
    println!("   Stack overflow is NOT in ConsciousnessState::new_with_config()");
    println!("   → Check Qwen integration or other components in emotional_influence");
}
