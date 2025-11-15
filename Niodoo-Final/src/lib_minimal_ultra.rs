//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

//! # Niodoo-Feeling: Minimal Core Consciousness Framework ULTRA
//!
//! Minimal build configuration for BUILD MASTER to get core consciousness framework compiling
//! This includes only the most essential modules that compile without any external dependencies

// Core system modules - ESSENTIAL ONLY (no external deps, no missing imports, no errors, no duplicates)
pub mod core;
pub mod error;
pub mod config;
pub mod consciousness;
pub mod memory;
pub mod brain;
pub mod brains;
pub mod empathy;
pub mod optimization;
pub mod events;
pub mod profiling;

// Advanced modules - ESSENTIAL ONLY (no external deps, no missing imports, no errors, no duplicates)
pub mod oscillatory;
pub mod personal_memory;
pub mod philosophy;
pub mod phase7;
pub mod soul_resonance;

// Mathematical and topological modules - ESSENTIAL ONLY (no external deps, no missing imports, no errors, no duplicates)
pub mod dual_mobius_gaussian;
pub mod sparse_gaussian_processes;
pub mod topology;
pub mod dynamics;
pub mod advanced_empathy;
pub mod bert_emotion;

// Error types
pub use error::*;

// Configuration
pub use crate::config::*;

// Core consciousness functionality
pub use consciousness::*;
pub use memory::*;
pub use brain::*;
pub use brains::*;
pub use empathy::*;
