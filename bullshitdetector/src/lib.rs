// Copyright (c) 2025 Jason Van Pham (ruffian-l on GitHub) @ The Niodoo Collaborative
// Licensed under the MIT License - See LICENSE file for details
// Attribution required for all derivative works

use serde::{Deserialize, Serialize};
use std::fmt;

/// Core bullshit detection types and structures
pub mod qwen_client;
pub mod topology_engine;
pub mod constants;

pub mod types {
    use super::*;

    /// Bullshit alert types
    #[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize)]
    pub enum BullshitType {
        FakeComplexity,
        CargoCult,
        OverEngineering,
        ArcAbuse,
        RwLockAbuse,
        SleepAbuse,
        UnwrapAbuse,
        DynTraitAbuse,
        CloneAbuse,
        MutexAbuse,
    }

    impl fmt::Display for BullshitType {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                BullshitType::FakeComplexity => write!(f, "FakeComplexity"),
                BullshitType::CargoCult => write!(f, "CargoCult"),
                BullshitType::OverEngineering => write!(f, "OverEngineering"),
                BullshitType::ArcAbuse => write!(f, "ArcAbuse"),
                BullshitType::RwLockAbuse => write!(f, "RwLockAbuse"),
                BullshitType::SleepAbuse => write!(f, "SleepAbuse"),
                BullshitType::UnwrapAbuse => write!(f, "UnwrapAbuse"),
                BullshitType::DynTraitAbuse => write!(f, "DynTraitAbuse"),
                BullshitType::CloneAbuse => write!(f, "CloneAbuse"),
                BullshitType::MutexAbuse => write!(f, "MutexAbuse"),
            }
        }
    }

    /// PAD (Pleasure-Arousal-Dominance) emotional valence model
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct PADValence {
        pub pleasure: f32,  // -1.0 to 1.0 (negative to positive)
        pub arousal: f32,   // 0.0 to 1.0 (calm to excited)
        pub dominance: f32, // -1.0 to 1.0 (submissive to dominant)
        pub total: f32,     // Computed total valence
    }

    impl PADValence {
        pub fn new(pleasure: f32, arousal: f32, dominance: f32) -> Self {
            let total = (pleasure * 0.4) + (arousal * 0.3) + (dominance * 0.3);
            Self { pleasure, arousal, dominance, total }
        }

        pub fn total(&self) -> f32 {
            self.total
        }

        pub fn negative() -> Self {
            Self::new(-0.8, 0.2, -0.3)
        }

        pub fn positive() -> Self {
            Self::new(0.6, 0.4, 0.5)
        }
    }

    /// Bullshit alert with confidence and suggestions
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct BullshitAlert {
        pub issue_type: BullshitType,
        pub confidence: f32,
        pub location: (usize, usize), // (line, column)
        pub context_snippet: String,
        pub why_bs: String,
        pub sug: String,
        pub severity: f32,
    }

    /// Review request for RAG generation
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct RagReviewRequest {
        pub alerts: Vec<BullshitAlert>,
        pub suggestions: Vec<Suggestion>,
    }

    /// Code suggestion with reasoning
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Suggestion {
        pub suggestion: String,
        pub reasoning: String,
        pub confidence: f32,
        pub xp_reward: u32,
    }

    /// Review request to RAG system
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ReviewRequest {
        pub alerts: Vec<BullshitAlert>,
        pub suggestions: Vec<Suggestion>,
    }

    /// Final review response
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ReviewResponse {
        pub summary: String,
        pub severity: f32,
        pub recommendations: Vec<String>,
        pub coherence: f32,
        pub latency_ms: u64,
    }

    /// Möbius transformation for non-orientable embeddings
    #[derive(Debug, Clone)]
    pub struct MobiusTransform {
        pub a: f32,
        pub b: f32,
        pub c: f32,
        pub d: f32,
    }

    impl MobiusTransform {
        pub fn new(a: f32, b: f32, c: f32, d: f32) -> Self {
            Self { a, b, c, d }
        }

        pub fn identity() -> Self {
            Self::new(1.0, 0.0, 0.0, 1.0)
        }

        pub fn twist(phi: f32) -> Self {
            // Golden ratio based twist for non-orientable surfaces
            let phi_inv = 1.0 / (1.0 + phi);
            Self::new(phi_inv, phi, phi_inv, phi_inv)
        }

        pub fn apply(&self, z: (f32, f32)) -> (f32, f32) {
            let (x, y) = z;
            let denom = self.c * x + self.d;
            if denom.abs() < 1e-10 {
                (f32::INFINITY, f32::INFINITY)
            } else {
                let new_x = (self.a * x + self.b) / denom;
                let new_y = y; // Preserve y for 2D embedding
                (new_x, new_y)
            }
        }
    }
}

/// Configuration structures
pub mod config {
    use super::*;

    #[derive(Debug, Clone)]
    pub struct IntegrationConfig {
        pub enable_git_hooks: bool,
        pub enable_web_server: bool,
        pub web_port: u16,
        pub enable_metrics: bool,
        pub metrics_port: u16,
    }

    impl Default for IntegrationConfig {
        fn default() -> Self {
            Self {
                enable_git_hooks: true,
                enable_web_server: true,
                web_port: 3000,
                enable_metrics: true,
                metrics_port: 9090,
            }
        }
    }

    #[derive(Debug, Clone)]
    pub struct DetectConfig {
        pub confidence_threshold: f32,
        pub max_snippet_length: usize,
        pub enable_tree_sitter: bool,
        pub enable_regex_fallback: bool,
    }

    impl Default for DetectConfig {
        fn default() -> Self {
            Self {
                confidence_threshold: 0.618, // Golden ratio inverse
                max_snippet_length: 500,
                enable_tree_sitter: true,
                enable_regex_fallback: true,
            }
        }
    }

    #[derive(Debug, Clone)]
    pub struct SuggestConfig {
        pub max_suggestions: usize,
        pub min_confidence: f32,
        pub enable_docs_impact: bool,
        pub xp_multiplier: f32,
    }

    impl Default for SuggestConfig {
        fn default() -> Self {
            Self {
                max_suggestions: 5,
                min_confidence: 0.618,
                enable_docs_impact: true,
                xp_multiplier: 1.5,
            }
        }
    }

    #[derive(Debug, Clone)]
    pub struct GPConfig {
        pub jitter: f32,
        pub noise_variance: f32,
        pub length_scale: f32,
        pub max_points: usize,
    }

    impl Default for GPConfig {
        fn default() -> Self {
            Self {
                jitter: 1e-10,
                noise_variance: 0.1,
                length_scale: 1.0,
                max_points: 1000,
            }
        }
    }
}

/// Core bullshit detection functions
pub mod detect;
/// Emotional feeler probes
pub mod feeler;
/// Memory system for learning from past bullshit
pub mod memory;
/// RAG generation for code reviews
pub mod rag;
/// Suggestion generation system
pub mod suggest;
/// Gaussian Process implementation
pub mod gp;
/// Hyperbolic embeddings for code structure
pub mod hyperbolic;
/// LSP server implementation
pub mod lsp;
/// Integration utilities (web server, git hooks, etc.)
pub mod integrate;
/// Dataset generation utilities
pub mod dataset;

/// Re-export main types for convenience
pub use types::*;
pub use config::*;

/// Re-export functions from modules
pub use detect::{scan_code, score_bs_confidence};
pub use suggest::{generate_sugs, add_docs_impact};
pub use integrate::{setup_git_hook, ci_pipeline_review};
pub use memory::SixLayerMemory;
pub use feeler::{spawn_probes, simulate_trajectory, score_and_filter, fuse_top_three, topology_similarity};
pub use rag::{generate_review, build_prompt};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_functionality() {
        // Basic test to ensure the module compiles
        let config = DetectConfig::default();
        assert_eq!(config.confidence_threshold, 0.618);
    }
}
