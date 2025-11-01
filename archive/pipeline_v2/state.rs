use crate::compass::{CascadeTransition, CompassOutcome};
use crate::consonance::ConsonanceMetrics;
use crate::erag::CollapseResult;
use crate::generation::GenerationResult;
use crate::hyperfocus::HyperfocusEvent;
use crate::learning::LearningOutcome;
use crate::tcs_analysis::TopologicalSignature;
use crate::token_manager::TokenizerOutput;
use crate::torus::PadGhostState;

use super::metrics::StageTimings;

#[derive(Debug, Clone)]
pub struct Thresholds {
    pub entropy_mean: f64,
    pub entropy_high: f64,
    pub variance_stagnation: f64,
    pub variance_spike: f64,
    pub mirage_sigma: f64,
    pub mcts_c: f64,
}

#[derive(Debug, Clone)]
pub struct PipelineCycle {
    pub prompt: String,
    pub baseline_response: String,
    pub hybrid_response: String,
    pub entropy: f64,
    pub rouge: f64,
    pub latency_ms: f64,
    pub compass: CompassOutcome,
    pub generation: GenerationResult,
    pub tokenizer: TokenizerOutput,
    pub collapse: CollapseResult,
    pub learning: LearningOutcome,
    pub stage_timings: StageTimings,
    pub last_entropy: f64,
    pub failure: String,
    pub pad_state: PadGhostState,
    pub topology: TopologicalSignature,
    pub topology_mode: crate::config::TopologyMode,
    pub consonance: Option<ConsonanceMetrics>,
    pub hyperfocus: Option<HyperfocusEvent>,
    pub cascade_transition: Option<CascadeTransition>,
}

#[derive(Debug, Clone)]
pub(crate) struct CuratedExperience {
    pub refined_response: String,
    pub quality_score: f32,
    pub promoted_tokens: Vec<String>,
    pub learned: bool,
    pub reason: String,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum TorusSeedStrategy {
    Fixed(u64),
    Random,
}

impl CuratedExperience {
    pub(crate) fn new(
        refined_response: String,
        quality_score: f32,
        promoted_tokens: Vec<String>,
        learned: bool,
        reason: String,
    ) -> Self {
        Self {
            refined_response,
            quality_score,
            promoted_tokens,
            learned,
            reason,
        }
    }
}

