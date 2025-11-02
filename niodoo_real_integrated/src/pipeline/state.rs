use crate::compass::{CascadeTransition, CompassOutcome};
use crate::consonance::ConsonanceMetrics;
use crate::data::Experience;
use crate::erag::CollapseResult;
use crate::generation::GenerationResult;
use crate::hyperfocus::HyperfocusEvent;
use crate::learning::LearningOutcome;
use crate::tcs_analysis::TopologicalSignature;
use crate::token_manager::TokenizerOutput;
use crate::torus::PadGhostState;
use std::collections::{HashMap, VecDeque};

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
    pub experience: Option<Experience>,
}

/// Phase 4.2: Curator feedback controller for adaptive parameter adjustment
#[derive(Debug, Clone)]
pub struct CuratorFeedbackController {
    /// Recent quality scores (sliding window)
    quality_history: VecDeque<f32>,
    /// Recent learned flags (sliding window)
    learned_history: VecDeque<bool>,
    /// Window size for feedback tracking
    window_size: usize,
    /// Current adaptive quality threshold
    adaptive_threshold: f32,
    /// Base quality threshold from config
    base_threshold: f32,
    /// Quality trend: positive = improving, negative = degrading
    quality_trend: f32,
}

impl CuratorFeedbackController {
    pub fn new(base_threshold: f32, window_size: usize) -> Self {
        Self {
            quality_history: VecDeque::with_capacity(window_size),
            learned_history: VecDeque::with_capacity(window_size),
            window_size,
            adaptive_threshold: base_threshold,
            base_threshold,
            quality_trend: 0.0,
        }
    }

    /// Phase 4.2: Record curator feedback and update adaptive threshold
    pub fn record_feedback(&mut self, quality_score: f32, learned: bool) {
        // Add to history
        self.quality_history.push_back(quality_score);
        self.learned_history.push_back(learned);
        
        // Maintain window size
        if self.quality_history.len() > self.window_size {
            self.quality_history.pop_front();
        }
        if self.learned_history.len() > self.window_size {
            self.learned_history.pop_front();
        }

        // Compute quality trend (exponential moving average)
        if self.quality_history.len() >= 2 {
            let recent_avg: f32 = self.quality_history.iter().rev().take(5).sum::<f32>() / 
                self.quality_history.iter().rev().take(5).count().min(5) as f32;
            let older_avg: f32 = if self.quality_history.len() > 5 {
                self.quality_history.iter().rev().skip(5).take(5).sum::<f32>() / 5.0
            } else {
                recent_avg
            };
            self.quality_trend = recent_avg - older_avg;
        }

        // Update adaptive threshold based on quality trend
        // If quality is improving, raise threshold (stricter)
        // If quality is degrading, lower threshold (more lenient)
        let threshold_adjustment = self.quality_trend * 0.05; // 5% adjustment per trend unit
        self.adaptive_threshold = (self.base_threshold + threshold_adjustment)
            .clamp(0.3, 0.9); // Clamp between reasonable bounds
        
        // Phase 5.2: Record metrics
        crate::metrics::curator_feedback_metrics().record_feedback(
            self.adaptive_threshold,
            self.quality_trend,
            self.recent_quality_avg(),
            self.learned_rate(),
        );
    }

    /// Phase 4.2: Get adaptive quality threshold
    pub fn adaptive_threshold(&self) -> f32 {
        self.adaptive_threshold
    }

    /// Phase 4.2: Get quality trend
    pub fn quality_trend(&self) -> f32 {
        self.quality_trend
    }

    /// Phase 4.2: Get recent quality average
    pub fn recent_quality_avg(&self) -> f32 {
        if self.quality_history.is_empty() {
            return self.base_threshold;
        }
        self.quality_history.iter().sum::<f32>() / self.quality_history.len() as f32
    }

    /// Phase 4.2: Get learned rate (percentage of recent responses marked as learned)
    pub fn learned_rate(&self) -> f32 {
        if self.learned_history.is_empty() {
            return 0.0;
        }
        let learned_count = self.learned_history.iter().filter(|&&l| l).count();
        learned_count as f32 / self.learned_history.len() as f32
    }

    /// Phase 4.2: Generate feedback-based parameter adjustments
    pub fn compute_parameter_adjustments(&self) -> HashMap<String, f64> {
        let mut adjustments = HashMap::new();
        
        let quality_avg = self.recent_quality_avg();
        let learned_rate = self.learned_rate();

        // Adjust temperature based on quality trend
        // If quality is improving, reduce temperature (more focused)
        // If quality is degrading, increase temperature (more exploratory)
        if self.quality_trend.abs() > 0.05 {
            let temp_adjustment = -self.quality_trend * 0.1; // Inverse relationship
            adjustments.insert("temperature".to_string(), temp_adjustment as f64);
            crate::metrics::curator_feedback_metrics().record_parameter_adjustment("temperature");
        }

        // Adjust top_p based on learned rate
        // High learned rate = curator is happy, maintain current top_p
        // Low learned rate = curator is rejecting, increase top_p (more diverse)
        if learned_rate < 0.3 && quality_avg < 0.6 {
            adjustments.insert("top_p".to_string(), 0.05); // Increase diversity
            crate::metrics::curator_feedback_metrics().record_parameter_adjustment("top_p");
        } else if learned_rate > 0.7 && quality_avg > 0.7 {
            adjustments.insert("top_p".to_string(), -0.02); // Slightly reduce diversity
            crate::metrics::curator_feedback_metrics().record_parameter_adjustment("top_p");
        }

        // Adjust retrieval_top_k based on quality
        // Higher quality = curator is satisfied, maintain current k
        // Lower quality = need more context, increase k
        if quality_avg < 0.5 {
            adjustments.insert("retrieval_top_k".to_string(), 1.0); // Increase context
            crate::metrics::curator_feedback_metrics().record_parameter_adjustment("retrieval_top_k");
        } else if quality_avg > 0.8 && learned_rate > 0.6 {
            adjustments.insert("retrieval_top_k".to_string(), -0.5); // Reduce context (less noise)
            crate::metrics::curator_feedback_metrics().record_parameter_adjustment("retrieval_top_k");
        }

        adjustments
    }
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
            experience: None,
        }
    }
}
