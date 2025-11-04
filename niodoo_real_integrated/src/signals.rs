use std::fmt;

use crate::config::FailureSignalThresholds;

#[derive(Debug, Clone)]
pub struct FailureSignals {
    pub rouge: f64,
    pub entropy_delta: f64,
    pub min_ucb: Option<f64>,
    pub average_similarity: f32,
    pub curator_score: Option<f64>,
    pub fallback_source: bool,
    pub oov_rate: f64,
    pub low_quality_hits: usize,
    pub soft_triggers: Vec<&'static str>,
    pub hard_triggers: Vec<&'static str>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    None,
    Soft,
    Hard,
}

impl FailureSignals {
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate(
        rouge: f64,
        entropy_delta: f64,
        min_ucb: Option<f64>,
        average_similarity: f32,
        curator_score: Option<f64>,
        fallback_source: bool,
        oov_rate: f64,
        low_quality_hits: usize,
    ) -> Self {
        // Use default thresholds for backward compatibility
        Self::evaluate_with_thresholds(
            rouge,
            entropy_delta,
            min_ucb,
            average_similarity,
            curator_score,
            fallback_source,
            oov_rate,
            low_quality_hits,
            &FailureSignalThresholds::default(),
        )
    }

    /// Evaluate with custom thresholds (for configurable behavior)
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_with_thresholds(
        rouge: f64,
        entropy_delta: f64,
        min_ucb: Option<f64>,
        average_similarity: f32,
        curator_score: Option<f64>,
        fallback_source: bool,
        oov_rate: f64,
        low_quality_hits: usize,
        thresholds: &FailureSignalThresholds,
    ) -> Self {
        let mut soft_triggers = Vec::new();
        let mut hard_triggers = Vec::new();

        if rouge < thresholds.hard_rouge_threshold {
            hard_triggers.push("rouge_below_threshold");
        }

        if entropy_delta > thresholds.hard_entropy_delta_threshold {
            hard_triggers.push("entropy_delta_above_threshold");
        }

        if let Some(curator) = curator_score {
            if curator < thresholds.hard_curator_threshold {
                hard_triggers.push("curator_score_below_threshold");
            }
        }

        if let Some(ucb) = min_ucb {
            if ucb < thresholds.soft_ucb_threshold {
                soft_triggers.push("ucb1_below_threshold");
            }
        }

        if average_similarity < thresholds.soft_avg_similarity_threshold {
            soft_triggers.push("average_similarity_low");
        }

        if oov_rate > thresholds.soft_oov_threshold {
            soft_triggers.push("oov_rate_high");
        }

        if fallback_source {
            soft_triggers.push("fallback_generation");
        }

        if low_quality_hits >= thresholds.low_quality_hits_threshold {
            soft_triggers.push("many_low_quality_hits");
        }

        if !hard_triggers.is_empty() {
            soft_triggers.push("hard_trigger_present");
        }

        Self {
            rouge,
            entropy_delta,
            min_ucb,
            average_similarity,
            curator_score,
            fallback_source,
            oov_rate,
            low_quality_hits,
            soft_triggers,
            hard_triggers,
        }
    }

    pub fn severity(&self) -> Severity {
        if !self.hard_triggers.is_empty() {
            Severity::Hard
        } else if !self.soft_triggers.is_empty() {
            Severity::Soft
        } else {
            Severity::None
        }
    }

    pub fn summary(&self) -> String {
        let mut parts = Vec::new();
        if !self.hard_triggers.is_empty() {
            parts.push(format!("hard={}", self.hard_triggers.join("|")));
        }
        if !self.soft_triggers.is_empty() {
            parts.push(format!("soft={}", self.soft_triggers.join("|")));
        }
        if parts.is_empty() {
            "none".to_string()
        } else {
            parts.join(";")
        }
    }

    pub fn soft_summary(&self) -> String {
        if self.soft_triggers.is_empty() {
            "general low-confidence".to_string()
        } else {
            self.soft_triggers.join(", ")
        }
    }
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Severity::None => write!(f, "none"),
            Severity::Soft => write!(f, "soft"),
            Severity::Hard => write!(f, "hard"),
        }
    }
}
