use std::fmt;

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
        const HARD_ROUGE_THRESHOLD: f64 = 0.5;
        const HARD_ENTROPY_DELTA_THRESHOLD: f64 = 0.1;
        const HARD_CURATOR_THRESHOLD: f64 = 0.7;
        const SOFT_UCB_THRESHOLD: f64 = 0.3;
        const SOFT_AVG_SIMILARITY_THRESHOLD: f32 = 0.4;
        const SOFT_OOV_THRESHOLD: f64 = 0.2;
        const LOW_QUALITY_HITS_THRESHOLD: usize = 3;

        let mut soft_triggers = Vec::new();
        let mut hard_triggers = Vec::new();

        if rouge < HARD_ROUGE_THRESHOLD {
            hard_triggers.push("rouge_below_0_5");
        }

        if entropy_delta > HARD_ENTROPY_DELTA_THRESHOLD {
            hard_triggers.push("entropy_delta_above_0_1");
        }

        if let Some(curator) = curator_score {
            if curator < HARD_CURATOR_THRESHOLD {
                hard_triggers.push("curator_score_below_0_7");
            }
        }

        if let Some(ucb) = min_ucb {
            if ucb < SOFT_UCB_THRESHOLD {
                soft_triggers.push("ucb1_below_0_3");
            }
        }

        if average_similarity < SOFT_AVG_SIMILARITY_THRESHOLD {
            soft_triggers.push("average_similarity_low");
        }

        if oov_rate > SOFT_OOV_THRESHOLD {
            soft_triggers.push("oov_rate_high");
        }

        if fallback_source {
            soft_triggers.push("fallback_generation");
        }

        if low_quality_hits >= LOW_QUALITY_HITS_THRESHOLD {
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
