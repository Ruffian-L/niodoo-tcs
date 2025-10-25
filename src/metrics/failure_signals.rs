use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum FailureSignalCode {
    RougeBelowThreshold,
    EntropyDeltaSpike,
    EntropyDeltaFlatline,
    CompassThreat,
    CompassHealing,
    MctsConfidenceLow,
    VllmFallback,
    CuratorLowQuality,
    Custom(&'static str),
}

impl FailureSignalCode {
    pub fn as_str(&self) -> &'static str {
        match self {
            FailureSignalCode::RougeBelowThreshold => "rouge_low",
            FailureSignalCode::EntropyDeltaSpike => "entropy_spike",
            FailureSignalCode::EntropyDeltaFlatline => "entropy_flat",
            FailureSignalCode::CompassThreat => "compass_threat",
            FailureSignalCode::CompassHealing => "compass_healing",
            FailureSignalCode::MctsConfidenceLow => "mcts_confidence_low",
            FailureSignalCode::VllmFallback => "vllm_fallback",
            FailureSignalCode::CuratorLowQuality => "curator_low_quality",
            FailureSignalCode::Custom(_) => "custom",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum FailureSeverity {
    Soft,
    Hard,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureSignal {
    pub code: FailureSignalCode,
    pub severity: FailureSeverity,
    pub description: String,
    pub value: f64,
}

impl FailureSignal {
    pub fn new(
        code: FailureSignalCode,
        severity: FailureSeverity,
        description: impl Into<String>,
        value: f64,
    ) -> Self {
        Self {
            code,
            severity,
            description: description.into(),
            value,
        }
    }
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct AdaptiveMetricsSnapshot {
    pub rouge: Option<f64>,
    pub entropy: Option<f64>,
    pub entropy_delta: Option<f64>,
    pub ucb1: Option<f64>,
    pub compass_state: Option<String>,
    pub curator_quality: Option<f64>,
    pub fallbacks_triggered: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureSignalThresholds {
    pub rouge_min: f64,
    pub entropy_delta_spike: f64,
    pub entropy_delta_flatline: f64,
    pub ucb1_min: f64,
    pub curator_quality_min: f64,
}

impl Default for FailureSignalThresholds {
    fn default() -> Self {
        Self {
            rouge_min: 0.5,
            entropy_delta_spike: 0.1,
            entropy_delta_flatline: 0.01,
            ucb1_min: 0.3,
            curator_quality_min: 0.7,
        }
    }
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct AggregatedFailureSignals {
    pub soft_signals: Vec<FailureSignal>,
    pub hard_signals: Vec<FailureSignal>,
}

impl AggregatedFailureSignals {
    pub fn has_hard_failure(&self) -> bool {
        !self.hard_signals.is_empty()
    }

    pub fn has_soft_failure(&self) -> bool {
        !self.soft_signals.is_empty()
    }

    pub fn all_signals(&self) -> impl Iterator<Item = &FailureSignal> {
        self.soft_signals.iter().chain(self.hard_signals.iter())
    }
}

#[derive(Debug, Clone)]
pub struct FailureSignalAggregator {
    thresholds: FailureSignalThresholds,
    recent_history: HashMap<&'static str, f64>,
}

impl FailureSignalAggregator {
    pub fn new(thresholds: FailureSignalThresholds) -> Self {
        Self {
            thresholds,
            recent_history: HashMap::new(),
        }
    }

    pub fn update_thresholds(&mut self, thresholds: FailureSignalThresholds) {
        self.thresholds = thresholds;
    }

    pub fn aggregate(&mut self, snapshot: &AdaptiveMetricsSnapshot) -> AggregatedFailureSignals {
        let mut aggregated = AggregatedFailureSignals::default();

        if let Some(rouge) = snapshot.rouge {
            if rouge < self.thresholds.rouge_min {
                aggregated.hard_signals.push(FailureSignal::new(
                    FailureSignalCode::RougeBelowThreshold,
                    FailureSeverity::Hard,
                    format!("ROUGE below threshold: {:.3} < {:.3}", rouge, self.thresholds.rouge_min),
                    rouge,
                ));
            }
        }

        if let Some(delta) = snapshot.entropy_delta {
            if delta > self.thresholds.entropy_delta_spike {
                aggregated.hard_signals.push(FailureSignal::new(
                    FailureSignalCode::EntropyDeltaSpike,
                    FailureSeverity::Hard,
                    format!(
                        "Entropy delta spike detected: {:.3} > {:.3}",
                        delta, self.thresholds.entropy_delta_spike
                    ),
                    delta,
                ));
            } else if delta.abs() < self.thresholds.entropy_delta_flatline {
                aggregated.soft_signals.push(FailureSignal::new(
                    FailureSignalCode::EntropyDeltaFlatline,
                    FailureSeverity::Soft,
                    format!(
                        "Entropy delta flatline: |Δ| {:.4} < {:.4}",
                        delta.abs(),
                        self.thresholds.entropy_delta_flatline
                    ),
                    delta,
                ));
            }
        }

        if let Some(ucb1) = snapshot.ucb1 {
            if ucb1 < self.thresholds.ucb1_min {
                aggregated.soft_signals.push(FailureSignal::new(
                    FailureSignalCode::MctsConfidenceLow,
                    FailureSeverity::Soft,
                    format!("MCTS UCB1 low: {:.3} < {:.3}", ucb1, self.thresholds.ucb1_min),
                    ucb1,
                ));
            }
        }

        if let Some(compass_state) = snapshot.compass_state.as_deref() {
            if compass_state == "10" {
                aggregated.soft_signals.push(FailureSignal::new(
                    FailureSignalCode::CompassThreat,
                    FailureSeverity::Soft,
                    "Compass indicated threat state",
                    1.0,
                ));
            }
        }

        if snapshot.fallbacks_triggered > 0 {
            aggregated.soft_signals.push(FailureSignal::new(
                FailureSignalCode::VllmFallback,
                FailureSeverity::Soft,
                format!("{} fallback(s) triggered", snapshot.fallbacks_triggered),
                snapshot.fallbacks_triggered as f64,
            ));
        }

        if let Some(curator) = snapshot.curator_quality {
            if curator < self.thresholds.curator_quality_min {
                aggregated.hard_signals.push(FailureSignal::new(
                    FailureSignalCode::CuratorLowQuality,
                    FailureSeverity::Hard,
                    format!(
                        "Curator quality below threshold: {:.3} < {:.3}",
                        curator, self.thresholds.curator_quality_min
                    ),
                    curator,
                ));
            }
        }

        aggregated
    }
}
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Signals computed from various subsystems indicating potential failure modes.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum FailureSignalCode {
    RougeBelowThreshold,
    EntropyDeltaSpike,
    EntropyDeltaFlatline,
    CompassThreat,
    CompassHealing,
    MctsConfidenceLow,
    VllmFallback,
    CuratorLowQuality,
    Custom(&'static str),
}

impl FailureSignalCode {
    pub fn as_str(&self) -> &'static str {
        match self {
            FailureSignalCode::RougeBelowThreshold => "rouge_low",
            FailureSignalCode::EntropyDeltaSpike => "entropy_spike",
            FailureSignalCode::EntropyDeltaFlatline => "entropy_flat",
            FailureSignalCode::CompassThreat => "compass_threat",
            FailureSignalCode::CompassHealing => "compass_healing",
            FailureSignalCode::MctsConfidenceLow => "mcts_confidence_low",
            FailureSignalCode::VllmFallback => "vllm_fallback",
            FailureSignalCode::CuratorLowQuality => "curator_low_quality",
            FailureSignalCode::Custom(_) => "custom",
        }
    }
}

/// Severity level describing the impact of a failure signal.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum FailureSeverity {
    Soft,
    Hard,
}

/// Concrete failure signal data point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureSignal {
    pub code: FailureSignalCode,
    pub severity: FailureSeverity,
    pub description: String,
    pub value: f64,
}

impl FailureSignal {
    pub fn new(
        code: FailureSignalCode,
        severity: FailureSeverity,
        description: impl Into<String>,
        value: f64,
    ) -> Self {
        Self {
            code,
            severity,
            description: description.into(),
            value,
        }
    }
}

/// Snapshot of adaptive metrics that feed into failure detection logic.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct AdaptiveMetricsSnapshot {
    pub rouge: Option<f64>,
    pub entropy: Option<f64>,
    pub entropy_delta: Option<f64>,
    pub ucb1: Option<f64>,
    pub compass_state: Option<String>,
    pub curator_quality: Option<f64>,
    pub fallbacks_triggered: usize,
}

/// Threshold configuration for mapping metrics to failure signals.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureSignalThresholds {
    pub rouge_min: f64,
    pub entropy_delta_spike: f64,
    pub entropy_delta_flatline: f64,
    pub ucb1_min: f64,
    pub curator_quality_min: f64,
}

impl Default for FailureSignalThresholds {
    fn default() -> Self {
        Self {
            rouge_min: 0.5,
            entropy_delta_spike: 0.1,
            entropy_delta_flatline: 0.01,
            ucb1_min: 0.3,
            curator_quality_min: 0.7,
        }
    }
}

/// Aggregated representation of failure signals for the current step.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct AggregatedFailureSignals {
    pub soft_signals: Vec<FailureSignal>,
    pub hard_signals: Vec<FailureSignal>,
}

impl AggregatedFailureSignals {
    pub fn has_hard_failure(&self) -> bool {
        !self.hard_signals.is_empty()
    }

    pub fn has_soft_failure(&self) -> bool {
        !self.soft_signals.is_empty()
    }

    pub fn all_signals(&self) -> impl Iterator<Item = &FailureSignal> {
        self.soft_signals.iter().chain(self.hard_signals.iter())
    }
}

/// Aggregates metrics into soft/hard failure signals.
#[derive(Debug, Clone)]
pub struct FailureSignalAggregator {
    thresholds: FailureSignalThresholds,
    recent_history: HashMap<&'static str, f64>,
}

impl FailureSignalAggregator {
    pub fn new(thresholds: FailureSignalThresholds) -> Self {
        Self {
            thresholds,
            recent_history: HashMap::new(),
        }
    }

    pub fn update_thresholds(&mut self, thresholds: FailureSignalThresholds) {
        self.thresholds = thresholds;
    }

    pub fn aggregate(&mut self, snapshot: &AdaptiveMetricsSnapshot) -> AggregatedFailureSignals {
        let mut aggregated = AggregatedFailureSignals::default();

        if let Some(rouge) = snapshot.rouge {
            if rouge < self.thresholds.rouge_min {
                aggregated.hard_signals.push(FailureSignal::new(
                    FailureSignalCode::RougeBelowThreshold,
                    FailureSeverity::Hard,
                    format!("ROUGE below threshold: {:.3} < {:.3}", rouge, self.thresholds.rouge_min),
                    rouge,
                ));
            }
        }

        if let Some(delta) = snapshot.entropy_delta {
            if delta > self.thresholds.entropy_delta_spike {
                aggregated.hard_signals.push(FailureSignal::new(
                    FailureSignalCode::EntropyDeltaSpike,
                    FailureSeverity::Hard,
                    format!(
                        "Entropy delta spike detected: {:.3} > {:.3}",
                        delta, self.thresholds.entropy_delta_spike
                    ),
                    delta,
                ));
            } else if delta.abs() < self.thresholds.entropy_delta_flatline {
                aggregated.soft_signals.push(FailureSignal::new(
                    FailureSignalCode::EntropyDeltaFlatline,
                    FailureSeverity::Soft,
                    format!(
                        "Entropy delta flatline: |Δ| {:.4} < {:.4}",
                        delta.abs(),
                        self.thresholds.entropy_delta_flatline
                    ),
                    delta,
                ));
            }
        }

        if let Some(ucb1) = snapshot.ucb1 {
            if ucb1 < self.thresholds.ucb1_min {
                aggregated.soft_signals.push(FailureSignal::new(
                    FailureSignalCode::MctsConfidenceLow,
                    FailureSeverity::Soft,
                    format!("MCTS UCB1 low: {:.3} < {:.3}", ucb1, self.thresholds.ucb1_min),
                    ucb1,
                ));
            }
        }

        if let Some(compass_state) = snapshot.compass_state.as_deref() {
            if compass_state == "10" {
                aggregated.soft_signals.push(FailureSignal::new(
                    FailureSignalCode::CompassThreat,
                    FailureSeverity::Soft,
                    "Compass indicated threat state",
                    1.0,
                ));
            }
        }

        if snapshot.fallbacks_triggered > 0 {
            aggregated.soft_signals.push(FailureSignal::new(
                FailureSignalCode::VllmFallback,
                FailureSeverity::Soft,
                format!("{} fallback(s) triggered", snapshot.fallbacks_triggered),
                snapshot.fallbacks_triggered as f64,
            ));
        }

        if let Some(curator) = snapshot.curator_quality {
            if curator < self.thresholds.curator_quality_min {
                aggregated.hard_signals.push(FailureSignal::new(
                    FailureSignalCode::CuratorLowQuality,
                    FailureSeverity::Hard,
                    format!(
                        "Curator quality below threshold: {:.3} < {:.3}",
                        curator, self.thresholds.curator_quality_min
                    ),
                    curator,
                ));
            }
        }

        aggregated
    }
}

// Simple failure evaluation wrapper
pub mod simple {
    use crate::metrics;

    pub struct FailureSignals;

    impl FailureSignals {
        pub fn evaluate(rouge: f64, delta: f64, curator: f64, ucb1: f64) -> (String, String) {
            metrics::evaluate_failure(rouge, delta, curator, ucb1)
        }
    }
}
