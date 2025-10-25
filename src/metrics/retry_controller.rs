use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

use super::failure_signals::{AggregatedFailureSignals, FailureSeverity};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum AdaptiveRetryLevel {
    Level0,
    Level1,
    Level2,
    Level3,
    Level4,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryControllerConfig {
    pub base_retry_delay_ms: u64,
    pub max_retries: u32,
    pub jitter_pct_range: (f64, f64),
}

impl Default for RetryControllerConfig {
    fn default() -> Self {
        Self {
            base_retry_delay_ms: 200,
            max_retries: 10,
            jitter_pct_range: (0.1, 0.3),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveRetryDecision {
    pub level: AdaptiveRetryLevel,
    pub should_retry: bool,
    pub next_delay: Option<Duration>,
    pub parameter_adjustments: Vec<String>,
}

impl AdaptiveRetryDecision {
    pub fn no_retry() -> Self {
        Self {
            level: AdaptiveRetryLevel::Level0,
            should_retry: false,
            next_delay: None,
            parameter_adjustments: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AdaptiveRetryController {
    config: RetryControllerConfig,
    attempt: u32,
    hard_failures_in_row: u32,
    last_decision: Option<(Instant, AdaptiveRetryDecision)>,
}

impl AdaptiveRetryController {
    pub fn new(config: RetryControllerConfig) -> Self {
        Self {
            config,
            attempt: 0,
            hard_failures_in_row: 0,
            last_decision: None,
        }
    }

    pub fn reset(&mut self) {
        self.attempt = 0;
        self.hard_failures_in_row = 0;
        self.last_decision = None;
    }

    pub fn register_success(&mut self) {
        self.reset();
    }

    pub fn next_decision(&mut self, signals: &AggregatedFailureSignals) -> AdaptiveRetryDecision {
        let mut decision = AdaptiveRetryDecision::no_retry();

        if !signals.has_soft_failure() && !signals.has_hard_failure() {
            self.register_success();
            return decision;
        }

        if self.attempt >= self.config.max_retries {
            decision.should_retry = false;
            decision.level = AdaptiveRetryLevel::Level4;
            decision.parameter_adjustments.push("Max retries reached".to_string());
            return decision;
        }

        self.attempt += 1;

        if signals.has_hard_failure() {
            self.hard_failures_in_row += 1;
        } else {
            self.hard_failures_in_row = 0;
        }

        decision.should_retry = true;
        decision.level = self.determine_level(signals);
        decision.parameter_adjustments = self.derive_adjustments(decision.level, signals);

        let backoff = self.calculate_backoff(self.attempt);
        decision.next_delay = Some(backoff);
        self.last_decision = Some((Instant::now(), decision.clone()));

        decision
    }

    fn determine_level(&self, signals: &AggregatedFailureSignals) -> AdaptiveRetryLevel {
        if signals.hard_signals.iter().any(|s| matches!(s.severity, FailureSeverity::Hard)) {
            match self.hard_failures_in_row {
                0 | 1 => AdaptiveRetryLevel::Level2,
                2..=3 => AdaptiveRetryLevel::Level3,
                _ => AdaptiveRetryLevel::Level4,
            }
        } else if signals.has_soft_failure() {
            AdaptiveRetryLevel::Level1
        } else {
            AdaptiveRetryLevel::Level0
        }
    }

    fn derive_adjustments(
        &self,
        level: AdaptiveRetryLevel,
        signals: &AggregatedFailureSignals,
    ) -> Vec<String> {
        match level {
            AdaptiveRetryLevel::Level0 => Vec::new(),
            AdaptiveRetryLevel::Level1 => vec!["temperature += 0.1".to_string(), "add_prompt_noise = true".to_string()],
            AdaptiveRetryLevel::Level2 => vec![
                "novelty_threshold += entropy_delta * 0.05".to_string(),
                "reset_generation_seed".to_string(),
            ],
            AdaptiveRetryLevel::Level3 => vec![
                "mcts_c += 0.1".to_string(),
                "top_p -= 0.05".to_string(),
                "retrieval_top_k += 2".to_string(),
            ],
            AdaptiveRetryLevel::Level4 => signals
                .hard_signals
                .iter()
                .map(|s| format!("persist_failure:{}", s.code.as_str()))
                .collect(),
        }
    }

    fn calculate_backoff(&self, attempt: u32) -> Duration {
        let base = self.config.base_retry_delay_ms as f64;
        let exponent = 2_u32.saturating_pow(attempt.min(10));
        let jitter_range = self.config.jitter_pct_range;
        let jitter = jitter_range.0
            + (rand::random::<f64>() * (jitter_range.1 - jitter_range.0));
        let delay_ms = base * exponent as f64 * (1.0 + jitter);
        Duration::from_millis(delay_ms.min((u64::MAX / 2) as f64) as u64)
    }
}
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

use super::failure_signals::{AggregatedFailureSignals, FailureSeverity};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum AdaptiveRetryLevel {
    Level0, // No retry required
    Level1, // Soft adjustments
    Level2, // Core retry + tune
    Level3, // Escalated tuning
    Level4, // Systemic adaptation (LoRA/Meta)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryControllerConfig {
    pub base_retry_delay_ms: u64,
    pub max_retries: u32,
    pub jitter_pct_range: (f64, f64),
}

impl Default for RetryControllerConfig {
    fn default() -> Self {
        Self {
            base_retry_delay_ms: 200,
            max_retries: 10,
            jitter_pct_range: (0.1, 0.3),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveRetryDecision {
    pub level: AdaptiveRetryLevel,
    pub should_retry: bool,
    pub next_delay: Option<Duration>,
    pub parameter_adjustments: Vec<String>,
}

impl AdaptiveRetryDecision {
    pub fn no_retry() -> Self {
        Self {
            level: AdaptiveRetryLevel::Level0,
            should_retry: false,
            next_delay: None,
            parameter_adjustments: Vec::new(),
        }
    }
}

/// Adaptive retry controller implementing exponential backoff with jitter.
#[derive(Debug, Clone)]
pub struct AdaptiveRetryController {
    config: RetryControllerConfig,
    attempt: u32,
    hard_failures_in_row: u32,
    last_decision: Option<(Instant, AdaptiveRetryDecision)>,
}

impl AdaptiveRetryController {
    pub fn new(config: RetryControllerConfig) -> Self {
        Self {
            config,
            attempt: 0,
            hard_failures_in_row: 0,
            last_decision: None,
        }
    }

    pub fn reset(&mut self) {
        self.attempt = 0;
        self.hard_failures_in_row = 0;
        self.last_decision = None;
    }

    pub fn register_success(&mut self) {
        self.reset();
    }

    pub fn next_decision(&mut self, signals: &AggregatedFailureSignals) -> AdaptiveRetryDecision {
        let mut decision = AdaptiveRetryDecision::no_retry();

        if !signals.has_soft_failure() && !signals.has_hard_failure() {
            self.register_success();
            return decision;
        }

        if self.attempt >= self.config.max_retries {
            decision.should_retry = false;
            decision.level = AdaptiveRetryLevel::Level4;
            decision.parameter_adjustments.push("Max retries reached".to_string());
            return decision;
        }

        self.attempt += 1;

        let has_hard = signals.has_hard_failure();
        if has_hard {
            self.hard_failures_in_row += 1;
        } else {
            self.hard_failures_in_row = 0;
        }

        decision.should_retry = true;
        decision.level = self.determine_level(signals);
        decision.parameter_adjustments = self.derive_adjustments(decision.level, signals);

        let backoff = self.calculate_backoff(self.attempt);
        decision.next_delay = Some(backoff);
        self.last_decision = Some((Instant::now(), decision.clone()));

        decision
    }

    fn determine_level(&self, signals: &AggregatedFailureSignals) -> AdaptiveRetryLevel {
        if signals.hard_signals.iter().any(|s| matches!(s.severity, FailureSeverity::Hard)) {
            match self.hard_failures_in_row {
                0 | 1 => AdaptiveRetryLevel::Level2,
                2..=3 => AdaptiveRetryLevel::Level3,
                _ => AdaptiveRetryLevel::Level4,
            }
        } else if signals.has_soft_failure() {
            AdaptiveRetryLevel::Level1
        } else {
            AdaptiveRetryLevel::Level0
        }
    }

    fn derive_adjustments(
        &self,
        level: AdaptiveRetryLevel,
        signals: &AggregatedFailureSignals,
    ) -> Vec<String> {
        match level {
            AdaptiveRetryLevel::Level0 => Vec::new(),
            AdaptiveRetryLevel::Level1 => vec!["temperature += 0.1".to_string(), "add_prompt_noise = true".to_string()],
            AdaptiveRetryLevel::Level2 => vec![
                "novelty_threshold += entropy_delta * 0.05".to_string(),
                "reset_generation_seed".to_string(),
            ],
            AdaptiveRetryLevel::Level3 => vec![
                "mcts_c += 0.1".to_string(),
                "top_p -= 0.05".to_string(),
                "retrieval_top_k += 2".to_string(),
            ],
            AdaptiveRetryLevel::Level4 => {
                let repeated = signals
                    .hard_signals
                    .iter()
                    .map(|s| format!("persist_failure:{}", s.code.as_str()))
                    .collect();
                repeated
            }
        }
    }

    fn calculate_backoff(&self, attempt: u32) -> Duration {
        let base = self.config.base_retry_delay_ms as f64;
        let exponent = 2_u32.saturating_pow(attempt.min(10));
        let jitter_range = self.config.jitter_pct_range;
        let jitter = jitter_range.0
            + (rand::random::<f64>() * (jitter_range.1 - jitter_range.0));
        let delay_ms = base * exponent as f64 * (1.0 + jitter);
        Duration::from_millis(delay_ms.min((u64::MAX / 2) as f64) as u64)
    }
}

