//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

use std::collections::HashMap;
use tracing::{info, warn, error};
use std::time::{Duration, Instant};
use std::hash::Hash;

/// Anti-Insanity Loop Detection for AI Systems
/// Prevents doing the same stupid thing over and over expecting different results
pub struct AntiInsanityLoop<T>
where
    T: Clone + Hash + Eq + std::fmt::Debug,
{
    /// Track what we've tried and how many times
    attempt_history: HashMap<T, AttemptRecord>,
    /// Maximum attempts before we force a different approach
    max_attempts: usize,
    /// Time window for considering attempts "recent"
    time_window: Duration,
    /// Threshold for considering something a repeated failure (0.0-1.0)
    failure_rate_threshold: f32,
    /// Number of attempts that constitute a "tight loop"
    tight_loop_threshold: usize,
    /// Time window for tight loop detection
    tight_loop_window: Duration,
    /// Maximum number of outcomes to keep in history
    max_outcome_history: usize,
    /// Callback for when insanity is detected
    insanity_callback: Option<Box<dyn Fn(&T, usize) -> bool>>,
}

#[derive(Debug, Clone)]
struct AttemptRecord {
    count: usize,
    first_attempt: Instant,
    last_attempt: Instant,
    outcomes: Vec<AttemptOutcome>,
}

#[derive(Debug, Clone)]
pub enum AttemptOutcome {
    Success,
    Failure(String),
    Timeout,
    Stuck,
}

#[derive(Debug)]
pub enum InsanityDetection {
    /// We've tried this exact thing too many times
    RepeatedFailure { attempts: usize, action: String },
    /// We're stuck in a tight loop
    TightLoop { duration: Duration },
    /// Same failure pattern keeps happening
    PatternLoop { pattern: String, occurrences: usize },
}

impl<T> AntiInsanityLoop<T>
where
    T: Clone + Hash + Eq + std::fmt::Debug,
{
    /// Create with default reasonable values
    pub fn new(max_attempts: usize, time_window: Duration) -> Self {
        Self {
            attempt_history: HashMap::new(),
            max_attempts,
            time_window,
            failure_rate_threshold: 0.8,
            tight_loop_threshold: 3,
            tight_loop_window: Duration::from_secs(30),
            max_outcome_history: 10,
            insanity_callback: None,
        }
    }

    /// Create with full configuration control
    pub fn with_config(
        max_attempts: usize,
        time_window: Duration,
        failure_rate_threshold: f32,
        tight_loop_threshold: usize,
        tight_loop_window: Duration,
        max_outcome_history: usize,
    ) -> Self {
        Self {
            attempt_history: HashMap::new(),
            max_attempts,
            time_window,
            failure_rate_threshold,
            tight_loop_threshold,
            tight_loop_window,
            max_outcome_history,
            insanity_callback: None,
        }
    }

    /// Set a callback for when insanity is detected
    pub fn on_insanity<F>(mut self, callback: F) -> Self
    where
        F: Fn(&T, usize) -> bool + 'static,
    {
        self.insanity_callback = Some(Box::new(callback));
        self
    }

    /// Check if we should attempt this action or if it's insane
    pub fn should_attempt(&mut self, action: &T) -> Result<(), InsanityDetection> {
        let now = Instant::now();

        // Clean up old attempts outside time window
        self.cleanup_old_attempts(now);

        // Check if we have a record for this action
        if let Some(record) = self.attempt_history.get(action) {
            // Check for repeated failures
            if record.count >= self.max_attempts {
                let failure_rate = record.outcomes.iter()
                    .filter(|o| matches!(o, AttemptOutcome::Failure(_) | AttemptOutcome::Timeout | AttemptOutcome::Stuck))
                    .count();

                if (failure_rate as f32 / record.count as f32) > self.failure_rate_threshold {
                    return Err(InsanityDetection::RepeatedFailure {
                        attempts: record.count,
                        action: format!("{:?}", action),
                    });
                }
            }

            // Check for tight loops (too many attempts in short time)
            if record.count >= self.tight_loop_threshold
                && now.duration_since(record.first_attempt) < self.tight_loop_window {
                return Err(InsanityDetection::TightLoop {
                    duration: now.duration_since(record.first_attempt),
                });
            }
        }

        Ok(())
    }

    /// Record an attempt and its outcome
    pub fn record_attempt(&mut self, action: T, outcome: AttemptOutcome) {
        let now = Instant::now();

        let record = self.attempt_history.entry(action.clone()).or_insert(AttemptRecord {
            count: 0,
            first_attempt: now,
            last_attempt: now,
            outcomes: Vec::new(),
        });

        record.count += 1;
        record.last_attempt = now;
        record.outcomes.push(outcome.clone());

        // Keep only recent outcomes
        if record.outcomes.len() > self.max_outcome_history {
            record.outcomes.drain(0..(record.outcomes.len() - self.max_outcome_history));
        }

        // Check for insanity patterns
        if let Some(ref callback) = self.insanity_callback {
            if record.count >= self.max_attempts {
                let should_continue = callback(&action, record.count);
                if !should_continue {
                    // Force clear this action to prevent further attempts
                    self.attempt_history.remove(&action);
                }
            }
        }
    }

    /// Force a different approach by clearing history for an action
    pub fn force_new_approach(&mut self, action: &T) {
        self.attempt_history.remove(action);
    }

    /// Get statistics about our attempt patterns
    pub fn get_stats(&self) -> HashMap<String, usize> {
        let mut stats = HashMap::new();

        for (_, record) in &self.attempt_history {
            let success_count = record.outcomes.iter()
                .filter(|o| matches!(o, AttemptOutcome::Success))
                .count();
            let failure_count = record.outcomes.iter()
                .filter(|o| matches!(o, AttemptOutcome::Failure(_)))
                .count();
            let timeout_count = record.outcomes.iter()
                .filter(|o| matches!(o, AttemptOutcome::Timeout))
                .count();
            let stuck_count = record.outcomes.iter()
                .filter(|o| matches!(o, AttemptOutcome::Stuck))
                .count();

            *stats.entry("total_attempts".to_string()).or_insert(0) += record.count;
            *stats.entry("successes".to_string()).or_insert(0) += success_count;
            *stats.entry("failures".to_string()).or_insert(0) += failure_count;
            *stats.entry("timeouts".to_string()).or_insert(0) += timeout_count;
            *stats.entry("stuck_loops".to_string()).or_insert(0) += stuck_count;
        }

        stats
    }

    fn cleanup_old_attempts(&mut self, now: Instant) {
        self.attempt_history.retain(|_, record| {
            now.duration_since(record.last_attempt) < self.time_window
        });
    }
}

/// Macro for easy anti-insanity checking
#[macro_export]
macro_rules! check_insanity {
    ($loop_detector:expr, $action:expr) => {
        match $loop_detector.should_attempt(&$action) {
            Ok(()) => {},
            Err(detection) => {
                tracing::error!("🚨 INSANITY DETECTED: {:?}", detection);
                return Err(format!("Prevented insane loop: {:?}", detection).into());
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_prevents_repeated_failures() {
        let mut detector = AntiInsanityLoop::new(3, Duration::from_secs(60));

        let action = "curl_same_endpoint";

        // First few attempts should be allowed
        assert!(detector.should_attempt(&action).is_ok());
        detector.record_attempt(action, AttemptOutcome::Failure("timeout".to_string()));

        assert!(detector.should_attempt(&action).is_ok());
        detector.record_attempt(action, AttemptOutcome::Failure("timeout".to_string()));

        assert!(detector.should_attempt(&action).is_ok());
        detector.record_attempt(action, AttemptOutcome::Failure("timeout".to_string()));

        // Fourth attempt should be blocked
        assert!(detector.should_attempt(&action).is_err());
    }

    #[test]
    fn test_allows_successful_attempts() {
        let mut detector = AntiInsanityLoop::new(3, Duration::from_secs(60));

        let action = "working_endpoint";

        // Successful attempts should not be blocked
        for _ in 0..5 {
            assert!(detector.should_attempt(&action).is_ok());
            detector.record_attempt(action, AttemptOutcome::Success);
        }
    }

    #[test]
    fn test_tight_loop_detection() {
        let mut detector = AntiInsanityLoop::new(10, Duration::from_secs(60));

        let action = "rapid_fire";

        // Rapid attempts should trigger tight loop detection
        for _ in 0..4 {
            assert!(detector.should_attempt(&action).is_ok());
            detector.record_attempt(action, AttemptOutcome::Stuck);
            thread::sleep(Duration::from_millis(1)); // Very short delay
        }

        // Should detect tight loop
        assert!(matches!(
            detector.should_attempt(&action),
            Err(InsanityDetection::TightLoop { .. })
        ));
    }

    #[test]
    fn test_custom_thresholds() {
        let mut detector = AntiInsanityLoop::with_config(
            5,                              // max_attempts
            Duration::from_secs(120),       // time_window
            0.6,                            // failure_rate_threshold
            2,                              // tight_loop_threshold
            Duration::from_secs(10),        // tight_loop_window
            20,                             // max_outcome_history
        );

        let action = "custom_config";

        // Should respect custom thresholds
        detector.record_attempt(action, AttemptOutcome::Failure("test".to_string()));
        detector.record_attempt(action, AttemptOutcome::Failure("test".to_string()));

        // Tight loop with threshold of 2
        assert!(matches!(
            detector.should_attempt(&action),
            Err(InsanityDetection::TightLoop { .. })
        ));
    }
}
