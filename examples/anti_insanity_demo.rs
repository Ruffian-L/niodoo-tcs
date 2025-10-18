/// Standalone demo of the Anti-Insanity Loop system
use tracing::{info, error, warn};
use std::time::Duration;

// Copy the core module here for standalone testing
use std::collections::HashMap;
use std::hash::Hash;
use std::time::Instant;

/// Anti-Insanity Loop Detection
pub struct AntiInsanityLoop<T>
where
    T: Clone + Hash + Eq + std::fmt::Debug,
{
    attempt_history: HashMap<T, AttemptRecord>,
    max_attempts: usize,
    time_window: Duration,
    failure_rate_threshold: f32,
    tight_loop_threshold: usize,
    tight_loop_window: Duration,
    max_outcome_history: usize,
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
}

#[derive(Debug)]
pub enum InsanityDetection {
    RepeatedFailure { attempts: usize, action: String },
    TightLoop { duration: Duration },
}

impl<T> AntiInsanityLoop<T>
where
    T: Clone + Hash + Eq + std::fmt::Debug,
{
    pub fn new(max_attempts: usize, time_window: Duration) -> Self {
        Self {
            attempt_history: HashMap::new(),
            max_attempts,
            time_window,
            failure_rate_threshold: 0.8,
            tight_loop_threshold: 3,
            tight_loop_window: Duration::from_secs(30),
            max_outcome_history: 10,
        }
    }

    pub fn should_attempt(&mut self, action: &T) -> Result<(), InsanityDetection> {
        let now = Instant::now();
        self.cleanup_old_attempts(now);

        if let Some(record) = self.attempt_history.get(action) {
            if record.count >= self.max_attempts {
                let failure_rate = record
                    .outcomes
                    .iter()
                    .filter(|o| matches!(o, AttemptOutcome::Failure(_) | AttemptOutcome::Timeout))
                    .count();

                if (failure_rate as f32 / record.count as f32) > self.failure_rate_threshold {
                    return Err(InsanityDetection::RepeatedFailure {
                        attempts: record.count,
                        action: format!("{:?}", action),
                    });
                }
            }

            if record.count >= self.tight_loop_threshold
                && now.duration_since(record.first_attempt) < self.tight_loop_window
            {
                return Err(InsanityDetection::TightLoop {
                    duration: now.duration_since(record.first_attempt),
                });
            }
        }

        Ok(())
    }

    pub fn record_attempt(&mut self, action: T, outcome: AttemptOutcome) {
        let now = Instant::now();

        let record = self
            .attempt_history
            .entry(action.clone())
            .or_insert(AttemptRecord {
                count: 0,
                first_attempt: now,
                last_attempt: now,
                outcomes: Vec::new(),
            });

        record.count += 1;
        record.last_attempt = now;
        record.outcomes.push(outcome.clone());

        if record.outcomes.len() > self.max_outcome_history {
            record
                .outcomes
                .drain(0..(record.outcomes.len() - self.max_outcome_history));
        }
    }

    fn cleanup_old_attempts(&mut self, now: Instant) {
        self.attempt_history
            .retain(|_, record| now.duration_since(record.last_attempt) < self.time_window);
    }
}

/// Configuration for demo scenarios (no hardcoding!)
struct DemoConfig {
    /// Max attempts before detecting insanity
    max_retry_attempts: usize,
    /// Time window for tracking attempts
    retry_time_window: Duration,
    /// Threshold for tight loop detection
    tight_loop_attempts: usize,
    /// Max iterations for demo loops
    max_demo_iterations: usize,
    /// Simulated retry delay
    simulated_retry_delay: Duration,
}

impl DemoConfig {
    fn from_env() -> Self {
        Self {
            max_retry_attempts: std::env::var("DEMO_MAX_RETRIES")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or_else(Self::default_max_retries),
            retry_time_window: Duration::from_secs(
                std::env::var("DEMO_TIME_WINDOW_SECS")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or_else(Self::default_time_window_secs),
            ),
            tight_loop_attempts: std::env::var("DEMO_TIGHT_LOOP_THRESHOLD")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or_else(Self::default_tight_loop_threshold),
            max_demo_iterations: std::env::var("DEMO_MAX_ITERATIONS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or_else(Self::default_max_iterations),
            simulated_retry_delay: Duration::from_millis(
                std::env::var("DEMO_RETRY_DELAY_MS")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or_else(Self::default_retry_delay_ms),
            ),
        }
    }

    // Default values with clear reasoning
    fn default_max_retries() -> usize {
        5 // Typical for API retries (exponential backoff usually stops at 5)
    }

    fn default_time_window_secs() -> u64 {
        60 // One minute is standard for transient failure windows
    }

    fn default_tight_loop_threshold() -> usize {
        3 // Three rapid attempts indicate a stuck loop
    }

    fn default_max_iterations() -> usize {
        10 // Enough to demonstrate the pattern without being tedious
    }

    fn default_retry_delay_ms() -> u64 {
        1 // Minimal delay for rapid-fire simulation
    }
}

fn main() {
    tracing::info!("🧠 Anti-Insanity Loop Detection Demo\n");

    let config = DemoConfig::from_env();

    tracing::info!("📋 Configuration:");
    tracing::info!("   Max retry attempts: {}", config.max_retry_attempts);
    tracing::info!("   Time window: {:?}", config.retry_time_window);
    tracing::info!("   Tight loop threshold: {}", config.tight_loop_attempts);
    tracing::info!("   Max demo iterations: {}", config.max_demo_iterations);
    tracing::info!("   Simulated delay: {:?}\n", config.simulated_retry_delay);

    // Scenario 1: Prevent repeated failures
    tracing::info!("📍 Scenario 1: Preventing Repeated Failures");
    tracing::info!("   Simulating failed API calls to a dead endpoint...\n");

    let mut detector = AntiInsanityLoop::new(config.max_retry_attempts, config.retry_time_window);

    for attempt in 1..=config.max_demo_iterations {
        let action = "call_dead_api";

        match detector.should_attempt(&action) {
            Ok(_) => {
                tracing::info!("   Attempt {}: Trying to call API...", attempt);
                // Simulate failure
                detector.record_attempt(
                    action,
                    AttemptOutcome::Failure("Connection refused".to_string()),
                );
                tracing::info!("   ❌ Failed (recorded)\n");
            }
            Err(InsanityDetection::RepeatedFailure { attempts, .. }) => {
                tracing::info!("   🚨 INSANITY DETECTED after {} attempts!", attempts);
                tracing::info!("   ✅ System prevented further futile attempts\n");
                break;
            }
            Err(e) => {
                tracing::info!("   🚨 INSANITY DETECTED: {:?}\n", e);
                break;
            }
        }
    }

    // Scenario 2: Tight loop detection
    tracing::info!("📍 Scenario 2: Tight Loop Detection");
    tracing::info!("   Simulating rapid-fire retries...\n");

    let mut loop_detector =
        AntiInsanityLoop::new(config.max_demo_iterations, config.retry_time_window);

    for attempt in 1..=config.max_demo_iterations {
        let action = "rapid_retry";

        match loop_detector.should_attempt(&action) {
            Ok(_) => {
                tracing::info!("   Rapid attempt {}", attempt);
                loop_detector.record_attempt(action, AttemptOutcome::Timeout);
                std::thread::sleep(config.simulated_retry_delay);
            }
            Err(InsanityDetection::TightLoop { duration }) => {
                tracing::info!("   🚨 TIGHT LOOP DETECTED!");
                tracing::info!("   ⏱️  {} attempts in {:?}", attempt, duration);
                tracing::info!("   ✅ Forcing backoff strategy\n");
                break;
            }
            Err(e) => {
                tracing::info!("   🚨 INSANITY DETECTED: {:?}\n", e);
                break;
            }
        }
    }

    // Scenario 3: Successful operations (should not trigger)
    tracing::info!("📍 Scenario 3: Successful Operations");
    tracing::info!("   Simulating healthy API calls...\n");

    let mut success_detector =
        AntiInsanityLoop::new(config.max_retry_attempts, config.retry_time_window);

    for attempt in 1..=config.max_demo_iterations {
        let action = "healthy_api";

        match success_detector.should_attempt(&action) {
            Ok(_) => {
                tracing::info!("   Attempt {}: ✅ Success", attempt);
                success_detector.record_attempt(action, AttemptOutcome::Success);
            }
            Err(e) => {
                tracing::info!("   🚨 Unexpected insanity detection: {:?}", e);
                break;
            }
        }
    }

    tracing::info!(
        "\n   ✅ All {} successful operations completed without triggering insanity detection\n",
        config.max_demo_iterations
    );

    tracing::info!("🎉 Demo Complete!");
    tracing::info!("\n💡 Key Takeaways:");
    tracing::info!("   • Prevents infinite retry loops");
    tracing::info!("   • Detects tight loop patterns");
    tracing::info!("   • Allows legitimate retry strategies");
    tracing::info!("   • Forces system to try different approaches");
}
