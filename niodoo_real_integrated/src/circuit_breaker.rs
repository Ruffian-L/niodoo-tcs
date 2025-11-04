//! Circuit Breaker Pattern for External Service Resilience
//!
//! Implements circuit breaker pattern with exponential backoff for Qdrant and vLLM services.
//! Prevents cascading failures by failing fast when services are down.

use anyhow::{Context, Result};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Circuit breaker states
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    /// Circuit is closed - requests pass through normally
    Closed,
    /// Circuit is open - requests fail immediately
    Open,
    /// Circuit is half-open - testing if service recovered
    HalfOpen,
}

/// Circuit breaker configuration
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// Failure threshold - open circuit after this many failures
    pub failure_threshold: u32,
    /// Success threshold - close circuit after this many successes in half-open state
    pub success_threshold: u32,
    /// Timeout duration - how long circuit stays open before moving to half-open
    pub timeout: Duration,
    /// Base delay for exponential backoff
    pub base_delay: Duration,
    /// Maximum delay cap for exponential backoff
    pub max_delay: Duration,
    /// Backoff exponent multiplier
    pub backoff_exponent: f64,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self::from_env()
    }
}

impl CircuitBreakerConfig {
    /// Create circuit breaker config from environment variables with defaults
    /// Environment variables (all optional):
    /// - CIRCUIT_BREAKER_FAILURE_THRESHOLD: Failure threshold (default: 5)
    /// - CIRCUIT_BREAKER_SUCCESS_THRESHOLD: Success threshold (default: 2)
    /// - CIRCUIT_BREAKER_TIMEOUT_SECS: Timeout in seconds (default: 60)
    /// - CIRCUIT_BREAKER_BASE_DELAY_MS: Base delay in milliseconds (default: 100)
    /// - CIRCUIT_BREAKER_MAX_DELAY_SECS: Maximum delay in seconds (default: 30)
    /// - CIRCUIT_BREAKER_BACKOFF_EXPONENT: Backoff exponent (default: 2.0)
    pub fn from_env() -> Self {
        Self {
            failure_threshold: std::env::var("CIRCUIT_BREAKER_FAILURE_THRESHOLD")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(5),
            success_threshold: std::env::var("CIRCUIT_BREAKER_SUCCESS_THRESHOLD")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(2),
            timeout: Duration::from_secs(
                std::env::var("CIRCUIT_BREAKER_TIMEOUT_SECS")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(60)
            ),
            base_delay: Duration::from_millis(
                std::env::var("CIRCUIT_BREAKER_BASE_DELAY_MS")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(100)
            ),
            max_delay: Duration::from_secs(
                std::env::var("CIRCUIT_BREAKER_MAX_DELAY_SECS")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(30)
            ),
            backoff_exponent: std::env::var("CIRCUIT_BREAKER_BACKOFF_EXPONENT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(2.0),
        }
    }
}

/// Circuit breaker state tracking
#[derive(Debug)]
struct CircuitBreakerState {
    state: CircuitState,
    failure_count: u32,
    success_count: u32,
    last_failure_time: Option<Instant>,
    consecutive_failures: u32,
}

impl Default for CircuitBreakerState {
    fn default() -> Self {
        Self {
            state: CircuitState::Closed,
            failure_count: 0,
            success_count: 0,
            last_failure_time: None,
            consecutive_failures: 0,
        }
    }
}

/// Circuit breaker for external service calls
#[derive(Clone)]
pub struct CircuitBreaker {
    config: CircuitBreakerConfig,
    state: Arc<RwLock<CircuitBreakerState>>,
    service_name: String,
}

impl CircuitBreaker {
    /// Create a new circuit breaker
    pub fn new(service_name: impl Into<String>, config: CircuitBreakerConfig) -> Self {
        Self {
            config,
            state: Arc::new(RwLock::new(CircuitBreakerState::default())),
            service_name: service_name.into(),
        }
    }

    /// Execute a function with circuit breaker protection
    pub async fn call<F, Fut, T>(&self, f: F) -> Result<T>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<T>>,
    {
        // Check if circuit is open
        {
            let guard = self.state.read().await;
            match guard.state {
                CircuitState::Open => {
                    if let Some(last_failure) = guard.last_failure_time {
                        if last_failure.elapsed() >= self.config.timeout {
                            // Timeout expired - move to half-open
                            drop(guard);
                            let mut write_guard = self.state.write().await;
                            write_guard.state = CircuitState::HalfOpen;
                            write_guard.success_count = 0;
                            info!(
                                service = %self.service_name,
                                "Circuit breaker moving to half-open state"
                            );
                        } else {
                            // Still in timeout - fail fast
                            let remaining = self.config.timeout - last_failure.elapsed();
                            warn!(
                                service = %self.service_name,
                                remaining_secs = remaining.as_secs(),
                                "Circuit breaker is open - failing fast"
                            );
                            return Err(anyhow::anyhow!(
                                "Circuit breaker is open for {} (remaining: {:?})",
                                self.service_name,
                                remaining
                            ));
                        }
                    }
                }
                CircuitState::HalfOpen => {
                    // Allow request through to test recovery
                }
                CircuitState::Closed => {
                    // Normal operation
                }
            }
        }

        // Execute the function
        let result = f().await;

        // Update circuit state based on result
        let mut guard = self.state.write().await;
        match &result {
            Ok(_) => {
                // Success - reset failure count
                guard.consecutive_failures = 0;
                match guard.state {
                    CircuitState::Closed => {
                        // Reset failure count on success
                        guard.failure_count = 0;
                    }
                    CircuitState::HalfOpen => {
                        guard.success_count += 1;
                        if guard.success_count >= self.config.success_threshold {
                            info!(
                                service = %self.service_name,
                                "Circuit breaker recovered - moving to closed state"
                            );
                            guard.state = CircuitState::Closed;
                            guard.failure_count = 0;
                            guard.success_count = 0;
                        }
                    }
                    CircuitState::Open => {
                        // Shouldn't happen - handled above
                    }
                }
            }
            Err(e) => {
                // Failure - increment counters
                guard.failure_count += 1;
                guard.consecutive_failures += 1;
                guard.last_failure_time = Some(Instant::now());

                match guard.state {
                    CircuitState::Closed => {
                        if guard.failure_count >= self.config.failure_threshold {
                            warn!(
                                service = %self.service_name,
                                failures = guard.failure_count,
                                "Circuit breaker opening due to failures"
                            );
                            guard.state = CircuitState::Open;
                        }
                    }
                    CircuitState::HalfOpen => {
                        // Half-open test failed - back to open
                        warn!(
                            service = %self.service_name,
                            "Circuit breaker test failed - reopening circuit"
                        );
                        guard.state = CircuitState::Open;
                        guard.success_count = 0;
                    }
                    CircuitState::Open => {
                        // Already open - keep it open
                    }
                }

                debug!(
                    service = %self.service_name,
                    error = %e,
                    failures = guard.failure_count,
                    "Circuit breaker recorded failure"
                );
            }
        }

        result
    }

    /// Get current circuit state
    pub async fn state(&self) -> CircuitState {
        self.state.read().await.state
    }

    /// Get failure count
    pub async fn failure_count(&self) -> u32 {
        self.state.read().await.failure_count
    }

    /// Manually reset circuit breaker to closed state
    pub async fn reset(&self) {
        let mut guard = self.state.write().await;
        guard.state = CircuitState::Closed;
        guard.failure_count = 0;
        guard.success_count = 0;
        guard.consecutive_failures = 0;
        guard.last_failure_time = None;
        info!(service = %self.service_name, "Circuit breaker manually reset");
    }

    /// Calculate exponential backoff delay
    pub fn calculate_backoff(&self, attempt: u32) -> Duration {
        let delay_ms = self.config.base_delay.as_millis() as f64
            * self.config.backoff_exponent.powi(attempt as i32);
        let delay_ms = delay_ms.min(self.config.max_delay.as_millis() as f64);
        Duration::from_millis(delay_ms as u64)
    }
}

/// Retry helper with exponential backoff and circuit breaker
pub async fn retry_with_circuit_breaker<F, Fut, T>(
    circuit_breaker: &CircuitBreaker,
    max_attempts: u32,
    mut operation: F,
) -> Result<T>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<T>>,
{
    let mut last_error = None;

    for attempt in 0..max_attempts {
        match circuit_breaker.call(|| operation()).await {
            Ok(result) => {
                if attempt > 0 {
                    info!(
                        service = %circuit_breaker.service_name,
                        attempts = attempt + 1,
                        "Operation succeeded after retries"
                    );
                }
                return Ok(result);
            }
            Err(e) => {
                last_error = Some(e);

                if attempt < max_attempts - 1 {
                    let backoff = circuit_breaker.calculate_backoff(attempt);
                    debug!(
                        service = %circuit_breaker.service_name,
                        attempt = attempt + 1,
                        backoff_ms = backoff.as_millis(),
                        "Retrying after backoff"
                    );
                    tokio::time::sleep(backoff).await;
                }
            }
        }
    }

    Err(last_error
        .unwrap_or_else(|| anyhow::anyhow!("Operation failed after {} attempts", max_attempts)))
    .context(format!(
        "Circuit breaker protected operation failed for {}",
        circuit_breaker.service_name
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_circuit_breaker_closed_to_open() {
        let cb = CircuitBreaker::new(
            "test",
            CircuitBreakerConfig {
                failure_threshold: 3,
                success_threshold: 2,
                timeout: Duration::from_secs(1),
                ..Default::default()
            },
        );

        // Fail 3 times - should open circuit
        for _ in 0..3 {
            let _ = cb
                .call(|| async { Err::<(), _>(anyhow::anyhow!("test error")) })
                .await;
        }

        assert_eq!(cb.state().await, CircuitState::Open);
    }

    #[tokio::test]
    async fn test_circuit_breaker_recovery() {
        let cb = CircuitBreaker::new(
            "test",
            CircuitBreakerConfig {
                failure_threshold: 2,
                success_threshold: 1,
                timeout: Duration::from_millis(100),
                ..Default::default()
            },
        );

        // Open circuit
        for _ in 0..2 {
            let _ = cb
                .call(|| async { Err::<(), _>(anyhow::anyhow!("test error")) })
                .await;
        }
        assert_eq!(cb.state().await, CircuitState::Open);

        // Wait for timeout
        tokio::time::sleep(Duration::from_millis(150)).await;

        // Success should recover
        let result = cb.call(|| async { Ok::<(), _>(()) }).await;
        assert!(result.is_ok());
        assert_eq!(cb.state().await, CircuitState::Closed);
    }
}
