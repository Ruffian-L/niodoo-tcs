/*
 * 🕐 PERFORMANCE PROFILING SYSTEM 🕐
 *
 * Simple profiling utilities for performance optimization and monitoring
 * in the Niodoo consciousness engine.
 */

use once_cell::sync::OnceCell;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Performance timer for measuring execution time
#[derive(Debug, Clone)]
pub struct PerfTimer {
    start: Instant,
    label: String,
}

impl PerfTimer {
    /// Start a new performance timer
    pub fn start(label: &str) -> Self {
        Self {
            start: Instant::now(),
            label: label.to_string(),
        }
    }

    /// Stop the timer and return the elapsed duration
    pub fn stop(self) -> Duration {
        self.start.elapsed()
    }

    /// Stop the timer and log the result
    pub fn stop_and_log(self) -> Duration {
        let elapsed = self.start.elapsed();
        tracing::debug!("⏱️ {} took {:?}", self.label, elapsed);
        elapsed
    }
}

/// Global performance metrics collector
static PERF_METRICS: OnceCell<Arc<Mutex<HashMap<String, PerfMetrics>>>> = OnceCell::new();

#[derive(Debug, Clone)]
pub struct PerfMetrics {
    pub total_calls: u64,
    pub total_time: Duration,
    pub min_time: Duration,
    pub max_time: Duration,
    pub avg_time: Duration,
}

impl PerfMetrics {
    fn new() -> Self {
        Self {
            total_calls: 0,
            total_time: Duration::from_nanos(0),
            min_time: Duration::from_secs(3600), // 1 hour max initially
            max_time: Duration::from_nanos(0),
            avg_time: Duration::from_nanos(0),
        }
    }

    fn update(&mut self, duration: Duration) {
        self.total_calls += 1;
        self.total_time += duration;

        if duration < self.min_time {
            self.min_time = duration;
        }
        if duration > self.max_time {
            self.max_time = duration;
        }

        self.avg_time =
            Duration::from_nanos((self.total_time.as_nanos() / self.total_calls as u128) as u64);
    }
}

/// Initialize the global performance metrics collector
pub fn init_perf_metrics() {
    PERF_METRICS.get_or_init(|| Arc::new(Mutex::new(HashMap::new())));
}

/// Record a performance measurement
pub fn record_perf(label: &str, duration: Duration) {
    if let Some(metrics) = PERF_METRICS.get() {
        if let Ok(mut guard) = metrics.lock() {
            guard
                .entry(label.to_string())
                .or_insert_with(PerfMetrics::new)
                .update(duration);
        }
    }
}

/// Get performance metrics for a specific label
pub fn get_perf_metrics(label: &str) -> Option<PerfMetrics> {
    if let Some(metrics) = PERF_METRICS.get() {
        if let Ok(guard) = metrics.lock() {
            return guard.get(label).cloned();
        }
    }
    None
}

/// Get all performance metrics
pub fn get_all_perf_metrics() -> HashMap<String, PerfMetrics> {
    if let Some(metrics) = PERF_METRICS.get() {
        if let Ok(guard) = metrics.lock() {
            return guard.clone();
        }
    }
    HashMap::new()
}

/// Performance measurement macro for easy timing
#[macro_export]
macro_rules! time_block {
    ($label:expr, $block:block) => {{
        let _timer = $crate::profiling::PerfTimer::start($label);
        let result = $block;
        let elapsed = _timer.stop_and_log();
        $crate::profiling::record_perf($label, elapsed);
        result
    }};
}

/// Convenience macro for timing function execution
#[macro_export]
macro_rules! time_fn {
    ($label:expr, $fn:expr) => {{
        let _timer = $crate::profiling::PerfTimer::start($label);
        let result = $fn;
        let elapsed = _timer.stop_and_log();
        $crate::profiling::record_perf($label, elapsed);
        result
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_perf_timer() {
        let timer = PerfTimer::start("test_operation");
        thread::sleep(Duration::from_millis(10));
        let elapsed = timer.stop();

        assert!(elapsed >= Duration::from_millis(10));
        assert!(elapsed < Duration::from_millis(100));
    }

    #[test]
    fn test_perf_metrics() {
        init_perf_metrics();

        // Record some measurements
        record_perf("test_op", Duration::from_millis(10));
        record_perf("test_op", Duration::from_millis(20));
        record_perf("test_op", Duration::from_millis(15));

        let metrics = get_perf_metrics("test_op").unwrap();
        assert_eq!(metrics.total_calls, 3);
        assert!(metrics.avg_time >= Duration::from_millis(10));
        assert!(metrics.avg_time <= Duration::from_millis(20));
    }
}
