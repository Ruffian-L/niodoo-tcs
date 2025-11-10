//! Global Resource Budget Tracking
//!
//! Tracks system-wide resource availability including:
//! - Token budgets (remaining tokens for generation)
//! - API rate limits (headroom before hitting limits)
//! - Compute cycles (available processing capacity)
//! - Memory bandwidth (memory usage headroom)
//!
//! Used by ERAG fitness function to penalize memory retrieval when resources are constrained.

use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

/// Global resource budget tracking for the entire system
#[derive(Debug, Clone)]
pub struct GlobalResourceBudget {
    /// Remaining token budget (for generation)
    tokens_remaining: Arc<AtomicU64>,
    /// Maximum token budget
    tokens_max: u64,

    /// API rate limit headroom (requests remaining in current window)
    api_rate_limit_headroom: Arc<AtomicU64>,
    /// Maximum API rate limit per window
    api_rate_limit_max: u64,

    /// Available compute cycles (normalized 0-1)
    compute_cycles_available: Arc<AtomicU64>,
    /// Maximum compute cycles (for normalization)
    compute_cycles_max: u64,

    /// Memory bandwidth headroom (normalized 0-1)
    memory_bandwidth_headroom: Arc<AtomicU64>,
    /// Maximum memory bandwidth (for normalization)
    memory_bandwidth_max: u64,
}

impl GlobalResourceBudget {
    /// Create a new resource budget with specified maximums
    pub fn new(
        tokens_max: u64,
        api_rate_limit_max: u64,
        compute_cycles_max: u64,
        memory_bandwidth_max: u64,
    ) -> Self {
        Self {
            tokens_remaining: Arc::new(AtomicU64::new(tokens_max)),
            tokens_max,
            api_rate_limit_headroom: Arc::new(AtomicU64::new(api_rate_limit_max)),
            api_rate_limit_max,
            compute_cycles_available: Arc::new(AtomicU64::new(compute_cycles_max)),
            compute_cycles_max,
            memory_bandwidth_headroom: Arc::new(AtomicU64::new(memory_bandwidth_max)),
            memory_bandwidth_max,
        }
    }

    /// Create default resource budget with reasonable defaults
    pub fn default() -> Self {
        Self::new(
            100_000,   // 100k tokens
            100,       // 100 API requests per window
            1_000_000, // 1M compute cycles
            100_000,   // 100k memory bandwidth units
        )
    }

    /// Consume tokens from the budget
    pub fn consume_tokens(&self, amount: u64) -> bool {
        let current =
            self.tokens_remaining
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |val| {
                    if val >= amount {
                        Some(val - amount)
                    } else {
                        None
                    }
                });
        current.is_ok()
    }

    /// Consume API rate limit quota
    pub fn consume_api_rate_limit(&self, amount: u64) -> bool {
        let current =
            self.api_rate_limit_headroom
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |val| {
                    if val >= amount {
                        Some(val - amount)
                    } else {
                        None
                    }
                });
        current.is_ok()
    }

    /// Update compute cycles available
    pub fn set_compute_cycles(&self, available: u64) {
        self.compute_cycles_available
            .store(available.min(self.compute_cycles_max), Ordering::SeqCst);
    }

    /// Update memory bandwidth headroom
    pub fn set_memory_bandwidth(&self, headroom: u64) {
        self.memory_bandwidth_headroom
            .store(headroom.min(self.memory_bandwidth_max), Ordering::SeqCst);
    }

    /// Get current resource availability as normalized values (0.0 = exhausted, 1.0 = full)
    pub fn get_resource_availability(&self) -> ResourceAvailability {
        let tokens_ratio =
            self.tokens_remaining.load(Ordering::SeqCst) as f32 / self.tokens_max as f32;
        let api_ratio = self.api_rate_limit_headroom.load(Ordering::SeqCst) as f32
            / self.api_rate_limit_max as f32;
        let compute_ratio = self.compute_cycles_available.load(Ordering::SeqCst) as f32
            / self.compute_cycles_max as f32;
        let memory_ratio = self.memory_bandwidth_headroom.load(Ordering::SeqCst) as f32
            / self.memory_bandwidth_max as f32;

        ResourceAvailability {
            tokens: tokens_ratio.clamp(0.0, 1.0),
            api_rate_limit: api_ratio.clamp(0.0, 1.0),
            compute_cycles: compute_ratio.clamp(0.0, 1.0),
            memory_bandwidth: memory_ratio.clamp(0.0, 1.0),
        }
    }

    /// Reset token budget (called periodically or after cooldown)
    pub fn reset_tokens(&self, amount: Option<u64>) {
        let reset_amount = amount.unwrap_or(self.tokens_max);
        self.tokens_remaining
            .store(reset_amount.min(self.tokens_max), Ordering::SeqCst);
    }

    /// Reset API rate limit (called at start of new rate limit window)
    pub fn reset_api_rate_limit(&self, amount: Option<u64>) {
        let reset_amount = amount.unwrap_or(self.api_rate_limit_max);
        self.api_rate_limit_headroom
            .store(reset_amount.min(self.api_rate_limit_max), Ordering::SeqCst);
    }

    /// Get overall resource availability (minimum of all resources)
    pub fn get_overall_availability(&self) -> f32 {
        let avail = self.get_resource_availability();
        avail
            .tokens
            .min(avail.api_rate_limit)
            .min(avail.compute_cycles)
            .min(avail.memory_bandwidth)
    }
}

/// Resource availability metrics (normalized 0.0-1.0)
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ResourceAvailability {
    /// Token budget remaining (0.0 = exhausted, 1.0 = full)
    pub tokens: f32,
    /// API rate limit headroom (0.0 = at limit, 1.0 = full headroom)
    pub api_rate_limit: f32,
    /// Compute cycles available (0.0 = exhausted, 1.0 = full)
    pub compute_cycles: f32,
    /// Memory bandwidth headroom (0.0 = exhausted, 1.0 = full)
    pub memory_bandwidth: f32,
}

impl ResourceAvailability {
    /// Calculate average resource availability
    pub fn average(&self) -> f32 {
        (self.tokens + self.api_rate_limit + self.compute_cycles + self.memory_bandwidth) / 4.0
    }

    /// Calculate minimum resource availability (bottleneck)
    pub fn minimum(&self) -> f32 {
        self.tokens
            .min(self.api_rate_limit)
            .min(self.compute_cycles)
            .min(self.memory_bandwidth)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resource_budget_creation() {
        let budget = GlobalResourceBudget::default();
        let avail = budget.get_resource_availability();
        assert_eq!(avail.tokens, 1.0);
        assert_eq!(avail.api_rate_limit, 1.0);
    }

    #[test]
    fn test_token_consumption() {
        let budget = GlobalResourceBudget::new(1000, 100, 1000, 1000);
        assert!(budget.consume_tokens(500));
        let avail = budget.get_resource_availability();
        assert_eq!(avail.tokens, 0.5);

        assert!(budget.consume_tokens(400));
        assert!(!budget.consume_tokens(200)); // Should fail, only 100 left
    }

    #[test]
    fn test_overall_availability() {
        let budget = GlobalResourceBudget::new(1000, 100, 1000, 1000);
        budget.consume_tokens(800); // 20% remaining
        budget.set_compute_cycles(200); // 20% remaining
        budget.set_memory_bandwidth(300); // 30% remaining

        let overall = budget.get_overall_availability();
        assert!((overall - 0.2).abs() < 0.01); // Should be ~0.2 (minimum)
    }
}
