//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

/*
 * 🧠 SMART CAPACITY CALCULATION UTILITIES 🧠
 *
 * This module provides dynamic capacity calculation for Vec allocations
 * based on system memory, thread count, and workload characteristics.
 * Eliminates hardcoded Vec::with_capacity(1024) throughout the codebase.
 */

use anyhow::Result;
use sysinfo::System;
use tracing::{debug, info};

/// Smart capacity calculation strategies
#[derive(Debug, Clone, Copy)]
pub enum CapacityStrategy {
    /// Memory-based: total_memory_mb / divisor
    MemoryBased(usize),
    /// Thread-based: threads * multiplier
    ThreadBased(usize),
    /// Hybrid: memory / divisor + threads * multiplier
    Hybrid {
        memory_divisor: usize,
        thread_multiplier: usize,
    },
    /// Fixed: constant value for specific use cases
    Fixed(usize),
}

/// Configuration for capacity calculation
#[derive(Debug, Clone)]
pub struct CapacityConfig {
    /// Minimum capacity to prevent under-allocation
    pub min_capacity: usize,
    /// Maximum capacity to prevent over-allocation
    pub max_capacity: usize,
    /// Memory safety factor (0.1 = 10% of available memory)
    pub memory_safety_factor: f64,
}

impl Default for CapacityConfig {
    fn default() -> Self {
        Self {
            min_capacity: 64,
            max_capacity: 1_000_000,
            memory_safety_factor: 0.1, // 10% of available memory
        }
    }
}

/// Smart capacity calculator
pub struct CapacityCalculator {
    system: System,
    config: CapacityConfig,
}

impl CapacityCalculator {
    /// Create a new capacity calculator
    pub fn new() -> Result<Self> {
        let mut system = System::new_all();
        system.refresh_all();

        Ok(Self {
            system,
            config: CapacityConfig::default(),
        })
    }

    /// Create with custom configuration
    pub fn with_config(config: CapacityConfig) -> Result<Self> {
        let mut system = System::new_all();
        system.refresh_all();

        Ok(Self { system, config })
    }

    /// Get total available memory in MB
    pub fn total_memory_mb(&self) -> u64 {
        self.system.total_memory() / 1024 / 1024
    }

    /// Get available memory in MB (free + available)
    pub fn available_memory_mb(&self) -> u64 {
        self.system.available_memory() / 1024 / 1024
    }

    /// Get number of logical CPU cores
    pub fn logical_cores(&self) -> usize {
        self.system.cpus().len()
    }

    /// Get number of physical CPU cores
    pub fn physical_cores(&self) -> usize {
        self.system
            .physical_core_count()
            .unwrap_or_else(|| self.logical_cores())
    }

    /// Calculate capacity using specified strategy
    pub fn calculate_capacity(&self, strategy: CapacityStrategy) -> usize {
        let capacity = match strategy {
            CapacityStrategy::MemoryBased(divisor) => {
                let memory_mb = self.available_memory_mb();
                let base_capacity = memory_mb as usize / divisor;
                debug!(
                    "Memory-based capacity: {}MB / {} = {}",
                    memory_mb, divisor, base_capacity
                );
                base_capacity
            }
            CapacityStrategy::ThreadBased(multiplier) => {
                let threads = self.logical_cores();
                let base_capacity = threads * multiplier;
                debug!(
                    "Thread-based capacity: {} threads * {} = {}",
                    threads, multiplier, base_capacity
                );
                base_capacity
            }
            CapacityStrategy::Hybrid {
                memory_divisor,
                thread_multiplier,
            } => {
                let memory_mb = self.available_memory_mb();
                let threads = self.logical_cores();
                let memory_part = memory_mb as usize / memory_divisor;
                let thread_part = threads * thread_multiplier;
                let base_capacity = memory_part + thread_part;
                debug!(
                    "Hybrid capacity: {}MB / {} + {} threads * {} = {}",
                    memory_mb, memory_divisor, threads, thread_multiplier, base_capacity
                );
                base_capacity
            }
            CapacityStrategy::Fixed(capacity) => capacity,
        };

        // Apply bounds and safety factor
        let safe_capacity = ((capacity as f64) * self.config.memory_safety_factor) as usize;
        let bounded_capacity = safe_capacity
            .max(self.config.min_capacity)
            .min(self.config.max_capacity);

        debug!(
            "Final capacity: {} -> {} (safe) -> {} (bounded)",
            capacity, safe_capacity, bounded_capacity
        );

        bounded_capacity
    }

    /// Predefined capacity strategies for common use cases
    pub fn worker_pool_capacity(&self) -> usize {
        // Workers = threads * 8 (good balance for most workloads)
        self.calculate_capacity(CapacityStrategy::ThreadBased(8))
    }

    /// Memory vector capacity for large data structures
    pub fn memory_vector_capacity(&self) -> usize {
        // Memory vectors = available_memory_mb / 8 (1/8th of memory for vectors)
        self.calculate_capacity(CapacityStrategy::MemoryBased(8))
    }

    /// Performance history capacity for time series data
    pub fn performance_history_capacity(&self) -> usize {
        // Performance history = threads * 16 (more space for detailed metrics)
        self.calculate_capacity(CapacityStrategy::ThreadBased(16))
    }

    /// Violation tracking capacity for error monitoring
    pub fn violation_tracking_capacity(&self) -> usize {
        // Violations = threads * 4 (moderate space for violation tracking)
        self.calculate_capacity(CapacityStrategy::ThreadBased(4))
    }

    /// Guidance capacity for AI guidance systems
    pub fn guidance_capacity(&self) -> usize {
        // Guidance = threads * 2 (smaller for guidance data)
        self.calculate_capacity(CapacityStrategy::ThreadBased(2))
    }

    /// Recommendations capacity for suggestion systems
    pub fn recommendations_capacity(&self) -> usize {
        // Recommendations = threads * 8 (similar to worker pools)
        self.calculate_capacity(CapacityStrategy::ThreadBased(8))
    }

    /// Analysis results capacity for computation results
    pub fn analysis_results_capacity(&self) -> usize {
        // Analysis = threads * 4 (moderate for analysis results)
        self.calculate_capacity(CapacityStrategy::ThreadBased(4))
    }

    /// Stage breakdown capacity for pipeline tracking
    pub fn stage_breakdown_capacity(&self) -> usize {
        // Stages = threads * 2 (smaller for stage tracking)
        self.calculate_capacity(CapacityStrategy::ThreadBased(2))
    }

    /// Embeddings capacity for vector storage
    pub fn embeddings_capacity(&self) -> usize {
        // Embeddings = available_memory_mb / 16 (1/16th of memory for embeddings)
        self.calculate_capacity(CapacityStrategy::MemoryBased(16))
    }

    /// Print system information for debugging
    pub fn print_system_info(&self) {
        info!("🔧 System Information for Capacity Calculation:");
        info!("  Total Memory: {} MB", self.total_memory_mb());
        info!("  Available Memory: {} MB", self.available_memory_mb());
        info!("  Logical Cores: {}", self.logical_cores());
        info!("  Physical Cores: {}", self.physical_cores());
        info!(
            "  Memory Safety Factor: {}",
            self.config.memory_safety_factor
        );
        info!("  Min Capacity: {}", self.config.min_capacity);
        info!("  Max Capacity: {}", self.config.max_capacity);
    }
}

/// Global capacity calculator instance for easy access
pub fn get_capacity_calculator() -> Result<CapacityCalculator> {
    CapacityCalculator::new()
}

/// Convenience functions for common capacity calculations
pub mod convenience {
    use super::*;

    /// Get worker pool capacity
    pub fn worker_pool() -> usize {
        get_capacity_calculator()
            .map(|calc| calc.worker_pool_capacity())
            .unwrap_or(1024) // fallback to old default
    }

    /// Get memory vector capacity
    pub fn memory_vector() -> usize {
        get_capacity_calculator()
            .map(|calc| calc.memory_vector_capacity())
            .unwrap_or(1024) // fallback to old default
    }

    /// Get performance history capacity
    pub fn performance_history() -> usize {
        get_capacity_calculator()
            .map(|calc| calc.performance_history_capacity())
            .unwrap_or(1024) // fallback to old default
    }

    /// Get violation tracking capacity
    pub fn violation_tracking() -> usize {
        get_capacity_calculator()
            .map(|calc| calc.violation_tracking_capacity())
            .unwrap_or(1024) // fallback to old default
    }

    /// Get guidance capacity
    pub fn guidance() -> usize {
        get_capacity_calculator()
            .map(|calc| calc.guidance_capacity())
            .unwrap_or(1024) // fallback to old default
    }

    /// Get recommendations capacity
    pub fn recommendations() -> usize {
        get_capacity_calculator()
            .map(|calc| calc.recommendations_capacity())
            .unwrap_or(1024) // fallback to old default
    }

    /// Get analysis results capacity
    pub fn analysis_results() -> usize {
        get_capacity_calculator()
            .map(|calc| calc.analysis_results_capacity())
            .unwrap_or(1024) // fallback to old default
    }

    /// Get stage breakdown capacity
    pub fn stage_breakdown() -> usize {
        get_capacity_calculator()
            .map(|calc| calc.stage_breakdown_capacity())
            .unwrap_or(1024) // fallback to old default
    }

    /// Get embeddings capacity
    pub fn embeddings() -> usize {
        get_capacity_calculator()
            .map(|calc| calc.embeddings_capacity())
            .unwrap_or(1024) // fallback to old default
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capacity_calculator_creation() {
        let calc = CapacityCalculator::new();
        assert!(calc.is_ok());
    }

    #[test]
    fn test_system_info() {
        let calc = CapacityCalculator::new().unwrap();
        assert!(calc.total_memory_mb() > 0);
        assert!(calc.available_memory_mb() > 0);
        assert!(calc.logical_cores() > 0);
    }

    #[test]
    fn test_capacity_strategies() {
        let calc = CapacityCalculator::new().unwrap();

        // Test memory-based strategy
        let memory_capacity = calc.calculate_capacity(CapacityStrategy::MemoryBased(8));
        assert!(memory_capacity > 0);

        // Test thread-based strategy
        let thread_capacity = calc.calculate_capacity(CapacityStrategy::ThreadBased(4));
        assert!(thread_capacity > 0);

        // Test hybrid strategy
        let hybrid_capacity = calc.calculate_capacity(CapacityStrategy::Hybrid {
            memory_divisor: 16,
            thread_multiplier: 2,
        });
        assert!(hybrid_capacity > 0);
    }

    #[test]
    fn test_predefined_capacities() {
        let calc = CapacityCalculator::new().unwrap();

        assert!(calc.worker_pool_capacity() > 0);
        assert!(calc.memory_vector_capacity() > 0);
        assert!(calc.performance_history_capacity() > 0);
        assert!(calc.violation_tracking_capacity() > 0);
        assert!(calc.guidance_capacity() > 0);
        assert!(calc.recommendations_capacity() > 0);
        assert!(calc.analysis_results_capacity() > 0);
        assert!(calc.stage_breakdown_capacity() > 0);
        assert!(calc.embeddings_capacity() > 0);
    }

    #[test]
    fn test_convenience_functions() {
        // These should not panic and return reasonable values
        let worker = convenience::worker_pool();
        let memory = convenience::memory_vector();
        let perf = convenience::performance_history();
        let violation = convenience::violation_tracking();
        let guidance = convenience::guidance();
        let recs = convenience::recommendations();
        let analysis = convenience::analysis_results();
        let stages = convenience::stage_breakdown();
        let embeddings = convenience::embeddings();

        // Assert all return positive values
        assert!(worker > 0, "worker_pool should return positive capacity");
        assert!(memory > 0, "memory_vector should return positive capacity");
        assert!(
            perf > 0,
            "performance_history should return positive capacity"
        );
        assert!(
            violation > 0,
            "violation_tracking should return positive capacity"
        );
        assert!(guidance > 0, "guidance should return positive capacity");
        assert!(recs > 0, "recommendations should return positive capacity");
        assert!(
            analysis > 0,
            "analysis_results should return positive capacity"
        );
        assert!(
            stages > 0,
            "stage_breakdown should return positive capacity"
        );
        assert!(embeddings > 0, "embeddings should return positive capacity");
    }
}
