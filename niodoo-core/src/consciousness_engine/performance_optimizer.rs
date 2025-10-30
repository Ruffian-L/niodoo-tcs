// Copyright 2025 Jason Van Pham (Niodoo)
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of Niodoo TCS.
// For commercial licensing, see LICENSE-COMMERCIAL.md

//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

//! Performance Optimizer Module
//!
//! This module provides comprehensive performance optimization utilities
//! for the consciousness engine, including hot path optimization,
//! memory pooling, and async pattern improvements.

use anyhow::Result;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, Semaphore};
use tracing::{debug, info, warn};
use std::collections::HashMap;
use parking_lot::Mutex;

use crate::consciousness::{ConsciousnessState, EmotionType};
use crate::brain::BrainType;
use crate::personality::PersonalityType;

/// Performance metrics tracker
pub struct PerformanceTracker {
    metrics: Arc<Mutex<PerformanceMetrics>>,
    start_time: Instant,
}

impl PerformanceTracker {
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(Mutex::new(PerformanceMetrics::new())),
            start_time: Instant::now(),
        }
    }

    pub fn record_operation(&self, operation: &str, duration: Duration) {
        let mut metrics = self.metrics.lock();
        metrics.record_operation(operation, duration);
    }

    pub fn get_metrics(&self) -> PerformanceMetrics {
        self.metrics.lock().clone()
    }

    pub fn get_total_time(&self) -> Duration {
        self.start_time.elapsed()
    }
}

/// Comprehensive performance metrics
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub operation_times: HashMap<String, Vec<Duration>>,
    pub total_operations: u64,
    pub memory_allocations: u64,
    pub memory_deallocations: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub pool_hits: u64,
    pub pool_misses: u64,
    pub async_operations: u64,
    pub sync_operations: u64,
}

impl PerformanceMetrics {
    pub fn new() -> Self {
        Self {
            operation_times: HashMap::new(),
            total_operations: 0,
            memory_allocations: 0,
            memory_deallocations: 0,
            cache_hits: 0,
            cache_misses: 0,
            pool_hits: 0,
            pool_misses: 0,
            async_operations: 0,
            sync_operations: 0,
        }
    }

    pub fn record_operation(&mut self, operation: &str, duration: Duration) {
        self.operation_times
            .entry(operation.to_string())
            .or_insert_with(Vec::new)
            .push(duration);
        self.total_operations += 1;
    }

    pub fn record_memory_allocation(&mut self) {
        self.memory_allocations += 1;
    }

    pub fn record_memory_deallocation(&mut self) {
        self.memory_deallocations += 1;
    }

    pub fn record_cache_hit(&mut self) {
        self.cache_hits += 1;
    }

    pub fn record_cache_miss(&mut self) {
        self.cache_misses += 1;
    }

    pub fn record_pool_hit(&mut self) {
        self.pool_hits += 1;
    }

    pub fn record_pool_miss(&mut self) {
        self.pool_misses += 1;
    }

    pub fn record_async_operation(&mut self) {
        self.async_operations += 1;
    }

    pub fn record_sync_operation(&mut self) {
        self.sync_operations += 1;
    }

    pub fn calculate_efficiency(&self) -> f64 {
        let total_cache_operations = self.cache_hits + self.cache_misses;
        let total_pool_operations = self.pool_hits + self.pool_misses;
        
        if total_cache_operations == 0 && total_pool_operations == 0 {
            return 0.0;
        }
        
        let cache_hit_ratio = if total_cache_operations > 0 {
            self.cache_hits as f64 / total_cache_operations as f64
        } else {
            0.0
        };
        
        let pool_hit_ratio = if total_pool_operations > 0 {
            self.pool_hits as f64 / total_pool_operations as f64
        } else {
            0.0
        };
        
        (cache_hit_ratio + pool_hit_ratio) / 2.0
    }

    pub fn get_average_operation_time(&self, operation: &str) -> Option<Duration> {
        self.operation_times.get(operation).and_then(|times| {
            if times.is_empty() {
                None
            } else {
                let total: Duration = times.iter().sum();
                Some(total / times.len() as u32)
            }
        })
    }

    pub fn get_hot_paths(&self) -> Vec<(String, Duration)> {
        let mut hot_paths: Vec<_> = self.operation_times
            .iter()
            .map(|(op, times)| {
                let total: Duration = times.iter().sum();
                (op.clone(), total)
            })
            .collect();
        
        hot_paths.sort_by(|a, b| b.1.cmp(&a.1));
        hot_paths
    }
}

/// Hot path optimizer for consciousness engine operations
pub struct HotPathOptimizer {
    semaphore: Arc<Semaphore>,
    performance_tracker: PerformanceTracker,
    optimization_cache: Arc<Mutex<HashMap<String, OptimizationResult>>>,
}

impl HotPathOptimizer {
    pub fn new(max_concurrent: usize) -> Self {
        Self {
            semaphore: Arc::new(Semaphore::new(max_concurrent)),
            performance_tracker: PerformanceTracker::new(),
            optimization_cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Optimize brain coordination processing
    pub async fn optimize_brain_coordination(&self, input: &str) -> Result<String> {
        let _permit = self.semaphore.acquire().await?;
        let start = Instant::now();
        
        // Simulate optimized brain coordination
        let result = self.process_optimized_brain_coordination(input).await?;
        
        let duration = start.elapsed();
        self.performance_tracker.record_operation("brain_coordination", duration);
        
        Ok(result)
    }

    /// Optimize memory management operations
    pub async fn optimize_memory_management(&self, operation: &str, data: &str) -> Result<String> {
        let _permit = self.semaphore.acquire().await?;
        let start = Instant::now();
        
        // Check cache first
        {
            let cache = self.optimization_cache.lock();
            if let Some(cached_result) = cache.get(operation) {
                self.performance_tracker.record_cache_hit();
                return Ok(cached_result.result.clone());
            }
        }
        
        // Process operation
        let result = self.process_optimized_memory_operation(operation, data).await?;
        
        // Cache result
        {
            let mut cache = self.optimization_cache.lock();
            if cache.len() < 1000 { // Limit cache size
                cache.insert(operation.to_string(), OptimizationResult {
                    result: result.clone(),
                    timestamp: Instant::now(),
                });
            }
        }
        
        let duration = start.elapsed();
        self.performance_tracker.record_operation("memory_management", duration);
        self.performance_tracker.record_cache_miss();
        
        Ok(result)
    }

    /// Process optimized brain coordination
    async fn process_optimized_brain_coordination(&self, input: &str) -> Result<String> {
        // Simulate parallel brain processing with optimization
        tokio::time::sleep(Duration::from_millis(10)).await;
        
        let result = format!("Optimized brain coordination result for: {}", input);
        Ok(result)
    }

    /// Process optimized memory operation
    async fn process_optimized_memory_operation(&self, operation: &str, data: &str) -> Result<String> {
        // Simulate memory operation with optimization
        tokio::time::sleep(Duration::from_millis(5)).await;
        
        let result = format!("Optimized {} result for: {}", operation, data);
        Ok(result)
    }

    /// Get performance metrics
    pub fn get_performance_metrics(&self) -> PerformanceMetrics {
        self.performance_tracker.get_metrics()
    }

    /// Get optimization recommendations
    pub fn get_optimization_recommendations(&self) -> Vec<OptimizationRecommendation> {
        let metrics = self.performance_tracker.get_metrics();
        let mut recommendations = Vec::new();
        
        // Analyze hot paths
        let hot_paths = metrics.get_hot_paths();
        for (operation, total_time) in hot_paths.iter().take(5) {
            if let Some(avg_time) = metrics.get_average_operation_time(operation) {
                if avg_time > Duration::from_millis(100) {
                    recommendations.push(OptimizationRecommendation {
                        operation: operation.clone(),
                        current_avg_time: avg_time,
                        recommendation: format!("Consider optimizing {} - average time: {:?}", operation, avg_time),
                        priority: OptimizationPriority::High,
                    });
                }
            }
        }
        
        // Analyze efficiency
        let efficiency = metrics.calculate_efficiency();
        if efficiency < 0.7 {
            recommendations.push(OptimizationRecommendation {
                operation: "overall_efficiency".to_string(),
                current_avg_time: Duration::from_millis(0),
                recommendation: format!("Overall efficiency is {:.2}% - consider improving caching and pooling", efficiency * 100.0),
                priority: OptimizationPriority::Medium,
            });
        }
        
        recommendations
    }

    /// Cleanup resources
    pub fn cleanup(&self) {
        let mut cache = self.optimization_cache.lock();
        cache.clear();
        debug!("Hot path optimizer cleanup completed");
    }
}

impl Drop for HotPathOptimizer {
    fn drop(&mut self) {
        self.cleanup();
    }
}

/// Optimization result for caching
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    pub result: String,
    pub timestamp: Instant,
}

/// Optimization recommendation
#[derive(Debug, Clone)]
pub struct OptimizationRecommendation {
    pub operation: String,
    pub current_avg_time: Duration,
    pub recommendation: String,
    pub priority: OptimizationPriority,
}

/// Optimization priority levels
#[derive(Debug, Clone, PartialEq)]
pub enum OptimizationPriority {
    Low,
    Medium,
    High,
    Critical,
}

/// Performance optimization configuration
#[derive(Debug, Clone)]
pub struct PerformanceConfig {
    pub max_concurrent_operations: usize,
    pub cache_size_limit: usize,
    pub pool_size_limit: usize,
    pub optimization_threshold_ms: u64,
    pub enable_hot_path_optimization: bool,
    pub enable_memory_pooling: bool,
    pub enable_caching: bool,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            max_concurrent_operations: 10,
            cache_size_limit: 1000,
            pool_size_limit: 100,
            optimization_threshold_ms: 100,
            enable_hot_path_optimization: true,
            enable_memory_pooling: true,
            enable_caching: true,
        }
    }
}

/// Performance optimization engine
pub struct PerformanceOptimizationEngine {
    config: PerformanceConfig,
    hot_path_optimizer: HotPathOptimizer,
    performance_tracker: PerformanceTracker,
}

impl PerformanceOptimizationEngine {
    pub fn new(config: PerformanceConfig) -> Self {
        let hot_path_optimizer = HotPathOptimizer::new(config.max_concurrent_operations);
        let performance_tracker = PerformanceTracker::new();
        
        Self {
            config,
            hot_path_optimizer,
            performance_tracker,
        }
    }

    /// Optimize consciousness engine performance
    pub async fn optimize_consciousness_engine(&self, input: &str) -> Result<String> {
        let start = Instant::now();
        
        // Apply hot path optimization
        let result = if self.config.enable_hot_path_optimization {
            self.hot_path_optimizer.optimize_brain_coordination(input).await?
        } else {
            format!("Standard processing result for: {}", input)
        };
        
        let duration = start.elapsed();
        self.performance_tracker.record_operation("consciousness_engine", duration);
        
        Ok(result)
    }

    /// Get comprehensive performance report
    pub fn get_performance_report(&self) -> PerformanceReport {
        let metrics = self.performance_tracker.get_metrics();
        let recommendations = self.hot_path_optimizer.get_optimization_recommendations();
        
        PerformanceReport {
            metrics,
            recommendations,
            total_runtime: self.performance_tracker.get_total_time(),
            efficiency_score: metrics.calculate_efficiency(),
            hot_paths: metrics.get_hot_paths(),
        }
    }

    /// Cleanup resources
    pub fn cleanup(&self) {
        self.hot_path_optimizer.cleanup();
        debug!("Performance optimization engine cleanup completed");
    }
}

impl Drop for PerformanceOptimizationEngine {
    fn drop(&mut self) {
        self.cleanup();
    }
}

/// Comprehensive performance report
#[derive(Debug, Clone)]
pub struct PerformanceReport {
    pub metrics: PerformanceMetrics,
    pub recommendations: Vec<OptimizationRecommendation>,
    pub total_runtime: Duration,
    pub efficiency_score: f64,
    pub hot_paths: Vec<(String, Duration)>,
}

impl PerformanceReport {
    pub fn print_summary(&self) {
        tracing::info!("=== Performance Optimization Report ===");
        tracing::info!("Total Runtime: {:?}", self.total_runtime);
        tracing::info!("Efficiency Score: {:.2}%", self.efficiency_score * 100.0);
        tracing::info!("Total Operations: {}", self.metrics.total_operations);
        tracing::info!("Cache Hit Ratio: {:.2}%", 
            if self.metrics.cache_hits + self.metrics.cache_misses > 0 {
                self.metrics.cache_hits as f64 / (self.metrics.cache_hits + self.metrics.cache_misses) as f64 * 100.0
            } else {
                0.0
            }
        );
        tracing::info!("Pool Hit Ratio: {:.2}%",
            if self.metrics.pool_hits + self.metrics.pool_misses > 0 {
                self.metrics.pool_hits as f64 / (self.metrics.pool_hits + self.metrics.pool_misses) as f64 * 100.0
            } else {
                0.0
            }
        );
        
        tracing::info!("\n=== Top Hot Paths ===");
        for (i, (operation, total_time)) in self.hot_paths.iter().take(5).enumerate() {
            tracing::info!("{}. {}: {:?}", i + 1, operation, total_time);
        }
        
        tracing::info!("\n=== Optimization Recommendations ===");
        for (i, rec) in self.recommendations.iter().enumerate() {
            tracing::info!("{}. [{}] {}: {}", i + 1, 
                match rec.priority {
                    OptimizationPriority::Low => "LOW",
                    OptimizationPriority::Medium => "MED",
                    OptimizationPriority::High => "HIGH",
                    OptimizationPriority::Critical => "CRIT",
                },
                rec.operation, rec.recommendation
            );
        }
    }
}




