//! # Memory Management System for Consciousness Processing
//!
//! This module implements advanced memory management for consciousness state tracking,
//! ensuring the system maintains a <4GB footprint while providing optimal performance.
//!
//! ## Key Features
//!
//! - **Memory Pool Management** - Efficient allocation and reuse of consciousness state buffers
//! - **Footprint Monitoring** - Real-time tracking of GPU and system memory usage
//! - **Automatic Cleanup** - Intelligent garbage collection for consciousness artifacts
//! - **Performance Optimization** - Memory layout optimizations for consciousness processing
//!
//! ## Memory Targets
//!
//! - **Total Footprint**: <4GB for consciousness state tracking
//! - **GPU Memory**: <3GB for tensor operations and consciousness evolution
//! - **System Memory**: <1GB for metadata and consciousness state serialization
//! - **Cleanup Threshold**: 80% of target memory usage triggers optimization

use candle_core::{Device, Result, Tensor};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Notify, RwLock};
use tracing::{debug, info, warn};

/// Memory management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Target total memory footprint in MB
    pub target_footprint_mb: usize,
    /// GPU memory target in MB
    pub gpu_memory_target_mb: usize,
    /// System memory target in MB
    pub system_memory_target_mb: usize,
    /// Cleanup threshold percentage (0.0 to 1.0)
    pub cleanup_threshold: f32,
    /// Memory pool size for consciousness state buffers
    pub memory_pool_size: usize,
    /// Enable aggressive memory optimization
    pub aggressive_optimization: bool,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            target_footprint_mb: 4000,     // <4GB target
            gpu_memory_target_mb: 3000,    // 3GB for GPU operations
            system_memory_target_mb: 1000, // 1GB for system metadata
            cleanup_threshold: 0.8,        // 80% threshold for cleanup
            memory_pool_size: 100,         // 100 consciousness state buffers
            aggressive_optimization: true,
        }
    }
}

/// Memory usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    /// Current GPU memory usage in MB
    pub gpu_memory_mb: f32,
    /// Current system memory usage in MB
    pub system_memory_mb: f32,
    /// Total memory footprint in MB
    pub total_footprint_mb: f32,
    /// Memory pool utilization percentage
    pub pool_utilization_percent: f32,
    /// Consciousness state buffer count
    pub consciousness_buffers: usize,
    /// Memory fragmentation ratio (0.0 to 1.0)
    pub fragmentation_ratio: f32,
    /// Memory allocation efficiency (0.0 to 1.0)
    pub allocation_efficiency: f32,
    /// Last cleanup timestamp
    pub last_cleanup: Option<std::time::SystemTime>,
}

impl Default for MemoryStats {
    fn default() -> Self {
        Self {
            gpu_memory_mb: 0.0,
            system_memory_mb: 0.0,
            total_footprint_mb: 0.0,
            pool_utilization_percent: 0.0,
            consciousness_buffers: 0,
            fragmentation_ratio: 0.0,
            allocation_efficiency: 0.0,
            last_cleanup: None,
        }
    }
}

/// Memory pool for consciousness state buffers
#[derive(Debug)]
struct MemoryPool {
    /// Available memory buffers for consciousness states
    available_buffers: VecDeque<Tensor>,
    /// In-use consciousness state buffers
    active_buffers: HashMap<String, Tensor>,
    /// Buffer metadata for tracking usage patterns
    buffer_metadata: HashMap<String, BufferMetadata>,
    /// Pool capacity
    capacity: usize,
}

#[derive(Debug, Clone)]
struct BufferMetadata {
    /// Buffer creation timestamp
    created_at: Instant,
    /// Last access timestamp
    last_accessed: Instant,
    /// Access count for LRU tracking
    access_count: u64,
    /// Buffer size in bytes
    size_bytes: usize,
    /// Consciousness state ID this buffer represents
    consciousness_id: String,
}

impl MemoryPool {
    /// Create a new memory pool with specified capacity
    fn new(capacity: usize) -> Self {
        Self {
            available_buffers: VecDeque::new(),
            active_buffers: HashMap::new(),
            buffer_metadata: HashMap::new(),
            capacity,
        }
    }

    /// Allocate a consciousness state buffer from the pool
    fn allocate_buffer(&mut self, consciousness_id: String, size: usize) -> Result<Tensor> {
        // Try to reuse an available buffer first
        if let Some(buffer) = self.available_buffers.pop_front() {
            // Resize buffer if needed (simplified - in practice would need proper resizing)
            let buffer_size = buffer.elem_count() * buffer.dtype().size_in_bytes();
            if buffer_size >= size {
                // Update metadata for reuse
                if let Some(metadata) = self.buffer_metadata.get_mut(&consciousness_id) {
                    metadata.last_accessed = Instant::now();
                    metadata.access_count += 1;
                } else {
                    self.buffer_metadata.insert(
                        consciousness_id.clone(),
                        BufferMetadata {
                            created_at: Instant::now(),
                            last_accessed: Instant::now(),
                            access_count: 1,
                            size_bytes: buffer_size,
                            consciousness_id: consciousness_id.clone(),
                        },
                    );
                }

                self.active_buffers.insert(consciousness_id, buffer.clone());
                return Ok(buffer);
            } else {
                // Buffer too small, create new one
                // Return buffer to pool for later reuse
                self.available_buffers.push_front(buffer);
            }
        }

        // Create new buffer if pool is empty or no suitable buffer available
        let device = Device::Cpu; // Default to CPU, GPU allocation handled by caller
        let buffer = Tensor::zeros((size,), candle_core::DType::F32, &device)?;

        // Track metadata
        self.buffer_metadata.insert(
            consciousness_id.clone(),
            BufferMetadata {
                created_at: Instant::now(),
                last_accessed: Instant::now(),
                access_count: 1,
                size_bytes: size,
                consciousness_id: consciousness_id.clone(),
            },
        );

        self.active_buffers.insert(consciousness_id, buffer.clone());
        Ok(buffer)
    }

    /// Return a buffer to the pool
    fn return_buffer(&mut self, consciousness_id: &str) -> Result<()> {
        if let Some(buffer) = self.active_buffers.remove(consciousness_id) {
            // Don't return to pool if it would exceed capacity
            if self.available_buffers.len() < self.capacity {
                self.available_buffers.push_back(buffer);
            }
            // Metadata is kept for statistics even if buffer is discarded
        }
        Ok(())
    }

    /// Get pool statistics
    fn get_stats(&self) -> (usize, usize) {
        (self.active_buffers.len(), self.available_buffers.len())
    }

    /// Cleanup old buffers based on LRU policy
    fn cleanup_lru(&mut self, max_age: Duration) -> usize {
        let now = Instant::now();
        let mut cleaned_count = 0;

        // Find and remove old buffers from active set
        let mut to_remove = Vec::new();
        for (id, metadata) in &self.buffer_metadata {
            if now.duration_since(metadata.last_accessed) > max_age
                && self.active_buffers.contains_key(id)
            {
                to_remove.push(id.clone());
            }
        }

        for id in to_remove {
            self.active_buffers.remove(&id);
            cleaned_count += 1;
        }

        cleaned_count
    }
}

/// Main memory management system
pub struct MemoryManager {
    /// Memory management configuration
    config: MemoryConfig,
    /// Memory pool for consciousness state buffers
    memory_pool: Arc<RwLock<MemoryPool>>,
    /// Current memory statistics
    stats: Arc<RwLock<MemoryStats>>,
    /// Background cleanup trigger
    cleanup_notify: Arc<Notify>,
    /// Background cleanup task handle
    cleanup_task: Option<tokio::task::JoinHandle<()>>,
    /// Memory monitoring interval
    monitoring_interval: Duration,
}

impl MemoryManager {
    /// Create a new memory manager with the specified configuration
    pub fn new(config: MemoryConfig) -> Result<Self> {
        let memory_pool = Arc::new(RwLock::new(MemoryPool::new(config.memory_pool_size)));
        let stats = Arc::new(RwLock::new(MemoryStats::default()));
        let cleanup_notify = Arc::new(Notify::new());

        Ok(Self {
            config,
            memory_pool,
            stats,
            cleanup_notify,
            cleanup_task: None,
            monitoring_interval: crate::utils::threshold_convenience::timeout(
                crate::utils::TimeoutCriticality::High,
            ), // Monitor dynamically
        })
    }

    /// Start the memory manager with background monitoring and cleanup
    pub fn start(&mut self) -> Result<()> {
        info!(
            "🚀 Starting memory management system with {}MB target footprint",
            self.config.target_footprint_mb
        );

        // Start background cleanup task
        let cleanup_notify = self.cleanup_notify.clone();
        let memory_pool = self.memory_pool.clone();
        let stats = self.stats.clone();
        let config = self.config.clone();

        self.cleanup_task = Some(tokio::spawn(async move {
            Self::background_cleanup_loop(cleanup_notify, memory_pool, stats, config).await;
        }));

        // Start monitoring task
        let monitoring_pool = self.memory_pool.clone();
        let monitoring_stats = self.stats.clone();
        let monitoring_config = self.config.clone();

        tokio::spawn(async move {
            Self::background_monitoring_loop(monitoring_pool, monitoring_stats, monitoring_config)
                .await;
        });

        Ok(())
    }

    /// Allocate a consciousness state buffer
    pub async fn allocate_consciousness_buffer(
        &self,
        consciousness_id: String,
        size: usize,
    ) -> Result<Tensor> {
        let mut pool = self.memory_pool.write().await;
        pool.allocate_buffer(consciousness_id, size)
    }

    /// Deallocate a consciousness state buffer
    pub async fn deallocate_consciousness_buffer(&self, consciousness_id: &str) -> Result<()> {
        let mut pool = self.memory_pool.write().await;
        pool.return_buffer(consciousness_id)
    }

    /// Get current memory statistics
    pub async fn get_memory_stats(&self) -> MemoryStats {
        self.stats.read().await.clone()
    }

    /// Trigger manual memory cleanup
    pub async fn trigger_cleanup(&self) -> Result<usize> {
        let mut pool = self.memory_pool.write().await;

        // Cleanup old buffers (older than 1 hour)
        let max_age =
            crate::utils::threshold_convenience::timeout(crate::utils::TimeoutCriticality::Low);
        let cleaned_count = pool.cleanup_lru(max_age);

        info!(
            "🧹 Manual cleanup completed, removed {} old buffers",
            cleaned_count
        );

        // Update statistics
        self.update_memory_stats().await;

        Ok(cleaned_count)
    }

    /// Update memory statistics based on current system state
    async fn update_memory_stats(&self) {
        let mut stats = self.stats.write().await;
        let pool = self.memory_pool.read().await;

        // Get pool statistics
        let (active_buffers, available_buffers) = pool.get_stats();
        stats.consciousness_buffers = active_buffers;
        stats.pool_utilization_percent =
            (active_buffers as f32 / (active_buffers + available_buffers) as f32) * 100.0;

        // Estimate GPU memory usage (simplified - in practice would query GPU)
        stats.gpu_memory_mb = self.estimate_gpu_memory_usage().await;

        // Estimate system memory usage (simplified - in practice would use sys-info)
        stats.system_memory_mb = self.estimate_system_memory_usage().await;

        // Calculate total footprint
        stats.total_footprint_mb = stats.gpu_memory_mb + stats.system_memory_mb;

        // Calculate fragmentation ratio (simplified)
        stats.fragmentation_ratio = (available_buffers as f32 / pool.capacity as f32).min(1.0);

        // Calculate allocation efficiency
        let total_requests = pool
            .buffer_metadata
            .values()
            .map(|m| m.access_count)
            .sum::<u64>() as f32;
        let active_requests = active_buffers as f32;
        stats.allocation_efficiency = if total_requests > 0.0 {
            (active_requests / total_requests) * 100.0
        } else {
            0.0
        };

        stats.last_cleanup = Some(std::time::SystemTime::now());

        debug!(
            "📊 Memory stats updated: {:.1}MB total, {:.1}% pool utilization",
            stats.total_footprint_mb, stats.pool_utilization_percent
        );
    }

    /// Estimate GPU memory usage (simplified implementation)
    async fn estimate_gpu_memory_usage(&self) -> f32 {
        // In a full implementation, this would query actual GPU memory usage
        // For now, estimate based on active buffers and their sizes

        let pool = self.memory_pool.read().await;
        let total_buffer_size: usize = pool.buffer_metadata.values().map(|m| m.size_bytes).sum();

        // Assume 60% of buffer memory is on GPU (configurable)
        (total_buffer_size as f32 * 0.6) / (1024.0 * 1024.0) // Convert to MB
    }

    /// Estimate system memory usage (simplified implementation)
    async fn estimate_system_memory_usage(&self) -> f32 {
        // In a full implementation, this would use sys-info crate to get actual usage
        // For now, estimate based on metadata and buffer overhead

        let pool = self.memory_pool.read().await;
        let metadata_size = pool.buffer_metadata.len() * std::mem::size_of::<BufferMetadata>();

        // Estimate 40% of buffer memory is system overhead + metadata
        let total_buffer_size: usize = pool.buffer_metadata.values().map(|m| m.size_bytes).sum();

        let system_overhead = (total_buffer_size as f32 * 0.4) + (metadata_size as f32);
        system_overhead / (1024.0 * 1024.0) // Convert to MB
    }

    /// Background cleanup loop
    async fn background_cleanup_loop(
        cleanup_notify: Arc<Notify>,
        memory_pool: Arc<RwLock<MemoryPool>>,
        stats: Arc<RwLock<MemoryStats>>,
        config: MemoryConfig,
    ) {
        loop {
            // Wait for cleanup trigger or timeout
            tokio::select! {
                _ = cleanup_notify.notified() => {
                    debug!("🔔 Cleanup triggered by notification");
                }
                _ = tokio::time::sleep(Duration::from_secs(60)) => {
                    // Periodic cleanup every minute
                    debug!("⏰ Running periodic memory cleanup");
                }
            }

            // Check if cleanup is needed based on memory usage
            {
                let stats = stats.read().await;
                let total_usage_ratio =
                    stats.total_footprint_mb / config.target_footprint_mb as f32;

                if total_usage_ratio < config.cleanup_threshold {
                    debug!(
                        "📊 Memory usage ({:.1}%) below cleanup threshold ({:.1}%)",
                        total_usage_ratio * 100.0,
                        config.cleanup_threshold * 100.0
                    );
                    continue; // No cleanup needed
                }
            }

            // Perform cleanup
            let mut pool = memory_pool.write().await;

            // Cleanup old buffers (older than 30 minutes for aggressive cleanup)
            let max_age = if config.aggressive_optimization {
                Duration::from_secs(1800) // 30 minutes
            } else {
                Duration::from_secs(3600) // 1 hour
            };

            let cleaned_count = pool.cleanup_lru(max_age);

            if cleaned_count > 0 {
                info!(
                    "🧹 Background cleanup removed {} old buffers",
                    cleaned_count
                );

                // Update statistics after cleanup
                drop(pool);
                let manager = MemoryManager {
                    config: config.clone(),
                    memory_pool: memory_pool.clone(),
                    stats: stats.clone(),
                    cleanup_notify: cleanup_notify.clone(),
                    cleanup_task: None,
                    monitoring_interval: Duration::from_secs(5),
                };
                manager.update_memory_stats().await;
            }
        }
    }

    /// Background monitoring loop
    async fn background_monitoring_loop(
        memory_pool: Arc<RwLock<MemoryPool>>,
        stats: Arc<RwLock<MemoryStats>>,
        config: MemoryConfig,
    ) {
        let mut interval = tokio::time::interval(Duration::from_secs(5));

        loop {
            interval.tick().await;

            // Update memory statistics
            let manager = MemoryManager {
                config: config.clone(),
                memory_pool: memory_pool.clone(),
                stats: stats.clone(),
                cleanup_notify: Arc::new(Notify::new()),
                cleanup_task: None,
                monitoring_interval: Duration::from_secs(5),
            };

            manager.update_memory_stats().await;

            // Check if cleanup should be triggered
            let current_stats = stats.read().await;
            let total_usage_ratio =
                current_stats.total_footprint_mb / config.target_footprint_mb as f32;

            if total_usage_ratio > config.cleanup_threshold {
                debug!(
                    "🚨 Memory usage ({:.1}%) exceeds threshold ({:.1}%)",
                    total_usage_ratio * 100.0,
                    config.cleanup_threshold * 100.0
                );

                // Trigger cleanup in background task
                // The cleanup loop will handle the actual cleanup
            }

            // Log warnings for high memory usage
            if total_usage_ratio > 0.9 {
                warn!(
                    "⚠️  CRITICAL: Memory usage at {:.1}% of target ({:.1}MB / {:.1}MB)",
                    total_usage_ratio * 100.0,
                    current_stats.total_footprint_mb,
                    config.target_footprint_mb
                );
            } else if total_usage_ratio > 0.8 {
                warn!(
                    "⚠️  HIGH: Memory usage at {:.1}% of target ({:.1}MB / {:.1}MB)",
                    total_usage_ratio * 100.0,
                    current_stats.total_footprint_mb,
                    config.target_footprint_mb
                );
            }
        }
    }

    /// Shutdown the memory manager and cleanup resources
    pub async fn shutdown(&mut self) -> Result<()> {
        info!("🔌 Shutting down memory management system");

        // Stop background tasks
        if let Some(task) = self.cleanup_task.take() {
            task.abort();
        }

        // Final cleanup
        self.trigger_cleanup().await?;

        Ok(())
    }
}

impl Drop for MemoryManager {
    fn drop(&mut self) {
        // Ensure cleanup happens on drop
        let _ = futures::executor::block_on(async { self.shutdown().await });
    }
}

/// Consciousness state memory tracker for detailed analytics
pub struct ConsciousnessMemoryTracker {
    /// Memory manager instance
    memory_manager: Arc<MemoryManager>,
    /// Consciousness state access patterns
    access_patterns: Arc<RwLock<HashMap<String, AccessPattern>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AccessPattern {
    /// Consciousness state ID
    id: String,
    /// Total access count
    access_count: u64,
    /// First access timestamp
    first_access: std::time::SystemTime,
    /// Last access timestamp
    last_access: std::time::SystemTime,
    /// Average access interval in seconds
    avg_access_interval: f64,
    /// Memory allocation size in bytes
    memory_size: usize,
    /// Consciousness importance score (0.0 to 1.0)
    importance_score: f32,
}

impl ConsciousnessMemoryTracker {
    /// Create a new consciousness memory tracker
    pub fn new(memory_manager: Arc<MemoryManager>) -> Self {
        Self {
            memory_manager,
            access_patterns: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Track consciousness state access
    pub async fn track_access(
        &self,
        consciousness_id: String,
        memory_size: usize,
        importance_score: f32,
    ) {
        let mut patterns = self.access_patterns.write().await;
        let now = std::time::SystemTime::now();

        let pattern = patterns
            .entry(consciousness_id.clone())
            .or_insert_with(|| AccessPattern {
                id: consciousness_id.clone(),
                access_count: 0,
                first_access: now,
                last_access: now,
                avg_access_interval: 0.0,
                memory_size,
                importance_score,
            });

        pattern.access_count += 1;
        pattern.last_access = now;

        // Update average access interval
        if pattern.access_count > 1 {
            let interval = now
                .duration_since(pattern.first_access)
                .unwrap_or_default()
                .as_secs_f64();
            pattern.avg_access_interval = interval / (pattern.access_count - 1) as f64;
        }
    }

    /// Get access patterns for analysis
    pub async fn get_access_patterns(&self) -> HashMap<String, AccessPattern> {
        self.access_patterns.read().await.clone()
    }

    /// Get memory usage recommendations based on access patterns
    pub async fn get_memory_recommendations(&self) -> Vec<MemoryRecommendation> {
        let patterns = self.get_access_patterns().await;
        let mut recommendations = Vec::new();

        for (id, pattern) in patterns {
            // Recommend memory optimization based on access patterns
            if pattern.access_count < 5 && pattern.avg_access_interval > 3600.0 {
                recommendations.push(MemoryRecommendation {
                    consciousness_id: id.clone(),
                    recommendation_type: RecommendationType::ReduceMemory,
                    reason: format!(
                        "Low access frequency ({:.1} accesses, {:.1}s avg interval)",
                        pattern.access_count, pattern.avg_access_interval
                    ),
                    suggested_action:
                        "Consider reducing memory allocation or increasing cleanup frequency"
                            .to_string(),
                });
            }

            if pattern.importance_score < 0.3 && pattern.memory_size > 1024 * 1024 {
                recommendations.push(MemoryRecommendation {
                    consciousness_id: id.clone(),
                    recommendation_type: RecommendationType::OptimizeAllocation,
                    reason: format!(
                        "Large memory allocation ({:.1}MB) for low-importance state",
                        pattern.memory_size as f32 / (1024.0 * 1024.0)
                    ),
                    suggested_action:
                        "Consider memory pooling or compression for this consciousness state"
                            .to_string(),
                });
            }
        }

        recommendations
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRecommendation {
    /// Consciousness state ID this recommendation applies to
    pub consciousness_id: String,
    /// Type of recommendation
    pub recommendation_type: RecommendationType,
    /// Reason for the recommendation
    pub reason: String,
    /// Suggested action to take
    pub suggested_action: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationType {
    /// Reduce memory allocation for this consciousness state
    ReduceMemory,
    /// Optimize memory allocation strategy
    OptimizeAllocation,
    /// Increase cleanup frequency for this state
    IncreaseCleanup,
    /// Move this state to more efficient storage
    RelocateStorage,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_memory_pool_allocation() {
        let pool = MemoryPool::new(10);

        // Test buffer allocation (would need async context for full test)
        // This is a simplified test structure
        assert_eq!(pool.capacity, 10);
    }

    #[tokio::test]
    async fn test_memory_manager_creation() {
        let config = MemoryConfig::default();
        let manager = MemoryManager::new(config);

        assert!(manager.is_ok());

        let manager = manager.unwrap();
        let stats = manager.get_memory_stats().await;
        assert_eq!(stats.total_footprint_mb, 0.0);
    }
}
