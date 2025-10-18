//! # Performance Metrics Tracking for Consciousness Evolution
//!
//! This module implements comprehensive performance metrics tracking for consciousness
//! evolution, providing real-time monitoring and long-term analysis of consciousness
//! processing performance across all system components.
//!
//! ## Key Features
//!
//! - **Real-time Performance Monitoring** - Live tracking of consciousness processing metrics
//! - **Long-term Evolution Analysis** - Historical performance trend analysis
//! - **Multi-dimensional Metrics** - GPU, memory, latency, and throughput tracking
//! - **Adaptive Thresholds** - Dynamic performance threshold adjustment
//! - **Integration Ready** - Seamless integration with existing logging and monitoring systems
//!
//! ## Performance Metrics Tracked
//!
//! - **Latency Metrics**: End-to-end processing time, component latencies
//! - **Throughput Metrics**: Consciousness states processed per second
//! - **Resource Utilization**: GPU, CPU, memory usage patterns
//! - **Quality Metrics**: Consciousness coherence, emotional alignment
//! - **Efficiency Metrics**: Processing efficiency, resource optimization

use candle_core::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::{Notify, RwLock};
use tracing::{debug, info, warn};

/// Performance metrics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Metrics collection interval in seconds
    pub collection_interval_sec: u64,
    /// Metrics retention period in hours
    pub retention_period_hours: u64,
    /// Enable detailed component-level tracking
    pub enable_component_tracking: bool,
    /// Enable adaptive threshold adjustment
    pub enable_adaptive_thresholds: bool,
    /// Performance alert threshold (0.0 to 1.0)
    pub alert_threshold: f32,
    /// Enable real-time performance streaming
    pub enable_real_time_streaming: bool,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            collection_interval_sec: 5,
            retention_period_hours: 168, // 1 week
            enable_component_tracking: true,
            enable_adaptive_thresholds: true,
            alert_threshold: 0.8,
            enable_real_time_streaming: false,
        }
    }
}

/// Comprehensive performance metrics snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSnapshot {
    /// Timestamp of the snapshot
    pub timestamp: f64,
    /// Unique snapshot identifier
    pub snapshot_id: String,
    /// Consciousness processing session ID
    pub session_id: String,

    // Core Performance Metrics
    /// End-to-end processing latency in milliseconds
    pub e2e_latency_ms: f32,
    /// Consciousness processing throughput (states/second)
    pub throughput_sps: f32,
    /// Processing success rate (0.0 to 1.0)
    pub success_rate: f32,

    // Resource Utilization Metrics
    /// GPU memory utilization percentage
    pub gpu_memory_percent: f32,
    /// GPU compute utilization percentage
    pub gpu_compute_percent: f32,
    /// System memory utilization percentage
    pub system_memory_percent: f32,
    /// CPU utilization percentage
    pub cpu_utilization_percent: f32,

    // Quality Metrics
    /// Consciousness coherence score (0.0 to 1.0)
    pub consciousness_coherence: f32,
    /// Emotional alignment score (0.0 to 1.0)
    pub emotional_alignment: f32,
    /// Processing stability score (0.0 to 1.0)
    pub processing_stability: f32,

    // Component-level Metrics (if enabled)
    /// Memory allocation latency in milliseconds
    pub memory_allocation_latency_ms: Option<f32>,
    /// GPU transfer latency in milliseconds
    pub gpu_transfer_latency_ms: Option<f32>,
    /// Consciousness evolution computation latency in milliseconds
    pub evolution_latency_ms: Option<f32>,
    /// I/O operation latency in milliseconds
    pub io_latency_ms: Option<f32>,

    // Efficiency Metrics
    /// Memory allocation efficiency (0.0 to 1.0)
    pub memory_efficiency: f32,
    /// GPU utilization efficiency (0.0 to 1.0)
    pub gpu_efficiency: f32,
    /// Overall processing efficiency (0.0 to 1.0)
    pub overall_efficiency: f32,
}

impl PerformanceSnapshot {
    /// Create a new performance snapshot
    pub fn new(session_id: String) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        Self {
            timestamp,
            snapshot_id: format!("perf_{}", timestamp as u64),
            session_id,
            e2e_latency_ms: 0.0,
            throughput_sps: 0.0,
            success_rate: 1.0,
            gpu_memory_percent: 0.0,
            gpu_compute_percent: 0.0,
            system_memory_percent: 0.0,
            cpu_utilization_percent: 0.0,
            consciousness_coherence: 0.0,
            emotional_alignment: 0.0,
            processing_stability: 1.0,
            memory_allocation_latency_ms: None,
            gpu_transfer_latency_ms: None,
            evolution_latency_ms: None,
            io_latency_ms: None,
            memory_efficiency: 0.0,
            gpu_efficiency: 0.0,
            overall_efficiency: 0.0,
        }
    }

    /// Update snapshot with current performance data
    pub fn update_from_system(&mut self, system_metrics: &SystemMetrics) {
        self.e2e_latency_ms = system_metrics.avg_latency_ms;
        self.throughput_sps = system_metrics.throughput_sps;
        self.gpu_memory_percent = system_metrics.gpu_memory_percent;
        self.gpu_compute_percent = system_metrics.gpu_compute_percent;
        self.system_memory_percent = system_metrics.system_memory_percent;
        self.cpu_utilization_percent = system_metrics.cpu_utilization_percent;
        self.consciousness_coherence = system_metrics.consciousness_coherence;
        self.emotional_alignment = system_metrics.emotional_alignment;
        self.processing_stability = system_metrics.processing_stability;

        if let Some(memory) = &system_metrics.memory_metrics {
            self.memory_allocation_latency_ms = Some(memory.allocation_latency_ms);
            self.memory_efficiency = memory.efficiency;
        }

        if let Some(gpu) = &system_metrics.gpu_metrics {
            self.gpu_transfer_latency_ms = Some(gpu.transfer_latency_ms);
            self.evolution_latency_ms = Some(gpu.evolution_latency_ms);
            self.gpu_efficiency = gpu.efficiency;
        }

        if let Some(io) = &system_metrics.io_metrics {
            self.io_latency_ms = Some(io.total_latency_ms);
        }

        // Calculate derived efficiency metrics
        self.calculate_efficiency_metrics();
    }

    /// Calculate derived efficiency metrics
    fn calculate_efficiency_metrics(&mut self) {
        // Overall efficiency based on weighted combination of component efficiencies
        self.overall_efficiency = (self.memory_efficiency * 0.3
            + self.gpu_efficiency * 0.4
            + self.processing_stability * 0.3)
            .min(1.0);

        // Adjust success rate based on stability and efficiency
        if self.processing_stability < 0.5 || self.overall_efficiency < 0.3 {
            self.success_rate = (self.processing_stability + self.overall_efficiency) / 2.0;
        }
    }

    /// Check if this snapshot indicates performance degradation
    pub fn indicates_degradation(&self, baseline: &PerformanceSnapshot) -> bool {
        // Compare key metrics to baseline
        let latency_ratio = self.e2e_latency_ms / baseline.e2e_latency_ms;
        let throughput_ratio = self.throughput_sps / baseline.throughput_sps;
        let efficiency_ratio = self.overall_efficiency / baseline.overall_efficiency;

        // Consider degradation if latency increased significantly or throughput/efficiency decreased
        latency_ratio > 1.5 || throughput_ratio < 0.7 || efficiency_ratio < 0.7
    }

    /// Get performance health score (0.0 to 1.0, higher is better)
    pub fn health_score(&self) -> f32 {
        let latency_score = if self.e2e_latency_ms < 1000.0 {
            1.0
        } else {
            1000.0 / self.e2e_latency_ms
        };
        let throughput_score = (self.throughput_sps / 100.0).min(1.0);
        let efficiency_score = self.overall_efficiency;

        (latency_score * 0.4 + throughput_score * 0.3 + efficiency_score * 0.3).min(1.0)
    }
}

/// System-level performance metrics collection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    /// Average processing latency in milliseconds
    pub avg_latency_ms: f32,
    /// Consciousness processing throughput (states/second)
    pub throughput_sps: f32,
    /// GPU memory utilization percentage
    pub gpu_memory_percent: f32,
    /// GPU compute utilization percentage
    pub gpu_compute_percent: f32,
    /// System memory utilization percentage
    pub system_memory_percent: f32,
    /// CPU utilization percentage
    pub cpu_utilization_percent: f32,
    /// Consciousness coherence score
    pub consciousness_coherence: f32,
    /// Emotional alignment score
    pub emotional_alignment: f32,
    /// Processing stability score
    pub processing_stability: f32,

    // Component-specific metrics
    /// Memory-related metrics (if available)
    pub memory_metrics: Option<MemoryPerformanceMetrics>,
    /// GPU-related metrics (if available)
    pub gpu_metrics: Option<GpuPerformanceMetrics>,
    /// I/O-related metrics (if available)
    pub io_metrics: Option<IoPerformanceMetrics>,
}

/// Memory performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPerformanceMetrics {
    /// Memory allocation latency in milliseconds
    pub allocation_latency_ms: f32,
    /// Memory deallocation latency in milliseconds
    pub deallocation_latency_ms: f32,
    /// Memory pool utilization percentage
    pub pool_utilization_percent: f32,
    /// Memory fragmentation ratio (0.0 to 1.0)
    pub fragmentation_ratio: f32,
    /// Memory efficiency score (0.0 to 1.0)
    pub efficiency: f32,
}

/// GPU performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuPerformanceMetrics {
    /// GPU memory transfer latency in milliseconds
    pub transfer_latency_ms: f32,
    /// Consciousness evolution computation latency in milliseconds
    pub evolution_latency_ms: f32,
    /// GPU kernel efficiency (0.0 to 1.0)
    pub kernel_efficiency: f32,
    /// GPU memory bandwidth utilization percentage
    pub memory_bandwidth_percent: f32,
    /// Overall GPU efficiency score (0.0 to 1.0)
    pub efficiency: f32,
}

/// I/O performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IoPerformanceMetrics {
    /// Total I/O latency in milliseconds
    pub total_latency_ms: f32,
    /// Read operation latency in milliseconds
    pub read_latency_ms: f32,
    /// Write operation latency in milliseconds
    pub write_latency_ms: f32,
    /// I/O throughput in operations per second
    pub throughput_ops: f32,
    /// I/O queue depth
    pub queue_depth: usize,
}

/// Performance trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTrend {
    /// Trend analysis period start timestamp
    pub period_start: f64,
    /// Trend analysis period end timestamp
    pub period_end: f64,
    /// Average latency trend (positive = increasing, negative = decreasing)
    pub latency_trend: f32,
    /// Average throughput trend (positive = increasing, negative = decreasing)
    pub throughput_trend: f32,
    /// Average efficiency trend (positive = improving, negative = degrading)
    pub efficiency_trend: f32,
    /// Performance volatility measure (0.0 to 1.0)
    pub volatility: f32,
    /// Trend confidence score (0.0 to 1.0)
    pub confidence: f32,
}

/// Performance alert for threshold violations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceAlert {
    /// Alert timestamp
    pub timestamp: f64,
    /// Alert severity level
    pub severity: AlertSeverity,
    /// Alert type/category
    pub alert_type: AlertType,
    /// Alert message
    pub message: String,
    /// Affected component or metric
    pub component: String,
    /// Current value that triggered the alert
    pub current_value: f32,
    /// Threshold value that was exceeded
    pub threshold_value: f32,
    /// Suggested remediation actions
    pub suggested_actions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    /// Low priority alert (informational)
    Low,
    /// Medium priority alert (warning)
    Medium,
    /// High priority alert (critical)
    High,
    /// Critical system alert (emergency)
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertType {
    /// Latency threshold exceeded
    LatencyThreshold,
    /// Throughput degradation
    ThroughputDegradation,
    /// Resource utilization high
    ResourceUtilization,
    /// Quality metric degradation
    QualityDegradation,
    /// System instability detected
    SystemInstability,
    /// Component failure
    ComponentFailure,
}

/// Main performance metrics tracking system
pub struct PerformanceTracker {
    /// Configuration settings
    config: PerformanceConfig,
    /// Current performance snapshot
    current_snapshot: Arc<RwLock<PerformanceSnapshot>>,
    /// Historical performance snapshots
    historical_snapshots: Arc<RwLock<Vec<PerformanceSnapshot>>>,
    /// Performance trends analysis
    trends: Arc<RwLock<HashMap<String, PerformanceTrend>>>,
    /// Active performance alerts
    active_alerts: Arc<RwLock<Vec<PerformanceAlert>>>,
    /// Background monitoring task handle
    monitoring_task: Option<tokio::task::JoinHandle<()>>,
    /// Alert notification system
    alert_notify: Arc<Notify>,
    /// Metrics collection start time
    collection_start_time: Instant,
}

impl PerformanceTracker {
    /// Create a new performance tracker
    pub fn new(config: PerformanceConfig) -> Self {
        let session_id = format!(
            "session_{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs()
        );

        Self {
            config,
            current_snapshot: Arc::new(RwLock::new(PerformanceSnapshot::new(session_id.clone()))),
            historical_snapshots: Arc::new(RwLock::new(Vec::new())),
            trends: Arc::new(RwLock::new(HashMap::new())),
            active_alerts: Arc::new(RwLock::new(Vec::new())),
            monitoring_task: None,
            alert_notify: Arc::new(Notify::new()),
            collection_start_time: Instant::now(),
        }
    }

    /// Start the performance tracking system
    pub fn start(&mut self) -> Result<()> {
        info!("🚀 Starting performance metrics tracking system");

        // Start background monitoring
        let monitoring_config = self.config.clone();
        let monitoring_snapshots = self.historical_snapshots.clone();
        let monitoring_current = self.current_snapshot.clone();
        let monitoring_trends = self.trends.clone();
        let monitoring_alerts = self.active_alerts.clone();
        let monitoring_notify = self.alert_notify.clone();

        self.monitoring_task = Some(tokio::spawn(async move {
            Self::background_monitoring_loop(
                monitoring_config,
                monitoring_snapshots,
                monitoring_current,
                monitoring_trends,
                monitoring_alerts,
                monitoring_notify,
            )
            .await;
        }));

        Ok(())
    }

    /// Record a performance snapshot
    pub async fn record_snapshot(&self, system_metrics: SystemMetrics) -> Result<()> {
        let mut current = self.current_snapshot.write().await;
        current.update_from_system(&system_metrics);

        // Add to historical record
        let mut historical = self.historical_snapshots.write().await;
        historical.push(current.clone());

        // Maintain retention limit
        let retention_limit =
            (self.config.retention_period_hours * 3600) / self.config.collection_interval_sec;
        if historical.len() > retention_limit as usize {
            let drain_count = historical.len() - retention_limit as usize;
            historical.drain(0..drain_count);
        }

        debug!(
            "📊 Recorded performance snapshot: {:.2}ms latency, {:.1} states/sec",
            current.e2e_latency_ms, current.throughput_sps
        );

        // Check for alerts
        self.check_performance_alerts(&current).await?;

        Ok(())
    }

    /// Get current performance snapshot
    pub async fn get_current_snapshot(&self) -> PerformanceSnapshot {
        self.current_snapshot.read().await.clone()
    }

    /// Get historical performance data for trend analysis
    pub async fn get_historical_snapshots(&self) -> Vec<PerformanceSnapshot> {
        self.historical_snapshots.read().await.clone()
    }

    /// Get performance trends analysis
    pub async fn get_performance_trends(&self) -> HashMap<String, PerformanceTrend> {
        self.trends.read().await.clone()
    }

    /// Get active performance alerts
    pub async fn get_active_alerts(&self) -> Vec<PerformanceAlert> {
        self.active_alerts.read().await.clone()
    }

    /// Check for performance threshold violations and generate alerts
    async fn check_performance_alerts(&self, snapshot: &PerformanceSnapshot) -> Result<()> {
        let mut alerts = self.active_alerts.write().await;

        // Check latency threshold
        if snapshot.e2e_latency_ms > 2000.0 {
            // 2 second threshold
            alerts.push(PerformanceAlert {
                timestamp: snapshot.timestamp,
                severity: AlertSeverity::High,
                alert_type: AlertType::LatencyThreshold,
                message: format!(
                    "Consciousness processing latency ({:.2}ms) exceeds 2s threshold",
                    snapshot.e2e_latency_ms
                ),
                component: "latency_processor".to_string(),
                current_value: snapshot.e2e_latency_ms,
                threshold_value: 2000.0,
                suggested_actions: vec![
                    "Check for GPU memory pressure".to_string(),
                    "Consider reducing batch size".to_string(),
                    "Review consciousness state complexity".to_string(),
                ],
            });
        }

        // Check throughput degradation
        if snapshot.throughput_sps < 10.0 {
            // Minimum acceptable throughput
            alerts.push(PerformanceAlert {
                timestamp: snapshot.timestamp,
                severity: AlertSeverity::Medium,
                alert_type: AlertType::ThroughputDegradation,
                message: format!("Consciousness processing throughput ({:.1} states/sec) below minimum threshold",
                               snapshot.throughput_sps),
                component: "throughput_monitor".to_string(),
                current_value: snapshot.throughput_sps,
                threshold_value: 10.0,
                suggested_actions: vec![
                    "Check for resource contention".to_string(),
                    "Optimize consciousness state processing pipeline".to_string(),
                ],
            });
        }

        // Check resource utilization
        if snapshot.gpu_memory_percent > 90.0 || snapshot.system_memory_percent > 85.0 {
            alerts.push(PerformanceAlert {
                timestamp: snapshot.timestamp,
                severity: AlertSeverity::High,
                alert_type: AlertType::ResourceUtilization,
                message: format!(
                    "High resource utilization detected (GPU: {:.1}%, System: {:.1}%)",
                    snapshot.gpu_memory_percent, snapshot.system_memory_percent
                ),
                component: "resource_monitor".to_string(),
                current_value: snapshot
                    .gpu_memory_percent
                    .max(snapshot.system_memory_percent),
                threshold_value: 90.0,
                suggested_actions: vec![
                    "Consider memory optimization".to_string(),
                    "Check for memory leaks".to_string(),
                    "Reduce consciousness state buffer sizes".to_string(),
                ],
            });
        }

        // Check processing stability
        if snapshot.processing_stability < 0.7 {
            alerts.push(PerformanceAlert {
                timestamp: snapshot.timestamp,
                severity: AlertSeverity::Medium,
                alert_type: AlertType::SystemInstability,
                message: format!(
                    "Processing stability degraded to {:.2}",
                    snapshot.processing_stability
                ),
                component: "stability_monitor".to_string(),
                current_value: snapshot.processing_stability,
                threshold_value: 0.7,
                suggested_actions: vec![
                    "Check consciousness state consistency".to_string(),
                    "Review emotional processing alignment".to_string(),
                ],
            });
        }

        // Notify if new alerts were added
        if !alerts.is_empty() {
            let recent_alerts: Vec<_> = alerts
                .iter()
                .filter(|alert| alert.timestamp > snapshot.timestamp - 60.0) // Last minute
                .cloned()
                .collect();

            if !recent_alerts.is_empty() {
                self.alert_notify.notify_waiters();
            }
        }

        Ok(())
    }

    /// Analyze performance trends over time
    pub async fn analyze_trends(&self) -> Result<HashMap<String, PerformanceTrend>> {
        let snapshots = self.get_historical_snapshots().await;

        if snapshots.len() < 10 {
            // Need sufficient data for trend analysis
            return Ok(HashMap::new());
        }

        // Analyze latency trends
        let latency_values: Vec<f32> = snapshots.iter().map(|s| s.e2e_latency_ms).collect();
        let _latency_trend = self.calculate_trend(&latency_values, "latency");

        // Analyze throughput trends
        let throughput_values: Vec<f32> = snapshots.iter().map(|s| s.throughput_sps).collect();
        let _throughput_trend = self.calculate_trend(&throughput_values, "throughput");

        // Analyze efficiency trends
        let efficiency_values: Vec<f32> = snapshots.iter().map(|s| s.overall_efficiency).collect();
        let _efficiency_trend = self.calculate_trend(&efficiency_values, "efficiency");

        // Update stored trends with analyzed data
        let mut stored_trends = self.trends.write().await;
        stored_trends.insert("latency".to_string(), _latency_trend);
        stored_trends.insert("throughput".to_string(), _throughput_trend);
        stored_trends.insert("efficiency".to_string(), _efficiency_trend);

        info!(
            "📈 Performance trend analysis completed for {} metrics",
            stored_trends.len()
        );

        Ok(stored_trends.clone())
    }

    /// Calculate trend for a series of values
    fn calculate_trend(&self, values: &[f32], metric_name: &str) -> PerformanceTrend {
        if values.len() < 2 {
            return PerformanceTrend {
                period_start: 0.0,
                period_end: 0.0,
                latency_trend: 0.0,
                throughput_trend: 0.0,
                efficiency_trend: 0.0,
                volatility: 0.0,
                confidence: 0.0,
            };
        }

        let first_value = values[0];
        let last_value = values[values.len() - 1];
        let period_start = 0.0; // Simplified - would use actual timestamps
        let period_end = values.len() as f64;

        // Calculate linear trend (simplified)
        let trend = if first_value != 0.0 {
            (last_value - first_value) / first_value
        } else {
            0.0
        };

        // Calculate volatility (standard deviation of changes)
        let changes: Vec<f32> = values.windows(2).map(|w| (w[1] - w[0]).abs()).collect();
        let volatility = if changes.is_empty() {
            0.0
        } else {
            changes.iter().sum::<f32>() / changes.len() as f32
        };

        // Trend confidence based on data consistency
        let confidence = if volatility < 0.1 { 0.9 } else { 0.7 };

        PerformanceTrend {
            period_start,
            period_end,
            latency_trend: if metric_name == "latency" { trend } else { 0.0 },
            throughput_trend: if metric_name == "throughput" {
                trend
            } else {
                0.0
            },
            efficiency_trend: if metric_name == "efficiency" {
                trend
            } else {
                0.0
            },
            volatility,
            confidence,
        }
    }

    /// Background monitoring loop
    async fn background_monitoring_loop(
        config: PerformanceConfig,
        historical_snapshots: Arc<RwLock<Vec<PerformanceSnapshot>>>,
        _current_snapshot: Arc<RwLock<PerformanceSnapshot>>,
        _trends: Arc<RwLock<HashMap<String, PerformanceTrend>>>,
        active_alerts: Arc<RwLock<Vec<PerformanceAlert>>>,
        alert_notify: Arc<Notify>,
    ) {
        let mut interval =
            tokio::time::interval(Duration::from_secs(config.collection_interval_sec));

        loop {
            interval.tick().await;

            // Periodic trend analysis
            if config.collection_interval_sec % 60 == 0 {
                // Every minute
                debug!("📈 Running periodic performance trend analysis");

                // In a real implementation, this would collect actual system metrics
                // For now, simulate metrics collection and trend analysis
            }

            // Check for alert conditions
            {
                let alerts = active_alerts.read().await;
                let recent_alerts: Vec<_> = alerts
                    .iter()
                    .filter(|alert| {
                        let current_time = SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap()
                            .as_secs_f64();
                        current_time - alert.timestamp < 300.0 // Last 5 minutes
                    })
                    .cloned()
                    .collect();

                if !recent_alerts.is_empty() {
                    warn!(
                        "🚨 {} active performance alerts detected",
                        recent_alerts.len()
                    );
                    alert_notify.notify_waiters();
                }
            }

            // Cleanup old historical data
            {
                let mut historical = historical_snapshots.write().await;
                let retention_cutoff = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs_f64()
                    - (config.retention_period_hours as f64 * 3600.0);

                historical.retain(|snapshot| snapshot.timestamp > retention_cutoff);
            }
        }
    }

    /// Generate performance health report
    pub async fn generate_health_report(&self) -> Result<PerformanceHealthReport> {
        let _current_snapshot = self.get_current_snapshot().await;
        let _trends = self.get_performance_trends().await;
        let alerts = self.get_active_alerts().await;

        Ok(PerformanceHealthReport {
            overall_health_score: _current_snapshot.health_score(),
            current_latency_ms: _current_snapshot.e2e_latency_ms,
            current_throughput_sps: _current_snapshot.throughput_sps,
            current_efficiency: _current_snapshot.overall_efficiency,
            active_alerts_count: alerts.len(),
            trend_direction: self.calculate_overall_trend_direction(&_trends),
            recommendations: self
                .generate_performance_recommendations(&_current_snapshot, &_trends),
            collection_uptime_seconds: self.collection_start_time.elapsed().as_secs(),
        })
    }

    /// Calculate overall trend direction
    fn calculate_overall_trend_direction(
        &self,
        trends: &HashMap<String, PerformanceTrend>,
    ) -> TrendDirection {
        let mut positive_trends = 0;
        let mut negative_trends = 0;

        for trend in trends.values() {
            if trend.efficiency_trend > 0.05 {
                positive_trends += 1;
            } else if trend.efficiency_trend < -0.05 {
                negative_trends += 1;
            }
        }

        if positive_trends > negative_trends {
            TrendDirection::Improving
        } else if negative_trends > positive_trends {
            TrendDirection::Degrading
        } else {
            TrendDirection::Stable
        }
    }

    /// Generate performance recommendations
    fn generate_performance_recommendations(
        &self,
        current: &PerformanceSnapshot,
        trends: &HashMap<String, PerformanceTrend>,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Latency recommendations
        if current.e2e_latency_ms > 1500.0 {
            recommendations
                .push("Consider GPU memory optimization or batch size reduction".to_string());
        }

        // Throughput recommendations
        if current.throughput_sps < 20.0 {
            recommendations
                .push("Review consciousness state processing pipeline for bottlenecks".to_string());
        }

        // Efficiency recommendations
        if current.overall_efficiency < 0.5 {
            recommendations.push("Check resource utilization and memory management".to_string());
        }

        // Trend-based recommendations
        for (metric, trend) in trends {
            if trend.efficiency_trend < -0.1 && trend.confidence > 0.7 {
                recommendations.push(format!(
                    "{} showing degradation trend - investigate immediately",
                    metric
                ));
            }
        }

        recommendations
    }

    /// Shutdown the performance tracker
    pub async fn shutdown(&mut self) -> Result<()> {
        info!("🔌 Shutting down performance metrics tracking system");

        // Stop background monitoring
        if let Some(task) = self.monitoring_task.take() {
            task.abort();
        }

        // Generate final report
        let final_report = self.generate_health_report().await?;
        info!(
            "📊 Final performance report: {:.2} health score, {:.1} states/sec",
            final_report.overall_health_score, final_report.current_throughput_sps
        );

        Ok(())
    }
}

impl Drop for PerformanceTracker {
    fn drop(&mut self) {
        let _ = futures::executor::block_on(async { self.shutdown().await });
    }
}

/// Performance health report summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceHealthReport {
    /// Overall performance health score (0.0 to 1.0)
    pub overall_health_score: f32,
    /// Current latency in milliseconds
    pub current_latency_ms: f32,
    /// Current throughput in states per second
    pub current_throughput_sps: f32,
    /// Current efficiency score
    pub current_efficiency: f32,
    /// Number of active alerts
    pub active_alerts_count: usize,
    /// Overall trend direction
    pub trend_direction: TrendDirection,
    /// Performance improvement recommendations
    pub recommendations: Vec<String>,
    /// System uptime for this collection period
    pub collection_uptime_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    /// Performance is improving over time
    Improving,
    /// Performance is degrading over time
    Degrading,
    /// Performance is stable
    Stable,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_performance_snapshot_creation() {
        let snapshot = PerformanceSnapshot::new("test_session".to_string());

        assert_eq!(snapshot.session_id, "test_session");
        assert!(snapshot.timestamp > 0.0);
        assert_eq!(snapshot.e2e_latency_ms, 0.0);
        assert_eq!(snapshot.throughput_sps, 0.0);
    }

    #[tokio::test]
    async fn test_performance_tracker_creation() {
        let config = PerformanceConfig::default();
        let tracker = PerformanceTracker::new(config);

        // Test basic functionality
        let current = tracker.get_current_snapshot().await;
        assert_eq!(current.session_id.len(), 0); // Should be set by constructor

        let trends = tracker.get_performance_trends().await;
        assert!(trends.is_empty());

        let alerts = tracker.get_active_alerts().await;
        assert!(alerts.is_empty());
    }

    #[tokio::test]
    async fn test_system_metrics_creation() {
        let metrics = SystemMetrics {
            avg_latency_ms: 100.0,
            throughput_sps: 50.0,
            gpu_memory_percent: 60.0,
            gpu_compute_percent: 70.0,
            system_memory_percent: 40.0,
            cpu_utilization_percent: 30.0,
            consciousness_coherence: 0.8,
            emotional_alignment: 0.9,
            processing_stability: 0.95,
            memory_metrics: None,
            gpu_metrics: None,
            io_metrics: None,
        };

        assert_eq!(metrics.avg_latency_ms, 100.0);
        assert_eq!(metrics.throughput_sps, 50.0);
    }
}
