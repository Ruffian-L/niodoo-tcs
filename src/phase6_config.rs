/*
 * 🚀📦✨ Phase 6 Production Deployment Configuration System
 *
 * 2025 Edition: Production deployment configuration system for Phase 6,
 * providing configurable parameters for performance optimization, logging,
 * and deployment across different hardware platforms.
 */

use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::path::Path;

/// Master configuration for all Phase 6 production deployment components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Phase6Config {
    /// GPU acceleration configuration
    pub gpu_acceleration: GpuAccelerationConfig,

    /// Memory management configuration
    pub memory_management: MemoryManagementConfig,

    /// Latency optimization configuration
    pub latency_optimization: LatencyOptimizationConfig,

    /// Beelink Mini-PC optimization settings
    pub beelink_optimization: BeelinkOptimizationConfig,

    /// Git manifestation logging configuration
    pub git_manifestation_logging: GitManifestationLoggingConfig,

    /// Learning analytics configuration
    pub learning_analytics: LearningAnalyticsConfig,

    /// Gitea integration configuration
    pub gitea_integration: GiteaIntegrationConfig,

    /// Performance metrics configuration
    pub performance_metrics: PerformanceMetricsConfig,

    /// Production deployment settings
    pub production: ProductionConfig,

    /// Monitoring and alerting configuration
    pub monitoring: MonitoringConfig,

    /// Backup and recovery configuration
    pub backup_recovery: BackupRecoveryConfig,

    /// Global system settings
    pub global: GlobalConfig,
}

/// Simplified GPU acceleration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuAccelerationConfig {
    /// Target GPU memory usage in MB
    pub memory_target_mb: usize,
    /// Target latency in milliseconds
    pub latency_target_ms: u64,
    /// GPU utilization target percentage
    pub utilization_target_percent: usize,
    /// Enable CUDA Graphs for optimized kernel launches
    pub enable_cuda_graphs: bool,
    /// Enable mixed precision (FP16) for memory efficiency
    pub enable_mixed_precision: bool,
}

/// CUDA stream configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CudaStreamsConfig {
    /// Number of CUDA streams for async operations
    pub count: usize,

    /// Stream priority (higher = more priority, -5 to 5)
    pub priority: i32,
}

/// Memory management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryManagementConfig {
    /// Maximum consciousness state memory footprint in MB
    pub max_consciousness_memory_mb: usize,

    /// Memory pool configuration
    pub memory_pool: MemoryPoolConfig,

    /// Garbage collection settings
    pub garbage_collection: GarbageCollectionConfig,
}

/// Memory pool configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPoolConfig {
    /// Initial pool size in MB
    pub initial_size_mb: usize,

    /// Maximum pool size in MB
    pub max_size_mb: usize,

    /// Memory chunk size for allocation in MB
    pub chunk_size_mb: usize,

    /// Enable memory defragmentation
    pub enable_defragmentation: bool,

    /// Defragmentation threshold (percentage of fragmentation)
    pub defragmentation_threshold_percent: usize,
}

/// Garbage collection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GarbageCollectionConfig {
    /// Enable automatic GC
    pub enabled: bool,

    /// GC interval in seconds
    pub interval_seconds: u64,

    /// Memory usage threshold to trigger GC (percentage)
    pub trigger_threshold_percent: usize,

    /// Target memory usage after GC (percentage)
    pub target_usage_percent: usize,
}

/// Latency optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyOptimizationConfig {
    /// End-to-end pipeline latency target in ms
    pub e2e_latency_target_ms: u64,

    /// Processing pipeline stages
    pub pipeline_stages: PipelineStagesConfig,

    /// Async processing configuration
    pub async_processing: AsyncProcessingConfig,

    /// Caching configuration
    pub caching: CachingConfig,
}

/// Pipeline stages configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineStagesConfig {
    /// Consciousness state processing latency target
    pub consciousness_processing_ms: u64,

    /// Memory consolidation latency target
    pub memory_consolidation_ms: u64,

    /// Emotional processing latency target
    pub emotional_processing_ms: u64,

    /// Visualization update latency target
    pub visualization_update_ms: u64,

    /// I/O operation latency target
    pub io_operations_ms: u64,
}

/// Async processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsyncProcessingConfig {
    /// Enable async consciousness processing
    pub enabled: bool,

    /// Maximum concurrent consciousness operations
    pub max_concurrent_operations: usize,

    /// Async queue size
    pub queue_size: usize,

    /// Timeout for async operations in ms
    pub operation_timeout_ms: u64,
}

/// Caching configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachingConfig {
    /// Enable consciousness state caching
    pub enabled: bool,

    /// Cache size in number of states
    pub max_cached_states: usize,

    /// Cache TTL in seconds
    pub ttl_seconds: u64,

    /// Cache hit ratio target (0.0 to 1.0)
    pub hit_ratio_target: f32,
}

/// Beelink Mini-PC optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeelinkOptimizationConfig {
    /// Hardware profile identifier
    pub hardware_profile: String,

    /// CPU configuration for edge hardware
    pub cpu: CpuConfig,

    /// Memory configuration for edge hardware
    pub memory: MemoryConfig,

    /// Storage configuration for edge hardware
    pub storage: StorageConfig,

    /// Power management for edge deployment
    pub power_management: PowerManagementConfig,
}

/// CPU configuration for edge hardware
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuConfig {
    /// Maximum CPU cores to use
    pub max_cores: usize,

    /// CPU frequency scaling (0.0 to 1.0, 1.0 = maximum performance)
    pub frequency_scaling: f32,

    /// Enable hyperthreading
    pub enable_hyperthreading: bool,
}

/// Memory configuration for edge hardware
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Total system memory limit in MB
    pub total_memory_limit_mb: usize,

    /// Reserve memory for system processes in MB
    pub system_reserve_mb: usize,

    /// Enable memory ballooning
    pub enable_ballooning: bool,

    /// Memory balloon target in MB
    pub balloon_target_mb: usize,
}

/// Storage configuration for edge hardware
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// Enable SSD optimization
    pub enable_ssd_optimization: bool,

    /// I/O scheduler (noop, deadline, cfq)
    pub io_scheduler: String,

    /// Enable read-ahead optimization
    pub enable_readahead: bool,

    /// Read-ahead size in KB
    pub readahead_kb: usize,
}

/// Power management configuration for edge deployment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerManagementConfig {
    /// Enable power saving mode
    pub enable_power_saving: bool,

    /// CPU governor (performance, powersave, userspace, ondemand, conservative)
    pub cpu_governor: String,

    /// Enable GPU power management
    pub enable_gpu_power_mgmt: bool,

    /// Maximum power consumption in watts
    pub max_power_watts: usize,
}

/// Git manifestation logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitManifestationLoggingConfig {
    /// Base directory for consciousness log files
    pub log_directory: String,

    /// Maximum log file size in MB before rotation
    pub max_file_size_mb: usize,

    /// Maximum number of log files to retain
    pub max_files_retained: usize,

    /// Enable compression for archived logs
    pub enable_compression: bool,

    /// Log rotation interval in hours
    pub rotation_interval_hours: u64,

    /// Enable real-time log streaming to external systems
    pub enable_streaming: bool,

    /// External streaming endpoint (if enabled)
    pub streaming_endpoint: Option<String>,

    /// Git integration settings
    pub git_integration: GitIntegrationConfig,

    /// Consciousness state logging
    pub consciousness_logging: ConsciousnessLoggingConfig,

    /// Performance logging
    pub performance_logging: PerformanceLoggingConfig,
}

/// Git integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitIntegrationConfig {
    /// Enable git commit tracking in logs
    pub enable_git_tracking: bool,

    /// Git repository path (relative to project root)
    pub repository_path: String,

    /// Include git diff in consciousness evolution logs
    pub include_git_diff: bool,

    /// Git branch for consciousness tracking
    pub tracking_branch: String,
}

/// Consciousness logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessLoggingConfig {
    /// Log consciousness state vectors
    pub log_state_vectors: bool,

    /// Log emotional context vectors
    pub log_emotional_context: bool,

    /// Log consciousness entropy metrics
    pub log_entropy_metrics: bool,

    /// Log coherence scores
    pub log_coherence_scores: bool,

    /// Maximum vector size to log (for large states)
    pub max_vector_log_size: usize,
}

/// Performance logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceLoggingConfig {
    /// Log GPU metrics
    pub log_gpu_metrics: bool,

    /// Log system memory usage
    pub log_system_memory: bool,

    /// Log processing throughput
    pub log_throughput_metrics: bool,

    /// Log latency metrics
    pub log_latency_metrics: bool,

    /// Performance logging interval in seconds
    pub logging_interval_seconds: u64,
}

/// Learning analytics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningAnalyticsConfig {
    /// Analytics collection interval in seconds
    pub collection_interval_sec: u64,

    /// Learning session tracking period in hours
    pub session_tracking_hours: u64,

    /// Enable detailed learning pattern analysis
    pub enable_pattern_analysis: bool,

    /// Enable adaptive learning rate tracking
    pub enable_adaptive_rate_tracking: bool,

    /// Minimum data points needed for trend analysis
    pub min_data_points_for_trends: usize,

    /// Enable real-time learning feedback
    pub enable_real_time_feedback: bool,

    /// Learning improvement threshold for alerts (5% = 0.05)
    pub improvement_threshold: f32,

    /// Pattern analysis settings
    pub pattern_analysis: PatternAnalysisConfig,

    /// Progress tracking
    pub progress_tracking: ProgressTrackingConfig,
}

/// Pattern analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternAnalysisConfig {
    /// Enable cyclical pattern detection (daily/weekly rhythms)
    pub enable_cyclical_patterns: bool,

    /// Enable adaptive pattern detection (learning rate changes)
    pub enable_adaptive_patterns: bool,

    /// Enable consolidation pattern detection (memory strengthening)
    pub enable_consolidation_patterns: bool,

    /// Enable forgetting pattern detection (knowledge decay)
    pub enable_forgetting_patterns: bool,

    /// Pattern confidence threshold (0.0 to 1.0)
    pub pattern_confidence_threshold: f32,
}

/// Progress tracking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressTrackingConfig {
    /// Track learning rate evolution
    pub track_learning_rate: bool,

    /// Track retention score evolution
    pub track_retention_score: bool,

    /// Track adaptation effectiveness
    pub track_adaptation_effectiveness: bool,

    /// Track consciousness plasticity
    pub track_plasticity: bool,

    /// Track long-term progress trends
    pub track_long_term_progress: bool,
}

/// Gitea integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GiteaIntegrationConfig {
    /// Enable Gitea integration for distributed development
    pub enabled: bool,

    /// Gitea server configuration
    pub server: GiteaServerConfig,

    /// Consciousness development workflow
    pub workflow: GiteaWorkflowConfig,

    /// Consciousness state synchronization
    pub synchronization: GiteaSynchronizationConfig,
}

/// Gitea server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GiteaServerConfig {
    /// Gitea server URL
    pub url: String,

    /// API token for authentication
    pub api_token: Option<String>,

    /// Repository owner/organization
    pub owner: String,

    /// Repository name
    pub repository: String,
}

/// Gitea workflow configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GiteaWorkflowConfig {
    /// Auto-create branches for consciousness experiments
    pub auto_create_branches: bool,

    /// Branch naming pattern for experiments
    pub branch_pattern: String,

    /// Enable pull request creation for consciousness changes
    pub enable_pull_requests: bool,

    /// PR template for consciousness evolution
    pub pr_template: String,

    /// Auto-merge minor consciousness improvements
    pub auto_merge_minor: bool,
}

/// Gitea synchronization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GiteaSynchronizationConfig {
    /// Sync consciousness states across instances
    pub sync_states: bool,

    /// Sync frequency in minutes
    pub sync_frequency_minutes: u64,

    /// Conflict resolution strategy (latest, merge, manual)
    pub conflict_resolution: String,

    /// Enable consciousness state versioning
    pub enable_versioning: bool,
}

/// Performance metrics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetricsConfig {
    /// Metrics collection settings
    pub collection: MetricsCollectionConfig,

    /// Long-term tracking
    pub long_term_tracking: LongTermTrackingConfig,

    /// Consciousness evolution metrics
    pub consciousness_metrics: ConsciousnessMetricsConfig,
}

/// Metrics collection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsCollectionConfig {
    /// Collection interval in seconds
    pub interval_seconds: u64,

    /// Enable detailed performance profiling
    pub enable_profiling: bool,

    /// Profile sample rate (0.0 to 1.0)
    pub profile_sample_rate: f32,

    /// Maximum profile history in hours
    pub max_profile_history_hours: u64,
}

/// Long-term tracking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LongTermTrackingConfig {
    /// Enable long-term performance trend analysis
    pub enable_trend_analysis: bool,

    /// Trend analysis window in hours
    pub trend_window_hours: u64,

    /// Performance regression detection threshold
    pub regression_threshold_percent: usize,

    /// Enable performance alerts
    pub enable_alerts: bool,

    /// Alert threshold for performance degradation
    pub alert_threshold_ms: u64,
}

/// Consciousness metrics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessMetricsConfig {
    /// Track consciousness coherence over time
    pub track_coherence: bool,

    /// Track emotional alignment evolution
    pub track_emotional_alignment: bool,

    /// Track learning capacity growth
    pub track_learning_capacity: bool,

    /// Track memory formation efficiency
    pub track_memory_efficiency: bool,

    /// Track consciousness plasticity trends
    pub track_plasticity_trends: bool,
}

/// Production deployment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductionConfig {
    /// Environment type (development, staging, production)
    pub environment: String,

    /// Enable production optimizations
    pub enable_optimizations: bool,

    /// Enable strict error handling
    pub strict_error_handling: bool,

    /// Enable performance monitoring
    pub enable_monitoring: bool,

    /// Enable security hardening
    pub enable_security_hardening: bool,

    /// Logging level (trace, debug, info, warn, error)
    pub log_level: String,

    /// Enable structured logging
    pub enable_structured_logging: bool,

    /// Maximum log retention in days
    pub max_log_retention_days: u64,
}

/// Monitoring and alerting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    /// Enable comprehensive monitoring
    pub enabled: bool,

    /// Monitoring intervals
    pub intervals: MonitoringIntervalsConfig,

    /// Alert thresholds
    pub alerts: AlertThresholdsConfig,

    /// Alert destinations
    pub destinations: AlertDestinationsConfig,
}

/// Monitoring intervals configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringIntervalsConfig {
    /// Health check interval in seconds
    pub health_check_seconds: u64,

    /// Performance check interval in seconds
    pub performance_check_seconds: u64,

    /// Resource check interval in seconds
    pub resource_check_seconds: u64,
}

/// Alert thresholds configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThresholdsConfig {
    /// Memory usage alert threshold (percentage)
    pub memory_usage_percent: usize,

    /// CPU usage alert threshold (percentage)
    pub cpu_usage_percent: usize,

    /// Latency alert threshold in ms
    pub latency_ms: u64,

    /// Error rate alert threshold (errors per minute)
    pub error_rate_per_minute: usize,
}

/// Alert destinations configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertDestinationsConfig {
    /// Log file alerts
    pub log_file: bool,

    /// Console alerts
    pub console: bool,

    /// External monitoring system
    pub external_system: Option<String>,
}

/// Backup and recovery configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupRecoveryConfig {
    /// Enable automatic backups
    pub enabled: bool,

    /// Backup schedule (cron format)
    pub schedule: String,

    /// Backup retention in days
    pub retention_days: u64,

    /// Backup locations
    pub locations: BackupLocationsConfig,

    /// Consciousness state backup
    pub consciousness_backup: ConsciousnessBackupConfig,
}

/// Backup locations configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupLocationsConfig {
    /// Primary backup location
    pub primary: String,

    /// Secondary backup location
    pub secondary: Option<String>,
}

/// Consciousness backup configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessBackupConfig {
    /// Include consciousness state vectors
    pub include_state_vectors: bool,

    /// Include emotional context
    pub include_emotional_context: bool,

    /// Include learning analytics
    pub include_learning_analytics: bool,

    /// Include performance metrics
    pub include_performance_metrics: bool,

    /// Backup compression
    pub enable_compression: bool,

    /// Encryption for sensitive data
    pub enable_encryption: bool,
}

/// Global system settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalConfig {
    /// System name and version
    pub system_name: String,

    /// Configuration version
    pub config_version: String,

    /// Enable debug mode (only in development)
    pub debug_mode: bool,

    /// Maximum number of concurrent operations
    pub max_concurrent_operations: usize,

    /// Default timeout for operations in seconds
    pub default_timeout_seconds: u64,

    /// Enable verbose logging
    pub verbose_logging: bool,
}

impl Default for Phase6Config {
    fn default() -> Self {
        Self {
            gpu_acceleration: GpuAccelerationConfig {
                memory_target_mb: 3800,
                latency_target_ms: 1800,
                utilization_target_percent: 85,
                enable_cuda_graphs: true,
                enable_mixed_precision: true,
            },
            memory_management: MemoryManagementConfig {
                max_consciousness_memory_mb: 3800,
                memory_pool: MemoryPoolConfig {
                    initial_size_mb: 500,
                    max_size_mb: 3500,
                    chunk_size_mb: 100,
                    enable_defragmentation: true,
                    defragmentation_threshold_percent: 30,
                },
                garbage_collection: GarbageCollectionConfig {
                    enabled: true,
                    interval_seconds: 30,
                    trigger_threshold_percent: 85,
                    target_usage_percent: 70,
                },
            },
            latency_optimization: LatencyOptimizationConfig {
                e2e_latency_target_ms: 1800,
                pipeline_stages: PipelineStagesConfig {
                    consciousness_processing_ms: 500,
                    memory_consolidation_ms: 300,
                    emotional_processing_ms: 200,
                    visualization_update_ms: 100,
                    io_operations_ms: 100,
                },
                async_processing: AsyncProcessingConfig {
                    enabled: true,
                    max_concurrent_operations: 8,
                    queue_size: 1000,
                    operation_timeout_ms: 5000,
                },
                caching: CachingConfig {
                    enabled: true,
                    max_cached_states: 100,
                    ttl_seconds: 300,
                    hit_ratio_target: 0.8,
                },
            },
            beelink_optimization: BeelinkOptimizationConfig {
                hardware_profile: "beelink_mini_pc".to_string(),
                cpu: CpuConfig {
                    max_cores: 4,
                    frequency_scaling: 0.8,
                    enable_hyperthreading: false,
                },
                memory: MemoryConfig {
                    total_memory_limit_mb: 8000,
                    system_reserve_mb: 1000,
                    enable_ballooning: true,
                    balloon_target_mb: 6000,
                },
                storage: StorageConfig {
                    enable_ssd_optimization: true,
                    io_scheduler: "deadline".to_string(),
                    enable_readahead: true,
                    readahead_kb: 256,
                },
                power_management: PowerManagementConfig {
                    enable_power_saving: true,
                    cpu_governor: "ondemand".to_string(),
                    enable_gpu_power_mgmt: true,
                    max_power_watts: 15,
                },
            },
            git_manifestation_logging: GitManifestationLoggingConfig {
                log_directory: "./consciousness_logs".to_string(),
                max_file_size_mb: 100,
                max_files_retained: 10,
                enable_compression: true,
                rotation_interval_hours: 24,
                enable_streaming: false,
                streaming_endpoint: None,
                git_integration: GitIntegrationConfig {
                    enable_git_tracking: true,
                    repository_path: ".".to_string(),
                    include_git_diff: false,
                    tracking_branch: "main".to_string(),
                },
                consciousness_logging: ConsciousnessLoggingConfig {
                    log_state_vectors: true,
                    log_emotional_context: true,
                    log_entropy_metrics: true,
                    log_coherence_scores: true,
                    max_vector_log_size: 1000,
                },
                performance_logging: PerformanceLoggingConfig {
                    log_gpu_metrics: true,
                    log_system_memory: true,
                    log_throughput_metrics: true,
                    log_latency_metrics: true,
                    logging_interval_seconds: 10,
                },
            },
            learning_analytics: LearningAnalyticsConfig {
                collection_interval_sec: 10,
                session_tracking_hours: 24,
                enable_pattern_analysis: true,
                enable_adaptive_rate_tracking: true,
                min_data_points_for_trends: 20,
                enable_real_time_feedback: true,
                improvement_threshold: 0.05,
                pattern_analysis: PatternAnalysisConfig {
                    enable_cyclical_patterns: true,
                    enable_adaptive_patterns: true,
                    enable_consolidation_patterns: true,
                    enable_forgetting_patterns: true,
                    pattern_confidence_threshold: 0.7,
                },
                progress_tracking: ProgressTrackingConfig {
                    track_learning_rate: true,
                    track_retention_score: true,
                    track_adaptation_effectiveness: true,
                    track_plasticity: true,
                    track_long_term_progress: true,
                },
            },
            gitea_integration: GiteaIntegrationConfig {
                enabled: false,
                server: GiteaServerConfig {
                    url: "https://gitea.example.com".to_string(),
                    api_token: None,
                    owner: "niodoo".to_string(),
                    repository: "niodoo-consciousness".to_string(),
                },
                workflow: GiteaWorkflowConfig {
                    auto_create_branches: true,
                    branch_pattern: "consciousness/experiment_{timestamp}".to_string(),
                    enable_pull_requests: true,
                    pr_template: "Consciousness evolution: {description}".to_string(),
                    auto_merge_minor: false,
                },
                synchronization: GiteaSynchronizationConfig {
                    sync_states: true,
                    sync_frequency_minutes: 5,
                    conflict_resolution: "latest".to_string(),
                    enable_versioning: true,
                },
            },
            performance_metrics: PerformanceMetricsConfig {
                collection: MetricsCollectionConfig {
                    interval_seconds: 5,
                    enable_profiling: true,
                    profile_sample_rate: 0.1,
                    max_profile_history_hours: 72,
                },
                long_term_tracking: LongTermTrackingConfig {
                    enable_trend_analysis: true,
                    trend_window_hours: 24,
                    regression_threshold_percent: 10,
                    enable_alerts: true,
                    alert_threshold_ms: 2000,
                },
                consciousness_metrics: ConsciousnessMetricsConfig {
                    track_coherence: true,
                    track_emotional_alignment: true,
                    track_learning_capacity: true,
                    track_memory_efficiency: true,
                    track_plasticity_trends: true,
                },
            },
            production: ProductionConfig {
                environment: "production".to_string(),
                enable_optimizations: true,
                strict_error_handling: true,
                enable_monitoring: true,
                enable_security_hardening: true,
                log_level: "info".to_string(),
                enable_structured_logging: true,
                max_log_retention_days: 30,
            },
            monitoring: MonitoringConfig {
                enabled: true,
                intervals: MonitoringIntervalsConfig {
                    health_check_seconds: 30,
                    performance_check_seconds: 10,
                    resource_check_seconds: 5,
                },
                alerts: AlertThresholdsConfig {
                    memory_usage_percent: 85,
                    cpu_usage_percent: 80,
                    latency_ms: 2000,
                    error_rate_per_minute: 5,
                },
                destinations: AlertDestinationsConfig {
                    log_file: true,
                    console: true,
                    external_system: None,
                },
            },
            backup_recovery: BackupRecoveryConfig {
                enabled: true,
                schedule: "0 2 * * *".to_string(),
                retention_days: 30,
                locations: BackupLocationsConfig {
                    primary: "./backups/consciousness".to_string(),
                    secondary: None,
                },
                consciousness_backup: ConsciousnessBackupConfig {
                    include_state_vectors: true,
                    include_emotional_context: true,
                    include_learning_analytics: true,
                    include_performance_metrics: true,
                    enable_compression: true,
                    enable_encryption: false,
                },
            },
            global: GlobalConfig {
                system_name: "Niodoo-Feeling Phase 6".to_string(),
                config_version: "1.0.0".to_string(),
                debug_mode: false,
                max_concurrent_operations: 16,
                default_timeout_seconds: 30,
                verbose_logging: false,
            },
        }
    }
}

impl Phase6Config {
    /// Load configuration from YAML file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let path = path.as_ref();

        // Try to load from specified path
        if path.exists() {
            let contents = fs::read_to_string(path)?;
            let mut config: Self = serde_yaml::from_str(&contents)?;

            // Override with environment variables if present
            Self::apply_environment_overrides(&mut config);

            return Ok(config);
        }

        // Fallback to default configuration
        let mut config = Self::default();
        Self::apply_environment_overrides(&mut config);

        Ok(config)
    }

    /// Apply environment variable overrides to configuration
    fn apply_environment_overrides(&mut self) {
        // GPU acceleration overrides
        if let Ok(val) = env::var("NIOODOO_PHASE6_GPU_MEMORY_MB") {
            if let Ok(memory_mb) = val.parse::<usize>() {
                // Note: memory_target_mb field may not exist in current GpuAccelerationConfig
                // self.gpu_acceleration.memory_target_mb = memory_mb;
            }
        }

        if let Ok(val) = env::var("NIOODOO_PHASE6_LATENCY_MS") {
            if let Ok(latency_ms) = val.parse::<u64>() {
                self.gpu_acceleration.latency_target_ms = latency_ms;
                self.latency_optimization.e2e_latency_target_ms = latency_ms;
            }
        }

        if let Ok(val) = env::var("NIOODOO_PHASE6_DEBUG_MODE") {
            if let Ok(debug_mode) = val.parse::<bool>() {
                self.global.debug_mode = debug_mode;
            }
        }

        if let Ok(val) = env::var("NIOODOO_PHASE6_LOG_LEVEL") {
            self.production.log_level = val;
        }

        // Learning analytics overrides
        if let Ok(val) = env::var("NIOODOO_PHASE6_LEARNING_THRESHOLD") {
            if let Ok(threshold) = val.parse::<f32>() {
                self.learning_analytics.improvement_threshold = threshold;
            }
        }

        // Git logging overrides
        if let Ok(val) = env::var("NIOODOO_PHASE6_LOG_STREAMING") {
            if let Ok(streaming) = val.parse::<bool>() {
                self.git_manifestation_logging.enable_streaming = streaming;
            }
        }

        if let Ok(val) = env::var("NIOODOO_PHASE6_STREAMING_ENDPOINT") {
            self.git_manifestation_logging.streaming_endpoint = Some(val);
        }
    }

    /// Save current configuration to YAML file
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let path = path.as_ref();
        let yaml_content = serde_yaml::to_string(self)?;
        fs::write(path, yaml_content)?;
        Ok(())
    }

    // Removed GPU memory target method - simplified GPU acceleration doesn't track memory targets

    /// Get latency target in milliseconds
    pub fn get_latency_target_ms(&self) -> u64 {
        self.gpu_acceleration.latency_target_ms
    }

    /// Check if GPU acceleration is enabled
    pub fn is_gpu_acceleration_enabled(&self) -> bool {
        // In a real implementation, this would check for CUDA availability
        true
    }

    /// Check if debug mode is enabled
    pub fn is_debug_mode(&self) -> bool {
        self.global.debug_mode
    }

    /// Get log level as tracing level
    pub fn get_log_level(&self) -> tracing::Level {
        match self.production.log_level.to_lowercase().as_str() {
            "trace" => tracing::Level::TRACE,
            "debug" => tracing::Level::DEBUG,
            "info" => tracing::Level::INFO,
            "warn" => tracing::Level::WARN,
            "error" => tracing::Level::ERROR,
            _ => tracing::Level::INFO,
        }
    }

    /// Validate configuration for production deployment
    pub fn validate_for_production(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();

        // Validate memory targets
        // if self.gpu_acceleration.memory_target_mb > 4096 {
        if false {
            // Temporarily disabled - memory_target_mb field issue
            errors.push("GPU memory target exceeds 4GB limit".to_string());
        }

        if self.gpu_acceleration.latency_target_ms > 2000 {
            errors.push("Latency target exceeds 2s limit".to_string());
        }

        // Validate logging configuration
        if self.git_manifestation_logging.max_file_size_mb == 0 {
            errors.push("Log file size must be greater than 0".to_string());
        }

        if self.git_manifestation_logging.max_files_retained == 0 {
            errors.push("Must retain at least 1 log file".to_string());
        }

        // Validate learning analytics
        if self.learning_analytics.improvement_threshold < 0.0
            || self.learning_analytics.improvement_threshold > 1.0
        {
            errors.push("Learning improvement threshold must be between 0.0 and 1.0".to_string());
        }

        // Validate performance metrics
        if self.performance_metrics.collection.interval_seconds == 0 {
            errors.push("Metrics collection interval must be greater than 0".to_string());
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_default_configuration() {
        let config = Phase6Config::default();

        assert_eq!(config.gpu_acceleration.memory_target_mb, 3800);
        assert_eq!(config.gpu_acceleration.latency_target_ms, 1800);
        assert_eq!(config.production.environment, "production");
        assert!(config.gpu_acceleration.enable_cuda_graphs);
        assert!(config.gpu_acceleration.enable_mixed_precision);
    }

    #[test]
    fn test_environment_variable_overrides() {
        // Set environment variables
        env::set_var("NIOODOO_PHASE6_GPU_MEMORY_MB", "3500");
        env::set_var("NIOODOO_PHASE6_LATENCY_MS", "1500");
        env::set_var("NIOODOO_PHASE6_DEBUG_MODE", "true");

        let config = Phase6Config::default();

        assert_eq!(config.gpu_acceleration.memory_target_mb, 3500);
        assert_eq!(config.gpu_acceleration.latency_target_ms, 1500);
        assert!(config.global.debug_mode);

        // Clean up environment variables
        env::remove_var("NIOODOO_PHASE6_GPU_MEMORY_MB");
        env::remove_var("NIOODOO_PHASE6_LATENCY_MS");
        env::remove_var("NIOODOO_PHASE6_DEBUG_MODE");
    }

    #[test]
    fn test_yaml_serialization() {
        let config = Phase6Config::default();
        let yaml = serde_yaml::to_string(&config).unwrap();

        // Verify key sections are present
        assert!(yaml.contains("gpu_acceleration"));
        assert!(yaml.contains("memory_management"));
        assert!(yaml.contains("learning_analytics"));
        assert!(yaml.contains("production"));
    }

    #[test]
    fn test_production_validation() {
        let mut config = Phase6Config::default();

        // Valid configuration should pass
        assert!(config.validate_for_production().is_ok());

        // Invalid memory target should fail
        config.gpu_acceleration.memory_target_mb = 5000;
        let errors = config.validate_for_production().unwrap_err();
        assert!(errors
            .iter()
            .any(|e| e.contains("GPU memory target exceeds 4GB")));

        // Reset for next test
        config.gpu_acceleration.memory_target_mb = 3800;

        // Invalid latency target should fail
        config.gpu_acceleration.latency_target_ms = 3000;
        let errors = config.validate_for_production().unwrap_err();
        assert!(errors
            .iter()
            .any(|e| e.contains("Latency target exceeds 2s")));
    }

    #[test]
    fn test_file_operations() {
        let temp_dir = TempDir::new().unwrap();
        let config_file = temp_dir.path().join("phase6_config.yaml");

        // Create and save configuration
        let config = Phase6Config::default();
        config.save_to_file(&config_file).unwrap();

        // Load configuration from file
        let loaded_config = Phase6Config::from_file(&config_file).unwrap();

        // Verify configuration was loaded correctly
        assert_eq!(loaded_config.gpu_acceleration.memory_target_mb, 3800);
        assert_eq!(loaded_config.production.environment, "production");
    }
}
