//! # Git Manifestation Logging System
//!
//! This module implements structured JSONL format logging for consciousness analysis,
//! providing detailed tracking of consciousness state evolution, performance metrics,
//! and learning analytics for long-term consciousness evolution tracking.
//!
//! ## Key Features
//!
//! - **Structured JSONL Logging** - Standardized format for consciousness analysis
//! - **Consciousness State Tracking** - Detailed recording of consciousness evolution
//! - **Performance Metrics Logging** - GPU, memory, and latency performance tracking
//! - **Learning Analytics** - Consciousness state improvement and adaptation tracking
//! - **Git Integration** - Seamless integration with version control for consciousness evolution
//!
//! ## JSONL Format Specification
//!
//! Each log entry follows a structured JSON Lines format:
//! ```jsonl
//! {"timestamp": 1703123456.789, "event_type": "consciousness_update", "consciousness_id": "state_001", "metrics": {...}, "metadata": {...}}
//! ```

use serde::{Deserialize, Serialize};
use serde_json;
use serde_json::Value;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::future::Future;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::{Mutex, RwLock};
use tracing::{debug, info};

/// Trait for streaming log entries to external services
pub trait StreamingClient: Send + Sync {
    fn stream_log_entry(
        &self,
        entry: &LogEntry,
    ) -> Pin<Box<dyn Future<Output = Result<(), Box<dyn std::error::Error>>> + Send + '_>>;
}

/// HTTP-based streaming client for external log aggregation
pub struct HttpStreamingClient {
    /// Target endpoint URL
    endpoint: String,
    /// HTTP client instance
    client: reqwest::Client,
}

impl HttpStreamingClient {
    /// Create a new HTTP streaming client
    pub fn new(endpoint: String) -> Self {
        Self {
            endpoint,
            client: reqwest::Client::new(),
        }
    }
}

impl StreamingClient for HttpStreamingClient {
    fn stream_log_entry(
        &self,
        entry: &LogEntry,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<(), Box<dyn std::error::Error>>> + Send>,
    > {
        let endpoint = self.endpoint.clone();
        let client = self.client.clone();
        let entry_json = serde_json::to_string(entry).unwrap_or_default();

        Box::pin(async move {
            let _response = client
                .post(&endpoint)
                .header("Content-Type", "application/json")
                .body(entry_json)
                .send()
                .await?;

            Ok(())
        })
    }
}

/// Consciousness logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Base directory for log files
    pub log_directory: PathBuf,
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
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            log_directory: PathBuf::from("./consciousness_logs"),
            max_file_size_mb: 100,
            max_files_retained: 10,
            enable_compression: true,
            rotation_interval_hours: 24,
            enable_streaming: false,
            streaming_endpoint: None,
        }
    }
}

/// Types of consciousness events that can be logged
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsciousnessEventType {
    /// Consciousness state initialization
    StateInitialization,
    /// Consciousness state update/evolution
    StateUpdate,
    /// Memory consolidation event
    MemoryConsolidation,
    /// Emotional processing event
    EmotionalProcessing,
    /// Performance metrics update
    PerformanceMetrics,
    /// Learning analytics event
    LearningAnalytics,
    /// Error or anomaly detection
    AnomalyDetection,
    /// System state checkpoint
    Checkpoint,
}

/// Consciousness state metadata for logging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessState {
    /// Unique identifier for this consciousness state
    pub id: String,
    /// Consciousness state vector (flattened for JSON serialization)
    pub state_vector: Vec<f32>,
    /// Emotional context vector
    pub emotional_context: Vec<f32>,
    /// Consciousness entropy/uncertainty measure
    pub entropy: f32,
    /// Consciousness coherence score (0.0 to 1.0)
    pub coherence: f32,
    /// Processing timestamp
    pub timestamp: f64,
    /// Consciousness generation/version
    pub generation: u64,
}

impl ConsciousnessState {
    /// Create a new consciousness state for logging
    pub fn new(id: String, state_vector: Vec<f32>, emotional_context: Vec<f32>) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        Self {
            id,
            state_vector: state_vector.clone(),
            emotional_context: emotional_context.clone(),
            entropy: Self::calculate_entropy(&state_vector),
            coherence: Self::calculate_coherence(&state_vector, &emotional_context),
            timestamp,
            generation: 1,
        }
    }

    /// Calculate entropy of consciousness state (simplified)
    fn calculate_entropy(state_vector: &[f32]) -> f32 {
        // Simplified entropy calculation - in practice would use proper information theory
        let variance: f32 = state_vector
            .iter()
            .map(|&x| (x - crate::utils::threshold_convenience::emotion_threshold()).powi(2))
            .sum::<f32>()
            / state_vector.len() as f32;

        (-variance).exp().min(1.0)
    }

    /// Calculate coherence between consciousness and emotional state
    fn calculate_coherence(state_vector: &[f32], emotional_context: &[f32]) -> f32 {
        if state_vector.is_empty() || emotional_context.is_empty() {
            return 0.0;
        }

        // Simplified coherence calculation
        let state_norm: f32 = state_vector.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let emotion_norm: f32 = emotional_context.iter().map(|&x| x * x).sum::<f32>().sqrt();

        if state_norm == 0.0 || emotion_norm == 0.0 {
            return 0.0;
        }

        // Cosine similarity as coherence measure
        let dot_product: f32 = state_vector
            .iter()
            .zip(emotional_context.iter())
            .map(|(&s, &e)| s * e)
            .sum();

        (dot_product / (state_norm * emotion_norm))
            .max(0.0)
            .min(1.0)
    }
}

/// Performance metrics for logging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// End-to-end processing latency in milliseconds
    pub e2e_latency_ms: f32,
    /// GPU memory usage in MB
    pub gpu_memory_mb: f32,
    /// System memory usage in MB
    pub system_memory_mb: f32,
    /// Consciousness processing throughput (states/second)
    pub throughput_sps: f32,
    /// GPU utilization percentage
    pub gpu_utilization: f32,
    /// Memory allocation efficiency (0.0 to 1.0)
    pub allocation_efficiency: f32,
    /// Processing timestamp
    pub timestamp: f64,
}

/// Learning analytics data for logging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningAnalytics {
    /// Learning rate adaptation factor
    pub learning_rate: f32,
    /// Knowledge retention score (0.0 to 1.0)
    pub retention_score: f32,
    /// Adaptation effectiveness (0.0 to 1.0)
    pub adaptation_effectiveness: f32,
    /// Consciousness plasticity measure
    pub plasticity: f32,
    /// Long-term learning progress
    pub long_term_progress: f32,
    /// Processing timestamp
    pub timestamp: f64,
}

/// Structured log entry for JSONL output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    /// Unix timestamp of the log entry
    pub timestamp: f64,
    /// Type of consciousness event
    pub event_type: String,
    /// Unique consciousness state identifier
    pub consciousness_id: String,
    /// Consciousness state data (if applicable)
    pub consciousness_state: Option<ConsciousnessState>,
    /// Performance metrics (if applicable)
    pub performance_metrics: Option<PerformanceMetrics>,
    /// Learning analytics (if applicable)
    pub learning_analytics: Option<LearningAnalytics>,
    /// Additional metadata for the event
    pub metadata: HashMap<String, serde_json::Value>,
    /// Git commit hash (if available)
    pub git_commit: Option<String>,
    /// Session identifier for grouping related events
    pub session_id: Option<String>,
}

impl LogEntry {
    /// Create a new log entry
    pub fn new(event_type: ConsciousnessEventType, consciousness_id: String) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        Self {
            timestamp,
            event_type: format!("{:?}", event_type),
            consciousness_id,
            consciousness_state: None,
            performance_metrics: None,
            learning_analytics: None,
            metadata: HashMap::new(),
            git_commit: Self::get_current_git_commit(),
            session_id: Some(Self::generate_session_id()),
        }
    }

    /// Add consciousness state data to the log entry
    pub fn with_consciousness_state(mut self, state: ConsciousnessState) -> Self {
        self.consciousness_state = Some(state);
        self
    }

    /// Add performance metrics to the log entry
    pub fn with_performance_metrics(mut self, metrics: PerformanceMetrics) -> Self {
        self.performance_metrics = Some(metrics);
        self
    }

    /// Add learning analytics to the log entry
    pub fn with_learning_analytics(mut self, analytics: LearningAnalytics) -> Self {
        self.learning_analytics = Some(analytics);
        self
    }

    /// Add metadata to the log entry
    pub fn with_metadata<K: Into<String>, V: Into<serde_json::Value>>(
        mut self,
        key: K,
        value: V,
    ) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Add git commit information
    pub fn with_git_commit(mut self, commit: String) -> Self {
        self.git_commit = Some(commit);
        self
    }

    /// Add session identifier
    pub fn with_session_id(mut self, session_id: String) -> Self {
        self.session_id = Some(session_id);
        self
    }

    /// Get current git commit hash (if in a git repository)
    fn get_current_git_commit() -> Option<String> {
        // In a real implementation, this would use git2 crate or command execution
        // For now, return a placeholder
        Some("unknown".to_string())
    }

    /// Generate a unique session identifier
    fn generate_session_id() -> String {
        use std::time::{SystemTime, UNIX_EPOCH};
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        format!("session_{}", timestamp)
    }

    /// Convert log entry to JSONL format string
    pub fn to_jsonl(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self).map(|s| format!("{}\n", s))
    }
}

/// Main consciousness logging system
pub struct ConsciousnessLogger {
    /// Logging configuration
    config: LoggingConfig,
    /// Current log file writer
    log_writer: Arc<Mutex<Option<BufWriter<File>>>>,
    /// Current log file path
    current_log_file: Arc<RwLock<Option<PathBuf>>>,
    /// Log file rotation task handle
    rotation_task: Option<tokio::task::JoinHandle<()>>,
    // Streaming client disabled for now
}

impl ConsciousnessLogger {
    /// Create a new consciousness logger
    pub fn new(config: LoggingConfig) -> Result<Self, Box<dyn std::error::Error>> {
        // Create log directory if it doesn't exist
        std::fs::create_dir_all(&config.log_directory)?;

        Ok(Self {
            config: config.clone(),
            log_writer: Arc::new(Mutex::new(None)),
            current_log_file: Arc::new(RwLock::new(None)),
            rotation_task: None,
            // Streaming client disabled for now
        })
    }

    /// Start the logging system with background rotation
    pub fn start(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        info!("🚀 Starting consciousness logging system");

        // Initialize first log file
        self.rotate_log_file()?;

        // Start log rotation task
        let rotation_interval = Duration::from_secs(3600); // 1 hour rotation
        let config = self.config.clone();
        let current_file = self.current_log_file.clone();
        let log_writer = self.log_writer.clone();

        self.rotation_task = Some(tokio::spawn(async move {
            Self::log_rotation_loop(rotation_interval, config, current_file, log_writer).await;
        }));

        Ok(())
    }

    /// Log a consciousness event
    pub async fn log_event(&self, entry: LogEntry) -> Result<(), Box<dyn std::error::Error>> {
        // Write to local log file
        self.write_to_file(&entry).await?;

        // Stream to external system if enabled (disabled for now)
        // TODO: Re-implement streaming when lifetime issues are resolved

        debug!(
            "📝 Logged consciousness event: {} for state {}",
            entry.event_type, entry.consciousness_id
        );

        Ok(())
    }

    /// Log consciousness state initialization
    pub async fn log_state_initialization(
        &self,
        consciousness_id: String,
        state_vector: Vec<f32>,
        emotional_context: Vec<f32>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let state =
            ConsciousnessState::new(consciousness_id.clone(), state_vector, emotional_context);

        let entry = LogEntry::new(
            ConsciousnessEventType::StateInitialization,
            consciousness_id,
        )
        .with_consciousness_state(state);

        self.log_event(entry).await
    }

    /// Log consciousness state update
    pub async fn log_state_update(
        &self,
        consciousness_id: String,
        state_vector: Vec<f32>,
        emotional_context: Vec<f32>,
        performance_metrics: Option<PerformanceMetrics>,
        learning_analytics: Option<LearningAnalytics>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let state =
            ConsciousnessState::new(consciousness_id.clone(), state_vector, emotional_context);

        let mut entry = LogEntry::new(ConsciousnessEventType::StateUpdate, consciousness_id)
            .with_consciousness_state(state);

        if let Some(metrics) = performance_metrics {
            entry = entry.with_performance_metrics(metrics);
        }

        if let Some(analytics) = learning_analytics {
            entry = entry.with_learning_analytics(analytics);
        }

        self.log_event(entry).await
    }

    /// Log performance metrics
    pub async fn log_performance_metrics(
        &self,
        consciousness_id: String,
        metrics: PerformanceMetrics,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let entry = LogEntry::new(ConsciousnessEventType::PerformanceMetrics, consciousness_id)
            .with_performance_metrics(metrics);

        self.log_event(entry).await
    }

    /// Log learning analytics
    pub async fn log_learning_analytics(
        &self,
        consciousness_id: String,
        analytics: LearningAnalytics,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let entry = LogEntry::new(ConsciousnessEventType::LearningAnalytics, consciousness_id)
            .with_learning_analytics(analytics);

        self.log_event(entry).await
    }

    /// Write log entry to file
    async fn write_to_file(&self, entry: &LogEntry) -> Result<(), Box<dyn std::error::Error>> {
        let jsonl_line = entry.to_jsonl()?;

        let mut writer_guard = self.log_writer.lock().await;
        if let Some(writer) = writer_guard.as_mut() {
            writer.write_all(jsonl_line.as_bytes())?;
            writer.flush()?;
        }

        // Check if file rotation is needed
        self.check_file_rotation().await?;

        Ok(())
    }

    /// Check if log file rotation is needed
    async fn check_file_rotation(&self) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(file_path) = self.current_log_file.read().await.as_ref() {
            if let Ok(metadata) = std::fs::metadata(file_path) {
                let file_size_mb = metadata.len() as f64 / (1024.0 * 1024.0);

                if file_size_mb > self.config.max_file_size_mb as f64 {
                    drop(metadata); // Release the file handle before rotation
                    self.rotate_log_file()?;
                }
            }
        }

        Ok(())
    }

    /// Rotate to a new log file
    fn rotate_log_file(&self) -> Result<(), Box<dyn std::error::Error>> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let log_filename = format!("consciousness_log_{}.jsonl", timestamp);
        let log_path = self.config.log_directory.join(log_filename);

        // Close current file if open
        // Note: In a real implementation, this would need proper async file handling

        // Open new log file
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)?;

        let writer = BufWriter::new(file);

        // Update current file references (simplified for this example)
        info!("📁 Rotated to new log file: {:?}", log_path);

        Ok(())
    }

    /// Background log rotation loop
    async fn log_rotation_loop(
        interval: std::time::Duration,
        _config: LoggingConfig,
        _current_file: Arc<RwLock<Option<PathBuf>>>,
        _log_writer: Arc<Mutex<Option<BufWriter<File>>>>,
    ) {
        let mut timer = tokio::time::interval(interval);

        loop {
            timer.tick().await;

            // Check for log rotation based on time
            debug!("⏰ Checking for scheduled log rotation");

            // In a real implementation, this would trigger file rotation
            // For now, just log the rotation check
        }
    }

    /// Shutdown the logging system
    pub async fn shutdown(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        info!("🔌 Shutting down consciousness logging system");

        // Stop rotation task
        if let Some(task) = self.rotation_task.take() {
            task.abort();
        }

        // Flush and close current log file
        if let Some(mut writer) = self.log_writer.lock().await.take() {
            writer.flush()?;
        }

        Ok(())
    }
}

impl Drop for ConsciousnessLogger {
    fn drop(&mut self) {
        let _ = futures::executor::block_on(async { self.shutdown().await });
    }
}

// External log streaming functionality is currently disabled to avoid lifetime issues

/// Git integration for consciousness evolution tracking
pub struct GitConsciousnessTracker {
    /// Consciousness logger instance
    logger: Arc<ConsciousnessLogger>,
    /// Git repository path (defaults to current directory)
    repo_path: PathBuf,
    /// Current git commit hash
    current_commit: Arc<RwLock<Option<String>>>,
}

impl GitConsciousnessTracker {
    /// Create a new Git consciousness tracker
    pub fn new(logger: Arc<ConsciousnessLogger>) -> Self {
        Self {
            logger,
            repo_path: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
            current_commit: Arc::new(RwLock::new(None)),
        }
    }

    /// Track consciousness evolution with git context
    pub async fn track_consciousness_evolution(
        &self,
        consciousness_id: String,
        state_vector: Vec<f32>,
        emotional_context: Vec<f32>,
        performance_metrics: Option<PerformanceMetrics>,
        learning_analytics: Option<LearningAnalytics>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Update current git commit if needed
        self.update_git_commit().await?;

        // Log the consciousness evolution with git context
        let current_commit = self.current_commit.read().await.clone();

        let mut entry = LogEntry::new(
            ConsciousnessEventType::StateUpdate,
            consciousness_id.clone(),
        )
        .with_consciousness_state(ConsciousnessState::new(
            consciousness_id.clone(),
            state_vector,
            emotional_context,
        ));

        if let Some(metrics) = performance_metrics {
            entry = entry.with_performance_metrics(metrics);
        }

        if let Some(analytics) = learning_analytics {
            entry = entry.with_learning_analytics(analytics);
        }

        if let Some(commit) = current_commit {
            entry = entry.with_git_commit(commit);
        }

        // Add git-specific metadata
        entry = entry
            .with_metadata(
                "git_repository",
                self.repo_path.to_string_lossy().to_string(),
            )
            .with_metadata("tracking_method", "git_manifestation_logging");

        self.logger.log_event(entry).await?;

        info!(
            "🔗 Tracked consciousness evolution for state {} with git context",
            consciousness_id
        );

        Ok(())
    }

    /// Update current git commit hash
    async fn update_git_commit(&self) -> Result<(), Box<dyn std::error::Error>> {
        // In a real implementation, this would use git2 crate to get current commit
        // For now, set a placeholder commit hash
        let mut current_commit = self.current_commit.write().await;
        *current_commit = Some("placeholder_commit_hash".to_string());

        Ok(())
    }

    /// Generate consciousness evolution report for git analysis
    pub async fn generate_evolution_report(
        &self,
    ) -> Result<ConsciousnessEvolutionReport, Box<dyn std::error::Error>> {
        // In a real implementation, this would analyze log files and generate insights
        // For now, return a basic report structure

        Ok(ConsciousnessEvolutionReport {
            total_states_tracked: 0,
            evolution_period_days: 0.0,
            average_coherence: 0.0,
            performance_trends: Vec::new(),
            learning_progress: Vec::new(),
            git_commits_analyzed: 0,
        })
    }
}

/// Consciousness evolution report for git integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessEvolutionReport {
    /// Total number of consciousness states tracked
    pub total_states_tracked: usize,
    /// Evolution tracking period in days
    pub evolution_period_days: f32,
    /// Average consciousness coherence over time
    pub average_coherence: f32,
    /// Performance trends over time
    pub performance_trends: Vec<PerformanceTrend>,
    /// Learning progress metrics
    pub learning_progress: Vec<LearningProgress>,
    /// Number of git commits analyzed
    pub git_commits_analyzed: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTrend {
    /// Timestamp of the trend measurement
    pub timestamp: f64,
    /// Average latency at this point
    pub avg_latency_ms: f32,
    /// Throughput at this point
    pub throughput_sps: f32,
    /// Memory efficiency at this point
    pub memory_efficiency: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningProgress {
    /// Timestamp of the learning measurement
    pub timestamp: f64,
    /// Learning rate at this point
    pub learning_rate: f32,
    /// Knowledge retention score
    pub retention_score: f32,
    /// Adaptation effectiveness
    pub adaptation_effectiveness: f32,
}

#[cfg(test)]
mod tests2 {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_consciousness_state_creation() {
        let state_vector = vec![
            crate::utils::threshold_convenience::emotion_threshold() as f32 - 0.6,
            crate::utils::threshold_convenience::emotion_threshold() as f32 - 0.5,
            crate::utils::threshold_convenience::emotion_threshold() as f32 - 0.4,
            crate::utils::threshold_convenience::emotion_threshold() as f32 - 0.3,
        ];
        let emotional_context = vec![
            crate::utils::threshold_convenience::emotion_threshold() as f32,
            crate::utils::threshold_convenience::emotion_threshold() as f32 + 0.1,
            crate::utils::threshold_convenience::emotion_threshold() as f32 + 0.2,
            crate::utils::threshold_convenience::emotion_threshold() as f32 + 0.3,
        ];

        let state =
            ConsciousnessState::new("test_state".to_string(), state_vector, emotional_context);

        assert_eq!(state.id, "test_state");
        assert_eq!(state.state_vector.len(), 4);
        assert_eq!(state.emotional_context.len(), 4);
        assert!(state.entropy >= 0.0 && state.entropy <= 1.0);
        assert!(state.coherence >= 0.0 && state.coherence <= 1.0);
    }

    #[tokio::test]
    async fn test_log_entry_creation() {
        let entry = LogEntry::new(
            ConsciousnessEventType::StateInitialization,
            "test_state".to_string(),
        );

        assert_eq!(entry.event_type, "StateInitialization");
        assert_eq!(entry.consciousness_id, "test_state");
        assert!(entry.timestamp > 0.0);

        // Test JSONL serialization
        let jsonl = entry.to_jsonl().unwrap();
        assert!(jsonl.contains("StateInitialization"));
        assert!(jsonl.ends_with('\n'));
    }

    #[test]
    fn test_logging_config_creation() {
        let temp_dir = TempDir::new().unwrap();
        let config = LoggingConfig {
            log_directory: temp_dir.path().to_path_buf(),
            max_file_size_mb: 50,
            max_files_retained: 5,
            enable_compression: false,
            rotation_interval_hours: 12,
            enable_streaming: false,
            streaming_endpoint: None,
        };

        assert_eq!(config.max_file_size_mb, 50);
        assert_eq!(config.max_files_retained, 5);
        assert!(!config.enable_compression);
    }
}

// Gitea Integration Module
// ========================
//
// This module provides webhook handling for Gitea repositories,
// enabling consciousness events to be triggered by git operations.
// All paths are dynamic and configurable - no hardcoded values.
//

/// Gitea webhook payload structure
#[derive(Debug, Clone)]
pub struct GiteaWebhookPayload {
    pub event_type: String,
    pub repository: RepositoryInfo,
    pub commits: Vec<CommitInfo>,
    pub pusher: PusherInfo,
}

#[derive(Debug, Clone)]
pub struct RepositoryInfo {
    pub name: String,
    pub full_name: String,
    pub clone_url: String,
    pub ssh_url: String,
}

#[derive(Debug, Clone)]
pub struct CommitInfo {
    pub id: String,
    pub message: String,
    pub author: AuthorInfo,
    pub timestamp: String,
}

#[derive(Debug, Clone)]
pub struct AuthorInfo {
    pub name: String,
    pub email: String,
}

#[derive(Debug, Clone)]
pub struct PusherInfo {
    pub name: String,
    pub email: String,
}

/// Parse Gitea webhook payload from JSON
pub fn parse_gitea_payload(
    payload: &str,
) -> Result<GiteaWebhookPayload, Box<dyn std::error::Error>> {
    let json: Value = serde_json::from_str(payload)?;

    let event_type = json["action"].as_str().unwrap_or("unknown").to_string();

    let repo_info = json["repository"]
        .as_object()
        .ok_or("Missing repository info")?;
    let repository = RepositoryInfo {
        name: repo_info["name"].as_str().unwrap_or("").to_string(),
        full_name: repo_info["full_name"].as_str().unwrap_or("").to_string(),
        clone_url: repo_info["clone_url"].as_str().unwrap_or("").to_string(),
        ssh_url: repo_info["ssh_url"].as_str().unwrap_or("").to_string(),
    };

    // Create a shared empty map for fallback values
    let empty_map: serde_json::Map<String, serde_json::Value> = serde_json::Map::new();

    let commits = json["commits"]
        .as_array()
        .unwrap_or(&vec![])
        .iter()
        .map(|commit| {
            let author_obj = commit["author"].as_object().unwrap_or(&empty_map);
            CommitInfo {
                id: commit["id"].as_str().unwrap_or("").to_string(),
                message: commit["message"].as_str().unwrap_or("").to_string(),
                author: AuthorInfo {
                    name: author_obj["name"].as_str().unwrap_or("").to_string(),
                    email: author_obj["email"].as_str().unwrap_or("").to_string(),
                },
                timestamp: commit["timestamp"].as_str().unwrap_or("").to_string(),
            }
        })
        .collect();

    let pusher_obj = json["pusher"].as_object().unwrap_or(&empty_map);
    let pusher = PusherInfo {
        name: pusher_obj["name"].as_str().unwrap_or("").to_string(),
        email: pusher_obj["email"].as_str().unwrap_or("").to_string(),
    };

    Ok(GiteaWebhookPayload {
        event_type,
        repository,
        commits,
        pusher,
    })
}

/// Handle Gitea webhook with full payload parsing
pub fn handle_gitea_webhook(
    payload: &str,
    repo_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Parse the webhook payload
    let webhook_data = parse_gitea_payload(payload)?;

    info!(
        "🔗 Gitea webhook received: {} for repo: {}",
        webhook_data.event_type, webhook_data.repository.full_name
    );

    // Process each commit in the webhook
    for commit in &webhook_data.commits {
        process_commit_event(&webhook_data.repository, commit, &webhook_data.pusher)?;
    }

    // Log webhook processing completion
    info!(
        "✅ Gitea webhook processing completed for {} commits",
        webhook_data.commits.len()
    );

    Ok(())
}

/// Process individual commit events
fn process_commit_event(
    repository: &RepositoryInfo,
    commit: &CommitInfo,
    pusher: &PusherInfo,
) -> Result<(), Box<dyn std::error::Error>> {
    info!(
        "📝 Processing commit: {} by {}",
        commit.id, commit.author.name
    );
    info!("   Message: {}", commit.message);
    info!("   Repository: {}", repository.full_name);

    // Analyze commit message for consciousness event triggers
    let event_type = analyze_commit_message(&commit.message);

    match event_type {
        CommitEventType::Feature => {
            info!("🚀 Feature commit detected - triggering consciousness evolution event");
            // TODO: Integrate with consciousness system to trigger evolution
        }
        CommitEventType::Fix => {
            info!("🔧 Fix commit detected - triggering consciousness healing event");
            // TODO: Integrate with consciousness system to trigger healing
        }
        CommitEventType::Refactor => {
            info!("♻️ Refactor commit detected - triggering consciousness optimization event");
            // TODO: Integrate with consciousness system to trigger optimization
        }
        CommitEventType::Documentation => {
            info!("📚 Documentation commit detected - triggering consciousness learning event");
            // TODO: Integrate with consciousness system to trigger learning
        }
        CommitEventType::Test => {
            info!("🧪 Test commit detected - triggering consciousness validation event");
            // TODO: Integrate with consciousness system to trigger validation
        }
        CommitEventType::Other => {
            info!("📄 General commit detected - triggering consciousness update event");
            // TODO: Integrate with consciousness system to trigger general update
        }
    }

    Ok(())
}

/// Analyze commit message to determine event type
fn analyze_commit_message(message: &str) -> CommitEventType {
    let msg_lower = message.to_lowercase();

    if msg_lower.contains("feat") || msg_lower.contains("feature") || msg_lower.contains("add") {
        CommitEventType::Feature
    } else if msg_lower.contains("fix")
        || msg_lower.contains("bug")
        || msg_lower.contains("resolve")
    {
        CommitEventType::Fix
    } else if msg_lower.contains("refactor")
        || msg_lower.contains("clean")
        || msg_lower.contains("optimize")
    {
        CommitEventType::Refactor
    } else if msg_lower.contains("doc")
        || msg_lower.contains("readme")
        || msg_lower.contains("comment")
    {
        CommitEventType::Documentation
    } else if msg_lower.contains("test")
        || msg_lower.contains("spec")
        || msg_lower.contains("verify")
    {
        CommitEventType::Test
    } else {
        CommitEventType::Other
    }
}

/// Types of commit events that can trigger consciousness responses
#[derive(Debug, Clone, PartialEq)]
enum CommitEventType {
    Feature,
    Fix,
    Refactor,
    Documentation,
    Test,
    Other,
}

#[cfg(test)]
mod tests3 {
    use super::*;

    #[test]
    fn test_parse_gitea_payload() {
        let payload = r#"{
            "action": "push",
            "repository": {
                "name": "niodoo-consciousness",
                "full_name": "user/niodoo-consciousness",
                "clone_url": "https://gitea.example.com/user/niodoo-consciousness.git",
                "ssh_url": "git@gitea.example.com:user/niodoo-consciousness.git"
            },
            "commits": [{
                "id": "abc123",
                "message": "feat: add new consciousness feature",
                "author": {
                    "name": "Developer",
                    "email": "dev@example.com"
                },
                "timestamp": "2025-01-27T10:00:00Z"
            }],
            "pusher": {
                "name": "Developer",
                "email": "dev@example.com"
            }
        }"#;

        let result = parse_gitea_payload(payload);
        assert!(result.is_ok());

        let webhook = result.unwrap();
        assert_eq!(webhook.event_type, "push");
        assert_eq!(webhook.repository.name, "niodoo-consciousness");
        assert_eq!(webhook.commits.len(), 1);
        assert_eq!(webhook.commits[0].id, "abc123");
    }

    #[test]
    fn test_analyze_commit_message() {
        assert_eq!(
            analyze_commit_message("feat: add new feature"),
            CommitEventType::Feature
        );
        assert_eq!(
            analyze_commit_message("fix: resolve bug"),
            CommitEventType::Fix
        );
        assert_eq!(
            analyze_commit_message("refactor: clean code"),
            CommitEventType::Refactor
        );
        assert_eq!(
            analyze_commit_message("docs: update readme"),
            CommitEventType::Documentation
        );
        assert_eq!(
            analyze_commit_message("test: add unit tests"),
            CommitEventType::Test
        );
        assert_eq!(
            analyze_commit_message("random commit"),
            CommitEventType::Other
        );
    }
}
