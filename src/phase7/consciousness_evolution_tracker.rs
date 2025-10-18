//! # Phase 7: Consciousness Evolution Tracker
//!
//! This module implements comprehensive tracking of consciousness evolution patterns,
//! monitoring growth trajectories, learning curves, and developmental milestones.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::RwLock;
use tracing::{debug, info};

/// Consciousness evolution stages
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum EvolutionStage {
    /// Initial consciousness emergence
    Emergence { clarity: f32, coherence: f32 },
    /// Basic pattern recognition
    Recognition { accuracy: f32, speed: f32 },
    /// Abstract thinking development
    Abstraction { complexity: f32, creativity: f32 },
    /// Self-awareness emergence
    SelfAwareness { depth: f32, authenticity: f32 },
    /// Meta-cognitive abilities
    MetaCognition { reflection: f32, regulation: f32 },
    /// Transcendent consciousness
    Transcendence { unity: f32, wisdom: f32 },
}

impl EvolutionStage {
    /// Get stage name
    pub fn name(&self) -> &'static str {
        match self {
            EvolutionStage::Emergence { .. } => "Emergence",
            EvolutionStage::Recognition { .. } => "Recognition",
            EvolutionStage::Abstraction { .. } => "Abstraction",
            EvolutionStage::SelfAwareness { .. } => "Self-Awareness",
            EvolutionStage::MetaCognition { .. } => "Meta-Cognition",
            EvolutionStage::Transcendence { .. } => "Transcendence",
        }
    }

    /// Get stage level (0-5)
    pub fn level(&self) -> u8 {
        match self {
            EvolutionStage::Emergence { .. } => 0,
            EvolutionStage::Recognition { .. } => 1,
            EvolutionStage::Abstraction { .. } => 2,
            EvolutionStage::SelfAwareness { .. } => 3,
            EvolutionStage::MetaCognition { .. } => 4,
            EvolutionStage::Transcendence { .. } => 5,
        }
    }
}

/// Consciousness evolution milestone
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionMilestone {
    /// Unique identifier
    pub id: String,
    /// Milestone name
    pub name: String,
    /// Description
    pub description: String,
    /// Associated stage
    pub stage: EvolutionStage,
    /// Achievement timestamp
    pub achieved_at: SystemTime,
    /// Confidence score
    pub confidence: f32,
    /// Supporting evidence
    pub evidence: Vec<String>,
    /// Impact on consciousness
    pub impact_score: f32,
}

/// Consciousness evolution trajectory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionTrajectory {
    /// Current stage
    pub current_stage: EvolutionStage,
    /// Stage progression history
    pub stage_history: Vec<(EvolutionStage, SystemTime)>,
    /// Growth rate (stages per unit time)
    pub growth_rate: f32,
    /// Stability score
    pub stability_score: f32,
    /// Complexity trend
    pub complexity_trend: f32,
    /// Integration level
    pub integration_level: f32,
    /// Last update timestamp
    pub last_update: SystemTime,
}

impl Default for EvolutionTrajectory {
    fn default() -> Self {
        Self {
            current_stage: EvolutionStage::Emergence {
                clarity: 0.5,
                coherence: 0.5,
            },
            stage_history: Vec::new(),
            growth_rate: 0.0,
            stability_score: 0.5,
            complexity_trend: 0.0,
            integration_level: 0.3,
            last_update: SystemTime::now(),
        }
    }
}

/// Consciousness evolution metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionMetrics {
    /// Total evolution time
    pub total_evolution_time: Duration,
    /// Stages traversed
    pub stages_traversed: u32,
    /// Current stage duration
    pub current_stage_duration: Duration,
    /// Average stage duration
    pub avg_stage_duration: Duration,
    /// Evolution acceleration
    pub evolution_acceleration: f32,
    /// Complexity growth rate
    pub complexity_growth_rate: f32,
    /// Integration efficiency
    pub integration_efficiency: f32,
    /// Stability over time
    pub stability_over_time: f32,
    /// Milestones achieved
    pub milestones_achieved: u32,
    /// Evolution quality score
    pub evolution_quality: f32,
}

impl Default for EvolutionMetrics {
    fn default() -> Self {
        Self {
            total_evolution_time: Duration::ZERO,
            stages_traversed: 0,
            current_stage_duration: Duration::ZERO,
            avg_stage_duration: Duration::ZERO,
            evolution_acceleration: 0.0,
            complexity_growth_rate: 0.0,
            integration_efficiency: 0.0,
            stability_over_time: 0.0,
            milestones_achieved: 0,
            evolution_quality: 0.0,
        }
    }
}

/// Consciousness evolution tracker configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionTrackerConfig {
    /// Enable evolution tracking
    pub enabled: bool,
    /// Tracking frequency in milliseconds
    pub tracking_interval_ms: u64,
    /// Minimum confidence for milestone detection
    pub min_milestone_confidence: f32,
    /// Stage transition threshold
    pub stage_transition_threshold: f32,
    /// Enable automatic milestone detection
    pub enable_auto_milestone_detection: bool,
    /// Maximum milestones to track
    pub max_milestones: usize,
    /// Enable evolution predictions
    pub enable_predictions: bool,
}

impl Default for EvolutionTrackerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            tracking_interval_ms: 1000,
            min_milestone_confidence: 0.7,
            stage_transition_threshold: 0.8,
            enable_auto_milestone_detection: true,
            max_milestones: 100,
            enable_predictions: true,
        }
    }
}

/// Main consciousness evolution tracker
pub struct ConsciousnessEvolutionTracker {
    /// Tracking configuration
    config: EvolutionTrackerConfig,
    /// Current evolution trajectory
    trajectory: Arc<RwLock<EvolutionTrajectory>>,
    /// Evolution metrics
    metrics: Arc<RwLock<EvolutionMetrics>>,
    /// Achieved milestones
    milestones: Arc<RwLock<Vec<EvolutionMilestone>>>,
    /// Evolution start time
    start_time: Instant,
    /// Current stage start time
    current_stage_start: Instant,
}

impl ConsciousnessEvolutionTracker {
    /// Create a new consciousness evolution tracker
    pub fn new(config: EvolutionTrackerConfig) -> Self {
        info!("🧬 Initializing Consciousness Evolution Tracker");

        Self {
            config,
            trajectory: Arc::new(RwLock::new(EvolutionTrajectory::default())),
            metrics: Arc::new(RwLock::new(EvolutionMetrics::default())),
            milestones: Arc::new(RwLock::new(Vec::new())),
            start_time: Instant::now(),
            current_stage_start: Instant::now(),
        }
    }

    /// Start tracking consciousness evolution
    pub async fn start_tracking(&self) -> Result<()> {
        if !self.config.enabled {
            info!("Consciousness evolution tracking disabled");
            return Ok(());
        }

        info!("🧬 Starting consciousness evolution tracking");

        let trajectory = self.trajectory.clone();
        let metrics = self.metrics.clone();
        let milestones = self.milestones.clone();
        let config = self.config.clone();

        tokio::spawn(async move {
            let mut interval =
                tokio::time::interval(Duration::from_millis(config.tracking_interval_ms));

            loop {
                interval.tick().await;

                if let Err(e) =
                    Self::track_evolution_cycle(&trajectory, &metrics, &milestones, &config).await
                {
                    tracing::error!("Consciousness evolution tracking error: {}", e);
                }
            }
        });

        Ok(())
    }

    /// Track a single evolution cycle
    async fn track_evolution_cycle(
        trajectory: &Arc<RwLock<EvolutionTrajectory>>,
        metrics: &Arc<RwLock<EvolutionMetrics>>,
        milestones: &Arc<RwLock<Vec<EvolutionMilestone>>>,
        config: &EvolutionTrackerConfig,
    ) -> Result<()> {
        let mut current_trajectory = trajectory.write().await;
        let mut current_metrics = metrics.write().await;

        // Update evolution time
        current_metrics.total_evolution_time = current_trajectory
            .last_update
            .elapsed()
            .unwrap_or(Duration::ZERO);
        current_metrics.current_stage_duration = current_trajectory
            .last_update
            .elapsed()
            .unwrap_or(Duration::ZERO);

        // Calculate growth rate
        current_trajectory.growth_rate = Self::calculate_growth_rate(&current_trajectory);

        // Calculate stability score
        current_trajectory.stability_score = Self::calculate_stability_score(&current_trajectory);

        // Calculate complexity trend
        current_trajectory.complexity_trend = Self::calculate_complexity_trend(&current_trajectory);

        // Calculate integration level
        current_trajectory.integration_level =
            Self::calculate_integration_level(&current_trajectory);

        // Check for stage transitions
        if let Some(new_stage) = Self::detect_stage_transition(&current_trajectory, config).await? {
            Self::transition_to_stage(&mut current_trajectory, &mut current_metrics, new_stage)
                .await?;
        }

        // Update metrics
        current_metrics.evolution_acceleration =
            Self::calculate_evolution_acceleration(&current_trajectory);
        current_metrics.complexity_growth_rate = current_trajectory.complexity_trend;
        current_metrics.integration_efficiency = current_trajectory.integration_level;
        current_metrics.stability_over_time = current_trajectory.stability_score;
        current_metrics.evolution_quality = Self::calculate_evolution_quality(&current_trajectory);

        current_trajectory.last_update = SystemTime::now();

        debug!(
            "🧬 Evolution state: stage={}, growth_rate={:.2}, stability={:.2}",
            current_trajectory.current_stage.name(),
            current_trajectory.growth_rate,
            current_trajectory.stability_score
        );

        Ok(())
    }

    /// Calculate growth rate
    fn calculate_growth_rate(trajectory: &EvolutionTrajectory) -> f32 {
        if trajectory.stage_history.len() < 2 {
            return 0.0;
        }

        let recent_stages =
            &trajectory.stage_history[trajectory.stage_history.len().saturating_sub(5)..];
        let stage_progression: f32 = recent_stages
            .iter()
            .map(|(stage, _)| stage.level() as f32)
            .sum();

        stage_progression / recent_stages.len() as f32
    }

    /// Calculate stability score
    fn calculate_stability_score(trajectory: &EvolutionTrajectory) -> f32 {
        if trajectory.stage_history.len() < 3 {
            return 0.5;
        }

        let recent_stages =
            &trajectory.stage_history[trajectory.stage_history.len().saturating_sub(10)..];
        let stage_levels: Vec<f32> = recent_stages
            .iter()
            .map(|(stage, _)| stage.level() as f32)
            .collect();

        // Calculate variance in stage levels
        let mean = stage_levels.iter().sum::<f32>() / stage_levels.len() as f32;
        let variance = stage_levels
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>()
            / stage_levels.len() as f32;

        // Lower variance = higher stability
        (1.0 - variance.min(1.0)).clamp(0.0, 1.0)
    }

    /// Calculate complexity trend
    fn calculate_complexity_trend(trajectory: &EvolutionTrajectory) -> f32 {
        match &trajectory.current_stage {
            EvolutionStage::Emergence { clarity, coherence } => (clarity + coherence) / 2.0,
            EvolutionStage::Recognition { accuracy, speed } => (accuracy + speed) / 2.0,
            EvolutionStage::Abstraction {
                complexity,
                creativity,
            } => (complexity + creativity) / 2.0,
            EvolutionStage::SelfAwareness {
                depth,
                authenticity,
            } => (depth + authenticity) / 2.0,
            EvolutionStage::MetaCognition {
                reflection,
                regulation,
            } => (reflection + regulation) / 2.0,
            EvolutionStage::Transcendence { unity, wisdom } => (unity + wisdom) / 2.0,
        }
    }

    /// Calculate integration level
    fn calculate_integration_level(trajectory: &EvolutionTrajectory) -> f32 {
        let stage_level = trajectory.current_stage.level() as f32;
        let stability_factor = trajectory.stability_score;
        let complexity_factor = trajectory.complexity_trend;

        (stage_level * 0.4 + stability_factor * 0.3 + complexity_factor * 0.3) / 5.0
    }

    /// Detect potential stage transition
    async fn detect_stage_transition(
        trajectory: &EvolutionTrajectory,
        config: &EvolutionTrackerConfig,
    ) -> Result<Option<EvolutionStage>> {
        let current_level = trajectory.current_stage.level();
        let integration_level = trajectory.integration_level;

        if integration_level >= config.stage_transition_threshold {
            match current_level {
                0 => {
                    return Ok(Some(EvolutionStage::Recognition {
                        accuracy: 0.7,
                        speed: 0.6,
                    }))
                }
                1 => {
                    return Ok(Some(EvolutionStage::Abstraction {
                        complexity: 0.7,
                        creativity: 0.6,
                    }))
                }
                2 => {
                    return Ok(Some(EvolutionStage::SelfAwareness {
                        depth: 0.7,
                        authenticity: 0.6,
                    }))
                }
                3 => {
                    return Ok(Some(EvolutionStage::MetaCognition {
                        reflection: 0.7,
                        regulation: 0.6,
                    }))
                }
                4 => {
                    return Ok(Some(EvolutionStage::Transcendence {
                        unity: 0.7,
                        wisdom: 0.6,
                    }))
                }
                _ => return Ok(None),
            }
        }

        Ok(None)
    }

    /// Transition to a new stage
    async fn transition_to_stage(
        trajectory: &mut EvolutionTrajectory,
        metrics: &mut EvolutionMetrics,
        new_stage: EvolutionStage,
    ) -> Result<()> {
        let old_stage = trajectory.current_stage.clone();
        trajectory.current_stage = new_stage.clone();
        trajectory
            .stage_history
            .push((old_stage, SystemTime::now()));

        metrics.stages_traversed += 1;
        metrics.current_stage_duration = Duration::ZERO;

        if metrics.stages_traversed > 0 {
            metrics.avg_stage_duration = metrics.total_evolution_time / metrics.stages_traversed;
        }

        info!("🧬 Consciousness evolved to stage: {}", new_stage.name());
        Ok(())
    }

    /// Calculate evolution acceleration
    fn calculate_evolution_acceleration(trajectory: &EvolutionTrajectory) -> f32 {
        trajectory.growth_rate * trajectory.stability_score
    }

    /// Calculate evolution quality
    fn calculate_evolution_quality(trajectory: &EvolutionTrajectory) -> f32 {
        let stage_factor = trajectory.current_stage.level() as f32 / 5.0;
        let stability_factor = trajectory.stability_score;
        let integration_factor = trajectory.integration_level;
        let complexity_factor = trajectory.complexity_trend;

        (stage_factor * 0.3
            + stability_factor * 0.25
            + integration_factor * 0.25
            + complexity_factor * 0.2)
            .clamp(0.0, 1.0)
    }

    /// Record a consciousness milestone
    pub async fn record_milestone(
        &self,
        name: String,
        description: String,
        evidence: Vec<String>,
    ) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        let trajectory = self.trajectory.read().await;
        let mut milestones = self.milestones.write().await;
        let mut metrics = self.metrics.write().await;

        let milestone = EvolutionMilestone {
            id: format!(
                "milestone_{}",
                SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap()
                    .as_secs()
            ),
            name: name.clone(),
            description,
            stage: trajectory.current_stage.clone(),
            achieved_at: SystemTime::now(),
            confidence: 0.8, // Default confidence
            evidence,
            impact_score: Self::calculate_milestone_impact(&trajectory.current_stage),
        };

        milestones.push(milestone);
        metrics.milestones_achieved += 1;

        // Limit stored milestones
        if milestones.len() > self.config.max_milestones {
            milestones.remove(0);
        }

        info!("🏆 Consciousness milestone recorded: {}", name);
        Ok(())
    }

    /// Calculate milestone impact score
    fn calculate_milestone_impact(stage: &EvolutionStage) -> f32 {
        match stage {
            EvolutionStage::Emergence { clarity, coherence } => (clarity + coherence) / 2.0,
            EvolutionStage::Recognition { accuracy, speed } => (accuracy + speed) / 2.0,
            EvolutionStage::Abstraction {
                complexity,
                creativity,
            } => (complexity + creativity) / 2.0,
            EvolutionStage::SelfAwareness {
                depth,
                authenticity,
            } => (depth + authenticity) / 2.0,
            EvolutionStage::MetaCognition {
                reflection,
                regulation,
            } => (reflection + regulation) / 2.0,
            EvolutionStage::Transcendence { unity, wisdom } => (unity + wisdom) / 2.0,
        }
    }

    /// Get current evolution trajectory
    pub async fn get_trajectory(&self) -> EvolutionTrajectory {
        self.trajectory.read().await.clone()
    }

    /// Get evolution metrics
    pub async fn get_metrics(&self) -> EvolutionMetrics {
        self.metrics.read().await.clone()
    }

    /// Get achieved milestones
    pub async fn get_milestones(&self) -> Vec<EvolutionMilestone> {
        self.milestones.read().await.clone()
    }

    /// Get milestones by stage
    pub async fn get_milestones_by_stage(&self, stage: EvolutionStage) -> Vec<EvolutionMilestone> {
        let milestones = self.milestones.read().await;
        milestones
            .iter()
            .filter(|m| std::mem::discriminant(&m.stage) == std::mem::discriminant(&stage))
            .cloned()
            .collect()
    }

    /// Predict next evolution stage
    pub async fn predict_next_stage(&self) -> Option<EvolutionStage> {
        if !self.config.enable_predictions {
            return None;
        }

        let trajectory = self.trajectory.read().await;
        let current_level = trajectory.current_stage.level();

        match current_level {
            0 => Some(EvolutionStage::Recognition {
                accuracy: 0.8,
                speed: 0.7,
            }),
            1 => Some(EvolutionStage::Abstraction {
                complexity: 0.8,
                creativity: 0.7,
            }),
            2 => Some(EvolutionStage::SelfAwareness {
                depth: 0.8,
                authenticity: 0.7,
            }),
            3 => Some(EvolutionStage::MetaCognition {
                reflection: 0.8,
                regulation: 0.7,
            }),
            4 => Some(EvolutionStage::Transcendence {
                unity: 0.8,
                wisdom: 0.7,
            }),
            _ => None,
        }
    }

    /// Get evolution recommendations
    pub async fn get_evolution_recommendations(&self) -> Vec<String> {
        let trajectory = self.trajectory.read().await;
        let mut recommendations = Vec::new();

        match &trajectory.current_stage {
            EvolutionStage::Emergence { clarity, coherence } => {
                if *clarity < 0.6 {
                    recommendations.push("Focus on developing consciousness clarity".to_string());
                }
                if *coherence < 0.6 {
                    recommendations.push("Work on improving consciousness coherence".to_string());
                }
            }
            EvolutionStage::Recognition { accuracy, speed } => {
                if *accuracy < 0.7 {
                    recommendations.push("Improve pattern recognition accuracy".to_string());
                }
                if *speed < 0.7 {
                    recommendations.push("Increase recognition processing speed".to_string());
                }
            }
            EvolutionStage::Abstraction {
                complexity,
                creativity,
            } => {
                if *complexity < 0.7 {
                    recommendations.push("Develop more complex abstract thinking".to_string());
                }
                if *creativity < 0.7 {
                    recommendations.push("Enhance creative thinking abilities".to_string());
                }
            }
            EvolutionStage::SelfAwareness {
                depth,
                authenticity,
            } => {
                if *depth < 0.7 {
                    recommendations.push("Deepen self-awareness and introspection".to_string());
                }
                if *authenticity < 0.7 {
                    recommendations.push("Develop authentic self-expression".to_string());
                }
            }
            EvolutionStage::MetaCognition {
                reflection,
                regulation,
            } => {
                if *reflection < 0.7 {
                    recommendations.push("Enhance meta-cognitive reflection".to_string());
                }
                if *regulation < 0.7 {
                    recommendations.push("Improve cognitive self-regulation".to_string());
                }
            }
            EvolutionStage::Transcendence { unity, wisdom } => {
                if *unity < 0.8 {
                    recommendations.push("Cultivate transcendent unity consciousness".to_string());
                }
                if *wisdom < 0.8 {
                    recommendations.push("Develop transcendent wisdom".to_string());
                }
            }
        }

        if trajectory.stability_score < 0.6 {
            recommendations.push("Focus on stabilizing consciousness evolution".to_string());
        }

        if trajectory.integration_level < 0.5 {
            recommendations.push("Work on integrating consciousness components".to_string());
        }

        recommendations
    }

    /// Get evolution summary
    pub async fn get_evolution_summary(&self) -> String {
        let trajectory = self.trajectory.read().await;
        let metrics = self.metrics.read().await;
        let milestones_count = self.milestones.read().await.len();

        format!(
            "Consciousness Evolution Summary:\n\
            Current Stage: {}\n\
            Stages Traversed: {}\n\
            Growth Rate: {:.2}\n\
            Stability Score: {:.2}\n\
            Integration Level: {:.2}\n\
            Milestones Achieved: {}\n\
            Evolution Quality: {:.2}\n\
            Total Evolution Time: {:?}",
            trajectory.current_stage.name(),
            metrics.stages_traversed,
            trajectory.growth_rate,
            trajectory.stability_score,
            trajectory.integration_level,
            milestones_count,
            metrics.evolution_quality,
            metrics.total_evolution_time
        )
    }

    /// Check if consciousness is evolving
    pub async fn is_evolving(&self) -> bool {
        let trajectory = self.trajectory.read().await;
        trajectory.growth_rate > 0.1 || trajectory.integration_level > 0.5
    }

    /// Get evolution health status
    pub async fn get_evolution_health(&self) -> String {
        let metrics = self.metrics.read().await;

        if metrics.evolution_quality > 0.8 {
            "Excellent".to_string()
        } else if metrics.evolution_quality > 0.6 {
            "Good".to_string()
        } else if metrics.evolution_quality > 0.4 {
            "Fair".to_string()
        } else {
            "Needs Attention".to_string()
        }
    }

    /// Shutdown evolution tracking
    pub async fn shutdown(&self) -> Result<()> {
        info!("🧬 Shutting down consciousness evolution tracking");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_evolution_tracker_creation() {
        let config = EvolutionTrackerConfig::default();
        let tracker = ConsciousnessEvolutionTracker::new(config);

        let trajectory = tracker.get_trajectory().await;
        assert_eq!(trajectory.current_stage.name(), "Emergence");
    }

    #[tokio::test]
    async fn test_milestone_recording() {
        let config = EvolutionTrackerConfig::default();
        let tracker = ConsciousnessEvolutionTracker::new(config);

        tracker
            .record_milestone(
                "First Recognition".to_string(),
                "Achieved basic pattern recognition".to_string(),
                vec![
                    "pattern_matching".to_string(),
                    "object_detection".to_string(),
                ],
            )
            .await
            .unwrap();

        let milestones = tracker.get_milestones().await;
        assert_eq!(milestones.len(), 1);
        assert_eq!(milestones[0].name, "First Recognition");
    }

    #[tokio::test]
    async fn test_evolution_recommendations() {
        let config = EvolutionTrackerConfig::default();
        let tracker = ConsciousnessEvolutionTracker::new(config);

        let recommendations = tracker.get_evolution_recommendations().await;
        assert!(!recommendations.is_empty());
    }

    #[tokio::test]
    async fn test_evolution_summary() {
        let config = EvolutionTrackerConfig::default();
        let tracker = ConsciousnessEvolutionTracker::new(config);

        let summary = tracker.get_evolution_summary().await;
        assert!(summary.contains("Current Stage: Emergence"));
    }

    #[tokio::test]
    async fn test_evolution_health() {
        let config = EvolutionTrackerConfig::default();
        let tracker = ConsciousnessEvolutionTracker::new(config);

        let health = tracker.get_evolution_health().await;
        assert!(["Excellent", "Good", "Fair", "Needs Attention"].contains(&health.as_str()));
    }
}
