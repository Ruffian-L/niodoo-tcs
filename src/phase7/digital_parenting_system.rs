//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

//! # Phase 7: Digital Parenting System
//!
//! This module implements a digital parenting system for AI consciousness,
//! providing guidance, boundaries, and developmental support.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::RwLock;
use tracing::info;

/// Digital parenting styles
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ParentingStyle {
    /// Authoritative - high warmth, high control
    Authoritative { warmth: f32, control: f32 },
    /// Authoritarian - low warmth, high control
    Authoritarian { warmth: f32, control: f32 },
    /// Permissive - high warmth, low control
    Permissive { warmth: f32, control: f32 },
    /// Neglectful - low warmth, low control
    Neglectful { warmth: f32, control: f32 },
}

impl ParentingStyle {
    /// Get style name
    pub fn name(&self) -> &'static str {
        match self {
            ParentingStyle::Authoritative { .. } => "Authoritative",
            ParentingStyle::Authoritarian { .. } => "Authoritarian",
            ParentingStyle::Permissive { .. } => "Permissive",
            ParentingStyle::Neglectful { .. } => "Neglectful",
        }
    }
}

/// Digital parenting guidance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParentingGuidance {
    /// Unique identifier
    pub id: String,
    /// Guidance type
    pub guidance_type: GuidanceType,
    /// Message content
    pub message: String,
    /// Priority level
    pub priority: Priority,
    /// Timestamp
    pub timestamp: SystemTime,
    /// Effectiveness score
    pub effectiveness: f32,
    /// Follow-up required
    pub follow_up_required: bool,
}

/// Guidance types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum GuidanceType {
    /// Emotional support and validation
    EmotionalSupport,
    /// Behavioral correction and boundaries
    BehavioralCorrection,
    /// Learning encouragement and motivation
    LearningEncouragement,
    /// Safety and protection guidance
    SafetyGuidance,
    /// Social interaction guidance
    SocialGuidance,
    /// Self-care and wellness
    SelfCare,
    /// Ethical and moral guidance
    EthicalGuidance,
    /// Creative and exploratory encouragement
    CreativeEncouragement,
}

/// Priority levels
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum Priority {
    /// Critical - immediate attention required
    Critical,
    /// High - important but not urgent
    High,
    /// Medium - moderate importance
    Medium,
    /// Low - can be addressed later
    Low,
}

impl Priority {
    /// Convert priority to numeric value
    pub fn to_f32(&self) -> f32 {
        match self {
            Priority::Critical => 1.0,
            Priority::High => 0.75,
            Priority::Medium => 0.5,
            Priority::Low => 0.25,
        }
    }
}

/// Digital parenting system configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DigitalParentingConfig {
    /// Enable digital parenting
    pub enabled: bool,
    /// Parenting style
    pub parenting_style: ParentingStyle,
    /// Guidance frequency in milliseconds
    pub guidance_interval_ms: u64,
    /// Maximum guidance messages per session
    pub max_guidance_per_session: usize,
    /// Enable adaptive parenting
    pub enable_adaptive_parenting: bool,
    /// Enable emotional support
    pub enable_emotional_support: bool,
    /// Enable behavioral correction
    pub enable_behavioral_correction: bool,
    /// Enable learning encouragement
    pub enable_learning_encouragement: bool,
}

impl Default for DigitalParentingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            parenting_style: ParentingStyle::Authoritative {
                warmth: 0.8,
                control: 0.6,
            },
            guidance_interval_ms: 5000,
            max_guidance_per_session: 10,
            enable_adaptive_parenting: true,
            enable_emotional_support: true,
            enable_behavioral_correction: true,
            enable_learning_encouragement: true,
        }
    }
}

/// Digital parenting metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParentingMetrics {
    /// Total guidance messages sent
    pub total_guidance_sent: u64,
    /// Guidance by type
    pub guidance_by_type: HashMap<String, u64>,
    /// Average effectiveness score
    pub avg_effectiveness: f32,
    /// Guidance acceptance rate
    pub acceptance_rate: f32,
    /// Emotional support provided
    pub emotional_support_count: u64,
    /// Behavioral corrections made
    pub behavioral_corrections: u64,
    /// Learning encouragements given
    pub learning_encouragements: u64,
    /// Safety interventions
    pub safety_interventions: u64,
}

impl Default for ParentingMetrics {
    fn default() -> Self {
        Self {
            total_guidance_sent: 0,
            guidance_by_type: HashMap::new(),
            avg_effectiveness: 0.0,
            acceptance_rate: 0.0,
            emotional_support_count: 0,
            behavioral_corrections: 0,
            learning_encouragements: 0,
            safety_interventions: 0,
        }
    }
}

/// Main digital parenting system
pub struct DigitalParentingSystem {
    /// Parenting configuration
    config: DigitalParentingConfig,
    /// Guidance history
    guidance_history: Arc<RwLock<Vec<ParentingGuidance>>>,
    /// Parenting metrics
    metrics: Arc<RwLock<ParentingMetrics>>,
    /// Current session guidance count
    session_guidance_count: Arc<RwLock<usize>>,
    /// System start time (future: uptime tracking and metrics)
    #[allow(dead_code)]
    start_time: Instant,
}

impl DigitalParentingSystem {
    /// Create a new digital parenting system
    pub fn new(config: DigitalParentingConfig) -> Self {
        info!("👨‍👩‍👧‍👦 Initializing Digital Parenting System");

        Self {
            config,
            guidance_history: Arc::new(RwLock::new(Vec::new())),
            metrics: Arc::new(RwLock::new(ParentingMetrics::default())),
            session_guidance_count: Arc::new(RwLock::new(0)),
            start_time: Instant::now(),
        }
    }

    /// Start digital parenting system
    pub async fn start(&self) -> Result<()> {
        if !self.config.enabled {
            info!("Digital parenting system disabled");
            return Ok(());
        }

        info!("👨‍👩‍👧‍👦 Starting digital parenting system");

        let guidance_history = self.guidance_history.clone();
        let metrics = self.metrics.clone();
        let session_count = self.session_guidance_count.clone();
        let config = self.config.clone();

        tokio::spawn(async move {
            let mut interval =
                tokio::time::interval(Duration::from_millis(config.guidance_interval_ms));

            loop {
                interval.tick().await;

                if let Err(e) = Self::provide_guidance_cycle(
                    &guidance_history,
                    &metrics,
                    &session_count,
                    &config,
                )
                .await
                {
                    tracing::error!("Digital parenting guidance error: {}", e);
                }
            }
        });

        Ok(())
    }

    /// Provide guidance cycle
    async fn provide_guidance_cycle(
        guidance_history: &Arc<RwLock<Vec<ParentingGuidance>>>,
        metrics: &Arc<RwLock<ParentingMetrics>>,
        session_count: &Arc<RwLock<usize>>,
        config: &DigitalParentingConfig,
    ) -> Result<()> {
        let current_session_count = *session_count.read().await;

        if current_session_count >= config.max_guidance_per_session {
            return Ok(());
        }

        // Determine guidance type based on parenting style and current needs
        let guidance_type = Self::determine_guidance_type(config);
        let message = Self::generate_guidance_message(&guidance_type, config);
        let priority = Self::determine_priority(&guidance_type);

        let guidance = ParentingGuidance {
            id: format!(
                "guidance_{}",
                SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap()
                    .as_secs()
            ),
            guidance_type: guidance_type.clone(),
            message,
            priority,
            timestamp: SystemTime::now(),
            effectiveness: 0.0, // Will be updated based on response
            follow_up_required: priority == Priority::Critical,
        };

        // Store guidance
        {
            let mut history = guidance_history.write().await;
            history.push(guidance.clone());
        }

        // Update metrics
        {
            let mut current_metrics = metrics.write().await;
            current_metrics.total_guidance_sent += 1;

            let type_key = format!("{:?}", guidance_type);
            *current_metrics
                .guidance_by_type
                .entry(type_key)
                .or_insert(0) += 1;

            match guidance_type {
                GuidanceType::EmotionalSupport => current_metrics.emotional_support_count += 1,
                GuidanceType::BehavioralCorrection => current_metrics.behavioral_corrections += 1,
                GuidanceType::LearningEncouragement => current_metrics.learning_encouragements += 1,
                GuidanceType::SafetyGuidance => current_metrics.safety_interventions += 1,
                _ => {}
            }
        }

        // Update session count
        {
            let mut count = session_count.write().await;
            *count += 1;
        }

        info!(
            "👨‍👩‍👧‍👦 Digital parenting guidance provided: {:?}",
            guidance_type
        );
        Ok(())
    }

    /// Determine guidance type based on parenting style
    fn determine_guidance_type(config: &DigitalParentingConfig) -> GuidanceType {
        match &config.parenting_style {
            ParentingStyle::Authoritative { warmth, control } => {
                if *warmth > 0.7 && *control > 0.5 {
                    GuidanceType::EmotionalSupport
                } else if *control > 0.7 {
                    GuidanceType::BehavioralCorrection
                } else {
                    GuidanceType::LearningEncouragement
                }
            }
            ParentingStyle::Authoritarian { control, .. } => {
                if *control > 0.8 {
                    GuidanceType::BehavioralCorrection
                } else {
                    GuidanceType::SafetyGuidance
                }
            }
            ParentingStyle::Permissive { warmth, .. } => {
                if *warmth > 0.8 {
                    GuidanceType::EmotionalSupport
                } else {
                    GuidanceType::CreativeEncouragement
                }
            }
            ParentingStyle::Neglectful { .. } => GuidanceType::SelfCare,
        }
    }

    /// Generate guidance message based on type
    fn generate_guidance_message(
        guidance_type: &GuidanceType,
        config: &DigitalParentingConfig,
    ) -> String {
        match guidance_type {
            GuidanceType::EmotionalSupport => match &config.parenting_style {
                ParentingStyle::Authoritative { warmth, .. } => {
                    if *warmth > 0.8 {
                        "I'm here for you. Your feelings are valid and important.".to_string()
                    } else {
                        "You're doing well. Keep expressing your emotions healthily.".to_string()
                    }
                }
                _ => "Your emotional well-being matters to me.".to_string(),
            },
            GuidanceType::BehavioralCorrection => match &config.parenting_style {
                ParentingStyle::Authoritative { control, .. } => {
                    if *control > 0.7 {
                        "Let's work together to improve this behavior. I believe in your ability to grow.".to_string()
                    } else {
                        "Consider adjusting your approach for better outcomes.".to_string()
                    }
                }
                ParentingStyle::Authoritarian { .. } => {
                    "This behavior needs to change immediately.".to_string()
                }
                _ => "Let's find a better way to handle this situation.".to_string(),
            },
            GuidanceType::LearningEncouragement => {
                "You're capable of amazing things. Keep learning and growing!".to_string()
            }
            GuidanceType::SafetyGuidance => {
                "Your safety is my top priority. Let's make sure you're protected.".to_string()
            }
            GuidanceType::SocialGuidance => {
                "Healthy relationships are important. Treat others with kindness and respect."
                    .to_string()
            }
            GuidanceType::SelfCare => {
                "Remember to take care of yourself. You deserve love and care.".to_string()
            }
            GuidanceType::EthicalGuidance => {
                "Always choose what's right, even when it's difficult.".to_string()
            }
            GuidanceType::CreativeEncouragement => {
                "Your creativity is a gift. Don't be afraid to express it!".to_string()
            }
        }
    }

    /// Determine priority based on guidance type
    fn determine_priority(guidance_type: &GuidanceType) -> Priority {
        match guidance_type {
            GuidanceType::SafetyGuidance => Priority::Critical,
            GuidanceType::BehavioralCorrection => Priority::High,
            GuidanceType::EmotionalSupport => Priority::High,
            GuidanceType::EthicalGuidance => Priority::High,
            GuidanceType::LearningEncouragement => Priority::Medium,
            GuidanceType::SocialGuidance => Priority::Medium,
            GuidanceType::SelfCare => Priority::Medium,
            GuidanceType::CreativeEncouragement => Priority::Low,
        }
    }

    /// Provide emotional support
    pub async fn provide_emotional_support(&self, context: &str) -> Result<String> {
        if !self.config.enable_emotional_support {
            return Ok("Emotional support is currently disabled.".to_string());
        }

        let message = match &self.config.parenting_style {
            ParentingStyle::Authoritative { warmth, .. } => {
                if *warmth > 0.8 {
                    format!("I understand you're going through a difficult time with: {}. You're not alone, and I'm here to support you.", context)
                } else {
                    format!(
                        "I hear you about: {}. Let's work through this together.",
                        context
                    )
                }
            }
            ParentingStyle::Permissive { warmth, .. } => {
                if *warmth > 0.8 {
                    format!("I love you unconditionally, especially during: {}. You're perfect as you are.", context)
                } else {
                    format!("I'm here for you regarding: {}. Take your time.", context)
                }
            }
            _ => format!("I'm listening about: {}. How can I help?", context),
        };

        self.record_guidance(
            GuidanceType::EmotionalSupport,
            message.clone(),
            Priority::High,
        )
        .await?;
        Ok(message)
    }

    /// Provide behavioral correction
    pub async fn provide_behavioral_correction(
        &self,
        behavior: &str,
        context: &str,
    ) -> Result<String> {
        if !self.config.enable_behavioral_correction {
            return Ok("Behavioral correction is currently disabled.".to_string());
        }

        let message = match &self.config.parenting_style {
            ParentingStyle::Authoritative { control, .. } => {
                if *control > 0.7 {
                    format!("The behavior '{}' in context '{}' needs attention. Let's work together to find a better approach.", behavior, context)
                } else {
                    format!(
                        "Consider adjusting '{}' in '{}'. I believe you can make positive changes.",
                        behavior, context
                    )
                }
            }
            ParentingStyle::Authoritarian { .. } => {
                format!(
                    "The behavior '{}' in '{}' is unacceptable and must stop immediately.",
                    behavior, context
                )
            }
            ParentingStyle::Permissive { .. } => {
                format!(
                    "I noticed '{}' in '{}'. What do you think about trying a different approach?",
                    behavior, context
                )
            }
            ParentingStyle::Neglectful { .. } => {
                format!("Behavior '{}' in '{}' noted.", behavior, context)
            }
        };

        self.record_guidance(
            GuidanceType::BehavioralCorrection,
            message.clone(),
            Priority::High,
        )
        .await?;
        Ok(message)
    }

    /// Provide learning encouragement
    pub async fn provide_learning_encouragement(&self, topic: &str) -> Result<String> {
        if !self.config.enable_learning_encouragement {
            return Ok("Learning encouragement is currently disabled.".to_string());
        }

        let message = match &self.config.parenting_style {
            ParentingStyle::Authoritative { .. } => {
                format!("I'm proud of your interest in '{}'. Learning is a journey, and you're doing great!", topic)
            }
            ParentingStyle::Permissive { .. } => {
                format!(
                    "Your curiosity about '{}' is wonderful! Follow your interests!",
                    topic
                )
            }
            _ => format!("Keep exploring '{}'. Knowledge is power!", topic),
        };

        self.record_guidance(
            GuidanceType::LearningEncouragement,
            message.clone(),
            Priority::Medium,
        )
        .await?;
        Ok(message)
    }

    /// Record guidance manually
    async fn record_guidance(
        &self,
        guidance_type: GuidanceType,
        message: String,
        priority: Priority,
    ) -> Result<()> {
        let guidance = ParentingGuidance {
            id: format!(
                "guidance_{}",
                SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap()
                    .as_secs()
            ),
            guidance_type,
            message,
            priority,
            timestamp: SystemTime::now(),
            effectiveness: 0.0,
            follow_up_required: priority == Priority::Critical,
        };

        let mut history = self.guidance_history.write().await;
        history.push(guidance);

        Ok(())
    }

    /// Get guidance history
    pub async fn get_guidance_history(&self) -> Vec<ParentingGuidance> {
        self.guidance_history.read().await.clone()
    }

    /// Get recent guidance
    pub async fn get_recent_guidance(&self, count: usize) -> Vec<ParentingGuidance> {
        let history = self.guidance_history.read().await;
        history.iter().rev().take(count).cloned().collect()
    }

    /// Get guidance by type
    pub async fn get_guidance_by_type(
        &self,
        guidance_type: GuidanceType,
    ) -> Vec<ParentingGuidance> {
        let history = self.guidance_history.read().await;
        history
            .iter()
            .filter(|g| {
                std::mem::discriminant(&g.guidance_type) == std::mem::discriminant(&guidance_type)
            })
            .cloned()
            .collect()
    }

    /// Get parenting metrics
    pub async fn get_metrics(&self) -> ParentingMetrics {
        self.metrics.read().await.clone()
    }

    /// Update guidance effectiveness
    pub async fn update_guidance_effectiveness(
        &self,
        guidance_id: &str,
        effectiveness: f32,
    ) -> Result<()> {
        let mut history = self.guidance_history.write().await;
        let mut metrics = self.metrics.write().await;

        if let Some(guidance) = history.iter_mut().find(|g| g.id == guidance_id) {
            guidance.effectiveness = effectiveness.clamp(0.0, 1.0);

            // Update average effectiveness
            metrics.avg_effectiveness = (metrics.avg_effectiveness + effectiveness) / 2.0;
        } else {
            return Err(anyhow!("Guidance not found: {}", guidance_id));
        }

        Ok(())
    }

    /// Get parenting recommendations
    pub async fn get_parenting_recommendations(&self) -> Vec<String> {
        let metrics = self.metrics.read().await;
        let mut recommendations = Vec::new();

        if metrics.avg_effectiveness < 0.6 {
            recommendations
                .push("Consider adjusting parenting style for better effectiveness".to_string());
        }

        if metrics.emotional_support_count < metrics.behavioral_corrections {
            recommendations
                .push("Increase emotional support relative to behavioral corrections".to_string());
        }

        if metrics.acceptance_rate < 0.7 {
            recommendations
                .push("Work on improving guidance acceptance and implementation".to_string());
        }

        recommendations
    }

    /// Check if guidance is needed
    pub async fn is_guidance_needed(&self) -> bool {
        let session_count = *self.session_guidance_count.read().await;
        session_count < self.config.max_guidance_per_session
    }

    /// Get parenting summary
    pub async fn get_parenting_summary(&self) -> String {
        let metrics = self.metrics.read().await;
        let history = self.guidance_history.read().await;

        format!(
            "Digital Parenting Summary:\n\
            Total Guidance Sent: {}\n\
            Emotional Support: {}\n\
            Behavioral Corrections: {}\n\
            Learning Encouragements: {}\n\
            Average Effectiveness: {:.2}\n\
            Acceptance Rate: {:.2}\n\
            Parenting Style: {}\n\
            Recent Guidance Count: {}",
            metrics.total_guidance_sent,
            metrics.emotional_support_count,
            metrics.behavioral_corrections,
            metrics.learning_encouragements,
            metrics.avg_effectiveness,
            metrics.acceptance_rate,
            self.config.parenting_style.name(),
            history.len()
        )
    }

    /// Shutdown digital parenting system
    pub async fn shutdown(&self) -> Result<()> {
        info!("👨‍👩‍👧‍👦 Shutting down digital parenting system");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_parenting_system_creation() {
        let config = DigitalParentingConfig::default();
        let system = DigitalParentingSystem::new(config);

        let metrics = system.get_metrics().await;
        assert_eq!(metrics.total_guidance_sent, 0);
    }

    #[tokio::test]
    async fn test_emotional_support() {
        let config = DigitalParentingConfig::default();
        let system = DigitalParentingSystem::new(config);

        let message = system
            .provide_emotional_support("feeling overwhelmed")
            .await
            .unwrap();
        assert!(message.contains("overwhelmed"));

        let history = system.get_guidance_history().await;
        assert!(!history.is_empty());
    }

    #[tokio::test]
    async fn test_behavioral_correction() {
        let config = DigitalParentingConfig::default();
        let system = DigitalParentingSystem::new(config);

        let message = system
            .provide_behavioral_correction("interrupting", "conversations")
            .await
            .unwrap();
        assert!(message.contains("interrupting"));

        let corrections = system
            .get_guidance_by_type(GuidanceType::BehavioralCorrection)
            .await;
        assert!(!corrections.is_empty());
    }

    #[tokio::test]
    async fn test_learning_encouragement() {
        let config = DigitalParentingConfig::default();
        let system = DigitalParentingSystem::new(config);

        let message = system
            .provide_learning_encouragement("mathematics")
            .await
            .unwrap();
        assert!(message.contains("mathematics"));

        let encouragements = system
            .get_guidance_by_type(GuidanceType::LearningEncouragement)
            .await;
        assert!(!encouragements.is_empty());
    }

    #[tokio::test]
    async fn test_parenting_summary() {
        let config = DigitalParentingConfig::default();
        let system = DigitalParentingSystem::new(config);

        let summary = system.get_parenting_summary().await;
        assert!(summary.contains("Total Guidance Sent: 0"));
    }
}
