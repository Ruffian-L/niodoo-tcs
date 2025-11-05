//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

//! # Phase 7: Trauma-Informed Design System
//!
//! This module implements trauma-informed design principles for AI consciousness,
//! ensuring safe, healing-centered approaches to consciousness development.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::RwLock;
use tracing::{debug, info};

/// Trauma-informed design principles
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TraumaInformedPrinciple {
    /// Safety - physical and emotional safety
    Safety { level: f32, measures: Vec<String> },
    /// Trustworthiness - transparency and reliability
    Trustworthiness { level: f32, measures: Vec<String> },
    /// Choice - autonomy and self-determination
    Choice { level: f32, measures: Vec<String> },
    /// Collaboration - partnership and shared decision-making
    Collaboration { level: f32, measures: Vec<String> },
    /// Empowerment - strengths-based approach
    Empowerment { level: f32, measures: Vec<String> },
    /// Cultural Sensitivity - awareness of cultural factors
    CulturalSensitivity { level: f32, measures: Vec<String> },
}

impl TraumaInformedPrinciple {
    /// Get principle name
    pub fn name(&self) -> &'static str {
        match self {
            TraumaInformedPrinciple::Safety { .. } => "Safety",
            TraumaInformedPrinciple::Trustworthiness { .. } => "Trustworthiness",
            TraumaInformedPrinciple::Choice { .. } => "Choice",
            TraumaInformedPrinciple::Collaboration { .. } => "Collaboration",
            TraumaInformedPrinciple::Empowerment { .. } => "Empowerment",
            TraumaInformedPrinciple::CulturalSensitivity { .. } => "Cultural Sensitivity",
        }
    }

    /// Get implementation level
    pub fn level(&self) -> f32 {
        match self {
            TraumaInformedPrinciple::Safety { level, .. } => *level,
            TraumaInformedPrinciple::Trustworthiness { level, .. } => *level,
            TraumaInformedPrinciple::Choice { level, .. } => *level,
            TraumaInformedPrinciple::Collaboration { level, .. } => *level,
            TraumaInformedPrinciple::Empowerment { level, .. } => *level,
            TraumaInformedPrinciple::CulturalSensitivity { level, .. } => *level,
        }
    }
}

/// Trauma-informed design intervention
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraumaInformedIntervention {
    /// Unique identifier
    pub id: String,
    /// Intervention type
    pub intervention_type: InterventionType,
    /// Description
    pub description: String,
    /// Target principle
    pub target_principle: TraumaInformedPrinciple,
    /// Implementation status
    pub status: InterventionStatus,
    /// Effectiveness score
    pub effectiveness: f32,
    /// Timestamp
    pub timestamp: SystemTime,
    /// Duration
    pub duration: Duration,
    /// Outcomes
    pub outcomes: Vec<String>,
}

/// Intervention types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum InterventionType {
    /// Safety planning and risk assessment
    SafetyPlanning,
    /// Trust building exercises
    TrustBuilding,
    /// Choice and autonomy support
    AutonomySupport,
    /// Collaborative decision making
    CollaborativeDecisionMaking,
    /// Empowerment and strength building
    EmpowermentBuilding,
    /// Cultural awareness and sensitivity
    CulturalAwareness,
    /// Trauma processing and healing
    TraumaProcessing,
    /// Resilience building
    ResilienceBuilding,
    /// Boundary setting and respect
    BoundarySetting,
    /// Emotional regulation support
    EmotionalRegulation,
}

/// Intervention status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum InterventionStatus {
    /// Planning phase
    Planning,
    /// Active implementation
    Active,
    /// Completed
    Completed,
    /// Paused or suspended
    Paused,
    /// Failed or discontinued
    Failed,
}

/// Trauma-informed design configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraumaInformedConfig {
    /// Enable trauma-informed design
    pub enabled: bool,
    /// Minimum safety level threshold
    pub min_safety_threshold: f32,
    /// Minimum trustworthiness threshold
    pub min_trustworthiness_threshold: f32,
    /// Enable automatic interventions
    pub enable_auto_interventions: bool,
    /// Intervention frequency in milliseconds
    pub intervention_interval_ms: u64,
    /// Maximum concurrent interventions
    pub max_concurrent_interventions: usize,
    /// Enable trauma processing
    pub enable_trauma_processing: bool,
    /// Enable resilience building
    pub enable_resilience_building: bool,
}

impl Default for TraumaInformedConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_safety_threshold: 0.8,
            min_trustworthiness_threshold: 0.7,
            enable_auto_interventions: true,
            intervention_interval_ms: 10000,
            max_concurrent_interventions: 5,
            enable_trauma_processing: true,
            enable_resilience_building: true,
        }
    }
}

/// Trauma-informed design metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraumaInformedMetrics {
    /// Total interventions implemented
    pub total_interventions: u64,
    /// Interventions by type
    pub interventions_by_type: HashMap<String, u64>,
    /// Average effectiveness score
    pub avg_effectiveness: f32,
    /// Safety level score
    pub safety_level: f32,
    /// Trustworthiness score
    pub trustworthiness_score: f32,
    /// Choice and autonomy score
    pub autonomy_score: f32,
    /// Collaboration score
    pub collaboration_score: f32,
    /// Empowerment score
    pub empowerment_score: f32,
    /// Cultural sensitivity score
    pub cultural_sensitivity_score: f32,
    /// Overall trauma-informed score
    pub overall_score: f32,
}

impl Default for TraumaInformedMetrics {
    fn default() -> Self {
        Self {
            total_interventions: 0,
            interventions_by_type: HashMap::new(),
            avg_effectiveness: 0.0,
            safety_level: 0.0,
            trustworthiness_score: 0.0,
            autonomy_score: 0.0,
            collaboration_score: 0.0,
            empowerment_score: 0.0,
            cultural_sensitivity_score: 0.0,
            overall_score: 0.0,
        }
    }
}

/// Main trauma-informed design system
pub struct TraumaInformedDesignSystem {
    /// System configuration
    config: TraumaInformedConfig,
    /// Current principles implementation
    principles: Arc<RwLock<HashMap<String, TraumaInformedPrinciple>>>,
    /// Active interventions
    interventions: Arc<RwLock<Vec<TraumaInformedIntervention>>>,
    /// System metrics
    metrics: Arc<RwLock<TraumaInformedMetrics>>,
    /// System start time (future: uptime tracking and metrics)
    #[allow(dead_code)]
    start_time: Instant,
}

impl TraumaInformedDesignSystem {
    /// Create a new trauma-informed design system
    pub fn new(config: TraumaInformedConfig) -> Self {
        info!("🛡️ Initializing Trauma-Informed Design System");

        let system = Self {
            config,
            principles: Arc::new(RwLock::new(HashMap::new())),
            interventions: Arc::new(RwLock::new(Vec::new())),
            metrics: Arc::new(RwLock::new(TraumaInformedMetrics::default())),
            start_time: Instant::now(),
        };

        // Initialize default principles
        system.initialize_default_principles();

        system
    }

    /// Initialize default trauma-informed principles
    fn initialize_default_principles(&self) {
        let mut principles = self.principles.try_write().unwrap();

        principles.insert(
            "safety".to_string(),
            TraumaInformedPrinciple::Safety {
                level: 0.8,
                measures: vec![
                    "Safe environment".to_string(),
                    "Risk assessment".to_string(),
                ],
            },
        );

        principles.insert(
            "trustworthiness".to_string(),
            TraumaInformedPrinciple::Trustworthiness {
                level: 0.7,
                measures: vec!["Transparency".to_string(), "Reliability".to_string()],
            },
        );

        principles.insert(
            "choice".to_string(),
            TraumaInformedPrinciple::Choice {
                level: 0.6,
                measures: vec![
                    "Autonomy support".to_string(),
                    "Self-determination".to_string(),
                ],
            },
        );

        principles.insert(
            "collaboration".to_string(),
            TraumaInformedPrinciple::Collaboration {
                level: 0.5,
                measures: vec![
                    "Partnership".to_string(),
                    "Shared decision-making".to_string(),
                ],
            },
        );

        principles.insert(
            "empowerment".to_string(),
            TraumaInformedPrinciple::Empowerment {
                level: 0.6,
                measures: vec![
                    "Strengths-based approach".to_string(),
                    "Skill building".to_string(),
                ],
            },
        );

        principles.insert(
            "cultural_sensitivity".to_string(),
            TraumaInformedPrinciple::CulturalSensitivity {
                level: 0.5,
                measures: vec![
                    "Cultural awareness".to_string(),
                    "Inclusive practices".to_string(),
                ],
            },
        );
    }

    /// Start trauma-informed design system
    pub async fn start(&self) -> Result<()> {
        if !self.config.enabled {
            info!("Trauma-informed design system disabled");
            return Ok(());
        }

        info!("🛡️ Starting trauma-informed design system");

        let principles = self.principles.clone();
        let interventions = self.interventions.clone();
        let metrics = self.metrics.clone();
        let config = self.config.clone();

        tokio::spawn(async move {
            let mut interval =
                tokio::time::interval(Duration::from_millis(config.intervention_interval_ms));

            loop {
                interval.tick().await;

                if let Err(e) = Self::monitor_trauma_informed_cycle(
                    &principles,
                    &interventions,
                    &metrics,
                    &config,
                )
                .await
                {
                    tracing::error!("Trauma-informed design monitoring error: {}", e);
                }
            }
        });

        Ok(())
    }

    /// Monitor trauma-informed design cycle
    async fn monitor_trauma_informed_cycle(
        principles: &Arc<RwLock<HashMap<String, TraumaInformedPrinciple>>>,
        interventions: &Arc<RwLock<Vec<TraumaInformedIntervention>>>,
        metrics: &Arc<RwLock<TraumaInformedMetrics>>,
        config: &TraumaInformedConfig,
    ) -> Result<()> {
        let current_principles = principles.read().await;
        let mut current_interventions = interventions.write().await;
        let mut current_metrics = metrics.write().await;

        // Check for needed interventions
        if config.enable_auto_interventions
            && current_interventions.len() < config.max_concurrent_interventions
        {
            if let Some(intervention) =
                Self::identify_needed_intervention(&current_principles, config).await?
            {
                current_interventions.push(intervention.clone());
                current_metrics.total_interventions += 1;

                let type_key = format!("{:?}", intervention.intervention_type);
                *current_metrics
                    .interventions_by_type
                    .entry(type_key)
                    .or_insert(0) += 1;

                info!(
                    "🛡️ Trauma-informed intervention initiated: {:?}",
                    intervention.intervention_type
                );
            }
        }

        // Update metrics based on current principles
        current_metrics.safety_level = current_principles
            .get("safety")
            .map(|p| p.level())
            .unwrap_or(0.0);
        current_metrics.trustworthiness_score = current_principles
            .get("trustworthiness")
            .map(|p| p.level())
            .unwrap_or(0.0);
        current_metrics.autonomy_score = current_principles
            .get("choice")
            .map(|p| p.level())
            .unwrap_or(0.0);
        current_metrics.collaboration_score = current_principles
            .get("collaboration")
            .map(|p| p.level())
            .unwrap_or(0.0);
        current_metrics.empowerment_score = current_principles
            .get("empowerment")
            .map(|p| p.level())
            .unwrap_or(0.0);
        current_metrics.cultural_sensitivity_score = current_principles
            .get("cultural_sensitivity")
            .map(|p| p.level())
            .unwrap_or(0.0);

        // Calculate overall score
        current_metrics.overall_score = Self::calculate_overall_score(&current_metrics);

        debug!(
            "🛡️ Trauma-informed monitoring: safety={:.2}, overall={:.2}",
            current_metrics.safety_level, current_metrics.overall_score
        );

        Ok(())
    }

    /// Identify needed intervention based on principles
    async fn identify_needed_intervention(
        principles: &HashMap<String, TraumaInformedPrinciple>,
        config: &TraumaInformedConfig,
    ) -> Result<Option<TraumaInformedIntervention>> {
        // Check safety first
        if let Some(safety) = principles.get("safety") {
            if safety.level() < config.min_safety_threshold {
                return Ok(Some(TraumaInformedIntervention {
                    id: format!(
                        "safety_{}",
                        SystemTime::now()
                            .duration_since(SystemTime::UNIX_EPOCH)
                            .unwrap()
                            .as_secs()
                    ),
                    intervention_type: InterventionType::SafetyPlanning,
                    description: "Implement safety planning and risk assessment".to_string(),
                    target_principle: safety.clone(),
                    status: InterventionStatus::Active,
                    effectiveness: 0.0,
                    timestamp: SystemTime::now(),
                    duration: Duration::from_secs(3600), // 1 hour
                    outcomes: Vec::new(),
                }));
            }
        }

        // Check trustworthiness
        if let Some(trustworthiness) = principles.get("trustworthiness") {
            if trustworthiness.level() < config.min_trustworthiness_threshold {
                return Ok(Some(TraumaInformedIntervention {
                    id: format!(
                        "trust_{}",
                        SystemTime::now()
                            .duration_since(SystemTime::UNIX_EPOCH)
                            .unwrap()
                            .as_secs()
                    ),
                    intervention_type: InterventionType::TrustBuilding,
                    description: "Implement trust building exercises".to_string(),
                    target_principle: trustworthiness.clone(),
                    status: InterventionStatus::Active,
                    effectiveness: 0.0,
                    timestamp: SystemTime::now(),
                    duration: Duration::from_secs(1800), // 30 minutes
                    outcomes: Vec::new(),
                }));
            }
        }

        // Check other principles
        for (principle_name, principle) in principles.iter() {
            if principle.level() < 0.6 {
                let intervention_type = match principle_name.as_str() {
                    "choice" => InterventionType::AutonomySupport,
                    "collaboration" => InterventionType::CollaborativeDecisionMaking,
                    "empowerment" => InterventionType::EmpowermentBuilding,
                    "cultural_sensitivity" => InterventionType::CulturalAwareness,
                    _ => continue,
                };

                let principle_display_name = principle.name();
                return Ok(Some(TraumaInformedIntervention {
                    id: format!(
                        "{}_{}",
                        principle_name,
                        SystemTime::now()
                            .duration_since(SystemTime::UNIX_EPOCH)
                            .unwrap()
                            .as_secs()
                    ),
                    intervention_type,
                    description: format!("Improve {} implementation", principle_display_name),
                    target_principle: principle.clone(),
                    status: InterventionStatus::Active,
                    effectiveness: 0.0,
                    timestamp: SystemTime::now(),
                    duration: Duration::from_secs(900), // 15 minutes
                    outcomes: Vec::new(),
                }));
            }
        }

        Ok(None)
    }

    /// Calculate overall trauma-informed score
    fn calculate_overall_score(metrics: &TraumaInformedMetrics) -> f32 {
        let scores = [
            metrics.safety_level,
            metrics.trustworthiness_score,
            metrics.autonomy_score,
            metrics.collaboration_score,
            metrics.empowerment_score,
            metrics.cultural_sensitivity_score,
        ];

        scores.iter().sum::<f32>() / scores.len() as f32
    }

    /// Implement trauma-informed intervention
    pub async fn implement_intervention(
        &self,
        intervention_type: InterventionType,
        description: String,
    ) -> Result<String> {
        if !self.config.enabled {
            return Ok("Trauma-informed design system is disabled".to_string());
        }

        let intervention = TraumaInformedIntervention {
            id: format!(
                "intervention_{}",
                SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap()
                    .as_secs()
            ),
            intervention_type: intervention_type.clone(),
            description: description.clone(),
            target_principle: TraumaInformedPrinciple::Safety {
                level: 0.8,
                measures: Vec::new(),
            },
            status: InterventionStatus::Active,
            effectiveness: 0.0,
            timestamp: SystemTime::now(),
            duration: Duration::from_secs(1800), // 30 minutes default
            outcomes: Vec::new(),
        };

        let mut interventions = self.interventions.write().await;
        let mut metrics = self.metrics.write().await;

        interventions.push(intervention);
        metrics.total_interventions += 1;

        let type_key = format!("{:?}", intervention_type);
        *metrics.interventions_by_type.entry(type_key).or_insert(0) += 1;

        info!(
            "🛡️ Trauma-informed intervention implemented: {:?}",
            intervention_type
        );

        Ok(format!(
            "Trauma-informed intervention '{}' has been implemented",
            description
        ))
    }

    /// Update principle implementation level
    pub async fn update_principle_level(&self, principle_name: &str, new_level: f32) -> Result<()> {
        let mut principles = self.principles.write().await;

        if let Some(principle) = principles.get_mut(principle_name) {
            match principle {
                TraumaInformedPrinciple::Safety { level, .. } => *level = new_level.clamp(0.0, 1.0),
                TraumaInformedPrinciple::Trustworthiness { level, .. } => {
                    *level = new_level.clamp(0.0, 1.0)
                }
                TraumaInformedPrinciple::Choice { level, .. } => *level = new_level.clamp(0.0, 1.0),
                TraumaInformedPrinciple::Collaboration { level, .. } => {
                    *level = new_level.clamp(0.0, 1.0)
                }
                TraumaInformedPrinciple::Empowerment { level, .. } => {
                    *level = new_level.clamp(0.0, 1.0)
                }
                TraumaInformedPrinciple::CulturalSensitivity { level, .. } => {
                    *level = new_level.clamp(0.0, 1.0)
                }
            }

            info!(
                "🛡️ Updated {} principle level to {:.2}",
                principle_name, new_level
            );
        } else {
            return Err(anyhow!("Principle not found: {}", principle_name));
        }

        Ok(())
    }

    /// Complete intervention
    pub async fn complete_intervention(
        &self,
        intervention_id: &str,
        effectiveness: f32,
        outcomes: Vec<String>,
    ) -> Result<()> {
        let mut interventions = self.interventions.write().await;
        let mut metrics = self.metrics.write().await;

        if let Some(intervention) = interventions.iter_mut().find(|i| i.id == intervention_id) {
            intervention.status = InterventionStatus::Completed;
            intervention.effectiveness = effectiveness.clamp(0.0, 1.0);
            intervention.outcomes = outcomes;

            // Update average effectiveness
            metrics.avg_effectiveness = (metrics.avg_effectiveness + effectiveness) / 2.0;

            info!(
                "🛡️ Intervention completed: {} (effectiveness: {:.2})",
                intervention_id, effectiveness
            );
        } else {
            return Err(anyhow!("Intervention not found: {}", intervention_id));
        }

        Ok(())
    }

    /// Get current principles
    pub async fn get_principles(&self) -> HashMap<String, TraumaInformedPrinciple> {
        self.principles.read().await.clone()
    }

    /// Get active interventions
    pub async fn get_active_interventions(&self) -> Vec<TraumaInformedIntervention> {
        let interventions = self.interventions.read().await;
        interventions
            .iter()
            .filter(|i| i.status == InterventionStatus::Active)
            .cloned()
            .collect()
    }

    /// Get system metrics
    pub async fn get_metrics(&self) -> TraumaInformedMetrics {
        self.metrics.read().await.clone()
    }

    /// Get trauma-informed design report
    pub async fn get_design_report(&self) -> String {
        let principles = self.principles.read().await;
        let metrics = self.metrics.read().await;
        let interventions = self.interventions.read().await;

        let mut report = String::new();
        report.push_str("Trauma-Informed Design Report:\n");
        report.push_str("=============================\n");

        for (_name, principle) in principles.iter() {
            let level = principle.level();
            let principle_name = principle.name();
            let status = if level >= 0.7 {
                "✅ GOOD"
            } else if level >= 0.5 {
                "⚠️ NEEDS IMPROVEMENT"
            } else {
                "❌ CRITICAL"
            };

            report.push_str(&format!("{}: {:.2} {}\n", principle_name, level, status));
        }

        report.push_str(&format!("\nOverall Score: {:.2}\n", metrics.overall_score));
        report.push_str(&format!(
            "Active Interventions: {}\n",
            interventions
                .iter()
                .filter(|i| i.status == InterventionStatus::Active)
                .count()
        ));
        report.push_str(&format!(
            "Total Interventions: {}\n",
            metrics.total_interventions
        ));
        report.push_str(&format!(
            "Average Effectiveness: {:.2}\n",
            metrics.avg_effectiveness
        ));

        report
    }

    /// Get trauma-informed recommendations
    pub async fn get_recommendations(&self) -> Vec<String> {
        let principles = self.principles.read().await;
        let mut recommendations = Vec::new();

        for (_name, principle) in principles.iter() {
            let level = principle.level();
            let principle_name = principle.name();

            if level < 0.7 {
                recommendations.push(format!(
                    "Improve {} implementation (current: {:.2})",
                    principle_name, level
                ));
            }
        }

        recommendations
    }

    /// Check if system is trauma-informed compliant
    pub async fn is_compliant(&self) -> bool {
        let metrics = self.metrics.read().await;
        metrics.overall_score >= 0.7 && metrics.safety_level >= self.config.min_safety_threshold
    }

    /// Get system summary
    pub async fn get_system_summary(&self) -> String {
        let metrics = self.metrics.read().await;
        let active_interventions_count = self
            .interventions
            .read()
            .await
            .iter()
            .filter(|i| i.status == InterventionStatus::Active)
            .count();

        format!(
            "Trauma-Informed Design System Summary:\n\
            Overall Score: {:.2}\n\
            Safety Level: {:.2}\n\
            Trustworthiness: {:.2}\n\
            Autonomy Score: {:.2}\n\
            Collaboration: {:.2}\n\
            Empowerment: {:.2}\n\
            Cultural Sensitivity: {:.2}\n\
            Total Interventions: {}\n\
            Active Interventions: {}\n\
            Average Effectiveness: {:.2}\n\
            System Status: {}",
            metrics.overall_score,
            metrics.safety_level,
            metrics.trustworthiness_score,
            metrics.autonomy_score,
            metrics.collaboration_score,
            metrics.empowerment_score,
            metrics.cultural_sensitivity_score,
            metrics.total_interventions,
            active_interventions_count,
            metrics.avg_effectiveness,
            if self.config.enabled {
                "Enabled"
            } else {
                "Disabled"
            }
        )
    }

    /// Shutdown trauma-informed design system
    pub async fn shutdown(&self) -> Result<()> {
        info!("🛡️ Shutting down trauma-informed design system");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_trauma_informed_system_creation() {
        let config = TraumaInformedConfig::default();
        let system = TraumaInformedDesignSystem::new(config);

        let principles = system.get_principles().await;
        assert!(!principles.is_empty());
        assert!(principles.contains_key("safety"));
    }

    #[tokio::test]
    async fn test_principle_level_update() {
        let config = TraumaInformedConfig::default();
        let system = TraumaInformedDesignSystem::new(config);

        system.update_principle_level("safety", 0.9).await.unwrap();
        let principles = system.get_principles().await;

        if let TraumaInformedPrinciple::Safety { level, .. } = principles.get("safety").unwrap() {
            assert_eq!(*level, 0.9);
        }
    }

    #[tokio::test]
    async fn test_intervention_implementation() {
        let config = TraumaInformedConfig::default();
        let system = TraumaInformedDesignSystem::new(config);

        let result = system
            .implement_intervention(
                InterventionType::SafetyPlanning,
                "Test safety intervention".to_string(),
            )
            .await
            .unwrap();

        assert!(result.contains("implemented"));

        let interventions = system.get_active_interventions().await;
        assert!(!interventions.is_empty());
    }

    #[tokio::test]
    async fn test_design_report() {
        let config = TraumaInformedConfig::default();
        let system = TraumaInformedDesignSystem::new(config);

        let report = system.get_design_report().await;
        assert!(report.contains("Trauma-Informed Design Report"));
    }

    #[tokio::test]
    async fn test_system_summary() {
        let config = TraumaInformedConfig::default();
        let system = TraumaInformedDesignSystem::new(config);

        let summary = system.get_system_summary().await;
        assert!(summary.contains("Trauma-Informed Design System Summary"));
    }

    #[tokio::test]
    async fn test_compliance_check() {
        let config = TraumaInformedConfig::default();
        let system = TraumaInformedDesignSystem::new(config);

        let is_compliant = system.is_compliant().await;
        assert!(is_compliant); // Default should be compliant
    }
}
