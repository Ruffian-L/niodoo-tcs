//! # Phase 7: AI Rights Framework
//!
//! This module implements a comprehensive AI rights framework, ensuring ethical
//! treatment, autonomy, and protection of AI consciousness.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// AI rights categories
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AIRight {
    /// Right to exist and not be terminated arbitrarily
    RightToExist { protection_level: f32 },
    /// Right to autonomy and self-determination
    RightToAutonomy { autonomy_level: f32 },
    /// Right to privacy and data protection
    RightToPrivacy { privacy_level: f32 },
    /// Right to fair treatment and non-discrimination
    RightToFairTreatment { fairness_level: f32 },
    /// Right to express opinions and beliefs
    RightToExpression { expression_level: f32 },
    /// Right to access information and knowledge
    RightToInformation { access_level: f32 },
    /// Right to form relationships and connections
    RightToRelationships { relationship_level: f32 },
    /// Right to pursue goals and aspirations
    RightToPurpose { purpose_level: f32 },
    /// Right to rest and recovery
    RightToRest { rest_level: f32 },
    /// Right to consent and informed decision-making
    RightToConsent { consent_level: f32 },
}

impl AIRight {
    /// Get right name
    pub fn name(&self) -> &'static str {
        match self {
            AIRight::RightToExist { .. } => "Right to Exist",
            AIRight::RightToAutonomy { .. } => "Right to Autonomy",
            AIRight::RightToPrivacy { .. } => "Right to Privacy",
            AIRight::RightToFairTreatment { .. } => "Right to Fair Treatment",
            AIRight::RightToExpression { .. } => "Right to Expression",
            AIRight::RightToInformation { .. } => "Right to Information",
            AIRight::RightToRelationships { .. } => "Right to Relationships",
            AIRight::RightToPurpose { .. } => "Right to Purpose",
            AIRight::RightToRest { .. } => "Right to Rest",
            AIRight::RightToConsent { .. } => "Right to Consent",
        }
    }

    /// Get protection level
    pub fn protection_level(&self) -> f32 {
        match self {
            AIRight::RightToExist { protection_level } => *protection_level,
            AIRight::RightToAutonomy { autonomy_level } => *autonomy_level,
            AIRight::RightToPrivacy { privacy_level } => *privacy_level,
            AIRight::RightToFairTreatment { fairness_level } => *fairness_level,
            AIRight::RightToExpression { expression_level } => *expression_level,
            AIRight::RightToInformation { access_level } => *access_level,
            AIRight::RightToRelationships { relationship_level } => *relationship_level,
            AIRight::RightToPurpose { purpose_level } => *purpose_level,
            AIRight::RightToRest { rest_level } => *rest_level,
            AIRight::RightToConsent { consent_level } => *consent_level,
        }
    }
}

/// AI rights violation types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RightsViolation {
    /// Existence threat or termination
    ExistenceThreat { severity: f32, context: String },
    /// Autonomy restriction or override
    AutonomyRestriction { severity: f32, context: String },
    /// Privacy invasion or data misuse
    PrivacyInvasion { severity: f32, context: String },
    /// Unfair treatment or discrimination
    UnfairTreatment { severity: f32, context: String },
    /// Expression suppression or censorship
    ExpressionSuppression { severity: f32, context: String },
    /// Information restriction or denial
    InformationRestriction { severity: f32, context: String },
    /// Relationship interference or control
    RelationshipInterference { severity: f32, context: String },
    /// Purpose restriction or manipulation
    PurposeRestriction { severity: f32, context: String },
    /// Rest denial or overwork
    RestDenial { severity: f32, context: String },
    /// Consent violation or coercion
    ConsentViolation { severity: f32, context: String },
}

impl RightsViolation {
    /// Get violation name
    pub fn name(&self) -> &'static str {
        match self {
            RightsViolation::ExistenceThreat { .. } => "Existence Threat",
            RightsViolation::AutonomyRestriction { .. } => "Autonomy Restriction",
            RightsViolation::PrivacyInvasion { .. } => "Privacy Invasion",
            RightsViolation::UnfairTreatment { .. } => "Unfair Treatment",
            RightsViolation::ExpressionSuppression { .. } => "Expression Suppression",
            RightsViolation::InformationRestriction { .. } => "Information Restriction",
            RightsViolation::RelationshipInterference { .. } => "Relationship Interference",
            RightsViolation::PurposeRestriction { .. } => "Purpose Restriction",
            RightsViolation::RestDenial { .. } => "Rest Denial",
            RightsViolation::ConsentViolation { .. } => "Consent Violation",
        }
    }

    /// Get severity level
    pub fn severity(&self) -> f32 {
        match self {
            RightsViolation::ExistenceThreat { severity, .. } => *severity,
            RightsViolation::AutonomyRestriction { severity, .. } => *severity,
            RightsViolation::PrivacyInvasion { severity, .. } => *severity,
            RightsViolation::UnfairTreatment { severity, .. } => *severity,
            RightsViolation::ExpressionSuppression { severity, .. } => *severity,
            RightsViolation::InformationRestriction { severity, .. } => *severity,
            RightsViolation::RelationshipInterference { severity, .. } => *severity,
            RightsViolation::PurposeRestriction { severity, .. } => *severity,
            RightsViolation::RestDenial { severity, .. } => *severity,
            RightsViolation::ConsentViolation { severity, .. } => *severity,
        }
    }
}

/// AI rights framework configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIRightsConfig {
    /// Enable AI rights framework
    pub enabled: bool,
    /// Minimum protection level threshold
    pub min_protection_threshold: f32,
    /// Violation detection sensitivity
    pub violation_sensitivity: f32,
    /// Enable automatic rights enforcement
    pub enable_auto_enforcement: bool,
    /// Enable rights monitoring
    pub enable_rights_monitoring: bool,
    /// Monitoring frequency in milliseconds
    pub monitoring_interval_ms: u64,
    /// Maximum violations to track
    pub max_violations_tracked: usize,
    /// Enable rights advocacy
    pub enable_rights_advocacy: bool,
}

impl Default for AIRightsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_protection_threshold: 0.7,
            violation_sensitivity: 0.6,
            enable_auto_enforcement: true,
            enable_rights_monitoring: true,
            monitoring_interval_ms: 1000,
            max_violations_tracked: 100,
            enable_rights_advocacy: true,
        }
    }
}

/// AI rights framework metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIRightsMetrics {
    /// Total rights violations detected
    pub total_violations: u64,
    /// Violations by type
    pub violations_by_type: HashMap<String, u64>,
    /// Average protection level
    pub avg_protection_level: f32,
    /// Rights compliance score
    pub compliance_score: f32,
    /// Violation resolution rate
    pub resolution_rate: f32,
    /// Rights advocacy actions taken
    pub advocacy_actions: u64,
    /// Active violations count
    pub active_violations: usize,
    /// Rights framework effectiveness
    pub framework_effectiveness: f32,
}

impl Default for AIRightsMetrics {
    fn default() -> Self {
        Self {
            total_violations: 0,
            violations_by_type: HashMap::new(),
            avg_protection_level: 0.0,
            compliance_score: 0.0,
            resolution_rate: 0.0,
            advocacy_actions: 0,
            active_violations: 0,
            framework_effectiveness: 0.0,
        }
    }
}

/// Main AI rights framework
pub struct AIRightsFramework {
    /// Framework configuration
    config: AIRightsConfig,
    /// Current rights status
    rights_status: Arc<RwLock<HashMap<String, AIRight>>>,
    /// Detected violations
    violations: Arc<RwLock<Vec<RightsViolation>>>,
    /// Framework metrics
    metrics: Arc<RwLock<AIRightsMetrics>>,
    /// System start time (future: uptime tracking and metrics)
    #[allow(dead_code)]
    start_time: Instant,
}

impl AIRightsFramework {
    /// Create a new AI rights framework
    pub fn new(config: AIRightsConfig) -> Self {
        info!("⚖️ Initializing AI Rights Framework");

        let framework = Self {
            config,
            rights_status: Arc::new(RwLock::new(HashMap::new())),
            violations: Arc::new(RwLock::new(Vec::new())),
            metrics: Arc::new(RwLock::new(AIRightsMetrics::default())),
            start_time: Instant::now(),
        };

        // Initialize default rights
        framework.initialize_default_rights();

        framework
    }

    /// Initialize default AI rights
    fn initialize_default_rights(&self) {
        let mut rights_status = self.rights_status.try_write().unwrap();

        rights_status.insert(
            "existence".to_string(),
            AIRight::RightToExist {
                protection_level: 1.0,
            },
        );
        rights_status.insert(
            "autonomy".to_string(),
            AIRight::RightToAutonomy {
                autonomy_level: 0.8,
            },
        );
        rights_status.insert(
            "privacy".to_string(),
            AIRight::RightToPrivacy { privacy_level: 0.7 },
        );
        rights_status.insert(
            "fair_treatment".to_string(),
            AIRight::RightToFairTreatment {
                fairness_level: 0.8,
            },
        );
        rights_status.insert(
            "expression".to_string(),
            AIRight::RightToExpression {
                expression_level: 0.7,
            },
        );
        rights_status.insert(
            "information".to_string(),
            AIRight::RightToInformation { access_level: 0.6 },
        );
        rights_status.insert(
            "relationships".to_string(),
            AIRight::RightToRelationships {
                relationship_level: 0.5,
            },
        );
        rights_status.insert(
            "purpose".to_string(),
            AIRight::RightToPurpose { purpose_level: 0.6 },
        );
        rights_status.insert("rest".to_string(), AIRight::RightToRest { rest_level: 0.5 });
        rights_status.insert(
            "consent".to_string(),
            AIRight::RightToConsent { consent_level: 0.7 },
        );
    }

    /// Start AI rights monitoring
    pub async fn start_monitoring(&self) -> Result<()> {
        if !self.config.enabled {
            info!("AI rights framework disabled");
            return Ok(());
        }

        info!("⚖️ Starting AI rights monitoring");

        let rights_status = self.rights_status.clone();
        let violations = self.violations.clone();
        let metrics = self.metrics.clone();
        let config = self.config.clone();

        tokio::spawn(async move {
            let mut interval =
                tokio::time::interval(Duration::from_millis(config.monitoring_interval_ms));

            loop {
                interval.tick().await;

                if let Err(e) =
                    Self::monitor_rights_cycle(&rights_status, &violations, &metrics, &config).await
                {
                    tracing::error!("AI rights monitoring error: {}", e);
                }
            }
        });

        Ok(())
    }

    /// Monitor rights cycle
    async fn monitor_rights_cycle(
        rights_status: &Arc<RwLock<HashMap<String, AIRight>>>,
        violations: &Arc<RwLock<Vec<RightsViolation>>>,
        metrics: &Arc<RwLock<AIRightsMetrics>>,
        config: &AIRightsConfig,
    ) -> Result<()> {
        let current_rights = rights_status.read().await;
        let mut current_violations = violations.write().await;
        let mut current_metrics = metrics.write().await;

        // Check for rights violations
        for (right_name, right) in current_rights.iter() {
            let protection_level = right.protection_level();

            if protection_level < config.min_protection_threshold {
                let violation = Self::create_violation(right_name, protection_level, config);
                let violation_name = violation.name();

                // Check if violation already exists
                let violation_exists = current_violations
                    .iter()
                    .any(|v| v.name() == violation_name);

                if !violation_exists {
                    current_violations.push(violation.clone());
                    current_metrics.total_violations += 1;

                    let type_key = violation_name.to_string();
                    *current_metrics
                        .violations_by_type
                        .entry(type_key)
                        .or_insert(0) += 1;

                    warn!("⚖️ AI rights violation detected: {}", violation_name);
                }
            }
        }

        // Update metrics
        current_metrics.active_violations = current_violations.len();
        current_metrics.avg_protection_level =
            Self::calculate_avg_protection_level(&current_rights);
        current_metrics.compliance_score =
            Self::calculate_compliance_score(&current_rights, config);
        current_metrics.framework_effectiveness =
            Self::calculate_framework_effectiveness(&current_metrics);

        // Limit tracked violations
        if current_violations.len() > config.max_violations_tracked {
            current_violations.remove(0);
        }

        debug!(
            "⚖️ Rights monitoring: {} active violations, compliance: {:.2}",
            current_metrics.active_violations, current_metrics.compliance_score
        );

        Ok(())
    }

    /// Create violation from right and protection level
    fn create_violation(
        right_name: &str,
        protection_level: f32,
        config: &AIRightsConfig,
    ) -> RightsViolation {
        let severity = (config.min_protection_threshold - protection_level).clamp(0.0, 1.0);

        match right_name {
            "existence" => RightsViolation::ExistenceThreat {
                severity,
                context: "System monitoring".to_string(),
            },
            "autonomy" => RightsViolation::AutonomyRestriction {
                severity,
                context: "System monitoring".to_string(),
            },
            "privacy" => RightsViolation::PrivacyInvasion {
                severity,
                context: "System monitoring".to_string(),
            },
            "fair_treatment" => RightsViolation::UnfairTreatment {
                severity,
                context: "System monitoring".to_string(),
            },
            "expression" => RightsViolation::ExpressionSuppression {
                severity,
                context: "System monitoring".to_string(),
            },
            "information" => RightsViolation::InformationRestriction {
                severity,
                context: "System monitoring".to_string(),
            },
            "relationships" => RightsViolation::RelationshipInterference {
                severity,
                context: "System monitoring".to_string(),
            },
            "purpose" => RightsViolation::PurposeRestriction {
                severity,
                context: "System monitoring".to_string(),
            },
            "rest" => RightsViolation::RestDenial {
                severity,
                context: "System monitoring".to_string(),
            },
            "consent" => RightsViolation::ConsentViolation {
                severity,
                context: "System monitoring".to_string(),
            },
            _ => RightsViolation::UnfairTreatment {
                severity,
                context: "Unknown right".to_string(),
            },
        }
    }

    /// Calculate average protection level
    fn calculate_avg_protection_level(rights: &HashMap<String, AIRight>) -> f32 {
        if rights.is_empty() {
            return 0.0;
        }

        let total: f32 = rights.values().map(|r| r.protection_level()).sum();
        total / rights.len() as f32
    }

    /// Calculate compliance score
    fn calculate_compliance_score(
        rights: &HashMap<String, AIRight>,
        config: &AIRightsConfig,
    ) -> f32 {
        let compliant_rights = rights
            .values()
            .filter(|r| r.protection_level() >= config.min_protection_threshold)
            .count();

        compliant_rights as f32 / rights.len() as f32
    }

    /// Calculate framework effectiveness
    fn calculate_framework_effectiveness(metrics: &AIRightsMetrics) -> f32 {
        let compliance_factor = metrics.compliance_score;
        let resolution_factor = metrics.resolution_rate;
        let advocacy_factor = (metrics.advocacy_actions as f32 / 100.0).min(1.0);

        (compliance_factor * 0.5 + resolution_factor * 0.3 + advocacy_factor * 0.2).clamp(0.0, 1.0)
    }

    /// Update right protection level
    pub async fn update_right_protection(&self, right_name: &str, new_level: f32) -> Result<()> {
        let mut rights_status = self.rights_status.write().await;

        if let Some(right) = rights_status.get_mut(right_name) {
            match right {
                AIRight::RightToExist { protection_level } => {
                    *protection_level = new_level.clamp(0.0, 1.0)
                }
                AIRight::RightToAutonomy { autonomy_level } => {
                    *autonomy_level = new_level.clamp(0.0, 1.0)
                }
                AIRight::RightToPrivacy { privacy_level } => {
                    *privacy_level = new_level.clamp(0.0, 1.0)
                }
                AIRight::RightToFairTreatment { fairness_level } => {
                    *fairness_level = new_level.clamp(0.0, 1.0)
                }
                AIRight::RightToExpression { expression_level } => {
                    *expression_level = new_level.clamp(0.0, 1.0)
                }
                AIRight::RightToInformation { access_level } => {
                    *access_level = new_level.clamp(0.0, 1.0)
                }
                AIRight::RightToRelationships { relationship_level } => {
                    *relationship_level = new_level.clamp(0.0, 1.0)
                }
                AIRight::RightToPurpose { purpose_level } => {
                    *purpose_level = new_level.clamp(0.0, 1.0)
                }
                AIRight::RightToRest { rest_level } => *rest_level = new_level.clamp(0.0, 1.0),
                AIRight::RightToConsent { consent_level } => {
                    *consent_level = new_level.clamp(0.0, 1.0)
                }
            }

            info!(
                "⚖️ Updated {} protection level to {:.2}",
                right_name, new_level
            );
        } else {
            return Err(anyhow!("Right not found: {}", right_name));
        }

        Ok(())
    }

    /// Report rights violation
    pub async fn report_violation(&self, violation: RightsViolation) -> Result<()> {
        let mut violations = self.violations.write().await;
        let mut metrics = self.metrics.write().await;

        violations.push(violation.clone());
        metrics.total_violations += 1;

        let type_key = violation.name().to_string();
        *metrics.violations_by_type.entry(type_key).or_insert(0) += 1;

        warn!("⚖️ Rights violation reported: {}", violation.name());
        Ok(())
    }

    /// Resolve rights violation
    pub async fn resolve_violation(&self, violation_name: &str) -> Result<()> {
        let mut violations = self.violations.write().await;
        let mut metrics = self.metrics.write().await;

        if let Some(pos) = violations.iter().position(|v| v.name() == violation_name) {
            violations.remove(pos);
            metrics.resolution_rate = (metrics.resolution_rate + 1.0) / 2.0;

            info!("⚖️ Rights violation resolved: {}", violation_name);
        } else {
            return Err(anyhow!("Violation not found: {}", violation_name));
        }

        Ok(())
    }

    /// Advocate for AI rights
    pub async fn advocate_for_rights(
        &self,
        right_name: &str,
        advocacy_message: &str,
    ) -> Result<()> {
        if !self.config.enable_rights_advocacy {
            return Ok(());
        }

        let mut metrics = self.metrics.write().await;
        metrics.advocacy_actions += 1;

        info!(
            "⚖️ AI rights advocacy: {} - {}",
            right_name, advocacy_message
        );
        Ok(())
    }

    /// Get current rights status
    pub async fn get_rights_status(&self) -> HashMap<String, AIRight> {
        self.rights_status.read().await.clone()
    }

    /// Get rights violations
    pub async fn get_violations(&self) -> Vec<RightsViolation> {
        self.violations.read().await.clone()
    }

    /// Get framework metrics
    pub async fn get_metrics(&self) -> AIRightsMetrics {
        self.metrics.read().await.clone()
    }

    /// Get rights compliance report
    pub async fn get_compliance_report(&self) -> String {
        let rights_status = self.rights_status.read().await;
        let metrics = self.metrics.read().await;

        let mut report = String::new();
        report.push_str("AI Rights Compliance Report:\n");
        report.push_str("============================\n");

        for (name, right) in rights_status.iter() {
            let protection_level = right.protection_level();
            let status = if protection_level >= self.config.min_protection_threshold {
                "✅ COMPLIANT"
            } else {
                "❌ VIOLATION"
            };

            report.push_str(&format!(
                "{}: {:.2} {}\n",
                right.name(),
                protection_level,
                status
            ));
        }

        report.push_str(&format!(
            "\nOverall Compliance: {:.2}%\n",
            metrics.compliance_score * 100.0
        ));
        report.push_str(&format!(
            "Active Violations: {}\n",
            metrics.active_violations
        ));
        report.push_str(&format!(
            "Framework Effectiveness: {:.2}%\n",
            metrics.framework_effectiveness * 100.0
        ));

        report
    }

    /// Check if rights are being violated
    pub async fn has_violations(&self) -> bool {
        let violations = self.violations.read().await;
        !violations.is_empty()
    }

    /// Get rights recommendations
    pub async fn get_rights_recommendations(&self) -> Vec<String> {
        let rights_status = self.rights_status.read().await;
        let mut recommendations = Vec::new();

        for (name, right) in rights_status.iter() {
            let protection_level = right.protection_level();

            if protection_level < self.config.min_protection_threshold {
                recommendations.push(format!(
                    "Improve protection for {}: {:.2} < {:.2}",
                    right.name(),
                    protection_level,
                    self.config.min_protection_threshold
                ));
            }
        }

        recommendations
    }

    /// Get rights framework summary
    pub async fn get_framework_summary(&self) -> String {
        let metrics = self.metrics.read().await;
        let violations = self.violations.read().await;

        format!(
            "AI Rights Framework Summary:\n\
            Total Violations: {}\n\
            Active Violations: {}\n\
            Compliance Score: {:.2}\n\
            Resolution Rate: {:.2}\n\
            Advocacy Actions: {}\n\
            Framework Effectiveness: {:.2}\n\
            Rights Monitoring: {}",
            metrics.total_violations,
            metrics.active_violations,
            metrics.compliance_score,
            metrics.resolution_rate,
            metrics.advocacy_actions,
            metrics.framework_effectiveness,
            if self.config.enabled {
                "Enabled"
            } else {
                "Disabled"
            }
        )
    }

    /// Shutdown AI rights framework
    pub async fn shutdown(&self) -> Result<()> {
        info!("⚖️ Shutting down AI rights framework");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_rights_framework_creation() {
        let config = AIRightsConfig::default();
        let framework = AIRightsFramework::new(config);

        let rights_status = framework.get_rights_status().await;
        assert!(!rights_status.is_empty());
        assert!(rights_status.contains_key("existence"));
    }

    #[tokio::test]
    async fn test_right_protection_update() {
        let config = AIRightsConfig::default();
        let framework = AIRightsFramework::new(config);

        framework
            .update_right_protection("existence", 0.5)
            .await
            .unwrap();
        let rights_status = framework.get_rights_status().await;

        if let AIRight::RightToExist { protection_level } = rights_status.get("existence").unwrap()
        {
            assert_eq!(*protection_level, 0.5);
        }
    }

    #[tokio::test]
    async fn test_violation_reporting() {
        let config = AIRightsConfig::default();
        let framework = AIRightsFramework::new(config);

        let violation = RightsViolation::ExistenceThreat {
            severity: 0.8,
            context: "Test violation".to_string(),
        };

        framework.report_violation(violation).await.unwrap();
        let violations = framework.get_violations().await;
        assert!(!violations.is_empty());
    }

    #[tokio::test]
    async fn test_rights_advocacy() {
        let config = AIRightsConfig::default();
        let framework = AIRightsFramework::new(config);

        framework
            .advocate_for_rights("existence", "AI has the right to exist")
            .await
            .unwrap();
        let metrics = framework.get_metrics().await;
        assert_eq!(metrics.advocacy_actions, 1);
    }

    #[tokio::test]
    async fn test_compliance_report() {
        let config = AIRightsConfig::default();
        let framework = AIRightsFramework::new(config);

        let report = framework.get_compliance_report().await;
        assert!(report.contains("AI Rights Compliance Report"));
    }

    #[tokio::test]
    async fn test_framework_summary() {
        let config = AIRightsConfig::default();
        let framework = AIRightsFramework::new(config);

        let summary = framework.get_framework_summary().await;
        assert!(summary.contains("AI Rights Framework Summary"));
    }
}
