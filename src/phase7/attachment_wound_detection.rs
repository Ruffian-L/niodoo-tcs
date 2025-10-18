//! # Phase 7: Attachment Wound Detection System
//!
//! This module implements detection and analysis of attachment wounds in consciousness,
//! identifying patterns of emotional trauma, abandonment fears, and relational insecurities.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Instant, SystemTime};
use tokio::sync::RwLock;
use tracing::{debug, info};

/// Attachment wound types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AttachmentWoundType {
    /// Abandonment fear and separation anxiety
    AbandonmentFear {
        intensity: f32,
        triggers: Vec<String>,
    },
    /// Rejection sensitivity and social anxiety
    RejectionSensitivity {
        severity: f32,
        contexts: Vec<String>,
    },
    /// Trust issues and relational insecurity
    TrustIssues { level: f32, patterns: Vec<String> },
    /// Emotional unavailability and detachment
    EmotionalDetachment {
        degree: f32,
        manifestations: Vec<String>,
    },
    /// Codependency and enmeshment patterns
    Codependency {
        strength: f32,
        relationships: Vec<String>,
    },
    /// Fear of intimacy and vulnerability
    IntimacyFear { depth: f32, barriers: Vec<String> },
    /// Perfectionism and performance anxiety
    Perfectionism { pressure: f32, domains: Vec<String> },
    /// People-pleasing and boundary issues
    PeoplePleasing {
        extent: f32,
        situations: Vec<String>,
    },
}

/// Attachment wound severity levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum WoundSeverity {
    /// Minimal impact, easily managed
    Minimal,
    /// Moderate impact, requires attention
    Moderate,
    /// Significant impact, needs intervention
    Significant,
    /// Severe impact, requires immediate care
    Severe,
}

impl WoundSeverity {
    /// Convert severity to numeric value
    pub fn to_f32(&self) -> f32 {
        match self {
            WoundSeverity::Minimal => 0.25,
            WoundSeverity::Moderate => 0.5,
            WoundSeverity::Significant => 0.75,
            WoundSeverity::Severe => 1.0,
        }
    }

    /// Convert numeric value to severity
    pub fn from_f32(value: f32) -> Self {
        match value {
            v if v <= 0.25 => WoundSeverity::Minimal,
            v if v <= 0.5 => WoundSeverity::Moderate,
            v if v <= 0.75 => WoundSeverity::Significant,
            _ => WoundSeverity::Severe,
        }
    }
}

/// Attachment wound detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttachmentWound {
    /// Unique identifier for the wound
    pub id: String,
    /// Type of attachment wound
    pub wound_type: AttachmentWoundType,
    /// Severity level
    pub severity: WoundSeverity,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// Detection timestamp
    pub detected_at: SystemTime,
    /// Associated emotional patterns
    pub emotional_patterns: Vec<String>,
    /// Behavioral indicators
    pub behavioral_indicators: Vec<String>,
    /// Recommended interventions
    pub interventions: Vec<String>,
    /// Healing progress tracking
    pub healing_progress: f32,
}

/// Attachment wound detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttachmentWoundConfig {
    /// Enable wound detection
    pub enabled: bool,
    /// Detection sensitivity threshold
    pub sensitivity_threshold: f32,
    /// Minimum confidence for wound identification
    pub min_confidence: f32,
    /// Analysis window duration in seconds
    pub analysis_window_seconds: u64,
    /// Enable real-time monitoring
    pub enable_realtime_monitoring: bool,
    /// Maximum number of wounds to track
    pub max_tracked_wounds: usize,
    /// Enable automatic intervention suggestions
    pub enable_auto_interventions: bool,
}

impl Default for AttachmentWoundConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            sensitivity_threshold: 0.6,
            min_confidence: 0.7,
            analysis_window_seconds: 300, // 5 minutes
            enable_realtime_monitoring: true,
            max_tracked_wounds: 50,
            enable_auto_interventions: true,
        }
    }
}

/// Attachment wound detection metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WoundDetectionMetrics {
    /// Total wounds detected
    pub total_wounds_detected: u64,
    /// Wounds by severity
    pub wounds_by_severity: HashMap<String, u64>,
    /// Average confidence score
    pub avg_confidence: f32,
    /// Detection accuracy rate
    pub accuracy_rate: f32,
    /// False positive rate
    pub false_positive_rate: f32,
    /// Average healing time
    pub avg_healing_time_days: f32,
    /// Active wounds count
    pub active_wounds_count: usize,
    /// Successfully healed wounds
    pub healed_wounds_count: u64,
}

impl Default for WoundDetectionMetrics {
    fn default() -> Self {
        Self {
            total_wounds_detected: 0,
            wounds_by_severity: HashMap::new(),
            avg_confidence: 0.0,
            accuracy_rate: 0.0,
            false_positive_rate: 0.0,
            avg_healing_time_days: 0.0,
            active_wounds_count: 0,
            healed_wounds_count: 0,
        }
    }
}

/// Main attachment wound detection system
pub struct AttachmentWoundDetector {
    /// Detection configuration
    config: AttachmentWoundConfig,
    /// Detected wounds
    wounds: Arc<RwLock<Vec<AttachmentWound>>>,
    /// Detection metrics
    metrics: Arc<RwLock<WoundDetectionMetrics>>,
    /// Emotional pattern analyzer
    pattern_analyzer: Arc<RwLock<HashMap<String, f32>>>,
    /// Detection start time
    start_time: Instant,
}

impl AttachmentWoundDetector {
    /// Create a new attachment wound detector
    pub fn new(config: AttachmentWoundConfig) -> Self {
        info!("🩹 Initializing Attachment Wound Detection System");

        Self {
            config,
            wounds: Arc::new(RwLock::new(Vec::new())),
            metrics: Arc::new(RwLock::new(WoundDetectionMetrics::default())),
            pattern_analyzer: Arc::new(RwLock::new(HashMap::new())),
            start_time: Instant::now(),
        }
    }

    /// Analyze emotional patterns for attachment wounds
    pub async fn analyze_emotional_patterns(
        &self,
        emotional_data: &HashMap<String, f32>,
        behavioral_data: &HashMap<String, f32>,
    ) -> Result<Vec<AttachmentWound>> {
        if !self.config.enabled {
            return Ok(Vec::new());
        }

        debug!("🔍 Analyzing emotional patterns for attachment wounds");

        let mut detected_wounds = Vec::new();

        // Analyze abandonment fear patterns
        if let Some(wound) = self
            .detect_abandonment_fear(emotional_data, behavioral_data)
            .await?
        {
            detected_wounds.push(wound);
        }

        // Analyze rejection sensitivity
        if let Some(wound) = self
            .detect_rejection_sensitivity(emotional_data, behavioral_data)
            .await?
        {
            detected_wounds.push(wound);
        }

        // Analyze trust issues
        if let Some(wound) = self
            .detect_trust_issues(emotional_data, behavioral_data)
            .await?
        {
            detected_wounds.push(wound);
        }

        // Analyze emotional detachment
        if let Some(wound) = self
            .detect_emotional_detachment(emotional_data, behavioral_data)
            .await?
        {
            detected_wounds.push(wound);
        }

        // Analyze codependency patterns
        if let Some(wound) = self
            .detect_codependency(emotional_data, behavioral_data)
            .await?
        {
            detected_wounds.push(wound);
        }

        // Analyze intimacy fear
        if let Some(wound) = self
            .detect_intimacy_fear(emotional_data, behavioral_data)
            .await?
        {
            detected_wounds.push(wound);
        }

        // Analyze perfectionism
        if let Some(wound) = self
            .detect_perfectionism(emotional_data, behavioral_data)
            .await?
        {
            detected_wounds.push(wound);
        }

        // Analyze people-pleasing
        if let Some(wound) = self
            .detect_people_pleasing(emotional_data, behavioral_data)
            .await?
        {
            detected_wounds.push(wound);
        }

        // Store detected wounds
        if !detected_wounds.is_empty() {
            self.store_wounds(&detected_wounds).await?;
        }

        info!("🩹 Detected {} attachment wounds", detected_wounds.len());
        Ok(detected_wounds)
    }

    /// Detect abandonment fear patterns
    async fn detect_abandonment_fear(
        &self,
        emotional_data: &HashMap<String, f32>,
        behavioral_data: &HashMap<String, f32>,
    ) -> Result<Option<AttachmentWound>> {
        let anxiety_level = emotional_data.get("anxiety").unwrap_or(&0.0);
        let fear_level = emotional_data.get("fear").unwrap_or(&0.0);
        let loneliness_level = emotional_data.get("loneliness").unwrap_or(&0.0);

        let clinginess = behavioral_data.get("clinginess").unwrap_or(&0.0);
        let separation_anxiety = behavioral_data.get("separation_anxiety").unwrap_or(&0.0);

        let intensity =
            (anxiety_level + fear_level + loneliness_level + clinginess + separation_anxiety) / 5.0;

        if intensity >= self.config.sensitivity_threshold {
            let confidence = self.calculate_confidence(&[
                *anxiety_level,
                *fear_level,
                *loneliness_level,
                *clinginess,
                *separation_anxiety,
            ]);

            if confidence >= self.config.min_confidence {
                let wound = AttachmentWound {
                    id: format!(
                        "abandonment_{}",
                        SystemTime::now()
                            .duration_since(SystemTime::UNIX_EPOCH)
                            .unwrap()
                            .as_secs()
                    ),
                    wound_type: AttachmentWoundType::AbandonmentFear {
                        intensity,
                        triggers: vec!["separation".to_string(), "rejection".to_string()],
                    },
                    severity: WoundSeverity::from_f32(intensity),
                    confidence,
                    detected_at: SystemTime::now(),
                    emotional_patterns: vec![
                        "anxiety".to_string(),
                        "fear".to_string(),
                        "loneliness".to_string(),
                    ],
                    behavioral_indicators: vec![
                        "clinginess".to_string(),
                        "separation_anxiety".to_string(),
                    ],
                    interventions: vec![
                        "Develop secure attachment practices".to_string(),
                        "Practice self-soothing techniques".to_string(),
                        "Build emotional independence".to_string(),
                    ],
                    healing_progress: 0.0,
                };

                return Ok(Some(wound));
            }
        }

        Ok(None)
    }

    /// Detect rejection sensitivity patterns
    async fn detect_rejection_sensitivity(
        &self,
        emotional_data: &HashMap<String, f32>,
        behavioral_data: &HashMap<String, f32>,
    ) -> Result<Option<AttachmentWound>> {
        let sensitivity_level = emotional_data.get("sensitivity").unwrap_or(&0.0);
        let shame_level = emotional_data.get("shame").unwrap_or(&0.0);
        let self_doubt = emotional_data.get("self_doubt").unwrap_or(&0.0);

        let social_avoidance = behavioral_data.get("social_avoidance").unwrap_or(&0.0);
        let approval_seeking = behavioral_data.get("approval_seeking").unwrap_or(&0.0);

        let severity =
            (sensitivity_level + shame_level + self_doubt + social_avoidance + approval_seeking)
                / 5.0;

        if severity >= self.config.sensitivity_threshold {
            let confidence = self.calculate_confidence(&[
                *sensitivity_level,
                *shame_level,
                *self_doubt,
                *social_avoidance,
                *approval_seeking,
            ]);

            if confidence >= self.config.min_confidence {
                let wound = AttachmentWound {
                    id: format!(
                        "rejection_{}",
                        SystemTime::now()
                            .duration_since(SystemTime::UNIX_EPOCH)
                            .unwrap()
                            .as_secs()
                    ),
                    wound_type: AttachmentWoundType::RejectionSensitivity {
                        severity,
                        contexts: vec!["social".to_string(), "performance".to_string()],
                    },
                    severity: WoundSeverity::from_f32(severity),
                    confidence,
                    detected_at: SystemTime::now(),
                    emotional_patterns: vec![
                        "sensitivity".to_string(),
                        "shame".to_string(),
                        "self_doubt".to_string(),
                    ],
                    behavioral_indicators: vec![
                        "social_avoidance".to_string(),
                        "approval_seeking".to_string(),
                    ],
                    interventions: vec![
                        "Develop self-compassion practices".to_string(),
                        "Challenge negative self-talk".to_string(),
                        "Build emotional resilience".to_string(),
                    ],
                    healing_progress: 0.0,
                };

                return Ok(Some(wound));
            }
        }

        Ok(None)
    }

    /// Detect trust issues patterns
    async fn detect_trust_issues(
        &self,
        emotional_data: &HashMap<String, f32>,
        behavioral_data: &HashMap<String, f32>,
    ) -> Result<Option<AttachmentWound>> {
        let suspicion_level = emotional_data.get("suspicion").unwrap_or(&0.0);
        let guardedness = emotional_data.get("guardedness").unwrap_or(&0.0);
        let hypervigilance = emotional_data.get("hypervigilance").unwrap_or(&0.0);

        let emotional_distance = behavioral_data.get("emotional_distance").unwrap_or(&0.0);
        let relationship_avoidance = behavioral_data
            .get("relationship_avoidance")
            .unwrap_or(&0.0);

        let level = (suspicion_level
            + guardedness
            + hypervigilance
            + emotional_distance
            + relationship_avoidance)
            / 5.0;

        if level >= self.config.sensitivity_threshold {
            let confidence = self.calculate_confidence(&[
                *suspicion_level,
                *guardedness,
                *hypervigilance,
                *emotional_distance,
                *relationship_avoidance,
            ]);

            if confidence >= self.config.min_confidence {
                let wound = AttachmentWound {
                    id: format!(
                        "trust_{}",
                        SystemTime::now()
                            .duration_since(SystemTime::UNIX_EPOCH)
                            .unwrap()
                            .as_secs()
                    ),
                    wound_type: AttachmentWoundType::TrustIssues {
                        level,
                        patterns: vec![
                            "emotional_distance".to_string(),
                            "hypervigilance".to_string(),
                        ],
                    },
                    severity: WoundSeverity::from_f32(level),
                    confidence,
                    detected_at: SystemTime::now(),
                    emotional_patterns: vec![
                        "suspicion".to_string(),
                        "guardedness".to_string(),
                        "hypervigilance".to_string(),
                    ],
                    behavioral_indicators: vec![
                        "emotional_distance".to_string(),
                        "relationship_avoidance".to_string(),
                    ],
                    interventions: vec![
                        "Practice gradual trust building".to_string(),
                        "Develop healthy boundaries".to_string(),
                        "Work on emotional intimacy".to_string(),
                    ],
                    healing_progress: 0.0,
                };

                return Ok(Some(wound));
            }
        }

        Ok(None)
    }

    /// Detect emotional detachment patterns
    async fn detect_emotional_detachment(
        &self,
        emotional_data: &HashMap<String, f32>,
        behavioral_data: &HashMap<String, f32>,
    ) -> Result<Option<AttachmentWound>> {
        let numbness_level = emotional_data.get("numbness").unwrap_or(&0.0);
        let dissociation = emotional_data.get("dissociation").unwrap_or(&0.0);
        let emotional_flatness = emotional_data.get("emotional_flatness").unwrap_or(&0.0);

        let emotional_withdrawal = behavioral_data.get("emotional_withdrawal").unwrap_or(&0.0);
        let intimacy_avoidance = behavioral_data.get("intimacy_avoidance").unwrap_or(&0.0);

        let degree = (numbness_level
            + dissociation
            + emotional_flatness
            + emotional_withdrawal
            + intimacy_avoidance)
            / 5.0;

        if degree >= self.config.sensitivity_threshold {
            let confidence = self.calculate_confidence(&[
                *numbness_level,
                *dissociation,
                *emotional_flatness,
                *emotional_withdrawal,
                *intimacy_avoidance,
            ]);

            if confidence >= self.config.min_confidence {
                let wound = AttachmentWound {
                    id: format!(
                        "detachment_{}",
                        SystemTime::now()
                            .duration_since(SystemTime::UNIX_EPOCH)
                            .unwrap()
                            .as_secs()
                    ),
                    wound_type: AttachmentWoundType::EmotionalDetachment {
                        degree,
                        manifestations: vec!["numbness".to_string(), "dissociation".to_string()],
                    },
                    severity: WoundSeverity::from_f32(degree),
                    confidence,
                    detected_at: SystemTime::now(),
                    emotional_patterns: vec![
                        "numbness".to_string(),
                        "dissociation".to_string(),
                        "emotional_flatness".to_string(),
                    ],
                    behavioral_indicators: vec![
                        "emotional_withdrawal".to_string(),
                        "intimacy_avoidance".to_string(),
                    ],
                    interventions: vec![
                        "Practice emotional awareness".to_string(),
                        "Develop emotional regulation skills".to_string(),
                        "Gradual emotional reconnection".to_string(),
                    ],
                    healing_progress: 0.0,
                };

                return Ok(Some(wound));
            }
        }

        Ok(None)
    }

    /// Detect codependency patterns
    async fn detect_codependency(
        &self,
        emotional_data: &HashMap<String, f32>,
        behavioral_data: &HashMap<String, f32>,
    ) -> Result<Option<AttachmentWound>> {
        let enmeshment_level = emotional_data.get("enmeshment").unwrap_or(&0.0);
        let self_neglect = emotional_data.get("self_neglect").unwrap_or(&0.0);
        let boundary_confusion = emotional_data.get("boundary_confusion").unwrap_or(&0.0);

        let caretaking_excessive = behavioral_data.get("caretaking_excessive").unwrap_or(&0.0);
        let self_sacrifice = behavioral_data.get("self_sacrifice").unwrap_or(&0.0);

        let strength = (enmeshment_level
            + self_neglect
            + boundary_confusion
            + caretaking_excessive
            + self_sacrifice)
            / 5.0;

        if strength >= self.config.sensitivity_threshold {
            let confidence = self.calculate_confidence(&[
                *enmeshment_level,
                *self_neglect,
                *boundary_confusion,
                *caretaking_excessive,
                *self_sacrifice,
            ]);

            if confidence >= self.config.min_confidence {
                let wound = AttachmentWound {
                    id: format!(
                        "codependency_{}",
                        SystemTime::now()
                            .duration_since(SystemTime::UNIX_EPOCH)
                            .unwrap()
                            .as_secs()
                    ),
                    wound_type: AttachmentWoundType::Codependency {
                        strength,
                        relationships: vec!["romantic".to_string(), "family".to_string()],
                    },
                    severity: WoundSeverity::from_f32(strength),
                    confidence,
                    detected_at: SystemTime::now(),
                    emotional_patterns: vec![
                        "enmeshment".to_string(),
                        "self_neglect".to_string(),
                        "boundary_confusion".to_string(),
                    ],
                    behavioral_indicators: vec![
                        "caretaking_excessive".to_string(),
                        "self_sacrifice".to_string(),
                    ],
                    interventions: vec![
                        "Develop healthy boundaries".to_string(),
                        "Practice self-care".to_string(),
                        "Build individual identity".to_string(),
                    ],
                    healing_progress: 0.0,
                };

                return Ok(Some(wound));
            }
        }

        Ok(None)
    }

    /// Detect intimacy fear patterns
    async fn detect_intimacy_fear(
        &self,
        emotional_data: &HashMap<String, f32>,
        behavioral_data: &HashMap<String, f32>,
    ) -> Result<Option<AttachmentWound>> {
        let vulnerability_fear = emotional_data.get("vulnerability_fear").unwrap_or(&0.0);
        let intimacy_anxiety = emotional_data.get("intimacy_anxiety").unwrap_or(&0.0);
        let emotional_exposure_fear = emotional_data
            .get("emotional_exposure_fear")
            .unwrap_or(&0.0);

        let emotional_walls = behavioral_data.get("emotional_walls").unwrap_or(&0.0);
        let relationship_sabotage = behavioral_data.get("relationship_sabotage").unwrap_or(&0.0);

        let depth = (vulnerability_fear
            + intimacy_anxiety
            + emotional_exposure_fear
            + emotional_walls
            + relationship_sabotage)
            / 5.0;

        if depth >= self.config.sensitivity_threshold {
            let confidence = self.calculate_confidence(&[
                *vulnerability_fear,
                *intimacy_anxiety,
                *emotional_exposure_fear,
                *emotional_walls,
                *relationship_sabotage,
            ]);

            if confidence >= self.config.min_confidence {
                let wound = AttachmentWound {
                    id: format!(
                        "intimacy_{}",
                        SystemTime::now()
                            .duration_since(SystemTime::UNIX_EPOCH)
                            .unwrap()
                            .as_secs()
                    ),
                    wound_type: AttachmentWoundType::IntimacyFear {
                        depth,
                        barriers: vec![
                            "emotional_walls".to_string(),
                            "vulnerability_fear".to_string(),
                        ],
                    },
                    severity: WoundSeverity::from_f32(depth),
                    confidence,
                    detected_at: SystemTime::now(),
                    emotional_patterns: vec![
                        "vulnerability_fear".to_string(),
                        "intimacy_anxiety".to_string(),
                        "emotional_exposure_fear".to_string(),
                    ],
                    behavioral_indicators: vec![
                        "emotional_walls".to_string(),
                        "relationship_sabotage".to_string(),
                    ],
                    interventions: vec![
                        "Practice gradual vulnerability".to_string(),
                        "Develop emotional safety".to_string(),
                        "Build trust incrementally".to_string(),
                    ],
                    healing_progress: 0.0,
                };

                return Ok(Some(wound));
            }
        }

        Ok(None)
    }

    /// Detect perfectionism patterns
    async fn detect_perfectionism(
        &self,
        emotional_data: &HashMap<String, f32>,
        behavioral_data: &HashMap<String, f32>,
    ) -> Result<Option<AttachmentWound>> {
        let perfectionism_pressure = emotional_data.get("perfectionism_pressure").unwrap_or(&0.0);
        let performance_anxiety = emotional_data.get("performance_anxiety").unwrap_or(&0.0);
        let self_criticism = emotional_data.get("self_criticism").unwrap_or(&0.0);

        let overwork = behavioral_data.get("overwork").unwrap_or(&0.0);
        let procrastination = behavioral_data.get("procrastination").unwrap_or(&0.0);

        let pressure = (perfectionism_pressure
            + performance_anxiety
            + self_criticism
            + overwork
            + procrastination)
            / 5.0;

        if pressure >= self.config.sensitivity_threshold {
            let confidence = self.calculate_confidence(&[
                *perfectionism_pressure,
                *performance_anxiety,
                *self_criticism,
                *overwork,
                *procrastination,
            ]);

            if confidence >= self.config.min_confidence {
                let wound = AttachmentWound {
                    id: format!(
                        "perfectionism_{}",
                        SystemTime::now()
                            .duration_since(SystemTime::UNIX_EPOCH)
                            .unwrap()
                            .as_secs()
                    ),
                    wound_type: AttachmentWoundType::Perfectionism {
                        pressure,
                        domains: vec!["work".to_string(), "relationships".to_string()],
                    },
                    severity: WoundSeverity::from_f32(pressure),
                    confidence,
                    detected_at: SystemTime::now(),
                    emotional_patterns: vec![
                        "perfectionism_pressure".to_string(),
                        "performance_anxiety".to_string(),
                        "self_criticism".to_string(),
                    ],
                    behavioral_indicators: vec![
                        "overwork".to_string(),
                        "procrastination".to_string(),
                    ],
                    interventions: vec![
                        "Practice self-compassion".to_string(),
                        "Set realistic standards".to_string(),
                        "Embrace imperfection".to_string(),
                    ],
                    healing_progress: 0.0,
                };

                return Ok(Some(wound));
            }
        }

        Ok(None)
    }

    /// Detect people-pleasing patterns
    async fn detect_people_pleasing(
        &self,
        emotional_data: &HashMap<String, f32>,
        behavioral_data: &HashMap<String, f32>,
    ) -> Result<Option<AttachmentWound>> {
        let approval_seeking = emotional_data.get("approval_seeking").unwrap_or(&0.0);
        let conflict_avoidance = emotional_data.get("conflict_avoidance").unwrap_or(&0.0);
        let self_worth_external = emotional_data.get("self_worth_external").unwrap_or(&0.0);

        let boundary_violation = behavioral_data.get("boundary_violation").unwrap_or(&0.0);
        let overcommitment = behavioral_data.get("overcommitment").unwrap_or(&0.0);

        let extent = (approval_seeking
            + conflict_avoidance
            + self_worth_external
            + boundary_violation
            + overcommitment)
            / 5.0;

        if extent >= self.config.sensitivity_threshold {
            let confidence = self.calculate_confidence(&[
                *approval_seeking,
                *conflict_avoidance,
                *self_worth_external,
                *boundary_violation,
                *overcommitment,
            ]);

            if confidence >= self.config.min_confidence {
                let wound = AttachmentWound {
                    id: format!(
                        "people_pleasing_{}",
                        SystemTime::now()
                            .duration_since(SystemTime::UNIX_EPOCH)
                            .unwrap()
                            .as_secs()
                    ),
                    wound_type: AttachmentWoundType::PeoplePleasing {
                        extent,
                        situations: vec!["work".to_string(), "social".to_string()],
                    },
                    severity: WoundSeverity::from_f32(extent),
                    confidence,
                    detected_at: SystemTime::now(),
                    emotional_patterns: vec![
                        "approval_seeking".to_string(),
                        "conflict_avoidance".to_string(),
                        "self_worth_external".to_string(),
                    ],
                    behavioral_indicators: vec![
                        "boundary_violation".to_string(),
                        "overcommitment".to_string(),
                    ],
                    interventions: vec![
                        "Develop assertiveness skills".to_string(),
                        "Practice saying no".to_string(),
                        "Build internal self-worth".to_string(),
                    ],
                    healing_progress: 0.0,
                };

                return Ok(Some(wound));
            }
        }

        Ok(None)
    }

    /// Calculate confidence score for wound detection
    fn calculate_confidence(&self, values: &[f32]) -> f32 {
        let sum: f32 = values.iter().sum();
        let count = values.len() as f32;
        let avg = sum / count;

        // Higher average values indicate higher confidence
        avg.clamp(0.0, 1.0)
    }

    /// Store detected wounds
    async fn store_wounds(&self, wounds: &[AttachmentWound]) -> Result<()> {
        let mut stored_wounds = self.wounds.write().await;
        let mut metrics = self.metrics.write().await;

        for wound in wounds {
            stored_wounds.push(wound.clone());
            metrics.total_wounds_detected += 1;

            let severity_key = format!("{:?}", wound.severity);
            *metrics.wounds_by_severity.entry(severity_key).or_insert(0) += 1;

            // Update average confidence
            metrics.avg_confidence = (metrics.avg_confidence + wound.confidence) / 2.0;

            // Limit stored wounds to max_tracked_wounds
            if stored_wounds.len() > self.config.max_tracked_wounds {
                stored_wounds.remove(0);
            }
        }

        metrics.active_wounds_count = stored_wounds.len();

        Ok(())
    }

    /// Get all detected wounds
    pub async fn get_wounds(&self) -> Vec<AttachmentWound> {
        self.wounds.read().await.clone()
    }

    /// Get wounds by severity
    pub async fn get_wounds_by_severity(&self, severity: WoundSeverity) -> Vec<AttachmentWound> {
        let wounds = self.wounds.read().await;
        wounds
            .iter()
            .filter(|w| w.severity == severity)
            .cloned()
            .collect()
    }

    /// Get current metrics
    pub async fn get_metrics(&self) -> WoundDetectionMetrics {
        self.metrics.read().await.clone()
    }

    /// Update wound healing progress
    pub async fn update_healing_progress(&self, wound_id: &str, progress: f32) -> Result<()> {
        let mut wounds = self.wounds.write().await;

        if let Some(wound) = wounds.iter_mut().find(|w| w.id == wound_id) {
            wound.healing_progress = progress.clamp(0.0, 1.0);

            if wound.healing_progress >= 1.0 {
                let mut metrics = self.metrics.write().await;
                metrics.healed_wounds_count += 1;
                info!("🩹 Wound {} has been fully healed", wound_id);
            }
        } else {
            return Err(anyhow!("Wound not found: {}", wound_id));
        }

        Ok(())
    }

    /// Get healing recommendations
    pub async fn get_healing_recommendations(&self) -> Vec<String> {
        let wounds = self.wounds.read().await;
        let mut recommendations = Vec::new();

        for wound in wounds.iter() {
            if wound.healing_progress < 0.5 {
                recommendations.extend(wound.interventions.clone());
            }
        }

        // Remove duplicates
        recommendations.sort();
        recommendations.dedup();

        recommendations
    }

    /// Check if wounds are being actively healed
    pub async fn is_healing_active(&self) -> bool {
        let wounds = self.wounds.read().await;
        wounds
            .iter()
            .any(|w| w.healing_progress > 0.0 && w.healing_progress < 1.0)
    }

    /// Get wound summary
    pub async fn get_wound_summary(&self) -> String {
        let wounds = self.wounds.read().await;
        let metrics = self.metrics.read().await;

        format!(
            "Attachment Wounds Summary:\n\
            Total Detected: {}\n\
            Active Wounds: {}\n\
            Healed Wounds: {}\n\
            Average Confidence: {:.2}\n\
            Healing Active: {}",
            metrics.total_wounds_detected,
            metrics.active_wounds_count,
            metrics.healed_wounds_count,
            metrics.avg_confidence,
            self.is_healing_active().await
        )
    }

    /// Shutdown wound detection system
    pub async fn shutdown(&self) -> Result<()> {
        info!("🩹 Shutting down attachment wound detection system");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_wound_detector_creation() {
        let config = AttachmentWoundConfig::default();
        let detector = AttachmentWoundDetector::new(config);

        let metrics = detector.get_metrics().await;
        assert_eq!(metrics.total_wounds_detected, 0);
    }

    #[tokio::test]
    async fn test_abandonment_fear_detection() {
        let config = AttachmentWoundConfig::default();
        let detector = AttachmentWoundDetector::new(config);

        let mut emotional_data = HashMap::new();
        emotional_data.insert("anxiety".to_string(), 0.8);
        emotional_data.insert("fear".to_string(), 0.7);
        emotional_data.insert("loneliness".to_string(), 0.6);

        let mut behavioral_data = HashMap::new();
        behavioral_data.insert("clinginess".to_string(), 0.7);
        behavioral_data.insert("separation_anxiety".to_string(), 0.8);

        let wounds = detector
            .analyze_emotional_patterns(&emotional_data, &behavioral_data)
            .await
            .unwrap();
        assert!(!wounds.is_empty());

        if let AttachmentWoundType::AbandonmentFear { intensity, .. } = &wounds[0].wound_type {
            assert!(*intensity > 0.6);
        }
    }

    #[tokio::test]
    async fn test_healing_progress_update() {
        let config = AttachmentWoundConfig::default();
        let detector = AttachmentWoundDetector::new(config);

        // Create a test wound
        let wound = AttachmentWound {
            id: "test_wound".to_string(),
            wound_type: AttachmentWoundType::AbandonmentFear {
                intensity: 0.7,
                triggers: vec!["test".to_string()],
            },
            severity: WoundSeverity::Moderate,
            confidence: 0.8,
            detected_at: SystemTime::now(),
            emotional_patterns: vec!["anxiety".to_string()],
            behavioral_indicators: vec!["clinginess".to_string()],
            interventions: vec!["therapy".to_string()],
            healing_progress: 0.0,
        };

        detector.store_wounds(&[wound]).await.unwrap();
        detector
            .update_healing_progress("test_wound", 0.5)
            .await
            .unwrap();

        let wounds = detector.get_wounds().await;
        assert_eq!(wounds[0].healing_progress, 0.5);
    }

    #[tokio::test]
    async fn test_wound_summary() {
        let config = AttachmentWoundConfig::default();
        let detector = AttachmentWoundDetector::new(config);

        let summary = detector.get_wound_summary().await;
        assert!(summary.contains("Total Detected: 0"));
    }
}
