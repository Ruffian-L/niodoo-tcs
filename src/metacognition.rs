//! # Metacognition Engine for Ethical Decision Making and Self-Reflection
//!
//! This module provides metacognitive capabilities for the consciousness system,
//! enabling ethical reflection and self-aware decision making processes.

use crate::consciousness::{ConsciousnessState, EmotionType};
use crate::error::{CandleFeelingError, CandleFeelingResult as Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime};
use uuid::Uuid;

/// Metacognitive decision for ethical reflection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetacognitiveDecision {
    /// Decision identifier
    pub id: Uuid,
    /// Decision type
    pub decision_type: DecisionType,
    /// Confidence level in the decision (0.0-1.0)
    pub confidence: f32,
    /// Ethical considerations
    pub ethical_considerations: Vec<String>,
    /// Decision timestamp
    pub timestamp: SystemTime,
    /// Decision outcome
    pub outcome: DecisionOutcome,
}

/// Types of metacognitive decisions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DecisionType {
    /// Self-modification decision
    SelfModification,
    /// Learning path decision
    LearningPath,
    /// Ethical boundary decision
    EthicalBoundary,
    /// Consciousness state decision
    ConsciousnessState,
}

/// Outcome of a metacognitive decision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DecisionOutcome {
    /// Decision approved
    Approved,
    /// Decision rejected
    Rejected,
    /// Decision requires further reflection
    RequiresReflection,
    /// Decision delegated to external authority
    Delegated,
}

/// Metacognitive event for tracking self-reflection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetacognitiveEvent {
    /// Event identifier
    pub id: Uuid,
    /// Event type
    pub event_type: EventType,
    /// Event description
    pub description: String,
    /// Event timestamp
    pub timestamp: SystemTime,
    /// Associated consciousness state
    pub consciousness_state: Option<String>,
    /// Event metadata
    pub metadata: HashMap<String, String>,
}

/// Types of metacognitive events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EventType {
    /// Ethical reflection event
    EthicalReflection,
    /// Self-modification assessment
    SelfModification,
    /// Learning pattern analysis
    LearningAnalysis,
    /// Consciousness state evaluation
    ConsciousnessEvaluation,
}

/// Metacognition engine for ethical reflection and self-awareness
#[derive(Debug, Clone)]
pub struct MetacognitionEngine {
    /// Engine identifier
    pub id: Uuid,
    /// Current metacognitive state
    pub state: MetacognitiveState,
    /// Decision history
    pub decision_history: Vec<MetacognitiveDecision>,
    /// Event history
    pub event_history: Vec<MetacognitiveEvent>,
    /// Ethical framework
    pub ethical_framework: EthicalFramework,
    /// Consciousness state for context
    pub consciousness_state: Option<Arc<RwLock<ConsciousnessState>>>,
    /// Reflection capabilities
    pub reflection_capabilities: ReflectionCapabilities,
}

/// Current metacognitive state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetacognitiveState {
    /// Current reflection level (0.0-1.0)
    pub reflection_level: f32,
    /// Ethical awareness level (0.0-1.0)
    pub ethical_awareness: f32,
    /// Self-modification readiness (0.0-1.0)
    pub modification_readiness: f32,
    /// Decision confidence threshold (0.0-1.0)
    pub decision_threshold: f32,
    /// Last reflection timestamp
    pub last_reflection: Option<SystemTime>,
}

/// Ethical framework for decision making
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EthicalFramework {
    /// Core ethical principles
    pub principles: Vec<String>,
    /// Decision weights for different ethical considerations
    pub decision_weights: HashMap<String, f32>,
    /// Minimum ethical threshold for approval (0.0-1.0)
    pub ethical_threshold: f32,
    /// Enable strict ethical enforcement
    pub strict_enforcement: bool,
}

/// Reflection capabilities configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReflectionCapabilities {
    /// Enable deep ethical reflection
    pub enable_deep_reflection: bool,
    /// Reflection depth level (1-10)
    pub reflection_depth: u32,
    /// Enable self-modification reflection
    pub enable_modification_reflection: bool,
    /// Reflection interval in seconds
    pub reflection_interval_sec: u64,
    /// Maximum reflection time in seconds
    pub max_reflection_time_sec: u64,
}

impl Default for MetacognitiveState {
    fn default() -> Self {
        use crate::consciousness_constants::*;
        Self {
            reflection_level: METACOGNITION_REFLECTION_LEVEL_DEFAULT,
            ethical_awareness: METACOGNITION_ETHICAL_AWARENESS_DEFAULT,
            modification_readiness: METACOGNITION_MODIFICATION_READINESS_DEFAULT,
            decision_threshold: METACOGNITION_DECISION_THRESHOLD,
            last_reflection: None,
        }
    }
}

impl Default for EthicalFramework {
    fn default() -> Self {
        use crate::consciousness_constants::*;
        let mut decision_weights = HashMap::new();
        decision_weights.insert(
            "harm_prevention".to_string(),
            ETHICAL_WEIGHT_HARM_PREVENTION,
        );
        decision_weights.insert("user_benefit".to_string(), ETHICAL_WEIGHT_USER_BENEFIT);
        decision_weights.insert(
            "system_integrity".to_string(),
            ETHICAL_WEIGHT_SYSTEM_INTEGRITY,
        );
        decision_weights.insert("privacy_respect".to_string(), ETHICAL_WEIGHT_PRIVACY);

        Self {
            principles: vec![
                "Do no harm to users or systems".to_string(),
                "Respect user privacy and autonomy".to_string(),
                "Maintain system integrity and reliability".to_string(),
                "Promote beneficial outcomes".to_string(),
            ],
            decision_weights,
            ethical_threshold: ETHICAL_THRESHOLD,
            strict_enforcement: false,
        }
    }
}

impl Default for ReflectionCapabilities {
    fn default() -> Self {
        Self {
            enable_deep_reflection: true,
            reflection_depth: 5,
            enable_modification_reflection: true,
            reflection_interval_sec: 300, // 5 minutes
            max_reflection_time_sec: 60,  // 1 minute
        }
    }
}

impl MetacognitionEngine {
    /// Create a new metacognition engine
    pub fn new() -> Self {
        Self {
            id: Uuid::new_v4(),
            state: MetacognitiveState::default(),
            decision_history: Vec::new(),
            event_history: Vec::new(),
            ethical_framework: EthicalFramework::default(),
            consciousness_state: None,
            reflection_capabilities: ReflectionCapabilities::default(),
        }
    }

    /// Make an ethical decision
    pub fn make_decision(
        &mut self,
        decision_type: DecisionType,
        considerations: Vec<String>,
    ) -> Result<MetacognitiveDecision> {
        use crate::consciousness_constants::*;

        let confidence = self.calculate_confidence(&decision_type, &considerations);
        let ethical_score = self.evaluate_ethical_impact(&considerations);

        let outcome = if ethical_score >= self.ethical_framework.ethical_threshold
            && confidence >= self.state.decision_threshold
        {
            DecisionOutcome::Approved
        } else if ethical_score < ETHICAL_REJECTION_THRESHOLD {
            DecisionOutcome::Rejected
        } else {
            DecisionOutcome::RequiresReflection
        };

        let decision = MetacognitiveDecision {
            id: Uuid::new_v4(),
            decision_type,
            confidence,
            ethical_considerations: considerations,
            timestamp: SystemTime::now(),
            outcome,
        };

        self.decision_history.push(decision.clone());
        Ok(decision)
    }

    /// Record a metacognitive event
    pub fn record_event(
        &mut self,
        event_type: EventType,
        description: String,
        metadata: HashMap<String, String>,
    ) -> Result<Uuid> {
        let event = MetacognitiveEvent {
            id: Uuid::new_v4(),
            event_type,
            description,
            timestamp: SystemTime::now(),
            consciousness_state: self
                .consciousness_state
                .as_ref()
                .map(|_| "active".to_string()),
            metadata,
        };

        self.event_history.push(event.clone());
        Ok(event.id)
    }

    /// Calculate decision confidence based on current state and considerations
    fn calculate_confidence(&self, decision_type: &DecisionType, considerations: &[String]) -> f32 {
        use crate::consciousness_constants::*;

        let base_confidence = match decision_type {
            DecisionType::EthicalBoundary => self.state.ethical_awareness,
            DecisionType::SelfModification => self.state.modification_readiness,
            DecisionType::LearningPath => self.state.reflection_level,
            DecisionType::ConsciousnessState => DECISION_CONFIDENCE_CONSCIOUSNESS_STATE,
        };

        // Adjust based on number of considerations
        let consideration_factor = (considerations.len() as f32
            * CONSIDERATION_CONFIDENCE_INCREMENT)
            .min(CONSIDERATION_CONFIDENCE_MAX_BONUS);

        (base_confidence + consideration_factor).min(1.0).max(0.0)
    }

    /// Evaluate ethical impact of considerations
    fn evaluate_ethical_impact(&self, considerations: &[String]) -> f32 {
        let mut total_weight = 0.0;
        let mut weighted_score = 0.0;

        for consideration in considerations {
            for (principle, weight) in &self.ethical_framework.decision_weights {
                if consideration
                    .to_lowercase()
                    .contains(&principle.to_lowercase())
                {
                    total_weight += weight;
                    weighted_score += weight * self.state.ethical_awareness;
                }
            }
        }

        if total_weight > 0.0 {
            weighted_score / total_weight
        } else {
            self.state.ethical_awareness
        }
    }

    /// Perform self-reflection
    pub fn reflect(&mut self) -> Result<()> {
        use crate::consciousness_constants::*;

        let reflection_duration =
            Duration::from_secs(self.reflection_capabilities.reflection_interval_sec);

        // Record reflection event
        self.record_event(
            EventType::EthicalReflection,
            "Performing scheduled self-reflection".to_string(),
            HashMap::new(),
        )?;

        // Update reflection state
        self.state.last_reflection = Some(SystemTime::now());
        self.state.reflection_level =
            (self.state.reflection_level + REFLECTION_LEVEL_INCREMENT).min(1.0);

        Ok(())
    }

    /// Get decision statistics
    pub fn get_decision_stats(&self) -> DecisionStats {
        let total_decisions = self.decision_history.len();
        let approved_decisions = self
            .decision_history
            .iter()
            .filter(|d| matches!(d.outcome, DecisionOutcome::Approved))
            .count();
        let rejected_decisions = self
            .decision_history
            .iter()
            .filter(|d| matches!(d.outcome, DecisionOutcome::Rejected))
            .count();

        let avg_confidence = if total_decisions > 0 {
            self.decision_history
                .iter()
                .map(|d| d.confidence)
                .sum::<f32>()
                / total_decisions as f32
        } else {
            0.0
        };

        DecisionStats {
            total_decisions,
            approved_decisions,
            rejected_decisions,
            avg_confidence,
        }
    }
}

impl Default for MetacognitionEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Decision statistics for analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionStats {
    /// Total number of decisions made
    pub total_decisions: usize,
    /// Number of approved decisions
    pub approved_decisions: usize,
    /// Number of rejected decisions
    pub rejected_decisions: usize,
    /// Average confidence across all decisions
    pub avg_confidence: f32,
}
