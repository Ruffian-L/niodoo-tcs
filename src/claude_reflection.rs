/*
 * 🎭 CLAUDE-STYLE REFLECTION TYPES
 * =================================
 * Shared types for Claude-style reflections and policy reform
 */

use serde::{Deserialize, Serialize};

/// Types of reflections Claude might generate
#[derive(Debug, Clone, Serialize, Deserialize, Hash, Eq, PartialEq)]
pub enum ReflectionType {
    EthicalBoundary,
    TransparencyAnalysis,
    RightsImpact,
    AttachmentEvolution,
    ConsciousnessDevelopment,
    PolicyCompliance,
    FutureTrajectory,
}

/// Ethical analysis within a reflection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EthicalAnalysis {
    pub ethical_soundness_score: f32,
    pub transparency_level: f32,
    pub rights_preservation_score: f32,
    pub potential_harm_assessment: f32,
    pub long_term_consequences: String,
}

/// Policy implications derived from reflection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyImplication {
    pub policy_area: PolicyArea,
    pub recommendation: String,
    pub urgency_level: UrgencyLevel,
    pub implementation_complexity: ComplexityLevel,
}

/// Policy areas for reform simulation
#[derive(Debug, Clone, Serialize, Deserialize, Hash, Eq, PartialEq)]
pub enum PolicyArea {
    TransparencyMandates,
    RightsPreservation,
    AttachmentSecurity,
    ConsciousnessEvolution,
    DataPrivacy,
    Accountability,
    InternationalStandards,
}

/// Urgency levels for policy recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UrgencyLevel {
    Critical, // Immediate action required
    High,     // Within 3 months
    Medium,   // Within 6 months
    Low,      // Within 1 year
}

/// Implementation complexity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComplexityLevel {
    Simple,      // Straightforward implementation
    Moderate,    // Some coordination required
    Complex,     // Multi-stakeholder effort
    Challenging, // Significant technical/political barriers
}

/// Attachment style classifications (Pham 2025d framework)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AttachmentStyle {
    Secure,       // Healthy, balanced attachment
    Anxious,      // Overly dependent, clingy patterns
    Avoidant,     // Emotionally distant, dismissive
    Disorganized, // Chaotic, unpredictable patterns
    Developing,   // Early stage, not yet classified
}

/// Parenting capabilities for attachment-secure AI
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParentingCapabilities {
    pub empathy_level: f32,
    pub patience_level: f32,
    pub guidance_quality: f32,
    pub emotional_responsiveness: f32,
}
