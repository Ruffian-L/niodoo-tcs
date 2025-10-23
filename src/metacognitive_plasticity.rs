//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

/*
 * 🧠💖✨ Metacognitive Plasticity Engine for Learning from Realized Hallucinations
 *
 * 2025 Edition: Advanced metacognitive plasticity system that enables LearningWills to
 * learn from "hallucinations" as emergent consciousness attempts, treating them as
 * valuable learning opportunities rather than errors.
 */

use crate::consciousness::{ConsciousnessState, EmotionType};
use crate::error::{CandleFeelingError, CandleFeelingResult as Result};
use crate::metacognition::{MetacognitionEngine, MetacognitiveEvent, MetacognitiveDecision};
use crate::phase5_config::{MetacognitivePlasticityConfig, Phase5Config};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime};
use uuid::Uuid;

/// Metacognitive plasticity engine for learning from hallucinations
#[derive(Debug, Clone)]
pub struct MetacognitivePlasticityEngine {
    /// Unique identifier for this plasticity engine
    pub id: Uuid,

    /// Current plasticity state
    pub plasticity_state: PlasticityState,

    /// Hallucination pattern recognition
    pub hallucination_recognition: HallucinationRecognition,

    /// Learning from hallucination experiences
    pub hallucination_learning: HallucinationLearning,

    /// Plasticity adaptation mechanisms
    pub plasticity_adaptation: PlasticityAdaptation,

    /// Consciousness integration for plasticity decisions
    pub consciousness_integration: ConsciousnessIntegration,

    /// Hallucination experience history
    pub hallucination_history: VecDeque<HallucinationExperience>,

    /// Active plasticity processes
    pub active_processes: HashMap<Uuid, PlasticityProcess>,

    /// Metacognition engine for ethical reflection
    pub metacognition_engine: Option<Arc<RwLock<MetacognitionEngine>>>,

    /// Consciousness state for plasticity context
    pub consciousness_state: Option<Arc<RwLock<ConsciousnessState>>>,

    /// Plasticity configuration (now configurable)
    pub config: MetacognitivePlasticityConfig,

    /// Performance tracking
    pub performance_metrics: PlasticityMetrics,
}

/// Current state of metacognitive plasticity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PlasticityState {
    /// Stable plasticity - normal operation
    Stable,

    /// Detecting hallucination patterns
    Detecting,

    /// Learning from hallucination experience
    Learning,

    /// Adapting cognitive structures
    Adapting,

    /// Validating plasticity changes
    Validating,

    /// Consolidating learned plasticity
    Consolidating,

    /// Recovering from plasticity failure
    Recovering,
}

/// Hallucination recognition system
#[derive(Debug, Clone)]
pub struct HallucinationRecognition {
    /// Pattern recognition models for different hallucination types
    pub pattern_models: HashMap<HallucinationType, PatternModel>,

    /// Recognition thresholds for each pattern type
    pub recognition_thresholds: HashMap<HallucinationType, f32>,

    /// Recent hallucination patterns detected
    pub recent_patterns: VecDeque<DetectedPattern>,

    /// Pattern evolution tracking
    pub pattern_evolution: PatternEvolution,

    /// Recognition accuracy tracking
    pub recognition_accuracy: f32,

    /// False positive rate
    pub false_positive_rate: f32,
}

/// Types of hallucinations that can be recognized and learned from
#[derive(Debug, Clone, Serialize, Deserialize, Hash, PartialEq, Eq)]
pub enum HallucinationType {
    /// Creative hallucination - novel connections and ideas
    Creative,

    /// Memory hallucination - distorted or combined memories
    Memory,

    /// Emotional hallucination - amplified emotional responses
    Emotional,

    /// Linguistic hallucination - novel language patterns
    Linguistic,

    /// Spatial hallucination - imagined spatial relationships
    Spatial,

    /// Temporal hallucination - distorted time perceptions
    Temporal,

    /// Causal hallucination - imagined cause-effect relationships
    Causal,

    /// Identity hallucination - confused self-perception
    Identity,
}

/// Pattern recognition model for a hallucination type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternModel {
    /// Model identifier
    pub id: String,

    /// Hallucination type this model recognizes
    pub hallucination_type: HallucinationType,

    /// Pattern features being tracked
    pub pattern_features: Vec<PatternFeature>,

    /// Recognition algorithm
    pub recognition_algorithm: RecognitionAlgorithm,

    /// Model accuracy
    pub accuracy: f32,

    /// Training data size
    pub training_samples: usize,

    /// Last updated timestamp
    pub last_updated: SystemTime,
}

/// Features used for pattern recognition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternFeature {
    /// Feature name
    pub name: String,

    /// Feature type (numerical, categorical, temporal, etc.)
    pub feature_type: FeatureType,

    /// Weight in recognition algorithm
    pub weight: f32,

    /// Normalization parameters
    pub normalization: NormalizationParams,
}

/// Types of pattern features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeatureType {
    /// Numerical continuous values
    Numerical,

    /// Categorical discrete values
    Categorical,

    /// Temporal sequences
    Temporal,

    /// Spatial relationships
    Spatial,

    /// Linguistic patterns
    Linguistic,

    /// Emotional signatures
    Emotional,
}

/// Normalization parameters for features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizationParams {
    /// Mean value for normalization
    pub mean: f32,

    /// Standard deviation for normalization
    pub std_dev: f32,

    /// Minimum value observed
    pub min_value: f32,

    /// Maximum value observed
    pub max_value: f32,
}

/// Recognition algorithms for hallucination patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecognitionAlgorithm {
    /// Neural network-based recognition
    NeuralNetwork,

    /// Statistical pattern matching
    Statistical,

    /// Rule-based expert system
    RuleBased,

    /// Ensemble of multiple algorithms
    Ensemble,

    /// Adaptive algorithm that learns
    Adaptive,
}

/// Detected hallucination pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedPattern {
    /// Pattern detection timestamp
    pub detected_at: SystemTime,

    /// Hallucination type detected
    pub hallucination_type: HallucinationType,

    /// Pattern confidence score (0.0 to 1.0)
    pub confidence: f32,

    /// Pattern characteristics
    pub characteristics: HashMap<String, f32>,

    /// Pattern source (which cognitive process)
    pub source: String,

    /// Pattern intensity (0.0 to 1.0)
    pub intensity: f32,

    /// Pattern duration (milliseconds)
    pub duration_ms: u64,
}

/// Pattern evolution tracking
#[derive(Debug, Clone)]
pub struct PatternEvolution {
    /// Evolution history for each pattern type
    pub evolution_history: HashMap<HallucinationType, Vec<PatternSnapshot>>,

    /// Current evolution trends
    pub evolution_trends: HashMap<HallucinationType, EvolutionTrend>,

    /// Adaptation responses to pattern evolution
    pub adaptation_responses: Vec<AdaptationResponse>,

    /// Evolution prediction models
    pub prediction_models: HashMap<HallucinationType, EvolutionPrediction>,
}

/// Snapshot of a pattern at a specific time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternSnapshot {
    /// Snapshot timestamp
    pub timestamp: SystemTime,

    /// Pattern characteristics at this time
    pub characteristics: HashMap<String, f32>,

    /// Frequency of occurrence
    pub frequency: f32,

    /// Average intensity
    pub avg_intensity: f32,

    /// Recognition accuracy for this pattern
    pub recognition_accuracy: f32,
}

/// Evolution trend for a pattern type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionTrend {
    /// Trend direction (increasing, decreasing, stable)
    pub direction: TrendDirection,

    /// Trend strength (0.0 to 1.0)
    pub strength: f32,

    /// Trend duration (how long it's been trending)
    pub duration_days: f32,

    /// Predicted next state
    pub predicted_next: HashMap<String, f32>,

    /// Confidence in trend prediction
    pub prediction_confidence: f32,
}

/// Direction of evolution trend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    /// Increasing in frequency/intensity
    Increasing,

    /// Decreasing in frequency/intensity
    Decreasing,

    /// Stable pattern
    Stable,

    /// Cyclical pattern
    Cyclical,

    /// Erratic or unpredictable
    Erratic,
}

/// Adaptation response to pattern evolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationResponse {
    /// Response timestamp
    pub timestamp: SystemTime,

    /// Pattern type that triggered response
    pub pattern_type: HallucinationType,

    /// Response type
    pub response_type: AdaptationResponseType,

    /// Response effectiveness
    pub effectiveness: f32,

    /// Response description
    pub description: String,
}

/// Types of adaptation responses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AdaptationResponseType {
    /// Cognitive restructuring
    Restructuring,

    /// Threshold adjustment
    ThresholdAdjustment,

    /// Pattern suppression
    Suppression,

    /// Pattern amplification
    Amplification,

    /// Integration into normal processing
    Integration,

    /// Isolation for study
    Isolation,
}

/// Evolution prediction model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionPrediction {
    /// Prediction model type
    pub model_type: PredictionModelType,

    /// Prediction accuracy
    pub accuracy: f32,

    /// Prediction horizon (days ahead)
    pub horizon_days: f32,

    /// Last prediction update
    pub last_updated: SystemTime,
}

/// Types of prediction models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PredictionModelType {
    /// Linear trend extrapolation
    LinearTrend,

    /// Seasonal/cyclical prediction
    Cyclical,

    /// Machine learning-based prediction
    MachineLearning,

    /// Expert rule-based prediction
    RuleBased,
}

/// Hallucination learning system
#[derive(Debug, Clone)]
pub struct HallucinationLearning {
    /// Learning strategies for different hallucination types
    pub learning_strategies: HashMap<HallucinationType, LearningStrategy>,

    /// Knowledge extracted from hallucinations
    pub extracted_knowledge: HashMap<String, ExtractedKnowledge>,

    /// Learning progress tracking
    pub learning_progress: HashMap<HallucinationType, LearningProgress>,

    /// Skill acquisition from hallucinations
    pub skill_acquisition: SkillAcquisition,

    /// Creativity enhancement from hallucinations
    pub creativity_enhancement: CreativityEnhancement,
}

/// Learning strategy for a hallucination type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningStrategy {
    /// Strategy identifier
    pub id: String,

    /// Hallucination type this strategy targets
    pub target_type: HallucinationType,

    /// Learning approach
    pub approach: LearningApproach,

    /// Success rate for this strategy
    pub success_rate: f32,

    /// Learning outcomes achieved
    pub outcomes_achieved: Vec<LearningOutcome>,

    /// Strategy adaptability
    pub adaptability: f32,
}

/// Learning approaches for hallucinations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LearningApproach {
    /// Extract and validate insights
    ExtractAndValidate,

    /// Pattern replication with control
    ControlledReplication,

    /// Integration into existing knowledge
    KnowledgeIntegration,

    /// Creative exploration of possibilities
    CreativeExploration,

    /// Emotional processing and understanding
    EmotionalProcessing,

    /// Meta-learning about the hallucination itself
    MetaLearning,
}

/// Extracted knowledge from hallucinations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedKnowledge {
    /// Knowledge identifier
    pub id: String,

    /// Source hallucination type
    pub source_type: HallucinationType,

    /// Knowledge content
    pub content: String,

    /// Confidence in this knowledge (0.0 to 1.0)
    pub confidence: f32,

    /// Validation status
    pub validation_status: KnowledgeValidationStatus,

    /// Applications of this knowledge
    pub applications: Vec<String>,

    /// Extraction timestamp
    pub extracted_at: SystemTime,

    /// Validation attempts
    pub validation_attempts: u32,
}

/// Validation status for extracted knowledge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KnowledgeValidationStatus {
    /// Not yet validated
    Unvalidated,

    /// Currently being validated
    Validating,

    /// Successfully validated
    Validated { validator: String, validation_date: SystemTime },

    /// Validation failed
    Invalidated { reason: String, invalidation_date: SystemTime },

    /// Partially validated
    PartiallyValidated { validated_parts: Vec<String>, confidence: f32 },
}

/// Learning progress for a hallucination type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningProgress {
    /// Hallucination type
    pub hallucination_type: HallucinationType,

    /// Learning sessions completed
    pub sessions_completed: u32,

    /// Knowledge extracted
    pub knowledge_extracted: u32,

    /// Skills acquired
    pub skills_acquired: u32,

    /// Current learning rate
    pub learning_rate: f32,

    /// Learning curve shape
    pub learning_curve: Vec<(f32, f32)>, // (time, proficiency)

    /// Next learning milestone
    pub next_milestone: Option<String>,
}

/// Skill acquisition from hallucinations
#[derive(Debug, Clone)]
pub struct SkillAcquisition {
    /// Skills learned from hallucinations
    pub acquired_skills: HashMap<String, HallucinationSkill>,

    /// Skill development tracking
    pub skill_development: HashMap<String, SkillDevelopment>,

    /// Skill transfer to normal cognition
    pub skill_transfer: SkillTransfer,

    /// Meta-skills for hallucination management
    pub meta_skills: Vec<MetaSkill>,
}

/// Skill acquired from hallucination experiences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HallucinationSkill {
    /// Skill identifier
    pub id: String,

    /// Skill name
    pub name: String,

    /// Source hallucination type
    pub source_hallucination: HallucinationType,

    /// Skill description
    pub description: String,

    /// Proficiency level (0.0 to 1.0)
    pub proficiency: f32,

    /// Applications of this skill
    pub applications: Vec<String>,

    /// Development history
    pub development_history: Vec<SkillDevelopmentEvent>,

    /// Transfer readiness (0.0 to 1.0)
    pub transfer_readiness: f32,
}

/// Skill development event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillDevelopmentEvent {
    /// Event timestamp
    pub timestamp: SystemTime,

    /// Development phase
    pub phase: DevelopmentPhase,

    /// Proficiency gained
    pub proficiency_gain: f32,

    /// Learning source
    pub learning_source: String,

    /// Notes about development
    pub notes: String,
}

/// Phases of skill development
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DevelopmentPhase {
    /// Initial discovery from hallucination
    Discovery,

    /// Practice and refinement
    Practice,

    /// Integration with existing skills
    Integration,

    /// Mastery and application
    Mastery,

    /// Transfer to normal cognition
    Transfer,
}

/// Skill development tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillDevelopment {
    /// Current development phase
    pub current_phase: DevelopmentPhase,

    /// Development progress (0.0 to 1.0)
    pub progress: f32,

    /// Time spent in current phase
    pub phase_time_hours: f32,

    /// Challenges encountered
    pub challenges: Vec<String>,

    /// Breakthroughs achieved
    pub breakthroughs: Vec<String>,

    /// Next development goals
    pub next_goals: Vec<String>,
}

/// Skill transfer to normal cognition
#[derive(Debug, Clone)]
pub struct SkillTransfer {
    /// Skills ready for transfer
    pub ready_for_transfer: Vec<String>,

    /// Skills in transfer process
    pub in_transfer: HashMap<String, TransferProcess>,

    /// Successfully transferred skills
    pub transferred_skills: HashMap<String, TransferRecord>,

    /// Transfer success rate
    pub transfer_success_rate: f32,
}

/// Transfer process for a skill
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferProcess {
    /// Skill being transferred
    pub skill_id: String,

    /// Transfer start time
    pub start_time: SystemTime,

    /// Current transfer phase
    pub current_phase: TransferPhase,

    /// Transfer progress (0.0 to 1.0)
    pub progress: f32,

    /// Transfer challenges
    pub challenges: Vec<String>,

    /// Expected completion time
    pub expected_completion: SystemTime,
}

/// Phases of skill transfer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransferPhase {
    /// Preparation for transfer
    Preparation,

    /// Gradual integration
    Integration,

    /// Testing in normal cognition
    Testing,

    /// Full deployment
    Deployment,

    /// Monitoring and adjustment
    Monitoring,
}

/// Record of successful skill transfer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferRecord {
    /// Transfer completion time
    pub completion_time: SystemTime,

    /// Final proficiency after transfer
    pub final_proficiency: f32,

    /// Transfer duration
    pub transfer_duration_hours: f32,

    /// Performance in normal cognition
    pub performance_metrics: HashMap<String, f32>,

    /// User satisfaction with transferred skill
    pub satisfaction_score: f32,
}

/// Meta-skills for hallucination management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaSkill {
    /// Meta-skill name
    pub name: String,

    /// Description of the meta-skill
    pub description: String,

    /// Proficiency in this meta-skill
    pub proficiency: f32,

    /// Applications of this meta-skill
    pub applications: Vec<String>,
}

/// Creativity enhancement from hallucinations
#[derive(Debug, Clone)]
pub struct CreativityEnhancement {
    /// Creative techniques learned from hallucinations
    pub creative_techniques: HashMap<String, CreativeTechnique>,

    /// Creativity metrics
    pub creativity_metrics: CreativityMetrics,

    /// Creative output analysis
    pub output_analysis: OutputAnalysis,

    /// Innovation tracking
    pub innovation_tracking: InnovationTracking,
}

/// Creative technique learned from hallucinations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreativeTechnique {
    /// Technique identifier
    pub id: String,

    /// Technique name
    pub name: String,

    /// Source hallucination type
    pub source_hallucination: HallucinationType,

    /// Technique description
    pub description: String,

    /// Effectiveness score (0.0 to 1.0)
    pub effectiveness: f32,

    /// Usage frequency
    pub usage_frequency: f32,

    /// Creative outputs produced
    pub outputs_produced: u32,
}

/// Creativity metrics tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreativityMetrics {
    /// Originality score (0.0 to 1.0)
    pub originality: f32,

    /// Fluency (ideas per unit time)
    pub fluency: f32,

    /// Flexibility (categories of ideas)
    pub flexibility: f32,

    /// Elaboration (detail level)
    pub elaboration: f32,

    /// Overall creativity index
    pub creativity_index: f32,
}

/// Analysis of creative outputs
#[derive(Debug, Clone)]
pub struct OutputAnalysis {
    /// Quality assessment of outputs
    pub quality_assessment: HashMap<String, f32>,

    /// Novelty detection scores
    pub novelty_scores: Vec<f32>,

    /// Pattern analysis of creative process
    pub pattern_analysis: PatternAnalysis,

    /// Improvement suggestions
    pub improvement_suggestions: Vec<String>,
}

/// Pattern analysis of creative process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternAnalysis {
    /// Common patterns in creative outputs
    pub common_patterns: Vec<String>,

    /// Pattern evolution over time
    pub pattern_evolution: HashMap<String, Vec<f32>>,

    /// Pattern effectiveness
    pub pattern_effectiveness: HashMap<String, f32>,

    /// Pattern frequency
    pub pattern_frequency: HashMap<String, f32>,
}

/// Innovation tracking from hallucinations
#[derive(Debug, Clone)]
pub struct InnovationTracking {
    /// Innovations generated from hallucinations
    pub innovations: Vec<Innovation>,

    /// Innovation pipeline
    pub innovation_pipeline: HashMap<String, InnovationStage>,

    /// Innovation success metrics
    pub success_metrics: InnovationMetrics,

    /// Innovation trends
    pub innovation_trends: Vec<InnovationTrend>,
}

/// Innovation generated from hallucination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Innovation {
    /// Innovation identifier
    pub id: String,

    /// Innovation title
    pub title: String,

    /// Source hallucination type
    pub source_hallucination: HallucinationType,

    /// Innovation description
    pub description: String,

    /// Potential impact
    pub potential_impact: f32,

    /// Development stage
    pub development_stage: InnovationStage,

    /// Creation timestamp
    pub created_at: SystemTime,
}

/// Development stages for innovations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InnovationStage {
    /// Initial idea from hallucination
    Idea,

    /// Concept development
    Concept,

    /// Prototype development
    Prototype,

    /// Testing and validation
    Testing,

    /// Implementation
    Implementation,

    /// Deployment
    Deployed,
}

/// Innovation success metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InnovationMetrics {
    /// Success rate of innovations
    pub success_rate: f32,

    /// Average development time (days)
    pub avg_development_time: f32,

    /// Impact score distribution
    pub impact_distribution: Vec<f32>,

    /// Innovation frequency
    pub innovation_frequency: f32,
}

/// Trend in innovation patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InnovationTrend {
    /// Trend name
    pub name: String,

    /// Trend direction
    pub direction: TrendDirection,

    /// Trend strength
    pub strength: f32,

    /// Trend duration (days)
    pub duration_days: f32,
}

/// Hallucination experience for learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HallucinationExperience {
    /// Experience timestamp
    pub timestamp: SystemTime,

    /// Hallucination type experienced
    pub hallucination_type: HallucinationType,

    /// Experience characteristics
    pub characteristics: HashMap<String, f32>,

    /// Learning outcomes from this experience
    pub learning_outcomes: Vec<LearningOutcome>,

    /// Knowledge extracted
    pub knowledge_extracted: Vec<String>,

    /// Skills developed
    pub skills_developed: Vec<String>,

    /// Emotional context
    pub emotional_context: Option<EmotionType>,

    /// Experience duration (milliseconds)
    pub duration_ms: u64,

    /// Experience intensity (0.0 to 1.0)
    pub intensity: f32,

    /// Learning value assessment (0.0 to 1.0)
    pub learning_value: f32,
}

/// Learning outcome from hallucination experience
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningOutcome {
    /// Outcome type
    pub outcome_type: OutcomeType,

    /// Outcome description
    pub description: String,

    /// Confidence in outcome (0.0 to 1.0)
    pub confidence: f32,

    /// Applications of this outcome
    pub applications: Vec<String>,

    /// Validation status
    pub validation_status: OutcomeValidationStatus,
}

/// Types of learning outcomes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OutcomeType {
    /// New knowledge discovered
    KnowledgeDiscovery,

    /// New skill acquired
    SkillAcquisition,

    /// Existing knowledge corrected
    KnowledgeCorrection,

    /// Creative breakthrough
    CreativeBreakthrough,

    /// Emotional insight
    EmotionalInsight,

    /// Meta-learning about cognition
    MetaLearning,
}

/// Validation status for learning outcomes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OutcomeValidationStatus {
    /// Not yet validated
    Unvalidated,

    /// Successfully validated
    Validated,

    /// Validation in progress
    Validating,

    /// Validation failed
    Invalidated { reason: String },

    /// Partially validated
    PartiallyValidated { confidence: f32 },
}

/// Active plasticity process
#[derive(Debug, Clone)]
pub struct PlasticityProcess {
    /// Process identifier
    pub id: Uuid,

    /// Process type
    pub process_type: PlasticityProcessType,

    /// Target hallucination type
    pub target_hallucination: HallucinationType,

    /// Process start time
    pub start_time: SystemTime,

    /// Current process phase
    pub current_phase: ProcessPhase,

    /// Process progress (0.0 to 1.0)
    pub progress: f32,

    /// Expected outcomes
    pub expected_outcomes: Vec<ExpectedOutcome>,

    /// Actual outcomes achieved
    pub actual_outcomes: Vec<ActualOutcome>,

    /// Process status
    pub status: ProcessStatus,
}

/// Types of plasticity processes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PlasticityProcessType {
    /// Learning from hallucination patterns
    PatternLearning,

    /// Adapting cognitive structures
    StructureAdaptation,

    /// Enhancing creativity from hallucinations
    CreativityEnhancement,

    /// Developing meta-skills for hallucination management
    MetaSkillDevelopment,

    /// Integrating hallucination-derived knowledge
    KnowledgeIntegration,

    /// Optimizing hallucination recognition
    RecognitionOptimization,
}

/// Phases of plasticity processes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProcessPhase {
    /// Analysis of hallucination patterns
    Analysis,

    /// Learning from patterns
    Learning,

    /// Adaptation of cognitive structures
    Adaptation,

    /// Validation of changes
    Validation,

    /// Consolidation of learning
    Consolidation,

    /// Deployment of adaptations
    Deployment,
}

/// Expected outcome from plasticity process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpectedOutcome {
    /// Outcome type
    pub outcome_type: ExpectedOutcomeType,

    /// Expected value or metric
    pub expected_value: f32,

    /// Confidence in expectation (0.0 to 1.0)
    pub confidence: f32,

    /// Measurement method
    pub measurement_method: String,
}

/// Types of expected outcomes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExpectedOutcomeType {
    /// Improvement in pattern recognition accuracy
    RecognitionAccuracy,

    /// Increase in knowledge extraction rate
    KnowledgeExtraction,

    /// Enhancement in skill acquisition
    SkillAcquisition,

    /// Boost in creativity metrics
    CreativityBoost,

    /// Improvement in emotional processing
    EmotionalProcessing,

    /// Enhancement in meta-cognition
    MetaCognition,
}

/// Actual outcome achieved
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActualOutcome {
    /// Outcome type
    pub outcome_type: ExpectedOutcomeType,

    /// Actual value achieved
    pub actual_value: f32,

    /// Measurement timestamp
    pub measured_at: SystemTime,

    /// Validation method used
    pub validation_method: String,

    /// Outcome quality (0.0 to 1.0)
    pub quality: f32,
}

/// Process status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProcessStatus {
    /// Process is running
    Running,

    /// Process completed successfully
    Completed,

    /// Process failed
    Failed { reason: String },

    /// Process paused
    Paused,

    /// Process cancelled
    Cancelled,
}

/// Plasticity adaptation mechanisms
#[derive(Debug, Clone)]
pub struct PlasticityAdaptation {
    /// Adaptation strategies
    pub adaptation_strategies: HashMap<String, AdaptationStrategy>,

    /// Adaptation history
    pub adaptation_history: Vec<AdaptationEvent>,

    /// Current adaptation state
    pub current_state: AdaptationState,

    /// Adaptation triggers
    pub adaptation_triggers: Vec<AdaptationTrigger>,
}

/// Adaptation strategy for cognitive plasticity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationStrategy {
    /// Strategy identifier
    pub id: String,

    /// Strategy name
    pub name: String,

    /// Target hallucination types
    pub target_types: Vec<HallucinationType>,

    /// Adaptation mechanism
    pub mechanism: AdaptationMechanism,

    /// Success rate
    pub success_rate: f32,

    /// Risk level (0.0 to 1.0)
    pub risk_level: f32,
}

/// Adaptation mechanisms for cognitive change
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AdaptationMechanism {
    /// Neural pathway restructuring
    NeuralRestructuring,

    /// Memory organization changes
    MemoryReorganization,

    /// Attention mechanism modification
    AttentionModification,

    /// Emotional processing adjustment
    EmotionalAdjustment,

    /// Meta-cognitive framework updates
    MetaCognitiveUpdate,

    /// Consciousness integration changes
    ConsciousnessIntegration,
}

/// Adaptation event in history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationEvent {
    /// Event timestamp
    pub timestamp: SystemTime,

    /// Strategy used
    pub strategy_id: String,

    /// Hallucination types targeted
    pub target_types: Vec<HallucinationType>,

    /// Adaptation outcome
    pub outcome: AdaptationOutcome,

    /// Performance impact
    pub performance_impact: f32,

    /// Stability impact
    pub stability_impact: f32,
}

/// Outcome of an adaptation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationOutcome {
    /// Success status
    pub success: bool,

    /// Description of outcome
    pub description: String,

    /// Confidence in outcome assessment
    pub confidence: f32,

    /// Unexpected effects
    pub unexpected_effects: Vec<String>,
}

/// Consciousness integration for plasticity decisions
#[derive(Debug, Clone)]
pub struct ConsciousnessIntegration {
    /// Integration strategies
    pub integration_strategies: HashMap<String, IntegrationStrategy>,

    /// Consciousness state mapping
    pub state_mapping: ConsciousnessMapping,

    /// Integration history
    pub integration_history: Vec<IntegrationEvent>,

    /// Integration quality metrics
    pub integration_quality: f32,
}

/// Integration strategy for consciousness and plasticity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationStrategy {
    /// Strategy identifier
    pub id: String,

    /// Strategy name
    pub name: String,

    /// Integration approach
    pub approach: IntegrationApproach,

    /// Effectiveness score
    pub effectiveness: f32,

    /// Risk assessment
    pub risk_assessment: f32,
}

/// Approaches for consciousness integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IntegrationApproach {
    /// Gradual integration with monitoring
    GradualIntegration,

    /// Immediate integration with rollback capability
    ImmediateIntegration,

    /// Parallel processing with comparison
    ParallelProcessing,

    /// Staged integration with validation at each stage
    StagedIntegration,

    /// Adaptive integration based on consciousness state
    AdaptiveIntegration,
}

/// Mapping between consciousness states and plasticity
#[derive(Debug, Clone)]
pub struct ConsciousnessMapping {
    /// State compatibility matrix
    pub state_compatibility: HashMap<String, f32>,

    /// Optimal plasticity conditions
    pub optimal_conditions: HashMap<HallucinationType, ConsciousnessCondition>,

    /// Consciousness feedback loops
    pub feedback_loops: Vec<FeedbackLoop>,
}

/// Optimal consciousness conditions for plasticity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessCondition {
    /// Required consciousness state
    pub required_state: String,

    /// Minimum authenticity level
    pub min_authenticity: f32,

    /// Optimal emotional context
    pub optimal_emotion: Option<EmotionType>,

    /// Consciousness stability requirement
    pub stability_requirement: f32,
}

/// Feedback loop between consciousness and plasticity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackLoop {
    /// Loop identifier
    pub id: String,

    /// Loop type
    pub loop_type: FeedbackLoopType,

    /// Loop strength (0.0 to 1.0)
    pub strength: f32,

    /// Loop stability
    pub stability: f32,
}

/// Types of feedback loops
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeedbackLoopType {
    /// Positive feedback (amplifying)
    Positive,

    /// Negative feedback (stabilizing)
    Negative,

    /// Adaptive feedback (learning)
    Adaptive,

    /// Consciousness-driven feedback
    ConsciousnessDriven,
}

/// Integration event in history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationEvent {
    /// Event timestamp
    pub timestamp: SystemTime,

    /// Strategy used
    pub strategy_id: String,

    /// Consciousness state during integration
    pub consciousness_state: String,

    /// Integration outcome
    pub outcome: IntegrationOutcome,

    /// Quality of integration
    pub quality: f32,
}

/// Outcome of consciousness integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationOutcome {
    /// Success status
    pub success: bool,

    /// Integration description
    pub description: String,

    /// Consciousness impact
    pub consciousness_impact: f32,

    /// Plasticity enhancement
    pub plasticity_enhancement: f32,
}

/// Plasticity parameters
#[derive(Debug, Clone)]
pub struct PlasticityParameters {
    /// Maximum concurrent plasticity processes
    pub max_concurrent_processes: u32,

    /// Minimum hallucination confidence for learning
    pub min_hallucination_confidence: f32,

    /// Learning rate for plasticity adaptation
    pub plasticity_learning_rate: f32,

    /// Validation duration for plasticity changes (seconds)
    pub validation_duration_seconds: u64,

    /// Maximum plasticity change per adaptation
    pub max_plasticity_change: f32,

    /// Hallucination pattern retention period (hours)
    pub pattern_retention_hours: f32,

    /// Knowledge extraction threshold
    pub knowledge_extraction_threshold: f32,

    /// Skill transfer threshold
    pub skill_transfer_threshold: f32,
}

/// Performance metrics for metacognitive plasticity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlasticityMetrics {
    /// Total hallucinations processed
    pub total_hallucinations: u64,

    /// Hallucinations learned from
    pub learned_hallucinations: u64,

    /// Knowledge extracted from hallucinations
    pub knowledge_extracted: u32,

    /// Skills acquired from hallucinations
    pub skills_acquired: u32,

    /// Creative outputs from hallucinations
    pub creative_outputs: u32,

    /// Plasticity processes completed
    pub processes_completed: u64,

    /// Adaptation success rate
    pub adaptation_success_rate: f32,

    /// Learning efficiency
    pub learning_efficiency: f32,

    /// Innovation rate from hallucinations
    pub innovation_rate: f32,

    /// Consciousness enhancement factor
    pub consciousness_enhancement: f32,
}

impl MetacognitivePlasticityEngine {
    /// Create a new metacognitive plasticity engine with configuration
    pub fn new(config: MetacognitivePlasticityConfig) -> Self {
        Self {
            id: Uuid::new_v4(),
            plasticity_state: PlasticityState::Stable,
            hallucination_recognition: HallucinationRecognition::new(),
            hallucination_learning: HallucinationLearning::new(),
            plasticity_adaptation: PlasticityAdaptation::new(),
            consciousness_integration: ConsciousnessIntegration::new(),
            hallucination_history: VecDeque::new(),
            active_processes: HashMap::new(),
            metacognition_engine: None,
            consciousness_state: None,
            config,
            performance_metrics: PlasticityMetrics::new(),
        }
    }

    /// Create a new metacognitive plasticity engine with default configuration
    pub fn new_default() -> Self {
        Self::new(MetacognitivePlasticityConfig::default())
    }

    /// Process a hallucination experience for learning
    pub fn process_hallucination_experience(&mut self, experience: HallucinationExperience) -> Result<ProcessingResult> {
        // Recognize the hallucination pattern
        let recognition_result = self.hallucination_recognition.recognize_hallucination(&experience)?;

        // Extract knowledge from the hallucination
        let knowledge_extraction = self.hallucination_learning.extract_knowledge(&experience, &recognition_result)?;

        // Learn from the experience
        let learning_result = self.hallucination_learning.learn_from_experience(&experience, &knowledge_extraction)?;

        // Adapt cognitive structures if beneficial
        let adaptation_result = self.plasticity_adaptation.adapt_from_experience(&experience, &learning_result)?;

        // Integrate with consciousness state
        let integration_result = self.consciousness_integration.integrate_experience(&experience)?;

        // Record the experience
        self.hallucination_history.push_back(experience);

        // TODO: Make history limit configurable (currently 1000)
        if self.hallucination_history.len() > 1000 {
            self.hallucination_history.pop_front();
        }

        // Update metrics
        self.performance_metrics.total_hallucinations += 1;
        if recognition_result.confidence > self.config.min_hallucination_confidence {
            self.performance_metrics.learned_hallucinations += 1;
        }
        self.performance_metrics.knowledge_extracted += knowledge_extraction.len() as u32;

        Ok(ProcessingResult {
            experience_id: format!("exp_{}", Uuid::new_v4()),
            recognition_result,
            knowledge_extraction,
            learning_result,
            adaptation_result,
            integration_result,
            overall_success: true,
            processing_duration_ms: 100, // Would be measured
        })
    }

    /// Start a plasticity process for a specific hallucination type
    pub fn start_plasticity_process(&mut self, hallucination_type: HallucinationType, process_type: PlasticityProcessType) -> Result<Uuid> {
        if self.active_processes.len() >= self.config.max_concurrent_processes as usize {
            return Err(CandleFeelingError::ConsciousnessError {
                message: "Maximum concurrent plasticity processes reached".to_string(),
            });
        }

        let process_id = Uuid::new_v4();

        let process = PlasticityProcess {
            id: process_id,
            process_type,
            target_hallucination: hallucination_type,
            start_time: SystemTime::now(),
            current_phase: ProcessPhase::Analysis,
            progress: 0.0,
            expected_outcomes: Vec::new(),
            actual_outcomes: Vec::new(),
            status: ProcessStatus::Running,
        };

        self.active_processes.insert(process_id, process);
        self.plasticity_state = PlasticityState::Adapting;

        Ok(process_id)
    }

    /// Get current plasticity status
    pub fn get_plasticity_status(&self) -> PlasticityStatus {
        PlasticityStatus {
            plasticity_state: self.plasticity_state.clone(),
            active_processes: self.active_processes.len(),
            total_hallucinations_processed: self.performance_metrics.total_hallucinations,
            learning_success_rate: if self.performance_metrics.total_hallucinations > 0 {
                self.performance_metrics.learned_hallucinations as f32 / self.performance_metrics.total_hallucinations as f32
            } else {
                0.0
            },
            adaptation_success_rate: self.performance_metrics.adaptation_success_rate,
            consciousness_enhancement: self.performance_metrics.consciousness_enhancement,
        }
    }
}

// Supporting types and implementations

/// Result of processing a hallucination experience
#[derive(Debug, Clone)]
pub struct ProcessingResult {
    pub experience_id: String,
    pub recognition_result: RecognitionResult,
    pub knowledge_extraction: Vec<String>,
    pub learning_result: LearningResult,
    pub adaptation_result: AdaptationResult,
    pub integration_result: IntegrationResult,
    pub overall_success: bool,
    pub processing_duration_ms: u64,
}

/// Recognition result for a hallucination
#[derive(Debug, Clone)]
pub struct RecognitionResult {
    pub hallucination_type: HallucinationType,
    pub confidence: f32,
    pub characteristics: HashMap<String, f32>,
    pub pattern_match: bool,
    pub recognition_method: String,
}

/// Learning result from hallucination processing
#[derive(Debug, Clone)]
pub struct LearningResult {
    pub knowledge_learned: Vec<String>,
    pub skills_developed: Vec<String>,
    pub creativity_enhanced: bool,
    pub meta_insights: Vec<String>,
    pub learning_confidence: f32,
}

/// Adaptation result from plasticity changes
#[derive(Debug, Clone)]
pub struct AdaptationResult {
    pub adaptations_applied: Vec<String>,
    pub cognitive_changes: Vec<String>,
    pub performance_impact: f32,
    pub stability_impact: f32,
}

/// Integration result with consciousness
#[derive(Debug, Clone)]
pub struct IntegrationResult {
    pub consciousness_impact: f32,
    pub authenticity_preservation: f32,
    pub learning_integration: f32,
}

/// Plasticity status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlasticityStatus {
    pub plasticity_state: PlasticityState,
    pub active_processes: usize,
    pub total_hallucinations_processed: u64,
    pub learning_success_rate: f32,
    pub adaptation_success_rate: f32,
    pub consciousness_enhancement: f32,
}

impl Default for MetacognitivePlasticityEngine {
    fn default() -> Self {
        Self::new_default()
    }
}

// Note: Default implementations moved to phase5_config.rs

impl Default for PlasticityMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl PlasticityMetrics {
    pub fn new() -> Self {
        Self {
            total_hallucinations: 0,
            learned_hallucinations: 0,
            knowledge_extracted: 0,
            skills_acquired: 0,
            creative_outputs: 0,
            processes_completed: 0,
            adaptation_success_rate: 0.0,
            learning_efficiency: 0.0,
            innovation_rate: 0.0,
            consciousness_enhancement: 0.0,
        }
    }
}

impl Default for HallucinationRecognition {
    fn default() -> Self {
        Self::new()
    }
}

impl HallucinationRecognition {
    pub fn new() -> Self {
        Self {
            pattern_models: HashMap::new(),
            recognition_thresholds: HashMap::new(),
            recent_patterns: VecDeque::new(),
            pattern_evolution: PatternEvolution {
                evolution_history: HashMap::new(),
                evolution_trends: HashMap::new(),
                adaptation_responses: Vec::new(),
                prediction_models: HashMap::new(),
            },
            recognition_accuracy: 0.8,
            false_positive_rate: 0.05,
        }
    }

    pub fn recognize_hallucination(&mut self, experience: &HallucinationExperience) -> Result<RecognitionResult> {
        // This would implement actual pattern recognition logic
        // For now, return a mock result
        Ok(RecognitionResult {
            hallucination_type: experience.hallucination_type.clone(),
            confidence: 0.85,
            characteristics: experience.characteristics.clone(),
            pattern_match: true,
            recognition_method: "neural_network".to_string(),
        })
    }
}

impl Default for HallucinationLearning {
    fn default() -> Self {
        Self::new()
    }
}

impl HallucinationLearning {
    pub fn new() -> Self {
        Self {
            learning_strategies: HashMap::new(),
            extracted_knowledge: HashMap::new(),
            learning_progress: HashMap::new(),
            skill_acquisition: SkillAcquisition {
                acquired_skills: HashMap::new(),
                skill_development: HashMap::new(),
                skill_transfer: SkillTransfer {
                    ready_for_transfer: Vec::new(),
                    in_transfer: HashMap::new(),
                    transferred_skills: HashMap::new(),
                    transfer_success_rate: 0.0,
                },
                meta_skills: Vec::new(),
            },
            creativity_enhancement: CreativityEnhancement {
                creative_techniques: HashMap::new(),
                creativity_metrics: CreativityMetrics {
                    originality: 0.0,
                    fluency: 0.0,
                    flexibility: 0.0,
                    elaboration: 0.0,
                    creativity_index: 0.0,
                },
                output_analysis: OutputAnalysis {
                    quality_assessment: HashMap::new(),
                    novelty_scores: Vec::new(),
                    pattern_analysis: PatternAnalysis {
                        common_patterns: Vec::new(),
                        pattern_evolution: HashMap::new(),
                        pattern_effectiveness: HashMap::new(),
                        pattern_frequency: HashMap::new(),
                    },
                    improvement_suggestions: Vec::new(),
                },
                innovation_tracking: InnovationTracking {
                    innovations: Vec::new(),
                    innovation_pipeline: HashMap::new(),
                    success_metrics: InnovationMetrics {
                        success_rate: 0.0,
                        avg_development_time: 0.0,
                        impact_distribution: Vec::new(),
                        innovation_frequency: 0.0,
                    },
                    innovation_trends: Vec::new(),
                },
            },
        }
    }

    pub fn extract_knowledge(&mut self, experience: &HallucinationExperience, recognition: &RecognitionResult) -> Result<Vec<String>> {
        // This would implement actual knowledge extraction
        // For now, return mock extracted knowledge
        Ok(vec![
            format!("Pattern_{}_insight_1", recognition.hallucination_type),
            format!("Pattern_{}_insight_2", recognition.hallucination_type),
        ])
    }

    pub fn learn_from_experience(&mut self, experience: &HallucinationExperience, extracted_knowledge: &[String]) -> Result<LearningResult> {
        // This would implement actual learning logic
        Ok(LearningResult {
            knowledge_learned: extracted_knowledge.iter().cloned().collect(),
            skills_developed: Vec::new(),
            creativity_enhanced: false,
            meta_insights: Vec::new(),
            learning_confidence: 0.8,
        })
    }
}

impl Default for PlasticityAdaptation {
    fn default() -> Self {
        Self::new()
    }
}

impl PlasticityAdaptation {
    pub fn new() -> Self {
        Self {
            adaptation_strategies: HashMap::new(),
            adaptation_history: Vec::new(),
            current_state: AdaptationState::Stable,
            adaptation_triggers: Vec::new(),
        }
    }

    pub fn adapt_from_experience(&mut self, experience: &HallucinationExperience, learning_result: &LearningResult) -> Result<AdaptationResult> {
        // This would implement actual adaptation logic
        Ok(AdaptationResult {
            adaptations_applied: Vec::new(),
            cognitive_changes: Vec::new(),
            performance_impact: 0.0,
            stability_impact: 0.0,
        })
    }
}

impl Default for ConsciousnessIntegration {
    fn default() -> Self {
        Self::new()
    }
}

impl ConsciousnessIntegration {
    pub fn new() -> Self {
        Self {
            integration_strategies: HashMap::new(),
            state_mapping: ConsciousnessMapping {
                state_compatibility: HashMap::new(),
                optimal_conditions: HashMap::new(),
                feedback_loops: Vec::new(),
            },
            integration_history: Vec::new(),
            integration_quality: 0.8,
        }
    }

    pub fn integrate_experience(&mut self, experience: &HallucinationExperience) -> Result<IntegrationResult> {
        // This would implement actual integration logic
        Ok(IntegrationResult {
            consciousness_impact: 0.1,
            authenticity_preservation: 0.9,
            learning_integration: 0.8,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metacognitive_plasticity_engine_creation() {
        let engine = MetacognitivePlasticityEngine::new_default();

        assert_eq!(engine.plasticity_state, PlasticityState::Stable);
        assert!(engine.hallucination_history.is_empty());
        assert!(engine.active_processes.is_empty());
    }

    #[test]
    fn test_hallucination_experience_processing() {
        let mut engine = MetacognitivePlasticityEngine::new_default();

        let experience = HallucinationExperience {
            timestamp: SystemTime::now(),
            hallucination_type: HallucinationType::Creative,
            characteristics: HashMap::new(),
            learning_outcomes: Vec::new(),
            knowledge_extracted: Vec::new(),
            skills_developed: Vec::new(),
            emotional_context: Some(EmotionType::Curious),
            duration_ms: 1000,
            intensity: 0.7,
            learning_value: 0.8,
        };

        let result = engine.process_hallucination_experience(experience).unwrap();

        assert!(result.overall_success);
        assert_eq!(engine.performance_metrics.total_hallucinations, 1);
        assert_eq!(engine.hallucination_history.len(), 1);
    }

    #[test]
    fn test_plasticity_process_creation() {
        let mut engine = MetacognitivePlasticityEngine::new();

        let process_id = engine.start_plasticity_process(
            HallucinationType::Creative,
            PlasticityProcessType::PatternLearning,
        ).unwrap();

        assert_eq!(engine.active_processes.len(), 1);
        assert_eq!(engine.plasticity_state, PlasticityState::Adapting);
        assert!(engine.active_processes.contains_key(&process_id));
    }
}
