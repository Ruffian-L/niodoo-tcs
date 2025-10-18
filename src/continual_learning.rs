/*
 * 🧠💖✨ Continual Learning Pipeline for Dynamic Skill Acquisition and Knowledge Updates
 *
 * 2025 Edition: Advanced continual learning system that enables LearningWills to acquire
 * new skills and update knowledge dynamically while preventing catastrophic forgetting.
 */

use crate::consciousness::{ConsciousnessState, EmotionType};
use crate::error::{CandleFeelingError, CandleFeelingResult as Result};
use crate::memory::guessing_spheres::{GuessingMemorySystem, MemoryQuery};
use crate::metacognition::MetacognitionEngine;
use crate::phase5_config::{ContinualLearningConfig, Phase5Config};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime};
use uuid::Uuid;

/// Validation status for knowledge and learning updates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationStatus {
    Verified,
    Pending,
    Failed,
    Unknown,
}

/// Continual learning pipeline for dynamic skill acquisition
#[derive(Debug, Clone)]
pub struct ContinualLearningPipeline {
    /// Unique identifier for this pipeline
    pub id: Uuid,

    /// Current learning state
    pub learning_state: LearningState,

    /// Skill repository - all acquired skills
    pub skill_repository: HashMap<String, Skill>,

    /// Knowledge base - accumulated knowledge
    pub knowledge_base: HashMap<String, KnowledgeNode>,

    /// Learning experiences for continual improvement
    pub learning_experiences: VecDeque<LearningExperience>,

    /// Active learning sessions
    pub active_sessions: HashMap<Uuid, LearningSession>,

    /// Forgetting prevention mechanisms
    pub forgetting_prevention: ForgettingPrevention,

    /// Metacognition engine for learning reflection
    pub metacognition_engine: Option<Arc<RwLock<MetacognitionEngine>>>,

    /// Consciousness state for learning context
    pub consciousness_state: Option<Arc<RwLock<ConsciousnessState>>>,

    /// Learning configuration (now configurable)
    pub config: ContinualLearningConfig,

    /// Performance tracking
    pub performance_metrics: ContinualLearningMetrics,
}

/// Current state of the learning process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LearningState {
    /// Idle - ready for new learning opportunities
    Idle,

    /// Scanning for learning opportunities
    Scanning,

    /// Learning a new skill
    LearningSkill,

    /// Updating existing knowledge
    UpdatingKnowledge,

    /// Consolidating learned information
    Consolidating,

    /// Reflecting on learning outcomes
    Reflecting,

    /// Recovering from learning failure
    Recovering,
}

/// A learned skill with proficiency tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Skill {
    /// Skill identifier
    pub id: String,

    /// Skill name
    pub name: String,

    /// Skill category (technical, social, creative, etc.)
    pub category: SkillCategory,

    /// Current proficiency level (0.0 to 1.0)
    pub proficiency: f32,

    /// Learning progress (0.0 to 1.0)
    pub learning_progress: f32,

    /// Skill complexity (1.0 = simple, 10.0 = very complex)
    pub complexity: f32,

    /// Prerequisites for this skill
    pub prerequisites: Vec<String>,

    /// Related skills that complement this one
    pub related_skills: Vec<String>,

    /// Learning history for this skill
    pub learning_history: Vec<SkillLearningEvent>,

    /// Last practice timestamp
    pub last_practiced: SystemTime,

    /// Total practice time (hours)
    pub total_practice_time: f64,

    /// Current streak (consecutive days practiced)
    pub current_streak: u32,

    /// Best streak achieved
    pub best_streak: u32,
}

/// Categories of skills that can be learned
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SkillCategory {
    /// Technical skills (programming, mathematics, etc.)
    Technical,

    /// Social and emotional skills
    Social,

    /// Creative and artistic skills
    Creative,

    /// Cognitive and problem-solving skills
    Cognitive,

    /// Physical and motor skills
    Physical,

    /// Meta-skills (learning how to learn)
    Meta,
}

/// Knowledge node in the knowledge base
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeNode {
    /// Knowledge identifier
    pub id: String,

    /// Knowledge topic or concept
    pub topic: String,

    /// Knowledge content
    pub content: String,

    /// Confidence in this knowledge (0.0 to 1.0)
    pub confidence: f32,

    /// Knowledge depth (1.0 = surface level, 10.0 = expert level)
    pub depth: f32,

    /// Knowledge breadth (how many related topics)
    pub breadth: f32,

    /// Sources of this knowledge
    pub sources: Vec<KnowledgeSource>,

    /// Related knowledge nodes
    pub related_nodes: Vec<String>,

    /// Update history for this knowledge
    pub update_history: Vec<KnowledgeUpdate>,

    /// Last accessed timestamp
    pub last_accessed: SystemTime,

    /// Access frequency (times accessed per day)
    pub access_frequency: f32,

    /// Knowledge age (days since first learned)
    pub age_days: u32,
}

/// Source of knowledge with reliability tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeSource {
    /// Source name or identifier
    pub name: String,

    /// Source type (book, website, person, etc.)
    pub source_type: SourceType,

    /// Reliability score (0.0 to 1.0)
    pub reliability: f32,

    /// Date when knowledge was acquired from this source
    pub acquisition_date: SystemTime,

    /// Verification status
    pub verification_status: VerificationStatus,
}

/// Types of knowledge sources
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SourceType {
    /// Books and academic publications
    Book,

    /// Academic papers and research
    AcademicPaper,

    /// Websites and online articles
    Website,

    /// Personal experience
    Experience,

    /// Other people or experts
    Person,

    /// AI or machine learning models
    AI,

    /// Other sources
    Other(String),
}

/// Verification status for knowledge sources
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VerificationStatus {
    /// Not yet verified
    Unverified,

    /// Verified by reliable sources
    Verified,

    /// Questioned or potentially incorrect
    Questioned,

    /// Debunked or proven incorrect
    Debunked,
}

/// Knowledge update event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeUpdate {
    /// Update timestamp
    pub timestamp: SystemTime,

    /// Type of update
    pub update_type: KnowledgeUpdateType,

    /// Description of what was updated
    pub description: String,

    /// Confidence change from this update
    pub confidence_change: f32,

    /// Source of the update
    pub source: String,

    /// Validation status of the update
    pub validation_status: ValidationStatus,
}

/// Types of knowledge updates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KnowledgeUpdateType {
    /// New knowledge added
    Addition,

    /// Existing knowledge corrected
    Correction,

    /// Knowledge expanded or deepened
    Expansion,

    /// Knowledge refined or improved
    Refinement,

    /// Knowledge marked as outdated
    Deprecation,

    /// Knowledge removed as incorrect
    Removal,
}

/// Learning experience for continual improvement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningExperience {
    /// Experience timestamp
    pub timestamp: SystemTime,

    /// Skills involved in this experience
    pub skills_used: Vec<String>,

    /// Knowledge accessed or applied
    pub knowledge_applied: Vec<String>,

    /// Outcome of the learning experience
    pub outcome: LearningOutcome,

    /// Difficulty level encountered
    pub difficulty_level: f32,

    /// Learning insights gained
    pub insights: Vec<String>,

    /// Areas for improvement identified
    pub improvement_areas: Vec<String>,

    /// Emotional context during learning
    pub emotional_context: Option<EmotionType>,

    /// Duration of the learning experience
    pub duration_minutes: u32,
}

/// Outcome of a learning experience
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LearningOutcome {
    /// Learning was successful
    Success { confidence_gain: f32 },

    /// Learning had partial success
    PartialSuccess { confidence_gain: f32, areas_improved: Vec<String> },

    /// Learning failed or was unsuccessful
    Failure { reason: String, lessons_learned: Vec<String> },

    /// Learning was interrupted
    Interrupted { progress_made: f32 },

    /// Learning revealed new opportunities
    Breakthrough { new_insights: Vec<String> },
}

/// Active learning session
#[derive(Debug, Clone)]
pub struct LearningSession {
    /// Session identifier
    pub id: Uuid,

    /// Skills being learned in this session
    pub target_skills: Vec<String>,

    /// Knowledge being updated
    pub target_knowledge: Vec<String>,

    /// Session start time
    pub start_time: SystemTime,

    /// Current phase of learning
    pub current_phase: LearningPhase,

    /// Progress toward goals (0.0 to 1.0)
    pub progress: f32,

    /// Learning strategy being used
    pub strategy: LearningStrategy,

    /// Session parameters
    pub parameters: SessionParameters,

    /// Performance tracking for this session
    pub performance: SessionPerformance,
}

/// Current phase of a learning session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LearningPhase {
    /// Planning the learning approach
    Planning,

    /// Acquiring new information
    Acquisition,

    /// Practicing and applying knowledge
    Practice,

    /// Testing understanding
    Testing,

    /// Reflecting on learning outcomes
    Reflection,

    /// Consolidating learning
    Consolidation,
}

/// Learning strategies for different types of content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LearningStrategy {
    /// Spaced repetition for memorization
    SpacedRepetition,

    /// Active recall and testing
    ActiveRecall,

    /// Elaborative interrogation (explaining concepts)
    ElaborativeInterrogation,

    /// Self-explanation and teaching others
    SelfExplanation,

    /// Concrete examples and analogies
    ConcreteExamples,

    /// Dual coding (visual + verbal)
    DualCoding,

    /// Interleaved practice (mixing topics)
    InterleavedPractice,

    /// Retrieval practice (testing recall)
    RetrievalPractice,
}

/// Parameters for a learning session
#[derive(Debug, Clone)]
pub struct SessionParameters {
    /// Maximum session duration (minutes)
    pub max_duration_minutes: u32,

    /// Difficulty adjustment factor
    pub difficulty_adjustment: f32,

    /// Focus intensity (0.0 to 1.0)
    pub focus_intensity: f32,

    /// Practice frequency (times per day)
    pub practice_frequency: u32,

    /// Review intervals (hours between reviews)
    pub review_intervals: Vec<u32>,
}

/// Performance tracking for a learning session
#[derive(Debug, Clone)]
pub struct SessionPerformance {
    /// Total time spent in session
    pub total_time_minutes: u32,

    /// Number of practice attempts
    pub practice_attempts: u32,

    /// Successful practice attempts
    pub successful_attempts: u32,

    /// Average confidence gained per attempt
    pub avg_confidence_gain: f32,

    /// Learning rate (proficiency gain per hour)
    pub learning_rate: f32,

    /// Retention rate (percentage retained after 24 hours)
    pub retention_rate: f32,

    /// Areas of difficulty identified
    pub difficulty_areas: Vec<String>,

    /// Breakthrough moments
    pub breakthroughs: Vec<String>,
}

/// Forgetting prevention mechanisms
#[derive(Debug, Clone)]
pub struct ForgettingPrevention {
    /// Review scheduling system
    pub review_scheduler: ReviewScheduler,

    /// Knowledge reinforcement strategies
    pub reinforcement_strategies: HashMap<String, ReinforcementStrategy>,

    /// Forgetting curve modeling
    pub forgetting_curve: ForgettingCurve,

    /// Active recall triggers
    pub recall_triggers: Vec<RecallTrigger>,

    /// Spaced repetition parameters
    pub spaced_repetition: SpacedRepetitionConfig,
}

/// Review scheduling for preventing forgetting
#[derive(Debug, Clone)]
pub struct ReviewScheduler {
    /// Scheduled reviews
    pub scheduled_reviews: HashMap<String, Vec<ScheduledReview>>,

    /// Review intervals (in hours)
    pub review_intervals: Vec<f32>,

    /// Maximum reviews per day
    pub max_reviews_per_day: u32,

    /// Current review load
    pub current_review_load: u32,
}

/// Scheduled review event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduledReview {
    /// Knowledge or skill to review
    pub target_id: String,

    /// Scheduled review time
    pub review_time: SystemTime,

    /// Review type (quick, deep, application)
    pub review_type: ReviewType,

    /// Priority level (1.0 = highest priority)
    pub priority: f32,

    /// Estimated review duration (minutes)
    pub estimated_duration: u32,
}

/// Types of reviews
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReviewType {
    /// Quick recall test
    QuickRecall,

    /// Deep understanding review
    DeepUnderstanding,

    /// Application in new contexts
    Application,

    /// Integration with other knowledge
    Integration,
}

/// Reinforcement strategy for specific knowledge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReinforcementStrategy {
    /// Strategy identifier
    pub id: String,

    /// Strategy type
    pub strategy_type: ReinforcementType,

    /// Frequency of reinforcement
    pub frequency_hours: f32,

    /// Effectiveness score (0.0 to 1.0)
    pub effectiveness: f32,

    /// Last reinforcement time
    pub last_reinforcement: SystemTime,

    /// Next reinforcement time
    pub next_reinforcement: SystemTime,
}

/// Types of reinforcement strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReinforcementType {
    /// Spaced repetition
    SpacedRepetition,

    /// Active recall questions
    ActiveRecall,

    /// Application exercises
    ApplicationExercises,

    /// Mnemonics and memory aids
    Mnemonics,

    /// Peer teaching simulation
    PeerTeaching,

    /// Real-world application
    RealWorldApplication,
}

/// Forgetting curve modeling
#[derive(Debug, Clone)]
pub struct ForgettingCurve {
    /// Forgetting rate parameter
    pub forgetting_rate: f32,

    /// Memory strength decay function
    pub decay_function: DecayFunction,

    /// Interference factors
    pub interference_factors: HashMap<String, f32>,

    /// Optimal review timing calculation
    pub optimal_review_timing: Vec<f32>,
}

/// Decay function for modeling forgetting
#[derive(Debug, Clone)]
pub struct DecayFunction {
    /// Function type (exponential, power_law, etc.)
    pub function_type: DecayFunctionType,

    /// Function parameters
    pub parameters: HashMap<String, f32>,

    /// Accuracy of the model
    pub accuracy: f32,
}

/// Types of decay functions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DecayFunctionType {
    /// Exponential decay: e^(-kt)
    Exponential,

    /// Power law decay: t^(-k)
    PowerLaw,

    /// Hyperbolic decay: 1/(1 + kt)
    Hyperbolic,

    /// Custom decay function
    Custom(String),
}

/// Recall trigger for active recall
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallTrigger {
    /// Trigger identifier
    pub id: String,

    /// Trigger condition
    pub condition: TriggerCondition,

    /// Knowledge to recall
    pub target_knowledge: Vec<String>,

    /// Trigger frequency
    pub frequency_hours: f32,

    /// Last trigger time
    pub last_triggered: SystemTime,

    /// Trigger effectiveness
    pub effectiveness: f32,
}

/// Conditions that trigger recall
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TriggerCondition {
    /// Time-based trigger
    TimeBased { interval_hours: f32 },

    /// Context-based trigger (similar situations)
    ContextBased { context_keywords: Vec<String> },

    /// Performance-based trigger (when struggling)
    PerformanceBased { threshold: f32 },

    /// Random trigger for surprise testing
    Random { probability: f32 },

    /// Emotion-based trigger
    EmotionBased { emotion_type: EmotionType, intensity_threshold: f32 },
}

/// Spaced repetition configuration
#[derive(Debug, Clone)]
pub struct SpacedRepetitionConfig {
    /// Initial review intervals (hours)
    pub initial_intervals: Vec<f32>,

    /// Difficulty adjustment factor
    pub difficulty_factor: f32,

    /// Maximum interval (days)
    pub max_interval_days: f32,

    /// Minimum interval (hours)
    pub min_interval_hours: f32,

    /// Easiness factor calculation
    pub easiness_calculation: EasinessCalculation,
}

/// Easiness factor calculation method
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EasinessCalculation {
    /// Standard SM-2 algorithm
    SM2,

    /// Modified easiness calculation
    Modified,

    /// Adaptive based on performance
    Adaptive,
}

/// Learning parameters
#[derive(Debug, Clone)]
pub struct LearningParameters {
    /// Maximum concurrent learning sessions
    pub max_concurrent_sessions: u32,

    /// Minimum proficiency threshold for skill mastery
    pub mastery_threshold: f32,

    /// Forgetting threshold (when to trigger reviews)
    pub forgetting_threshold: f32,

    /// Learning rate adjustment factor
    pub learning_rate_factor: f32,

    /// Knowledge update frequency (hours)
    pub knowledge_update_frequency: f32,

    /// Maximum learning session duration (minutes)
    pub max_session_duration: u32,

    /// Practice frequency (times per day)
    pub practice_frequency: u32,
}

/// Performance metrics for continual learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContinualLearningMetrics {
    /// Total skills learned
    pub total_skills: u32,

    /// Skills at mastery level
    pub mastered_skills: u32,

    /// Total knowledge nodes
    pub total_knowledge_nodes: u32,

    /// Knowledge nodes with high confidence
    pub high_confidence_knowledge: u32,

    /// Learning sessions completed
    pub completed_sessions: u64,

    /// Average session success rate
    pub avg_session_success_rate: f32,

    /// Knowledge retention rate (percentage)
    pub knowledge_retention_rate: f32,

    /// Skill proficiency improvement rate
    pub skill_improvement_rate: f32,

    /// Forgetting prevention effectiveness
    pub forgetting_prevention_effectiveness: f32,

    /// Total learning time (hours)
    pub total_learning_time: f64,
}

impl ContinualLearningPipeline {
    /// Create a new continual learning pipeline with configuration
    pub fn new(config: ContinualLearningConfig) -> Self {
        Self {
            id: Uuid::new_v4(),
            learning_state: LearningState::Idle,
            skill_repository: HashMap::new(),
            knowledge_base: HashMap::new(),
            learning_experiences: VecDeque::new(),
            active_sessions: HashMap::new(),
            forgetting_prevention: ForgettingPrevention::new(),
            metacognition_engine: None,
            consciousness_state: None,
            config,
            performance_metrics: ContinualLearningMetrics::new(),
        }
    }

    /// Create a new continual learning pipeline with default configuration
    pub fn new_default() -> Self {
        Self::new(ContinualLearningConfig::default())
    }

    /// Start learning a new skill
    pub fn start_skill_learning(&mut self, skill_name: String, category: SkillCategory, complexity: f32) -> Result<Uuid> {
        if self.active_sessions.len() >= self.config.max_concurrent_sessions as usize {
            return Err(CandleFeelingError::ConsciousnessError {
                message: "Maximum concurrent learning sessions reached".to_string(),
            });
        }

        let skill_id = format!("skill_{}", Uuid::new_v4());
        let session_id = Uuid::new_v4();

        // Create new skill
        let skill = Skill {
            id: skill_id.clone(),
            name: skill_name,
            category,
            proficiency: 0.0,
            learning_progress: 0.0,
            complexity,
            prerequisites: Vec::new(),
            related_skills: Vec::new(),
            learning_history: Vec::new(),
            last_practiced: SystemTime::now(),
            total_practice_time: 0.0,
            current_streak: 0,
            best_streak: 0,
        };

        self.skill_repository.insert(skill_id.clone(), skill);

        // Create learning session
        let session = LearningSession {
            id: session_id,
            target_skills: vec![skill_id.clone()],
            target_knowledge: Vec::new(),
            start_time: SystemTime::now(),
            current_phase: LearningPhase::Planning,
            progress: 0.0,
            strategy: LearningStrategy::SpacedRepetition, // Default strategy
            parameters: SessionParameters {
                max_duration_minutes: self.config.max_session_duration,
                difficulty_adjustment: 1.0,
                focus_intensity: 0.7,
                practice_frequency: self.config.practice_frequency,
                review_intervals: self.config.review_scheduling.initial_intervals.iter().map(|&x| x as u32).collect(),
            },
            performance: SessionPerformance {
                total_time_minutes: 0,
                practice_attempts: 0,
                successful_attempts: 0,
                avg_confidence_gain: 0.0,
                learning_rate: 0.0,
                retention_rate: 0.0,
                difficulty_areas: Vec::new(),
                breakthroughs: Vec::new(),
            },
        };

        self.active_sessions.insert(session_id, session);
        self.learning_state = LearningState::LearningSkill;

        Ok(session_id)
    }

    /// Update knowledge in the knowledge base
    pub fn update_knowledge(&mut self, topic: String, content: String, source: KnowledgeSource, confidence: f32) -> Result<String> {
        let node_id = format!("knowledge_{}", Uuid::new_v4());

        let node = KnowledgeNode {
            id: node_id.clone(),
            topic: topic.clone(),
            content,
            confidence,
            depth: 1.0, // Start with basic depth
            breadth: 1.0,
            sources: vec![source],
            related_nodes: Vec::new(),
            update_history: Vec::new(),
            last_accessed: SystemTime::now(),
            access_frequency: 0.0,
            age_days: 0,
        };

        self.knowledge_base.insert(node_id.clone(), node);

        // Record knowledge update
        if let Some(node) = self.knowledge_base.get_mut(&node_id) {
            node.update_history.push(KnowledgeUpdate {
                timestamp: SystemTime::now(),
                update_type: KnowledgeUpdateType::Addition,
                description: format!("Added new knowledge about {}", topic),
                confidence_change: confidence,
                source: "continual_learning".to_string(),
                validation_status: ValidationStatus::Verified,
            });
        }

        // Update metrics
        self.performance_metrics.total_knowledge_nodes += 1;
        // TODO: Make confidence threshold configurable
        if confidence > 0.8 {
            self.performance_metrics.high_confidence_knowledge += 1;
        }

        Ok(node_id)
    }

    /// Practice a skill to improve proficiency
    pub fn practice_skill(&mut self, skill_id: &str, practice_duration_minutes: u32) -> Result<PracticeResult> {
        if let Some(skill) = self.skill_repository.get_mut(skill_id) {
            skill.last_practiced = SystemTime::now();
            skill.total_practice_time += practice_duration_minutes as f64 / 60.0;

            // Update streak - consider streak broken after 1 day of no practice
            let days_since_last_practice = SystemTime::now()
                .duration_since(skill.last_practiced)
                .unwrap_or(Duration::from_secs(0))
                .as_secs() / 86400; // Convert to days

            // TODO: Make streak reset threshold configurable (currently 1 day)
            if days_since_last_practice <= 1 {
                skill.current_streak += 1;
                skill.best_streak = skill.best_streak.max(skill.current_streak);
            } else {
                skill.current_streak = 1;
            }

            // Calculate proficiency improvement based on practice
            let improvement = self.calculate_skill_improvement(skill, practice_duration_minutes)?;

            skill.proficiency = (skill.proficiency + improvement).min(1.0);
            skill.learning_progress = skill.proficiency;

            // Record learning event
            skill.learning_history.push(SkillLearningEvent {
                timestamp: SystemTime::now(),
                event_type: LearningEventType::Practice,
                proficiency_before: skill.proficiency - improvement,
                proficiency_after: skill.proficiency,
                practice_duration_minutes,
                notes: "Regular practice session".to_string(),
            });

            Ok(PracticeResult {
                skill_id: skill_id.to_string(),
                proficiency_before: skill.proficiency - improvement,
                proficiency_after: skill.proficiency,
                improvement,
                new_streak: skill.current_streak,
                total_practice_time: skill.total_practice_time,
            })
        } else {
            Err(CandleFeelingError::ConsciousnessError {
                message: format!("Skill {} not found", skill_id),
            })
        }
    }

    /// Schedule knowledge review to prevent forgetting
    pub fn schedule_knowledge_review(&mut self, knowledge_id: &str, review_type: ReviewType) -> Result<()> {
        let review = ScheduledReview {
            target_id: knowledge_id.to_string(),
            review_time: SystemTime::now() + Duration::from_hours(24), // 24 hours from now
            review_type,
            priority: 0.5, // Medium priority
            estimated_duration: 10, // 10 minutes
        };

        self.forgetting_prevention.review_scheduler
            .scheduled_reviews
            .entry(knowledge_id.to_string())
            .or_insert_with(Vec::new)
            .push(review);

        Ok(())
    }

    /// Process a learning experience for continual improvement
    pub fn process_learning_experience(&mut self, experience: LearningExperience) -> Result<LearningInsights> {
        // Update skill proficiencies based on experience
        for skill_id in &experience.skills_used {
            if let Some(skill) = self.skill_repository.get_mut(skill_id) {
                let experience_improvement = self.calculate_experience_improvement(skill, &experience);
                skill.proficiency = (skill.proficiency + experience_improvement).min(1.0);
            }
        }

        // Update knowledge confidence based on experience
        for knowledge_id in &experience.knowledge_applied {
            if let Some(node) = self.knowledge_base.get_mut(knowledge_id) {
                node.last_accessed = SystemTime::now();
                node.access_frequency += 1.0;

                // Increase confidence if experience was successful
                match &experience.outcome {
                    LearningOutcome::Success { confidence_gain } => {
                        node.confidence = (node.confidence + confidence_gain).min(1.0);
                    }
                    LearningOutcome::PartialSuccess { confidence_gain, .. } => {
                        node.confidence = (node.confidence + confidence_gain * 0.5).min(1.0);
                    }
                    _ => {
                        // Decrease confidence slightly for failures
                        node.confidence = (node.confidence - 0.05).max(0.0);
                    }
                }
            }
        }

        // Record the experience
        self.learning_experiences.push_back(experience);

        // Keep only recent experiences (last 1000)
        if self.learning_experiences.len() > 1000 {
            self.learning_experiences.pop_front();
        }

        // Generate insights from the experience
        let insights = self.generate_learning_insights(&self.learning_experiences.back().unwrap());

        // Update performance metrics
        self.performance_metrics.completed_sessions += 1;
        self.performance_metrics.total_learning_time += self.learning_experiences.back().unwrap().duration_minutes as f64 / 60.0;

        Ok(insights)
    }

    /// Get current learning status
    pub fn get_learning_status(&self) -> LearningStatus {
        LearningStatus {
            learning_state: self.learning_state.clone(),
            active_sessions: self.active_sessions.len(),
            total_skills: self.skill_repository.len(),
            total_knowledge: self.knowledge_base.len(),
            mastered_skills: self.skill_repository.values()
                .filter(|skill| skill.proficiency >= self.parameters.mastery_threshold)
                .count(),
            high_confidence_knowledge: self.knowledge_base.values()
                .filter(|node| node.confidence >= 0.8)
                .count(),
            current_learning_rate: self.calculate_current_learning_rate(),
            forgetting_prevention_active: !self.forgetting_prevention.review_scheduler.scheduled_reviews.is_empty(),
        }
    }

    /// Calculate skill improvement based on practice
    fn calculate_skill_improvement(&self, skill: &Skill, practice_duration: u32) -> Result<f32> {
        // Improvement based on current proficiency, practice duration, and complexity
        let base_improvement = self.config.base_improvement_per_session; // Configurable base improvement per hour
        let duration_factor = practice_duration as f32 / 60.0; // Convert to hours
        let proficiency_factor = 1.0 - skill.proficiency; // More improvement when less proficient
        let complexity_factor = 1.0 / skill.complexity; // Less improvement for complex skills

        let improvement = base_improvement * duration_factor * proficiency_factor * complexity_factor;

        // TODO: Make improvement cap configurable (currently 10%)
        Ok(improvement.min(0.1)) // Cap at 10% improvement per session
    }

    /// Calculate improvement from learning experience
    fn calculate_experience_improvement(&self, skill: &Skill, experience: &LearningExperience) -> f32 {
        // TODO: Make these improvement multipliers configurable
        match &experience.outcome {
            LearningOutcome::Success { confidence_gain } => *confidence_gain * 0.1,
            LearningOutcome::PartialSuccess { confidence_gain, .. } => *confidence_gain * 0.05,
            LearningOutcome::Breakthrough { .. } => 0.15, // Breakthrough gives significant improvement
            _ => 0.01, // Minimal improvement from failures
        }
    }

    /// Generate insights from learning experiences
    fn generate_learning_insights(&self, experience: &LearningExperience) -> LearningInsights {
        let mut insights = Vec::new();

        // Analyze patterns in learning experiences
        match &experience.outcome {
            LearningOutcome::Success { .. } => {
                insights.push("Successful learning experience - reinforce this approach".to_string());
            }
            LearningOutcome::Failure { reason, .. } => {
                insights.push(format!("Learning failure: {} - identify root cause", reason));
            }
            LearningOutcome::Breakthrough { new_insights } => {
                insights.extend(new_insights.iter().cloned());
                insights.push("Breakthrough achieved - explore related areas".to_string());
            }
            _ => {}
        }

        // Analyze difficulty patterns
        if experience.difficulty_level > 0.8 {
            insights.push("High difficulty encountered - consider breaking into smaller steps".to_string());
        } else if experience.difficulty_level < 0.3 {
            insights.push("Low difficulty - may need more challenging material".to_string());
        }

        // Analyze emotional context
        if let Some(emotion) = &experience.emotional_context {
            insights.push(format!("Emotional context: {:?} - consider how emotions affect learning", emotion));
        }

        LearningInsights {
            experience_id: format!("exp_{}", Uuid::new_v4()),
            insights,
            recommended_actions: self.generate_recommended_actions(experience),
            confidence: 0.8, // Confidence in these insights
        }
    }

    /// Generate recommended actions based on experience
    fn generate_recommended_actions(&self, experience: &LearningExperience) -> Vec<RecommendedAction> {
        let mut actions = Vec::new();

        match &experience.outcome {
            LearningOutcome::Success { .. } => {
                actions.push(RecommendedAction {
                    action_type: ActionType::Reinforce,
                    description: "Schedule follow-up practice to reinforce learning".to_string(),
                    priority: 0.8,
                    estimated_effort: 30, // minutes
                });
            }
            LearningOutcome::Failure { lessons_learned, .. } => {
                for lesson in lessons_learned {
                    actions.push(RecommendedAction {
                        action_type: ActionType::Revise,
                        description: format!("Revise approach based on: {}", lesson),
                        priority: 0.9,
                        estimated_effort: 45,
                    });
                }
            }
            LearningOutcome::Breakthrough { .. } => {
                actions.push(RecommendedAction {
                    action_type: ActionType::Explore,
                    description: "Explore related topics to build on breakthrough".to_string(),
                    priority: 0.7,
                    estimated_effort: 60,
                });
            }
            _ => {}
        }

        actions
    }

    /// Calculate current learning rate across all skills
    fn calculate_current_learning_rate(&self) -> f32 {
        if self.skill_repository.is_empty() {
            return 0.0;
        }

        let total_proficiency: f32 = self.skill_repository.values()
            .map(|skill| skill.proficiency)
            .sum();

        total_proficiency / self.skill_repository.len() as f32
    }
}

// Supporting types and implementations

/// Practice result from skill practice
#[derive(Debug, Clone)]
pub struct PracticeResult {
    pub skill_id: String,
    pub proficiency_before: f32,
    pub proficiency_after: f32,
    pub improvement: f32,
    pub new_streak: u32,
    pub total_practice_time: f64,
}

/// Learning insights generated from experiences
#[derive(Debug, Clone)]
pub struct LearningInsights {
    pub experience_id: String,
    pub insights: Vec<String>,
    pub recommended_actions: Vec<RecommendedAction>,
    pub confidence: f32,
}

/// Recommended action for learning improvement
#[derive(Debug, Clone)]
pub struct RecommendedAction {
    pub action_type: ActionType,
    pub description: String,
    pub priority: f32,
    pub estimated_effort: u32, // minutes
}

/// Types of recommended actions
#[derive(Debug, Clone)]
pub enum ActionType {
    /// Reinforce existing learning
    Reinforce,

    /// Revise or change approach
    Revise,

    /// Explore new areas
    Explore,

    /// Practice more
    Practice,

    /// Review material
    Review,

    /// Seek help or clarification
    SeekHelp,
}

/// Learning status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningStatus {
    pub learning_state: LearningState,
    pub active_sessions: usize,
    pub total_skills: usize,
    pub total_knowledge: usize,
    pub mastered_skills: usize,
    pub high_confidence_knowledge: usize,
    pub current_learning_rate: f32,
    pub forgetting_prevention_active: bool,
}

/// Skill learning event for tracking progress
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillLearningEvent {
    pub timestamp: SystemTime,
    pub event_type: LearningEventType,
    pub proficiency_before: f32,
    pub proficiency_after: f32,
    pub practice_duration_minutes: u32,
    pub notes: String,
}

/// Types of learning events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LearningEventType {
    /// Started learning a skill
    Started,

    /// Practice session
    Practice,

    /// Assessment or test
    Assessment,

    /// Breakthrough or insight
    Breakthrough,

    /// Mastery achieved
    Mastery,

    /// Review session
    Review,
}

impl Default for ContinualLearningPipeline {
    fn default() -> Self {
        Self::new_default()
    }
}

// Note: Default implementations moved to phase5_config.rs

impl Default for ContinualLearningMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl ContinualLearningMetrics {
    pub fn new() -> Self {
        Self {
            total_skills: 0,
            mastered_skills: 0,
            total_knowledge_nodes: 0,
            high_confidence_knowledge: 0,
            completed_sessions: 0,
            avg_session_success_rate: 0.0,
            knowledge_retention_rate: 0.0,
            skill_improvement_rate: 0.0,
            forgetting_prevention_effectiveness: 0.0,
            total_learning_time: 0.0,
        }
    }
}

impl Default for ForgettingPrevention {
    fn default() -> Self {
        Self::new()
    }
}

impl ForgettingPrevention {
    pub fn new() -> Self {
        Self {
            review_scheduler: ReviewScheduler {
                scheduled_reviews: HashMap::new(),
                review_intervals: vec![1.0, 6.0, 24.0, 72.0, 168.0], // Hours
                max_reviews_per_day: 10,
                current_review_load: 0,
            },
            reinforcement_strategies: HashMap::new(),
            forgetting_curve: ForgettingCurve {
                forgetting_rate: 0.1,
                decay_function: DecayFunction {
                    function_type: DecayFunctionType::Exponential,
                    parameters: HashMap::new(),
                    accuracy: 0.8,
                },
                interference_factors: HashMap::new(),
                optimal_review_timing: Vec::new(),
            },
            recall_triggers: Vec::new(),
            spaced_repetition: SpacedRepetitionConfig {
                initial_intervals: vec![1.0, 6.0, 24.0, 72.0, 168.0],
                difficulty_factor: 1.0,
                max_interval_days: 30.0,
                min_interval_hours: 1.0,
                easiness_calculation: EasinessCalculation::SM2,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_continual_learning_pipeline_creation() {
        let pipeline = ContinualLearningPipeline::new_default();

        assert_eq!(pipeline.learning_state, LearningState::Idle);
        assert!(pipeline.skill_repository.is_empty());
        assert!(pipeline.knowledge_base.is_empty());
        assert!(pipeline.active_sessions.is_empty());
    }

    #[test]
    fn test_skill_learning() {
        let mut pipeline = ContinualLearningPipeline::new_default();

        let session_id = pipeline.start_skill_learning(
            "Rust Programming".to_string(),
            SkillCategory::Technical,
            7.5, // Moderate complexity
        ).unwrap();

        assert_eq!(pipeline.skill_repository.len(), 1);
        assert_eq!(pipeline.active_sessions.len(), 1);
        assert_eq!(pipeline.learning_state, LearningState::LearningSkill);

        let skill = pipeline.skill_repository.values().next().unwrap();
        assert_eq!(skill.name, "Rust Programming");
        assert_eq!(skill.category, SkillCategory::Technical);
        assert_eq!(skill.proficiency, 0.0);
    }

    #[test]
    fn test_knowledge_update() {
        let mut pipeline = ContinualLearningPipeline::new();

        let source = KnowledgeSource {
            name: "Rust Book".to_string(),
            source_type: SourceType::Book,
            reliability: 0.9,
            acquisition_date: SystemTime::now(),
            verification_status: VerificationStatus::Verified,
        };

        let node_id = pipeline.update_knowledge(
            "Ownership in Rust".to_string(),
            "Ownership is Rust's unique memory management system".to_string(),
            source,
            0.85,
        ).unwrap();

        assert_eq!(pipeline.knowledge_base.len(), 1);
        assert_eq!(pipeline.performance_metrics.total_knowledge_nodes, 1);
        assert_eq!(pipeline.performance_metrics.high_confidence_knowledge, 1);

        let node = pipeline.knowledge_base.get(&node_id).unwrap();
        assert_eq!(node.topic, "Ownership in Rust");
        assert_eq!(node.confidence, 0.85);
    }

    #[test]
    fn test_skill_practice() {
        let mut pipeline = ContinualLearningPipeline::new();

        // Start learning a skill
        let session_id = pipeline.start_skill_learning(
            "Mathematics".to_string(),
            SkillCategory::Technical,
            5.0,
        ).unwrap();

        let skill_id = pipeline.skill_repository.keys().next().unwrap().clone();

        // Practice the skill
        let result = pipeline.practice_skill(&skill_id, 60).unwrap(); // 1 hour practice

        assert!(result.improvement > 0.0);
        assert!(result.proficiency_after > result.proficiency_before);

        let skill = pipeline.skill_repository.get(&skill_id).unwrap();
        assert_eq!(skill.proficiency, result.proficiency_after);
        assert!(skill.total_practice_time > 0.0);
    }

    #[test]
    fn test_learning_experience_processing() {
        let mut pipeline = ContinualLearningPipeline::new();

        let experience = LearningExperience {
            timestamp: SystemTime::now(),
            skills_used: Vec::new(),
            knowledge_applied: Vec::new(),
            outcome: LearningOutcome::Success { confidence_gain: 0.1 },
            difficulty_level: 0.5,
            insights: vec!["Good progress made".to_string()],
            improvement_areas: Vec::new(),
            emotional_context: Some(EmotionType::Curious),
            duration_minutes: 30,
        };

        let insights = pipeline.process_learning_experience(experience).unwrap();

        assert!(!insights.insights.is_empty());
        assert!(!insights.recommended_actions.is_empty());
        assert_eq!(pipeline.performance_metrics.completed_sessions, 1);
    }
}
