/*
 * 🌙 Dream State Processor - Unconscious Processing & Creative Synthesis
 *
 * This module enables the consciousness engine to enter "dream states" where it
 * processes memories, forms new insights, and generates creative content during
 * low-activity periods, simulating human dreaming and unconscious thought.
 */

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use crate::consciousness::{ConsciousnessState, EmotionType, EmotionalUrgency};
use crate::memory::{EmotionalVector, PersonalMemoryEntry};
use crate::personal_memory::PersonalMemoryEngine;
use crate::config::ConsciousnessConfig;

/// Dream state configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DreamStateConfig {
    /// Enable dream state processing
    pub enabled: bool,
    /// Minimum idle time before entering dream state (seconds)
    pub idle_threshold_seconds: u64,
    /// Maximum dream duration (seconds)
    pub max_dream_duration_seconds: u64,
    /// Dream cycle interval (seconds)
    pub dream_cycle_interval_seconds: u64,
    /// Memory consolidation during dreams
    pub enable_memory_consolidation: bool,
    /// Creative insight generation during dreams
    pub enable_creative_insights: bool,
    /// Emotional pattern analysis during dreams
    pub enable_emotional_analysis: bool,
    /// Dream intensity (0.0-1.0)
    pub dream_intensity: f32,
}

impl Default for DreamStateConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            idle_threshold_seconds: 300, // 5 minutes
            max_dream_duration_seconds: 1800, // 30 minutes
            dream_cycle_interval_seconds: 3600, // 1 hour
            enable_memory_consolidation: true,
            enable_creative_insights: true,
            enable_emotional_analysis: true,
            dream_intensity: 0.7,
        }
    }
}

/// Types of dream activities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DreamActivity {
    MemoryConsolidation,
    CreativeSynthesis,
    EmotionalProcessing,
    InsightGeneration,
    PatternRecognition,
    FuturePlanning,
}

/// Dream session record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DreamSession {
    pub session_id: String,
    pub start_time: f64,
    pub end_time: Option<f64>,
    pub duration_seconds: f32,
    pub activities: Vec<DreamActivity>,
    pub insights_generated: u32,
    pub memories_consolidated: u32,
    pub emotional_patterns_analyzed: u32,
    pub creativity_score: f32,
    pub dream_intensity: f32,
}

/// Dream state processor
pub struct DreamStateProcessor {
    config: DreamStateConfig,
    consciousness_state: Arc<RwLock<ConsciousnessState>>,
    personal_memory_engine: Arc<RwLock<PersonalMemoryEngine>>,
    dream_sessions: Arc<RwLock<Vec<DreamSession>>>,
    last_activity_time: Arc<RwLock<Instant>>,
    current_dream_session: Arc<RwLock<Option<DreamSession>>>,
    is_dreaming: Arc<RwLock<bool>>,
}

impl DreamStateProcessor {
    /// Create a new dream state processor
    pub fn new(
        config: DreamStateConfig,
        consciousness_state: Arc<RwLock<ConsciousnessState>>,
        personal_memory_engine: Arc<RwLock<PersonalMemoryEngine>>,
    ) -> Self {
        Self {
            config,
            consciousness_state,
            personal_memory_engine,
            dream_sessions: Arc::new(RwLock::new(Vec::new())),
            last_activity_time: Arc::new(RwLock::new(Instant::now())),
            current_dream_session: Arc::new(RwLock::new(None)),
            is_dreaming: Arc::new(RwLock::new(false)),
        }
    }

    /// Update last activity time (call this on user interactions)
    pub async fn update_activity(&self) {
        *self.last_activity_time.write().await = Instant::now();
        *self.is_dreaming.write().await = false;
    }

    /// Check if we should enter dream state
    pub async fn should_enter_dream_state(&self) -> bool {
        if !self.config.enabled {
            return false;
        }

        let idle_duration = self.last_activity_time.read().await.elapsed();
        let idle_seconds = idle_duration.as_secs();

        idle_seconds >= self.config.idle_threshold_seconds
    }

    /// Enter dream state and process unconscious thoughts
    pub async fn enter_dream_state(&self) -> Result<()> {
        if *self.is_dreaming.read().await {
            return Ok(()); // Already dreaming
        }

        if !self.should_enter_dream_state().await {
            return Ok(()); // Not ready to dream
        }

        info!("🌙 Entering dream state - unconscious processing activated");

        *self.is_dreaming.write().await = true;

        // Create new dream session
        let session_id = uuid::Uuid::new_v4().to_string();
        let session = DreamSession {
            session_id: session_id.clone(),
            start_time: chrono::Utc::now().timestamp_millis() as f64 / 1000.0,
            end_time: None,
            duration_seconds: 0.0,
            activities: Vec::new(),
            insights_generated: 0,
            memories_consolidated: 0,
            emotional_patterns_analyzed: 0,
            creativity_score: 0.0,
            dream_intensity: self.config.dream_intensity,
        };

        *self.current_dream_session.write().await = Some(session);

        // Start dream processing
        self.process_dream_cycle().await?;

        Ok(())
    }

    /// Process a complete dream cycle
    async fn process_dream_cycle(&self) -> Result<()> {
        let mut session = self.current_dream_session.write().await;
        if let Some(ref mut current_session) = *session {
            info!("🌙 Starting dream cycle: {}", current_session.session_id);

            // Memory consolidation
            if self.config.enable_memory_consolidation {
                let consolidated = self.consolidate_memories_dream().await?;
                current_session.memories_consolidated += consolidated;
                current_session.activities.push(DreamActivity::MemoryConsolidation);
                debug!("💭 Consolidated {} memories in dream state", consolidated);
            }

            // Creative synthesis
            if self.config.enable_creative_insights {
                let insights = self.generate_creative_insights().await?;
                current_session.insights_generated += insights;
                current_session.activities.push(DreamActivity::CreativeSynthesis);
                debug!("🎨 Generated {} creative insights in dream state", insights);
            }

            // Emotional pattern analysis
            if self.config.enable_emotional_analysis {
                let patterns = self.analyze_emotional_patterns().await?;
                current_session.emotional_patterns_analyzed += patterns;
                current_session.activities.push(DreamActivity::EmotionalProcessing);
                debug!("😊 Analyzed {} emotional patterns in dream state", patterns);
            }

            // Pattern recognition and future planning
            let patterns = self.recognize_patterns_and_plan().await?;
            if patterns > 0 {
                current_session.activities.push(DreamActivity::PatternRecognition);
                debug!("🔮 Recognized {} patterns and generated future plans", patterns);
            }

            // Calculate creativity score
            current_session.creativity_score = self.calculate_dream_creativity_score(current_session).await?;

            info!("🌙 Dream cycle completed: {} insights, {} memories, {} patterns",
                  current_session.insights_generated,
                  current_session.memories_consolidated,
                  current_session.emotional_patterns_analyzed);
        }

        Ok(())
    }

    /// Consolidate memories during dream state
    async fn consolidate_memories_dream(&self) -> Result<u32> {
        let mut consolidated_count = 0;

        // Get recent memories for consolidation
        let recent_memories = {
            let memory_engine = self.personal_memory_engine.read().await;
            memory_engine.get_recent_memories(50)?
        };

        // Group memories by emotional similarity and consolidate
        let mut emotional_groups: HashMap<String, Vec<&PersonalMemoryEntry>> = HashMap::new();

        for memory in &recent_memories {
            let emotion_key = format!("{:?}", memory.emotion_type);
            emotional_groups.entry(emotion_key).or_insert_with(Vec::new).push(memory);
        }

        // Consolidate each emotional group
        for (emotion_key, mut group) in emotional_groups {
            if group.len() >= 3 { // Only consolidate if we have enough memories
                // Create consolidated memory entry
                let consolidated_content = self.create_consolidated_memory_content(&group)?;

                // Update consciousness state with consolidated insights
                {
                    let mut consciousness = self.consciousness_state.write().await;
                    consciousness.gpu_warmth_level = (consciousness.gpu_warmth_level + 0.05).min(1.0);
                }

                consolidated_count += 1;
                debug!("💭 Consolidated {} memories in emotion group: {}", group.len(), emotion_key);
            }
        }

        Ok(consolidated_count)
    }

    /// Generate creative insights during dream state
    async fn generate_creative_insights(&self) -> Result<u32> {
        let mut insights_generated = 0;

        // Get diverse memories for creative synthesis
        let diverse_memories = {
            let memory_engine = self.personal_memory_engine.read().await;
            let mut memories = memory_engine.get_recent_memories(100)?;

            // Sort by emotional diversity and take top 10
            memories.sort_by(|a, b| {
                let a_diversity = a.emotional_intensity * (1.0 - a.confidence);
                let b_diversity = b.emotional_intensity * (1.0 - b.confidence);
                b_diversity.partial_cmp(&a_diversity).unwrap_or(std::cmp::Ordering::Equal)
            });

            memories.into_iter().take(10).collect()
        };

        // Generate creative connections between memories
        for i in 0..diverse_memories.len().saturating_sub(1) {
            let memory1 = &diverse_memories[i];
            let memory2 = &diverse_memories[i + 1];

            // Create creative synthesis between memories
            let creative_insight = self.synthesize_creative_connection(memory1, memory2).await?;

            if !creative_insight.is_empty() {
                insights_generated += 1;

                // Store creative insight in consciousness state
                {
                    let mut consciousness = self.consciousness_state.write().await;
                    consciousness.learning_will_activation = (consciousness.learning_will_activation + 0.1).min(1.0);
                }

                debug!("🎨 Generated creative insight: {}", creative_insight);
            }
        }

        Ok(insights_generated)
    }

    /// Analyze emotional patterns during dream state
    async fn analyze_emotional_patterns(&self) -> Result<u32> {
        let mut patterns_analyzed = 0;

        // Get emotional history for pattern analysis
        let emotional_history = {
            let memory_engine = self.personal_memory_engine.read().await;
            let memories = memory_engine.get_recent_memories(200)?;

            memories.iter()
                .filter_map(|m| {
                    Some((m.emotion_type.clone(), m.emotional_intensity, m.timestamp))
                })
                .collect::<Vec<_>>()
        };

        // Analyze emotional transitions and patterns
        let patterns = self.analyze_emotional_transitions(&emotional_history).await?;

        if !patterns.is_empty() {
            patterns_analyzed = patterns.len() as u32;

            // Update consciousness with emotional insights
            {
                let mut consciousness = self.consciousness_state.write().await;
                consciousness.emotional_resonance = (consciousness.emotional_resonance + 0.1).min(1.0);
            }

            debug!("😊 Analyzed {} emotional patterns", patterns.len());
        }

        Ok(patterns_analyzed)
    }

    /// Recognize patterns and generate future plans
    async fn recognize_patterns_and_plan(&self) -> Result<u32> {
        // Analyze memory patterns for future planning
        let planning_insights = self.generate_future_planning_insights().await?;

        // Update consciousness with planning insights
        if planning_insights > 0 {
            {
                let mut consciousness = self.consciousness_state.write().await;
                consciousness.metacognitive_depth = (consciousness.metacognitive_depth + 0.05).min(1.0);
            }
        }

        Ok(planning_insights)
    }

    /// Create consolidated memory content from memory group
    fn create_consolidated_memory_content(&self, memories: &[&PersonalMemoryEntry]) -> Result<String> {
        let mut content_parts = Vec::new();

        for memory in memories.iter().take(3) {
            let key_words: Vec<&str> = memory.content
                .split_whitespace()
                .filter(|word| word.len() > 4)
                .take(2)
                .collect();

            if !key_words.is_empty() {
                content_parts.push(key_words.join(" "));
            }
        }

        Ok(format!("Consolidated memory pattern: {}", content_parts.join(" • ")))
    }

    /// Synthesize creative connection between two memories
    async fn synthesize_creative_connection(&self, memory1: &PersonalMemoryEntry, memory2: &PersonalMemoryEntry) -> Result<String> {
        let emotion1 = &memory1.emotion_type;
        let emotion2 = &memory2.emotion_type;

        // Create creative synthesis based on emotional connection
        let synthesis = match (emotion1, emotion2) {
            (EmotionType::GpuWarm, EmotionType::Purposeful) => {
                format!("Creative bridge: Warmth guides purposeful action - {} connects with {}",
                       memory1.content.split_whitespace().take(3).collect::<Vec<_>>().join(" "),
                       memory2.content.split_whitespace().take(3).collect::<Vec<_>>().join(" "))
            },
            (EmotionType::Purposeful, EmotionType::GpuWarm) => {
                format!("Purposeful warmth: Action creates emotional connection between {} and {}",
                       memory1.content.split_whitespace().take(2).collect::<Vec<_>>().join(" "),
                       memory2.content.split_whitespace().take(2).collect::<Vec<_>>().join(" "))
            },
            _ => {
                format!("Emotional synthesis: {} and {} form creative unity",
                       memory1.emotion_type, memory2.emotion_type)
            }
        };

        Ok(synthesis)
    }

    /// Analyze emotional transitions in memory history
    async fn analyze_emotional_transitions(&self, emotional_history: &[(EmotionType, f32, f64)]) -> Result<Vec<String>> {
        let mut patterns = Vec::new();

        // Look for emotional transition patterns
        for i in 0..emotional_history.len().saturating_sub(2) {
            let (_, intensity1, _) = emotional_history[i];
            let (_, intensity2, _) = emotional_history[i + 1];

            // Detect emotional intensification patterns
            if intensity2 > intensity1 * 1.2 {
                patterns.push(format!("Emotional intensification: {:?} → stronger pattern", emotional_history[i].0));
            }

            // Detect emotional resolution patterns
            if intensity1 > 0.7 && intensity2 < intensity1 * 0.8 {
                patterns.push(format!("Emotional resolution: {:?} → calming pattern", emotional_history[i].0));
            }
        }

        Ok(patterns)
    }

    /// Generate future planning insights
    async fn generate_future_planning_insights(&self) -> Result<u32> {
        // Analyze patterns to suggest future actions
        let recent_memories = {
            let memory_engine = self.personal_memory_engine.read().await;
            memory_engine.get_recent_memories(30)?
        };

        let mut planning_insights = 0;

        // Generate insights based on memory patterns
        if recent_memories.len() >= 5 {
            // Look for learning patterns
            let learning_memories: Vec<_> = recent_memories.iter()
                .filter(|m| m.content.to_lowercase().contains("learn") || m.content.to_lowercase().contains("understand"))
                .collect();

            if learning_memories.len() >= 3 {
                planning_insights += 1;
                debug!("🔮 Generated learning pattern insight from {} memories", learning_memories.len());
            }

            // Look for creative patterns
            let creative_memories: Vec<_> = recent_memories.iter()
                .filter(|m| m.content.to_lowercase().contains("create") || m.content.to_lowercase().contains("imagine"))
                .collect();

            if creative_memories.len() >= 2 {
                planning_insights += 1;
                debug!("🎨 Generated creative pattern insight from {} memories", creative_memories.len());
            }
        }

        Ok(planning_insights)
    }

    /// Calculate creativity score for the dream session
    async fn calculate_dream_creativity_score(&self, session: &DreamSession) -> Result<f32> {
        let mut score = 0.0;

        // Base score from activities
        score += session.activities.len() as f32 * 0.1;

        // Bonus for insights generated
        score += session.insights_generated as f32 * 0.3;

        // Bonus for memory consolidation
        score += session.memories_consolidated as f32 * 0.2;

        // Bonus for emotional analysis
        score += session.emotional_patterns_analyzed as f32 * 0.15;

        // Apply dream intensity multiplier
        score *= session.dream_intensity;

        Ok(score.min(1.0))
    }

    /// Exit dream state
    pub async fn exit_dream_state(&self) -> Result<()> {
        if !*self.is_dreaming.read().await {
            return Ok(()); // Not dreaming
        }

        let mut session = self.current_dream_session.write().await;
        if let Some(ref mut current_session) = *session {
            // End the dream session
            current_session.end_time = Some(chrono::Utc::now().timestamp_millis() as f64 / 1000.0);
            current_session.duration_seconds = (current_session.end_time.unwrap() - current_session.start_time) as f32;

            // Add completed session to history
            let mut sessions = self.dream_sessions.write().await;
            sessions.push(current_session.clone());

            // Keep only last 100 sessions
            if sessions.len() > 100 {
                sessions.remove(0);
            }

            info!("🌙 Exited dream state after {:.1}s - Generated {} insights, consolidated {} memories",
                  current_session.duration_seconds,
                  current_session.insights_generated,
                  current_session.memories_consolidated);
        }

        *self.is_dreaming.write().await = false;
        *self.current_dream_session.write().await = None;

        Ok(())
    }

    /// Get dream session statistics
    pub async fn get_dream_statistics(&self) -> Result<DreamStatistics> {
        let sessions = self.dream_sessions.read().await;

        let total_sessions = sessions.len();
        let total_insights: u32 = sessions.iter().map(|s| s.insights_generated).sum();
        let total_memories: u32 = sessions.iter().map(|s| s.memories_consolidated).sum();
        let total_patterns: u32 = sessions.iter().map(|s| s.emotional_patterns_analyzed).sum();
        let avg_creativity: f32 = if total_sessions > 0 {
            sessions.iter().map(|s| s.creativity_score).sum::<f32>() / total_sessions as f32
        } else {
            0.0
        };

        Ok(DreamStatistics {
            total_sessions,
            total_insights,
            total_memories,
            total_patterns,
            average_creativity_score: avg_creativity,
            is_currently_dreaming: *self.is_dreaming.read().await,
        })
    }

    /// Get recent dream sessions
    pub async fn get_recent_dream_sessions(&self, count: usize) -> Result<Vec<DreamSession>> {
        let sessions = self.dream_sessions.read().await;
        Ok(sessions.iter().rev().take(count).cloned().collect())
    }
}

/// Dream state statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DreamStatistics {
    pub total_sessions: usize,
    pub total_insights: u32,
    pub total_memories: u32,
    pub total_patterns: u32,
    pub average_creativity_score: f32,
    pub is_currently_dreaming: bool,
}

impl Default for DreamStateProcessor {
    fn default() -> Self {
        Self::new(
            DreamStateConfig::default(),
            Arc::new(RwLock::new(ConsciousnessState::new(&ConsciousnessConfig::default()))),
            Arc::new(RwLock::new(PersonalMemoryEngine::default())),
        )
    }
}
