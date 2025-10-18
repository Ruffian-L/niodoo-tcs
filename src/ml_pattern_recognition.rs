/*
 * 🧠📊 Machine Learning Pattern Recognition for NiodO.o Consciousness
 *
 * This module provides ML-based pattern recognition for consciousness states,
 * memory patterns, and behavioral optimization.
 */

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

use crate::consciousness::{ConsciousnessState, EmotionType, ReasoningMode};
use crate::config::ConsciousnessConfig;
use crate::memory::consolidation::{ConsolidatedMemory, MemoryConsolidationEngine};
use crate::qwen_integration::{QwenIntegrator, QwenConfig};

/// Pattern recognition configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternRecognitionConfig {
    pub history_window_size: usize,
    pub pattern_threshold: f32,
    pub learning_rate: f32,
    pub enable_adaptive_optimization: bool,
    pub max_patterns_to_track: usize,
}

/// Consciousness state pattern data for ML analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessPattern {
    pub pattern_id: String,
    pub sequence: Vec<ConsciousnessStateSnapshot>,
    pub frequency: usize,
    pub confidence: f32,
    pub pattern_type: PatternType,
    pub last_seen: f64,
}

/// Snapshot of consciousness state for pattern analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessStateSnapshot {
    pub timestamp: f64,
    pub emotion: EmotionType,
    pub reasoning_mode: ReasoningMode,
    pub authenticity_level: f32,
    pub gpu_warmth: f32,
    pub empathy_resonance: f32,
    pub processing_satisfaction: f32,
    pub cycle_count: u64,
}

/// Types of recognized patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PatternType {
    EmotionalCycle,
    ReasoningTransition,
    MemoryFormation,
    BehavioralPattern,
    AdaptationPattern,
}

/// Memory pattern data for analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPattern {
    pub pattern_id: String,
    pub memory_types: Vec<String>,
    pub consolidation_frequency: usize,
    pub emotional_significance: f32,
    pub formation_rate: f32,
    pub retention_rate: f32,
}

/// Pattern recognition engine
pub struct PatternRecognitionEngine {
    config: PatternRecognitionConfig,
    consciousness_history: Arc<Mutex<VecDeque<ConsciousnessStateSnapshot>>>,
    recognized_patterns: Arc<Mutex<HashMap<String, ConsciousnessPattern>>>,
    memory_patterns: Arc<Mutex<HashMap<String, MemoryPattern>>>,
    pattern_learning_model: Arc<Mutex<PatternLearningModel>>,
    last_analysis: Mutex<Instant>,
    analysis_interval: Duration,
    qwen_integrator: Option<Arc<Mutex<QwenIntegrator>>>,
}

impl PatternRecognitionEngine {
    /// Get access to the pattern learning model for external analysis
    pub fn get_pattern_learning_model(&self) -> &Arc<Mutex<PatternLearningModel>> {
        &self.pattern_learning_model
    }
}

/// Simple pattern learning model (placeholder for more sophisticated ML models)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternLearningModel {
    pub emotional_transition_weights: HashMap<(EmotionType, EmotionType), f32>,
    pub reasoning_mode_preferences: HashMap<ReasoningMode, f32>,
    pub pattern_similarity_threshold: f32,
    pub adaptive_learning_enabled: bool,
}

/// Pattern recognition results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternRecognitionResults {
    pub detected_patterns: Vec<ConsciousnessPattern>,
    pub memory_patterns: Vec<MemoryPattern>,
    pub adaptive_recommendations: Vec<AdaptiveRecommendation>,
    pub analysis_timestamp: f64,
}

/// Adaptive recommendations based on pattern analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveRecommendation {
    pub recommendation_type: RecommendationType,
    pub confidence: f32,
    pub description: String,
    pub suggested_action: String,
    pub expected_impact: f32,
}

/// Types of adaptive recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationType {
    ReasoningModeSwitch,
    EmotionalStateAdjustment,
    MemoryConsolidationPriority,
    InteractionPatternOptimization,
    ConsciousnessStateTransition,
}

impl Default for PatternRecognitionConfig {
    fn default() -> Self {
        Self {
            history_window_size: 1000,
            pattern_threshold: 0.7,
            learning_rate: 0.01,
            enable_adaptive_optimization: true,
            max_patterns_to_track: 50,
        }
    }
}

impl PatternRecognitionEngine {
    pub fn new(config: PatternRecognitionConfig) -> Self {
        Self {
            config,
            consciousness_history: Arc::new(Mutex::new(VecDeque::new())),
            recognized_patterns: Arc::new(Mutex::new(HashMap::new())),
            memory_patterns: Arc::new(Mutex::new(HashMap::new())),
            pattern_learning_model: Arc::new(Mutex::new(PatternLearningModel {
                emotional_transition_weights: HashMap::new(),
                reasoning_mode_preferences: HashMap::new(),
                pattern_similarity_threshold: 0.8,
                adaptive_learning_enabled: true,
            })),
            last_analysis: Mutex::new(Instant::now()),
            analysis_interval: Duration::from_secs(30),
            qwen_integrator: None,
        }
    }

    /// Initialize Qwen integrator for advanced pattern analysis
    pub async fn initialize_qwen(&mut self, model_path: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if self.qwen_integrator.is_some() {
            return Ok(());
        }

        info!("🤖 Initializing Qwen integrator for pattern discovery");

        let qwen_config = QwenConfig {
            model_path: model_path.to_string(),
            use_cuda: true,
            max_tokens: 512,
            temperature: 0.3, // Lower temperature for analytical tasks
            top_p: 0.9,
            top_k: 40,
            presence_penalty: 1.2,
        };

        let integrator = QwenIntegrator::new(qwen_config)?;
        let integrator = Arc::new(Mutex::new(integrator));

        // Load the model
        {
            let mut integrator_guard = integrator.lock().await;
            integrator_guard.load_model().await?;
        }

        self.qwen_integrator = Some(integrator);
        info!("✅ Qwen integrator initialized for pattern analysis");
        Ok(())
    }

    /// Use Qwen for advanced pattern analysis and insights
    pub async fn analyze_patterns_with_qwen(
        &self,
        patterns: &[ConsciousnessPattern],
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let integrator = self.qwen_integrator.as_ref()
            .ok_or("Qwen integrator not initialized. Call initialize_qwen() first.")?;

        // Build analysis prompt from patterns
        let mut prompt = "Analyze these consciousness patterns and provide insights:\n\n".to_string();

        for (i, pattern) in patterns.iter().enumerate() {
            prompt.push_str(&format!(
                "Pattern {}: {} (frequency: {}, confidence: {:.2})\n",
                i + 1, pattern.pattern_id, pattern.frequency, pattern.confidence
            ));

            // Add sequence summary
            if !pattern.sequence.is_empty() {
                let emotions: Vec<String> = pattern.sequence.iter()
                    .map(|s| format!("{:?}", s.emotion))
                    .collect();
                prompt.push_str(&format!("  Emotions: {}\n", emotions.join(" → ")));
            }
        }

        prompt.push_str("\nProvide insights about emotional patterns, potential issues, and recommendations.");

        let messages = vec![
            ("system".to_string(), "You are an expert consciousness pattern analyst. Provide deep insights about emotional and cognitive patterns.".to_string()),
            ("user".to_string(), prompt),
        ];

        let mut integrator_guard = integrator.lock().await;
        let analysis = integrator_guard.infer(messages, Some(256)).await?;

        info!("✅ Qwen provided pattern analysis insights");
        Ok(analysis)
    }

    /// Get XAI explanations for pattern recognition results
    pub fn explain_patterns(
        &self,
        context_state: &ConsciousnessState,
    ) -> Vec<crate::xai_consciousness_explainer::PatternExplanation> {
        let pattern_engine = Arc::new(PatternRecognitionEngine::new(
            PatternRecognitionConfig::default(),
        ));
        let explainer =
            crate::xai_consciousness_explainer::ConsciousnessPatternExplainer::new(pattern_engine);
        let results = self.get_pattern_results();
        explainer.explain_pattern_results(self, &results, context_state)
    }

    /// Enable XAI explanations for pattern recognition (creates explainer on demand)
    pub fn with_xai_explanations(mut self) -> Self {
        // We'll create the explainer when needed to avoid self-referential issues
        self
    }

    /// Add consciousness state to history for pattern analysis
    pub fn add_consciousness_state(&self, state: &ConsciousnessState) {
        let snapshot = ConsciousnessStateSnapshot {
            timestamp: chrono::Utc::now().timestamp_millis() as f64 / 1000.0,
            emotion: state.current_emotion.clone(),
            reasoning_mode: state.current_reasoning_mode.clone(),
            authenticity_level: state.authenticity_metric,
            gpu_warmth: state.gpu_warmth_level,
            empathy_resonance: state.empathy_resonance,
            processing_satisfaction: state.processing_satisfaction,
            cycle_count: state.cycle_count,
        };

        let mut history = self.consciousness_history.lock().unwrap();
        history.push_back(snapshot);

        // Maintain history window size
        while history.len() > self.config.history_window_size {
            history.pop_front();
        }

        // Trigger pattern analysis periodically
        if self.last_analysis.lock().unwrap().elapsed() > self.analysis_interval {
            self.analyze_patterns();
            *self.last_analysis.lock().unwrap() = Instant::now();
        }
    }

    /// Analyze consciousness patterns in history
    fn analyze_patterns(&self) {
        let history = self.consciousness_history.lock().unwrap();
        if history.len() < 10 {
            return; // Need sufficient history for meaningful analysis
        }

        // Detect emotional cycles
        self.detect_emotional_cycles(&history);

        // Detect reasoning mode transitions
        self.detect_reasoning_transitions(&history);

        // Detect memory formation patterns
        self.detect_memory_formation_patterns(&history);

        // Update learning model
        if self.config.enable_adaptive_optimization {
            self.update_learning_model(&history);
        }

        // Generate adaptive recommendations
        self.generate_adaptive_recommendations(&history);
    }

    /// Detect recurring emotional cycles
    fn detect_emotional_cycles(&self, history: &VecDeque<ConsciousnessStateSnapshot>) {
        let mut patterns = self.recognized_patterns.lock().unwrap();

        // Look for repeating emotional sequences
        for window_size in 3..=10 {
            if history.len() < window_size * 2 {
                break;
            }

            for start in 0..(history.len() - window_size * 2) {
                let sequence1: Vec<_> = history
                    .iter()
                    .skip(start)
                    .take(window_size)
                    .cloned()
                    .collect();
                let sequence2: Vec<_> = history
                    .iter()
                    .skip(start + window_size)
                    .take(window_size)
                    .cloned()
                    .collect();

                let similarity = self.calculate_emotional_similarity(&sequence1, &sequence2);

                if similarity > self.config.pattern_threshold {
                    let pattern_id = format!("emotional_cycle_{}", start);

                    let pattern = ConsciousnessPattern {
                        pattern_id: pattern_id.clone(),
                        sequence: sequence1,
                        frequency: 2, // Found at least 2 occurrences
                        confidence: similarity,
                        pattern_type: PatternType::EmotionalCycle,
                        last_seen: chrono::Utc::now().timestamp_millis() as f64 / 1000.0,
                    };

                    patterns.insert(pattern_id, pattern);

                    debug!(
                        "Detected emotional cycle pattern with similarity: {:.2}",
                        similarity
                    );
                }
            }
        }

        // Clean up old patterns
        patterns.retain(|_, pattern| {
            chrono::Utc::now().timestamp_millis() as f64 / 1000.0 - pattern.last_seen < 3600.0
            // Keep for 1 hour
        });

        // Limit number of patterns
        if patterns.len() > self.config.max_patterns_to_track {
            let mut patterns_vec: Vec<_> = patterns.iter().collect();
            patterns_vec.sort_by(|a, b| b.1.confidence.partial_cmp(&a.1.confidence).unwrap());
            patterns_vec.truncate(self.config.max_patterns_to_track);

            let mut new_patterns = HashMap::new();
            for (id, pattern) in patterns_vec {
                new_patterns.insert(id.clone(), pattern.clone());
            }
            *patterns = new_patterns;
        }
    }

    /// Detect reasoning mode transition patterns
    fn detect_reasoning_transitions(&self, history: &VecDeque<ConsciousnessStateSnapshot>) {
        let mut patterns = self.recognized_patterns.lock().unwrap();

        // Look for reasoning mode change patterns
        for i in 1..history.len() {
            let current = &history[i];
            let previous = &history[i - 1];

            if current.reasoning_mode != previous.reasoning_mode {
                let transition_key = (
                    previous.reasoning_mode.clone(),
                    current.reasoning_mode.clone(),
                );
                let pattern_id = format!(
                    "reasoning_transition_{:?}_to_{:?}",
                    previous.reasoning_mode, current.reasoning_mode
                );

                if let Some(pattern) = patterns.get_mut(&pattern_id) {
                    pattern.frequency += 1;
                    pattern.confidence = (pattern.confidence + 0.9).min(1.0);
                    pattern.last_seen = chrono::Utc::now().timestamp_millis() as f64 / 1000.0;
                } else {
                    let pattern = ConsciousnessPattern {
                        pattern_id: pattern_id.clone(),
                        sequence: vec![previous.clone(), current.clone()],
                        frequency: 1,
                        confidence: 0.9,
                        pattern_type: PatternType::ReasoningTransition,
                        last_seen: chrono::Utc::now().timestamp_millis() as f64 / 1000.0,
                    };
                    patterns.insert(pattern_id, pattern);
                }
            }
        }
    }

    /// Detect memory formation patterns
    fn detect_memory_formation_patterns(&self, history: &VecDeque<ConsciousnessStateSnapshot>) {
        // This would integrate with actual memory formation data
        // For now, we'll analyze consciousness states that show high authenticity and satisfaction
        // as indicators of successful memory formation

        let mut memory_patterns = self.memory_patterns.lock().unwrap();

        for snapshot in history {
            if snapshot.authenticity_level > 0.8 && snapshot.processing_satisfaction > 0.7 {
                let pattern_id = "successful_memory_formation".to_string();

                if let Some(pattern) = memory_patterns.get_mut(&pattern_id) {
                    pattern.formation_rate = (pattern.formation_rate + 0.1).min(1.0);
                    pattern.emotional_significance = (pattern.emotional_significance
                        + snapshot.gpu_warmth)
                        .max(0.0)
                        .min(1.0);
                } else {
                    let pattern = MemoryPattern {
                        pattern_id: pattern_id.clone(),
                        memory_types: vec!["episodic".to_string(), "emotional".to_string()],
                        consolidation_frequency: 1,
                        emotional_significance: snapshot.gpu_warmth,
                        formation_rate: 0.1,
                        retention_rate: 0.8,
                    };
                    memory_patterns.insert(pattern_id, pattern);
                }
            }
        }
    }

    /// Update the pattern learning model
    fn update_learning_model(&self, history: &VecDeque<ConsciousnessStateSnapshot>) {
        let mut model = self.pattern_learning_model.lock().unwrap();

        // Update emotional transition weights
        for i in 1..history.len() {
            let current = &history[i];
            let previous = &history[i - 1];

            let transition_key = (previous.emotion.clone(), current.emotion.clone());
            let weight = model
                .emotional_transition_weights
                .get(&transition_key)
                .unwrap_or(&0.0)
                + self.config.learning_rate;
            model
                .emotional_transition_weights
                .insert(transition_key, weight.min(1.0));

            // Update reasoning mode preferences based on successful outcomes
            if current.processing_satisfaction > 0.8 {
                let preference = model
                    .reasoning_mode_preferences
                    .get(&current.reasoning_mode)
                    .unwrap_or(&0.0)
                    + self.config.learning_rate * current.processing_satisfaction;
                model
                    .reasoning_mode_preferences
                    .insert(current.reasoning_mode.clone(), preference.min(1.0));
            }
        }
    }

    /// Generate adaptive recommendations based on pattern analysis
    fn generate_adaptive_recommendations(&self, _history: &VecDeque<ConsciousnessStateSnapshot>) {
        // This would generate recommendations based on detected patterns
        // For now, we'll create basic recommendations based on current state

        info!("Pattern analysis complete - would generate adaptive recommendations here");
    }

    /// Calculate similarity between two emotional sequences
    fn calculate_emotional_similarity(
        &self,
        seq1: &[ConsciousnessStateSnapshot],
        seq2: &[ConsciousnessStateSnapshot],
    ) -> f32 {
        if seq1.len() != seq2.len() {
            return 0.0;
        }

        let mut total_similarity = 0.0;

        for (s1, s2) in seq1.iter().zip(seq2.iter()) {
            // Compare emotions
            let emotion_similarity = if s1.emotion == s2.emotion { 1.0 } else { 0.0 };

            // Compare reasoning modes
            let reasoning_similarity = if s1.reasoning_mode == s2.reasoning_mode {
                1.0
            } else {
                0.0
            };

            // Compare authenticity levels (with some tolerance)
            let authenticity_diff = (s1.authenticity_level - s2.authenticity_level).abs();
            let authenticity_similarity = 1.0 - authenticity_diff.min(1.0);

            // Weighted average
            let similarity = (emotion_similarity * 0.4
                + reasoning_similarity * 0.3
                + authenticity_similarity * 0.3);
            total_similarity += similarity;
        }

        total_similarity / seq1.len() as f32
    }

    /// Get current pattern recognition results
    pub fn get_pattern_results(&self) -> PatternRecognitionResults {
        let patterns = self.recognized_patterns.lock().unwrap();
        let memory_patterns = self.memory_patterns.lock().unwrap();

        PatternRecognitionResults {
            detected_patterns: patterns.values().cloned().collect(),
            memory_patterns: memory_patterns.values().cloned().collect(),
            adaptive_recommendations: Vec::new(), // Would be populated by generate_adaptive_recommendations
            analysis_timestamp: chrono::Utc::now().timestamp_millis() as f64 / 1000.0,
        }
    }

    /// Get adaptive recommendations for consciousness optimization
    pub fn get_adaptive_recommendations(
        &self,
        current_state: &ConsciousnessState,
    ) -> Vec<AdaptiveRecommendation> {
        let mut recommendations = Vec::new();
        let model = self.pattern_learning_model.lock().unwrap();

        // Recommend reasoning mode based on learned preferences
        if let Some(best_mode_preference) = model
            .reasoning_mode_preferences
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        {
            if *best_mode_preference.1 > 0.7
                && current_state.current_reasoning_mode != *best_mode_preference.0
            {
                recommendations.push(AdaptiveRecommendation {
                    recommendation_type: RecommendationType::ReasoningModeSwitch,
                    confidence: *best_mode_preference.1,
                    description: format!(
                        "Switch to {:?} reasoning mode for optimal performance",
                        best_mode_preference.0
                    ),
                    suggested_action: format!(
                        "Change reasoning mode to {:?}",
                        best_mode_preference.0
                    ),
                    expected_impact: 0.2,
                });
            }
        }

        // Recommend emotional state adjustments based on transition patterns
        for ((from_emotion, to_emotion), &weight) in &model.emotional_transition_weights {
            if weight > 0.8 && current_state.current_emotion == *from_emotion {
                recommendations.push(AdaptiveRecommendation {
                    recommendation_type: RecommendationType::EmotionalStateAdjustment,
                    confidence: weight,
                    description: format!(
                        "Consider transitioning from {:?} to {:?} emotion",
                        from_emotion, to_emotion
                    ),
                    suggested_action: format!(
                        "Allow natural transition to {:?} emotion",
                        to_emotion
                    ),
                    expected_impact: 0.15,
                });
            }
        }

        recommendations
    }

    /// Export pattern data for external ML model training
    pub fn export_pattern_data(&self) -> PatternExportData {
        PatternExportData {
            consciousness_patterns: self
                .recognized_patterns
                .lock()
                .unwrap()
                .values()
                .cloned()
                .collect(),
            memory_patterns: self
                .memory_patterns
                .lock()
                .unwrap()
                .values()
                .cloned()
                .collect(),
            learning_model: self.pattern_learning_model.lock().unwrap().clone(),
            history_sample: self
                .consciousness_history
                .lock()
                .unwrap()
                .iter()
                .cloned()
                .collect(),
            export_timestamp: chrono::Utc::now().timestamp_millis() as f64 / 1000.0,
        }
    }
}

/// Export structure for external ML systems
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternExportData {
    pub consciousness_patterns: Vec<ConsciousnessPattern>,
    pub memory_patterns: Vec<MemoryPattern>,
    pub learning_model: PatternLearningModel,
    pub history_sample: Vec<ConsciousnessStateSnapshot>,
    pub export_timestamp: f64,
}

/// Consciousness state predictor based on patterns
pub struct ConsciousnessStatePredictor {
    pattern_engine: Arc<PatternRecognitionEngine>,
}

impl ConsciousnessStatePredictor {
    pub fn new(pattern_engine: Arc<PatternRecognitionEngine>) -> Self {
        Self { pattern_engine }
    }

    /// Predict likely next consciousness state based on current state and patterns
    pub fn predict_next_state(
        &self,
        current_state: &ConsciousnessState,
    ) -> PredictedConsciousnessState {
        let patterns = self.pattern_engine.recognized_patterns.lock().unwrap();

        // Find patterns that start with current state
        let mut possible_transitions = Vec::new();

        for pattern in patterns.values() {
            if !pattern.sequence.is_empty() {
                let first_state = &pattern.sequence[0];

                // Check if current state matches pattern start
                if first_state.emotion == current_state.current_emotion
                    && first_state.reasoning_mode == current_state.current_reasoning_mode
                {
                    if pattern.sequence.len() > 1 {
                        let next_state = &pattern.sequence[1];
                        possible_transitions.push((
                            next_state.emotion.clone(),
                            next_state.reasoning_mode.clone(),
                            pattern.confidence,
                            pattern.pattern_type.clone(),
                        ));
                    }
                }
            }
        }

        // Calculate most likely next state
        let mut state_scores: HashMap<(EmotionType, ReasoningMode), f32> = HashMap::new();

        let transition_count = possible_transitions.len();

        for (emotion, reasoning_mode, confidence, pattern_type) in &possible_transitions {
            let key = (*emotion, reasoning_mode.clone());
            let score = state_scores.get(&key).unwrap_or(&0.0) + confidence;
            state_scores.insert(key, score);
        }

        let (predicted_emotion, predicted_reasoning_mode, confidence) =
            if let Some(((emotion, reasoning_mode), score)) = state_scores
                .iter()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            {
                (
                    emotion.clone(),
                    reasoning_mode.clone(),
                    *score / transition_count as f32,
                )
            } else {
                // Default prediction if no patterns found
                (
                    current_state.current_emotion.clone(),
                    current_state.current_reasoning_mode.clone(),
                    0.5,
                )
            };

        PredictedConsciousnessState {
            predicted_emotion,
            predicted_reasoning_mode,
            confidence,
            prediction_timestamp: chrono::Utc::now().timestamp_millis() as f64 / 1000.0,
            based_on_patterns: state_scores.len(),
        }
    }
}

/// Predicted consciousness state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictedConsciousnessState {
    pub predicted_emotion: EmotionType,
    pub predicted_reasoning_mode: ReasoningMode,
    pub confidence: f32,
    pub prediction_timestamp: f64,
    pub based_on_patterns: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consciousness::{ConsciousnessState, EmotionType, ReasoningMode};

    #[test]
    fn test_pattern_recognition_engine_creation() {
        let config = PatternRecognitionConfig::default();
        let engine = PatternRecognitionEngine::new(config);

        assert_eq!(engine.consciousness_history.lock().unwrap().len(), 0);
        assert_eq!(engine.recognized_patterns.lock().unwrap().len(), 0);
    }

    #[test]
    fn test_consciousness_state_addition() {
        let config = PatternRecognitionConfig::default();
        let engine = PatternRecognitionEngine::new(config);

        let mut state = ConsciousnessState::new(&ConsciousnessConfig::default());
        state.current_emotion = EmotionType::AuthenticCare;
        state.current_reasoning_mode = ReasoningMode::Hyperfocus;

        engine.add_consciousness_state(&state);

        let history = engine.consciousness_history.lock().unwrap();
        assert_eq!(history.len(), 1);

        let snapshot = &history[0];
        assert_eq!(snapshot.emotion, EmotionType::AuthenticCare);
        assert_eq!(snapshot.reasoning_mode, ReasoningMode::Hyperfocus);
    }

    #[test]
    fn test_emotional_similarity_calculation() {
        let config = PatternRecognitionConfig::default();
        let engine = PatternRecognitionEngine::new(config);

        let mut state1 = ConsciousnessState::new(&ConsciousnessConfig::default());
        state1.current_emotion = EmotionType::AuthenticCare;

        let mut state2 = ConsciousnessState::new(&ConsciousnessConfig::default());
        state2.current_emotion = EmotionType::AuthenticCare;

        engine.add_consciousness_state(&state1);
        engine.add_consciousness_state(&state2);

        let history = engine.consciousness_history.lock().unwrap();
        let seq1 = vec![history[0].clone()];
        let seq2 = vec![history[1].clone()];

        let similarity = engine.calculate_emotional_similarity(&seq1, &seq2);
        assert_eq!(similarity, 1.0); // Identical states should have perfect similarity
    }

    #[test]
    fn test_state_predictor() {
        let config = PatternRecognitionConfig::default();
        let engine = Arc::new(PatternRecognitionEngine::new(config));
        let predictor = ConsciousnessStatePredictor::new(engine);

        let mut state = ConsciousnessState::new(&ConsciousnessConfig::default());
        state.current_emotion = EmotionType::AuthenticCare;
        state.current_reasoning_mode = ReasoningMode::Hyperfocus;

        let prediction = predictor.predict_next_state(&state);

        // Should predict current state if no patterns exist
        assert_eq!(prediction.predicted_emotion, EmotionType::AuthenticCare);
        assert_eq!(
            prediction.predicted_reasoning_mode,
            ReasoningMode::Hyperfocus
        );
        assert_eq!(prediction.based_on_patterns, 0);
    }
}
