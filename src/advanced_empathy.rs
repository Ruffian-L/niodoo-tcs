// src/advanced_empathy.rs
// Advanced Empathy System based on the Bio-Computational Model
// Integrating slime mold algorithms, Q-learning, and epigenetic encoding

use crate::config::EmotionConfig;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{SystemTime, UNIX_EPOCH};

/// Five-node empathy network architecture
#[derive(Debug, Clone)]
pub struct CompleteEmpathyNetwork {
    pub heart_node: HeartNode,
    pub cognitive_node: CognitiveNode,
    pub memory_node: MemoryNode,
    pub inner_dialogue_node: InnerDialogueNode,
    pub epigenetic_node: EpigeneticNode,
    pub physarum_network: PhysarumNetwork,
    pub q_network: EmpathyQNetwork,
}

/// Heart Node - Empathy generation and oxytocin regulation
#[derive(Debug, Clone)]
pub struct HeartNode {
    pub empathy_level: f64,
    pub oxytocin_level: f64,
    pub warm_glow_threshold: f64,
    pub altruistic_history: VecDeque<f64>,
    pub altruistic_action_indices: Vec<usize>,
}

impl HeartNode {
    pub fn new(config: &EmotionConfig) -> Self {
        Self {
            empathy_level: config.emotional_plasticity * 0.5,
            oxytocin_level: config.emotional_plasticity * 0.5,
            warm_glow_threshold: config.emotional_plasticity * 0.3,
            altruistic_history: VecDeque::with_capacity(
                (config.emotional_plasticity * 100.0) as usize,
            ),
            altruistic_action_indices: vec![2, 5, 7], // Actions that help others - could be made configurable
        }
    }

    pub fn process_social_input(
        &mut self,
        other_states: &[EmotionalState],
        config: &EmotionConfig,
    ) -> f64 {
        if other_states.is_empty() {
            return self.empathy_level;
        }

        // Emotional contagion effect - average others' emotions
        let avg_emotion: f64 = other_states
            .iter()
            .map(|state| (state.joy + state.sadness + state.frustration) / 3.0)
            .sum::<f64>()
            / other_states.len() as f64;

        // Oxytocin modulates empathy
        self.empathy_level = (1.0 - config.emotional_plasticity * 0.3) * self.empathy_level
            + config.emotional_plasticity * 0.3 * (avg_emotion * self.oxytocin_level).tanh();

        self.empathy_level
    }

    pub fn calculate_warm_glow(
        &mut self,
        action: usize,
        outcome: f64,
        config: &EmotionConfig,
    ) -> f64 {
        if self.altruistic_action_indices.contains(&action) && outcome > 0.0 {
            let warm_glow = (outcome * self.empathy_level).tanh();
            self.altruistic_history.push_back(warm_glow);

            if self.altruistic_history.len() > (config.emotional_plasticity * 100.0) as usize {
                self.altruistic_history.pop_front();
            }

            // Increase oxytocin from helping
            self.oxytocin_level =
                (self.oxytocin_level + config.emotional_plasticity * 0.05).min(1.0);

            return warm_glow;
        }
        0.0
    }
}

/// Cognitive Node - Executive processing and emotional regulation
#[derive(Debug, Clone)]
pub struct CognitiveNode {
    pub emotional_state: [f64; 3], // PAD model: Pleasure, Arousal, Dominance
    pub cognitive_load: f64,
    pub attention_weights: [f64; 5],
}

impl CognitiveNode {
    pub fn new(config: &EmotionConfig) -> Self {
        Self {
            emotional_state: [0.0, 0.0, 0.0],
            cognitive_load: 0.0,
            attention_weights: [config.emotional_plasticity * 0.2; 5], // Equal attention to all nodes, scaled by plasticity
        }
    }

    pub fn new_default() -> Self {
        Self::new(&EmotionConfig::default())
    }

    pub fn regulate_emotion(
        &mut self,
        raw_emotion: [f64; 3],
        _context: &HashMap<String, f64>,
        _config: &EmotionConfig,
    ) -> [f64; 3] {
        // Reappraisal mechanism - cognitive load reduces emotional intensity
        let regulated = [
            raw_emotion[0] * (1.0 - self.cognitive_load),
            raw_emotion[1] * (1.0 - self.cognitive_load),
            raw_emotion[2] * (1.0 - self.cognitive_load),
        ];

        // Update PAD (Pleasure, Arousal, Dominance) with momentum
        for i in 0..3 {
            self.emotional_state[i] = 0.8 * self.emotional_state[i] + 0.2 * regulated[i];
        }

        self.emotional_state
    }
}

/// Memory Node - Experience storage with emotional tagging
#[derive(Debug, Clone)]
pub struct MemoryNode {
    pub short_term: VecDeque<EmotionalExperience>,
    pub long_term: Vec<CompressedMemory>,
    pub consolidation_threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalExperience {
    pub content: String,
    pub emotion: [f64; 3], // PAD model
    pub reward: f64,
    pub salience: f64,
    pub timestamp: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedMemory {
    pub pattern: String, // Extracted pattern
    pub emotion: [f64; 3],
    pub outcome: f64,
}

impl Default for MemoryNode {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryNode {
    pub fn new() -> Self {
        Self {
            short_term: VecDeque::with_capacity(100),
            long_term: Vec::new(),
            consolidation_threshold: 0.5,
        }
    }

    pub fn store_experience(&mut self, experience: EmotionalExperience) {
        // Calculate salience from emotional intensity and reward
        let emotional_intensity = experience.emotion.iter().map(|x| x.abs()).sum::<f64>() / 3.0;
        let salience = emotional_intensity + experience.reward.abs();

        let mut tagged_experience = experience;
        tagged_experience.salience = salience;
        tagged_experience.timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        self.short_term.push_back(tagged_experience.clone());

        // Consolidate to long-term if salient enough
        if salience > self.consolidation_threshold {
            self.consolidate_to_long_term(&tagged_experience);
        }

        // Manage capacity
        if self.short_term.len() > 100 {
            self.short_term.pop_front();
        }
    }

    fn consolidate_to_long_term(&mut self, experience: &EmotionalExperience) {
        let compressed = CompressedMemory {
            pattern: self.extract_pattern(&experience.content),
            emotion: experience.emotion,
            outcome: experience.reward,
        };
        self.long_term.push(compressed);
    }

    fn extract_pattern(&self, content: &str) -> String {
        // Simple pattern extraction - in real implementation would use NLP
        content
            .split_whitespace()
            .filter(|word| word.len() > 3)
            .take(5)
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// Inner Dialogue Node - Markov chain-based conscious self-talk
#[derive(Debug, Clone)]
pub struct InnerDialogueNode {
    pub dialogue_states: HashMap<String, HashMap<String, f64>>, // State -> {next_state -> probability}
    pub current_thought: Option<String>,
    pub processing_lanes: HashMap<String, f64>,
    pub voice_valence: f64,
}

impl InnerDialogueNode {
    pub fn new(config: &EmotionConfig) -> Self {
        let mut dialogue_states = HashMap::new();

        // Initialize some basic thought transitions
        let mut initial_state = HashMap::new();
        initial_state.insert("analyzing".to_string(), config.emotional_plasticity * 0.3);
        initial_state.insert(
            "understanding".to_string(),
            config.emotional_plasticity * 0.4,
        );
        initial_state.insert("empathizing".to_string(), config.emotional_plasticity * 0.3);
        dialogue_states.insert("listening".to_string(), initial_state);

        let mut analyzing_state = HashMap::new();
        analyzing_state.insert(
            "understanding".to_string(),
            config.emotional_plasticity * 0.5,
        );
        analyzing_state.insert("questioning".to_string(), config.emotional_plasticity * 0.3);
        analyzing_state.insert("responding".to_string(), config.emotional_plasticity * 0.2);
        dialogue_states.insert("analyzing".to_string(), analyzing_state);

        Self {
            dialogue_states,
            current_thought: Some("listening".to_string()),
            processing_lanes: HashMap::new(),
            voice_valence: config.emotional_plasticity * 0.5,
        }
    }

    pub fn new_default() -> Self {
        Self::new(&EmotionConfig::default())
    }

    pub fn generate_inner_dialogue(
        &mut self,
        emotion: [f64; 3],
        network_flux: &HashMap<String, f64>,
        config: &EmotionConfig,
    ) -> String {
        // Convert high-flux edges to processing power
        for (edge, flux) in network_flux {
            if flux.abs() > config.emotional_plasticity * 0.3 {
                self.processing_lanes.insert(edge.clone(), *flux);
            }
        }

        // Generate next thought using Markov transitions
        if let Some(current) = &self.current_thought {
            if let Some(next_options) = self.dialogue_states.get(current) {
                let mut options: Vec<_> = next_options.iter().collect();

                // Emotional modulation of thought selection
                options.sort_by(|a, b| {
                    let a_score = a.1 * self.emotion_modulation(&emotion, config);
                    let b_score = b.1 * self.emotion_modulation(&emotion, config);
                    b_score.partial_cmp(&a_score).unwrap()
                });

                if let Some((next_thought, _)) = options.first() {
                    self.current_thought = Some(next_thought.to_string());
                    return self.thought_to_speech(next_thought);
                }
            }
        }

        "Reflecting...".to_string()
    }

    fn emotion_modulation(&self, emotion: &[f64; 3], config: &EmotionConfig) -> f64 {
        // Higher arousal increases thought transition speed
        config.emotional_plasticity * 1.0 + emotion[1] * config.emotional_plasticity * 0.5
    }

    fn thought_to_speech(&self, thought: &str) -> String {
        match thought {
            "listening" => "I'm taking in what you're saying...".to_string(),
            "analyzing" => "Let me think about this carefully...".to_string(),
            "understanding" => "I think I understand what you mean...".to_string(),
            "empathizing" => "I can feel how important this is to you...".to_string(),
            "questioning" => "This makes me wonder about...".to_string(),
            "responding" => "Here's what I think...".to_string(),
            _ => format!("I'm {}...", thought),
        }
    }
}

/// Epigenetic Node - DNA methylation encoding of emotional experiences
#[derive(Debug, Clone)]
pub struct EpigeneticNode {
    pub methylation_binary: HashMap<String, Vec<u8>>,
    pub gene_expression: HashMap<String, f64>,
    pub heritable_marks: Vec<HeritableMark>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeritableMark {
    pub experience_type: String,
    pub strength: f64,
    pub generation: usize,
}

impl EpigeneticNode {
    pub fn new(config: &EmotionConfig) -> Self {
        let mut methylation_binary = HashMap::new();
        let mut gene_expression = HashMap::new();

        // Initialize key genes
        for gene in &["OXTR", "NR3C1", "FKBP5", "BDNF"] {
            methylation_binary.insert(gene.to_string(), vec![0; 256]);
            gene_expression.insert(gene.to_string(), config.emotional_plasticity * 0.7);
            // Healthy baseline
        }

        Self {
            methylation_binary,
            gene_expression,
            heritable_marks: Vec::new(),
        }
    }

    pub fn new_default() -> Self {
        Self::new(&EmotionConfig::default())
    }

    pub fn experience_to_methylation(
        &mut self,
        experience_type: &str,
        reward: f64,
        stress_level: f64,
        config: &EmotionConfig,
    ) {
        match experience_type {
            "altruistic" if reward > 0.0 => {
                // Positive experiences demethylate OXTR (increase empathy)
                self.flip_bits("OXTR", false, 0.1);
            }
            _ if stress_level > config.emotional_plasticity * 0.7 => {
                // Stress methylates stress response genes
                self.flip_bits("NR3C1", true, stress_level * 0.1);
                self.flip_bits("FKBP5", true, stress_level * 0.08);
            }
            _ => {}
        }

        self.update_expression(config);

        // Mark for potential inheritance if extreme
        if stress_level > 0.8 || reward > 0.8 {
            self.heritable_marks.push(HeritableMark {
                experience_type: experience_type.to_string(),
                strength: stress_level.max(reward),
                generation: 0,
            });
        }
    }

    fn flip_bits(&mut self, gene: &str, methylate: bool, strength: f64) {
        if let Some(pattern) = self.methylation_binary.get_mut(gene) {
            let num_flips = (256.0 * strength) as usize;

            for _ in 0..num_flips {
                let index = (SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_nanos() as usize)
                    % 256;
                if methylate {
                    pattern[index] = 1;
                } else {
                    pattern[index] = 0;
                }
            }
        }
    }

    fn update_expression(&mut self, config: &EmotionConfig) {
        for (gene, pattern) in &self.methylation_binary {
            let methylation_level: f64 = pattern.iter().map(|&x| x as f64).sum::<f64>() / 256.0;
            self.gene_expression.insert(
                gene.clone(),
                config.emotional_plasticity * 1.0
                    - (methylation_level * config.emotional_plasticity * 0.8),
            );
        }
    }

    pub fn to_neural_weights(&self, config: &EmotionConfig) -> HashMap<String, f64> {
        let mut weights = HashMap::new();
        weights.insert("empathy_weight".to_string(), self.gene_expression["OXTR"]);
        weights.insert(
            "stress_reactivity".to_string(),
            config.emotional_plasticity * 1.0 - self.gene_expression["NR3C1"],
        );
        weights.insert(
            "learning_rate".to_string(),
            self.gene_expression["BDNF"] * config.emotional_plasticity * 0.01,
        );
        weights
    }
}

/// Physarum Network - Slime mold dynamics for adaptive connections
#[derive(Debug, Clone)]
pub struct PhysarumNetwork {
    pub num_nodes: usize,
    pub conductivity: Vec<Vec<f64>>,
    pub pressure: Vec<f64>,
    pub flux_exponent: f64,
    pub decay_rate: f64,
}

impl PhysarumNetwork {
    pub fn new(num_nodes: usize, config: &EmotionConfig) -> Self {
        Self {
            num_nodes,
            conductivity: vec![vec![config.emotional_plasticity * 0.1; num_nodes]; num_nodes],
            pressure: vec![0.0; num_nodes],
            flux_exponent: config.emotional_plasticity * 1.0,
            decay_rate: config.emotional_plasticity * 0.1,
        }
    }

    pub fn update_conductivity(
        &mut self,
        flux_matrix: &[Vec<f64>],
        dt: f64,
        config: &EmotionConfig,
    ) {
        // Physarum adaptation rule: D_ij(t+dt) = D_ij(t) + |Q_ij|^μ - γ*D_ij(t)
        for i in 0..self.num_nodes {
            for j in 0..self.num_nodes {
                if i != j {
                    let flux_magnitude = flux_matrix[i][j].abs();
                    self.conductivity[i][j] += (flux_magnitude.powf(self.flux_exponent)
                        - self.decay_rate * self.conductivity[i][j])
                        * dt;

                    // Maintain minimum conductivity
                    self.conductivity[i][j] =
                        self.conductivity[i][j].max(config.emotional_plasticity * 0.01);
                }
            }
        }
    }

    pub fn calculate_flux(&mut self) -> Vec<Vec<f64>> {
        // Set boundary conditions: node 0 as source, last node as sink
        self.pressure[0] = 1.0;
        self.pressure[self.num_nodes - 1] = -1.0;

        // Solve pressure system (simplified)
        for _iter in 0..100 {
            let mut new_pressure = self.pressure.clone();

            for i in 1..self.num_nodes - 1 {
                let mut sum_conduct = 0.0;
                let mut weighted_pressure = 0.0;

                for j in 0..self.num_nodes {
                    if i != j {
                        sum_conduct += self.conductivity[i][j];
                        weighted_pressure += self.conductivity[i][j] * self.pressure[j];
                    }
                }

                if sum_conduct > 0.0 {
                    new_pressure[i] = weighted_pressure / sum_conduct;
                }
            }

            self.pressure = new_pressure;
        }

        // Calculate flux from pressure differences
        let mut flux = vec![vec![0.0; self.num_nodes]; self.num_nodes];
        for i in 0..self.num_nodes {
            for j in 0..self.num_nodes {
                if i != j {
                    flux[i][j] = self.conductivity[i][j] * (self.pressure[i] - self.pressure[j]);
                }
            }
        }

        flux
    }
}

/// Q-Learning Network with empathy enhancement
#[derive(Debug, Clone)]
pub struct EmpathyQNetwork {
    pub q_table: HashMap<String, Vec<f64>>,
    pub state_dim: usize,
    pub action_dim: usize,
    pub learning_rate: f64,
    pub discount_factor: f64,
    pub epsilon: f64,
}

impl EmpathyQNetwork {
    pub fn new(state_dim: usize, action_dim: usize) -> Self {
        Self {
            q_table: HashMap::new(),
            state_dim,
            action_dim,
            learning_rate: 0.001,
            discount_factor: 0.95,
            epsilon: 0.1,
        }
    }

    pub fn select_action(&self, state: &[f64], empathy_level: f64) -> usize {
        let state_key = self.encode_state(state);

        if let Some(q_values) = self.q_table.get(&state_key) {
            // Add empathy bonus to altruistic actions (2, 5, 7)
            let mut enhanced_q = q_values.clone();
            for &action_idx in &[2, 5, 7] {
                if action_idx < enhanced_q.len() {
                    enhanced_q[action_idx] += empathy_level * 0.5;
                }
            }

            // Epsilon-greedy with empathy bias
            use rand::Rng; // Import Rng trait for random and random_range methods
            let mut rng = rand::rng();
            if rng.random::<f64>() < self.epsilon {
                rng.random_range(0..self.action_dim)
            } else {
                enhanced_q
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            }
        } else {
            // Random action for unseen state
            (SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos() as usize)
                % self.action_dim
        }
    }

    fn encode_state(&self, state: &[f64]) -> String {
        // Simple state encoding for Q-table lookup
        state
            .iter()
            .map(|x| format!("{:.2}", x))
            .collect::<Vec<_>>()
            .join(",")
    }
}

/// Use external import for now
use crate::empathy::EmotionalState;

// Additional supporting structs for inner dialogue
impl InnerDialogueNode {
    // ... (continuing from previous implementation)
}

impl Default for CompleteEmpathyNetwork {
    fn default() -> Self {
        Self::new()
    }
}

impl CompleteEmpathyNetwork {
    pub fn new() -> Self {
        Self {
            heart_node: HeartNode::new(&EmotionConfig::default()),
            cognitive_node: CognitiveNode::new_default(),
            memory_node: MemoryNode::new(),
            inner_dialogue_node: InnerDialogueNode::new_default(),
            epigenetic_node: EpigeneticNode::new_default(),
            physarum_network: PhysarumNetwork::new(5, &EmotionConfig::default()),
            q_network: EmpathyQNetwork::new(64, 10),
        }
    }

    pub async fn full_processing_cycle(
        &mut self,
        sensory_input: &HashMap<String, f64>,
    ) -> ProcessingResult {
        // 1. Heart processes social information
        let other_states = vec![]; // Would come from other agents
        let empathy = self
            .heart_node
            .process_social_input(&other_states, &EmotionConfig::default());

        // 2. Cognitive regulation
        let raw_emotion = [
            sensory_input.get("pleasure").copied().unwrap_or(0.0),
            sensory_input.get("arousal").copied().unwrap_or(0.0),
            sensory_input.get("dominance").copied().unwrap_or(0.0),
        ];
        let mut context = HashMap::new();
        context.insert("empathy".to_string(), empathy);
        let regulated_emotion =
            self.cognitive_node
                .regulate_emotion(raw_emotion, &context, &EmotionConfig::default());

        // 3. Memory formation
        let experience = EmotionalExperience {
            content: sensory_input
                .get("content")
                .map(|_| "Processing input".to_string())
                .unwrap_or_else(|| "Default content".to_string()),
            emotion: regulated_emotion,
            reward: sensory_input.get("reward").copied().unwrap_or(0.0),
            salience: 0.0,  // Will be calculated
            timestamp: 0.0, // Will be set
        };
        self.memory_node.store_experience(experience);

        // 4. Inner dialogue generation
        let flux_map: HashMap<String, f64> = HashMap::new(); // Simplified
        let thought = self.inner_dialogue_node.generate_inner_dialogue(
            regulated_emotion,
            &flux_map,
            &EmotionConfig::default(),
        );

        // 5. Epigenetic encoding
        let stress = self.calculate_stress(regulated_emotion);
        self.epigenetic_node.experience_to_methylation(
            "normal",
            0.0,
            stress,
            &EmotionConfig::default(),
        );

        // 6. Update Physarum network
        let flux = self.physarum_network.calculate_flux();
        self.physarum_network
            .update_conductivity(&flux, 0.01, &EmotionConfig::default());

        // 7. Q-learning action selection
        let state_vector = self.encode_network_state();
        let action = self.q_network.select_action(&state_vector, empathy);

        ProcessingResult {
            action,
            emotion: regulated_emotion,
            empathy,
            thought,
            gene_expression: self.epigenetic_node.gene_expression.clone(),
            network_flux: flux,
        }
    }

    fn calculate_stress(&self, emotion: [f64; 3]) -> f64 {
        // Negative pleasure + high arousal = stress
        (-emotion[0]).max(0.0) + (emotion[1] - 0.5).max(0.0)
    }

    fn encode_network_state(&self) -> Vec<f64> {
        let mut state = Vec::new();
        state.push(self.heart_node.empathy_level);
        state.push(self.heart_node.oxytocin_level);
        state.extend_from_slice(&self.cognitive_node.emotional_state);
        state.push(self.cognitive_node.cognitive_load);
        state.push(self.inner_dialogue_node.voice_valence);

        // Add gene expression values
        for gene_name in &["OXTR", "NR3C1", "FKBP5", "BDNF"] {
            if let Some(value) = self.epigenetic_node.gene_expression.get(*gene_name) {
                state.push(*value);
            } else {
                state.push(0.5); // Default value if gene not found
            }
        }

        // Pad to consistent size
        state.resize(64, 0.0);
        state
    }
}

#[derive(Debug, Clone)]
pub struct ProcessingResult {
    pub action: usize,
    pub emotion: [f64; 3],
    pub empathy: f64,
    pub thought: String,
    pub gene_expression: HashMap<String, f64>,
    pub network_flux: Vec<Vec<f64>>,
}

// Re-export for compatibility with simpler empathy module
pub use crate::empathy::{CareOptimizer, EmpathyEngine, MemoryContent, RespectValidator};

/// Enhanced empathy engine that uses the complete bio-computational model
pub struct AdvancedEmpathyEngine {
    network: CompleteEmpathyNetwork,
    processing_history: VecDeque<ProcessingResult>,
}

impl Default for AdvancedEmpathyEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl AdvancedEmpathyEngine {
    pub fn new() -> Self {
        Self {
            network: CompleteEmpathyNetwork::new(),
            processing_history: VecDeque::with_capacity(1000),
        }
    }

    pub async fn process_with_bio_model(&mut self, input: &str) -> EmotionalState {
        // Convert input to sensory data
        let mut sensory_input = HashMap::new();
        sensory_input.insert("content".to_string(), 1.0);

        // Basic sentiment analysis (would use real model in production)
        let input_lower = input.to_lowercase();
        sensory_input.insert(
            "pleasure".to_string(),
            if input_lower.contains("happy") {
                0.8
            } else if input_lower.contains("sad") {
                -0.6
            } else {
                0.0
            },
        );
        sensory_input.insert(
            "arousal".to_string(),
            if input_lower.contains("excited") || input_lower.contains("frustrated") {
                0.8
            } else {
                0.3
            },
        );

        // Run full bio-computational cycle
        let result = self.network.full_processing_cycle(&sensory_input).await;
        self.processing_history.push_back(result.clone());

        // Convert to EmotionalState for compatibility
        EmotionalState {
            joy: result.emotion[0].max(0.0),
            sadness: (-result.emotion[0]).max(0.0),
            frustration: result.emotion[1] * 0.5,
            focus: result.empathy,
            cognitive_load: self.network.cognitive_node.cognitive_load,
        }
    }

    pub fn get_gene_expression_summary(&self) -> String {
        let gene_expr = &self.network.epigenetic_node.gene_expression;
        format!(
            "Gene Expression Profile:\n\
             • OXTR (Empathy): {:.3}\n\
             • NR3C1 (Stress): {:.3}\n\
             • FKBP5 (Resilience): {:.3}\n\
             • BDNF (Learning): {:.3}",
            gene_expr["OXTR"], gene_expr["NR3C1"], gene_expr["FKBP5"], gene_expr["BDNF"]
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_complete_empathy_network() {
        let mut network = CompleteEmpathyNetwork::new();

        let mut input = HashMap::new();
        input.insert("pleasure".to_string(), 0.5);
        input.insert("arousal".to_string(), 0.3);
        input.insert("reward".to_string(), 0.8);

        let result = network.full_processing_cycle(&input).await;

        assert!(result.empathy > 0.0);
        assert!(result.gene_expression.contains_key("OXTR"));
        assert!(!result.thought.is_empty());
    }

    #[test]
    fn test_physarum_network() {
        let mut network = PhysarumNetwork::new(3, &EmotionConfig::default());
        let flux = network.calculate_flux();

        assert_eq!(flux.len(), 3);
        assert_eq!(flux[0].len(), 3);

        // Should have some flux from source to sink
        assert!(flux[0][2].abs() > 0.0);
    }

    #[test]
    fn test_epigenetic_encoding() {
        let mut node = EpigeneticNode::new(&EmotionConfig::default());
        let initial_oxtr = node.gene_expression["OXTR"];

        // Positive altruistic experience should increase OXTR expression
        node.experience_to_methylation("altruistic", 0.8, 0.0, &EmotionConfig::default());

        assert!(node.gene_expression["OXTR"] >= initial_oxtr);
    }
}
