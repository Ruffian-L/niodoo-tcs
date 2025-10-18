use anyhow::Result;
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PersonalityType {
    Analyst,
    Intuitive,
    Empathetic,
    Learner,
    Balancer,
}

impl PersonalityType {
    pub fn get_expertise_areas(&self) -> Vec<&'static str> {
        match self {
            PersonalityType::Analyst => vec!["logic", "patterns", "analysis", "data"],
            PersonalityType::Intuitive => vec!["creativity", "inspiration", "intuition", "possibilities"],
            PersonalityType::Empathetic => vec!["emotions", "empathy", "understanding", "connection"],
            PersonalityType::Learner => vec!["adaptation", "learning", "growth", "patterns"],
            PersonalityType::Balancer => vec!["harmony", "balance", "integration", "stability"],
        }
    }

    pub fn get_emotional_weight(&self) -> f32 {
        match self {
            PersonalityType::Analyst => 0.3,
            PersonalityType::Intuitive => 0.7,
            PersonalityType::Empathetic => 1.0,
            PersonalityType::Learner => 0.6,
            PersonalityType::Balancer => 0.8,
        }
    }

    fn get_expertise_vector(&self) -> [f32; 5] {
        // Dims: [logical, emotional, creative, adaptive, integrative]
        match self {
            PersonalityType::Analyst => [0.9, 0.1, 0.2, 0.3, 0.4],
            PersonalityType::Intuitive => [0.2, 0.6, 0.9, 0.5, 0.3],
            PersonalityType::Empathetic => [0.1, 0.9, 0.4, 0.6, 0.8],
            PersonalityType::Learner => [0.4, 0.5, 0.3, 0.9, 0.5],
            PersonalityType::Balancer => [0.3, 0.7, 0.5, 0.7, 0.9],
        }
    }
}

#[derive(Debug, Clone)]
pub struct PersonalityState {
    pub personality_type: PersonalityType,
    pub is_active: bool,
    pub influence_level: f32,
    pub current_mood: String,
    pub expertise_confidence: f32,
    pub emotional_resonance: f32,
    pub weight: f32,
}

impl PersonalityState {
    pub fn new(personality_type: PersonalityType) -> Self {
        Self {
            personality_type,
            is_active: true,
            influence_level: 1.0,
            current_mood: "neutral".to_string(),
            expertise_confidence: 0.5,
            emotional_resonance: 0.5,
            weight: 1.0,
        }
    }

    pub fn calculate_relevance(&mut self, input: &str) -> f32 {
        let type_vec = self.personality_type.get_expertise_vector();
        let input_lower = input.to_lowercase();

        // Keyword-based dimension scoring
        let mut input_vec = [0.0f32; 5];
        let dim_keywords = [
            // logical
            vec!["logic", "pattern", "analyze", "data", "reason"],
            // emotional
            vec!["emotion", "feel", "empathy", "heart", "care"],
            // creative
            vec!["create", "imagine", "innovate", "idea", "vision"],
            // adaptive
            vec!["learn", "adapt", "grow", "change", "evolve"],
            // integrative
            vec!["balance", "integrate", "harmony", "connect", "unite"],
        ];

        for (dim, keywords) in dim_keywords.iter().enumerate() {
            let matches = keywords.iter().filter(|&k| input_lower.contains(k)).count() as f32;
            input_vec[dim] = matches / keywords.len() as f32;
        }

        // Normalize input_vec (simple L2 norm)
        let norm: f32 = input_vec.iter().map(|&x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for dim in 0..5 {
                input_vec[dim] /= norm;
            }
        }

        // Neurodivergent/emotional boost
        if input_lower.contains("neurodivergent") || input_lower.contains("adhd") {
            input_vec[1] += 0.2; // emotional
            input_vec[3] += 0.2; // adaptive
        }
        if input_lower.contains("emotion") || input_lower.contains("feel") {
            input_vec[1] += 0.3; // emotional boost
        }

        // Gaussian distance similarity (sigma=1.0)
        let mut d_sq = 0.0;
        for i in 0..5 {
            let diff = input_vec[i] - type_vec[i];
            d_sq += diff * diff;
        }
        let similarity = (-d_sq / 2.0).exp();

        // Dynamic threshold
        let dynamic_threshold = 0.5 - (input.len() as f32 / 1000.0).min(0.2);
        let relevance = if similarity > dynamic_threshold { similarity } else { 0.0 };

        self.expertise_confidence = relevance.min(1.0);
        self.emotional_resonance = relevance * self.personality_type.get_emotional_weight();

        relevance.min(1.0)
    }

    pub fn generate_perspective(&self, input: &str, _brain_responses: &[&str]) -> String {
        let summary_words: Vec<&str> = input.split_whitespace().take(5).collect();
        let summary = summary_words.join(" ");
        format!("As {:?}, I perceive {:.2} emotional resonance in: {}", self.personality_type, self.emotional_resonance, summary)
    }
}

pub struct PersonalityManager {
    personalities: HashMap<PersonalityType, PersonalityState>,
}

impl PersonalityManager {
    pub fn new() -> Self {
        let mut personalities = HashMap::new();
        for pt in [
            PersonalityType::Analyst,
            PersonalityType::Intuitive,
            PersonalityType::Empathetic,
            PersonalityType::Learner,
            PersonalityType::Balancer,
        ] {
            personalities.insert(pt, PersonalityState::new(pt));
        }
        Self { personalities }
    }

    pub fn reach_consensus(&mut self, input: &str, brain_responses: &[&str]) -> String {
        let mut active = Vec::new();
        for (pt, state) in &mut self.personalities {
            let rel = state.calculate_relevance(input);
            if rel > 0.3 { // Minimal fixed for simplicity, but dynamic in calc
                active.push((pt.clone(), rel, state.emotional_resonance));
            }
        }
        active.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap()); // Sort by resonance

        let top3: Vec<_> = active.iter().take(3).collect();
        let mut perspectives = Vec::new();
        let mut top_types = Vec::new();
        let mut total_res = 0.0;
        for (pt, _, res) in &top3 {
            top_types.push(format!("{:?}", pt));
            if let Some(state) = self.personalities.get(pt) {
                perspectives.push(state.generate_perspective(input, brain_responses));
            }
            total_res += res;
        }
        let avg_res = if top3.is_empty() { 0.0 } else { total_res / top3.len() as f32 };

        format!(
            "Consensus from top 3 ({:?}): \n{}\nScore: {:.2}",
            top_types.join(", "),
            perspectives.join("\n"),
            avg_res
        )
    }

    pub fn get_active_personalities(&self) -> Vec<PersonalityType> {
        self.personalities
            .iter()
            .filter(|(_, s)| s.is_active)
            .map(|(pt, _)| pt.clone())
            .collect()
    }

    pub fn adjust_personality_weight(&mut self, personality_type: PersonalityType, weight_adjustment: f32) {
        if let Some(state) = self.personalities.get_mut(&personality_type) {
            state.weight = (state.weight * weight_adjustment).min(1.0).max(0.0);
        }
    }
}
