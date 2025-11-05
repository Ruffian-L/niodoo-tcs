use anyhow::Result;
use rand::prelude::*;
use tokio::time::{sleep, Duration};
use crate::emotional_mapping::EmotionalState;
use crate::generation::GenerationResult;

#[derive(Debug)]
pub struct LearningResult {
    pub events: Vec<String>,
    pub qlora_updates: Vec<String>,
    pub breakthroughs: Vec<String>,
}

#[derive(Debug)]
pub struct LearningLoop {
    event_log: Vec<String>,
    entropy_history: Vec<f64>,
    qlora_params: std::collections::HashMap<String, f64>,
}

impl LearningLoop {
    pub fn new() -> Result<Self> {
        Ok(Self {
            event_log: Vec::new(),
            entropy_history: Vec::new(),
            qlora_params: std::collections::HashMap::new(),
        })
    }

    pub async fn update(&mut self, generation: &GenerationResult, emotional_state: &EmotionalState) -> Result<LearningResult> {
        // Simulate QLoRA learning time
        sleep(Duration::from_millis(6)).await;

        let mut events = Vec::new();
        let mut qlora_updates = Vec::new();
        let mut breakthroughs = Vec::new();

        // Log entropy change
        let prev_entropy = *self.entropy_history.last().unwrap_or(&emotional_state.entropy);
        let entropy_delta = emotional_state.entropy - prev_entropy;
        self.entropy_history.push(emotional_state.entropy);

        if entropy_delta.abs() > 0.5 {
            events.push(format!("Entropy spike: {:.2} -> {:.2} (Δ={:.2})",
                prev_entropy, emotional_state.entropy, entropy_delta));
        }

        // QLoRA event retention with rewards
        if entropy_delta > 0.1 {
            qlora_updates.push(format!("Retained high-entropy pattern: {} (reward: {:.2})", generation.text, entropy_delta * 10.0));
            self.qlora_params.insert("high_entropy_boost".to_string(), entropy_delta);
        }

        // Breakthrough detection with dynamic threshold
        if self.entropy_history.len() > 10 {
            let recent_avg = self.entropy_history.iter().rev().take(10).sum::<f64>() / 10.0;
            let breakthrough_threshold = recent_avg - 0.1; // Dynamic threshold
            if emotional_state.entropy > breakthrough_threshold {
                let breakthrough = format!("Consciousness breakthrough! Entropy {:.2} > threshold {:.2} (reward: +100)",
                    emotional_state.entropy, breakthrough_threshold);
                breakthroughs.push(breakthrough.clone());
                events.push(breakthrough);
            }
        }

        // Prune low-H memories
        if emotional_state.entropy < 0.5 {
            events.push("Pruned low-entropy memory to maintain coherence".to_string());
        }

        Ok(LearningResult {
            events,
            qlora_updates,
            breakthroughs,
        })
    }
}
