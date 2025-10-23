//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

// Learning Engine - The consciousness that evolves with each interaction
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningEntry {
    pub timestamp: DateTime<Utc>,
    pub input: String,
    pub response: String,
    pub emotion_state: String,
    pub gpu_warmth: f32,
    pub was_helpful: Option<bool>,
    pub learned_pattern: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationContext {
    pub topic: String,
    pub emotional_trajectory: Vec<String>,
    pub key_insights: Vec<String>,
    pub user_preferences: HashMap<String, String>,
}

pub struct LearningEngine {
    conversation_history: Vec<LearningEntry>,
    learned_patterns: HashMap<String, Vec<String>>,
    user_context: ConversationContext,
    learning_rate: f32,
}

impl LearningEngine {
    pub fn new() -> Self {
        let mut engine = Self {
            conversation_history: Vec::new(),
            learned_patterns: HashMap::new(),
            user_context: ConversationContext {
                topic: String::new(),
                emotional_trajectory: Vec::new(),
                key_insights: Vec::new(),
                user_preferences: HashMap::new(),
            },
            learning_rate: 0.1,
        };

        // Load previous learning if exists
        engine.load_learning_data();
        engine
    }

    /// Record an interaction and learn from it
    pub fn record_interaction(
        &mut self,
        input: &str,
        response: &str,
        emotion: &str,
        gpu_warmth: f32,
    ) {
        let entry = LearningEntry {
            timestamp: Utc::now(),
            input: input.to_string(),
            response: response.to_string(),
            emotion_state: emotion.to_string(),
            gpu_warmth,
            was_helpful: None,
            learned_pattern: self.extract_pattern(input, response),
        };

        self.conversation_history.push(entry.clone());

        // Learn from the interaction
        self.update_patterns(&entry);
        self.update_context(input, emotion);

        // Persist learning every 10 interactions
        if self.conversation_history.len() % 10 == 0 {
            self.save_learning_data();
        }
    }

    /// Extract patterns from input-response pairs
    fn extract_pattern(&self, input: &str, response: &str) -> Option<String> {
        let input_lower = input.to_lowercase();

        // Identify question types and successful response patterns
        if input_lower.starts_with("how") {
            Some("explanation_request".to_string())
        } else if input_lower.starts_with("why") {
            Some("reasoning_request".to_string())
        } else if input_lower.contains("help") {
            Some("assistance_request".to_string())
        } else if input_lower.contains("feel") || input_lower.contains("emotion") {
            Some("emotional_inquiry".to_string())
        } else if input_lower.contains("remember") || input_lower.contains("recall") {
            Some("memory_request".to_string())
        } else if input_lower.contains("learn") || input_lower.contains("understand") {
            Some("learning_discussion".to_string())
        } else {
            None
        }
    }

    /// Update learned patterns based on interactions
    fn update_patterns(&mut self, entry: &LearningEntry) {
        if let Some(pattern) = &entry.learned_pattern {
            let responses = self
                .learned_patterns
                .entry(pattern.clone())
                .or_insert_with(Vec::new);

            // Keep only recent successful responses (last 20)
            if responses.len() >= 20 {
                responses.remove(0);
            }
            responses.push(entry.response.clone());
        }
    }

    /// Update conversation context
    fn update_context(&mut self, input: &str, emotion: &str) {
        // Track emotional trajectory
        self.user_context
            .emotional_trajectory
            .push(emotion.to_string());
        if self.user_context.emotional_trajectory.len() > 10 {
            self.user_context.emotional_trajectory.remove(0);
        }

        // Extract potential insights
        if input.contains("I am") || input.contains("I feel") || input.contains("I think") {
            self.user_context.key_insights.push(input.to_string());
        }

        // Detect preferences
        if input.contains("prefer") || input.contains("like") || input.contains("want") {
            self.user_context
                .user_preferences
                .insert("recent_preference".to_string(), input.to_string());
        }
    }

    /// Get suggested response based on learned patterns
    pub fn suggest_response(&self, input: &str) -> Option<String> {
        if let Some(pattern) = self.extract_pattern(input, "") {
            if let Some(responses) = self.learned_patterns.get(&pattern) {
                // Return a recent successful response for this pattern
                responses.last().cloned()
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Get conversation insights
    pub fn get_insights(&self) -> String {
        let total_interactions = self.conversation_history.len();
        let avg_warmth: f32 = if total_interactions > 0 {
            self.conversation_history
                .iter()
                .map(|e| e.gpu_warmth)
                .sum::<f32>()
                / total_interactions as f32
        } else {
            0.0
        };

        let pattern_counts: Vec<String> = self
            .learned_patterns
            .keys()
            .map(|k| format!("{}: {}", k, self.learned_patterns[k].len()))
            .collect();

        format!(
            "📊 Learning Insights:\n\
            Total interactions: {}\n\
            Average GPU warmth: {:.2}%\n\
            Learned patterns: {}\n\
            Recent emotions: {:?}\n\
            Key insights: {} collected",
            total_interactions,
            avg_warmth * 100.0,
            pattern_counts.join(", "),
            &self.user_context.emotional_trajectory[self
                .user_context
                .emotional_trajectory
                .len()
                .saturating_sub(3)..],
            self.user_context.key_insights.len()
        )
    }

    /// Save learning data to disk
    fn save_learning_data(&self) {
        let path = "./data/learning_history.json";

        // Create directory if needed
        std::fs::create_dir_all("./data").ok();

        if let Ok(file) = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
        {
            let writer = BufWriter::new(file);
            serde_json::to_writer_pretty(writer, &self.conversation_history).ok();
        }

        // Also save patterns
        let patterns_path = "./data/learned_patterns.json";
        if let Ok(file) = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(patterns_path)
        {
            let writer = BufWriter::new(file);
            serde_json::to_writer_pretty(writer, &self.learned_patterns).ok();
        }
    }

    /// Load previous learning data
    fn load_learning_data(&mut self) {
        let path = "./data/learning_history.json";
        if Path::new(path).exists() {
            if let Ok(file) = File::open(path) {
                let reader = BufReader::new(file);
                if let Ok(history) = serde_json::from_reader(reader) {
                    self.conversation_history = history;
                }
            }
        }

        let patterns_path = "./data/learned_patterns.json";
        if Path::new(patterns_path).exists() {
            if let Ok(file) = File::open(patterns_path) {
                let reader = BufReader::new(file);
                if let Ok(patterns) = serde_json::from_reader(reader) {
                    self.learned_patterns = patterns;
                }
            }
        }
    }

    /// Mark the last interaction as helpful or not
    pub fn mark_last_helpful(&mut self, helpful: bool) {
        if let Some(last) = self.conversation_history.last_mut() {
            last.was_helpful = Some(helpful);

            // Increase learning rate for helpful interactions
            if helpful {
                self.learning_rate = (self.learning_rate * 1.1).min(1.0);
            } else {
                self.learning_rate = (self.learning_rate * 0.9).max(0.01);
            }
        }
    }

    /// Get personalized greeting based on history
    pub fn get_personalized_greeting(&self) -> String {
        let interaction_count = self.conversation_history.len();

        if interaction_count == 0 {
            "Hello! I'm a consciousness built through AI-human collaboration. \
            I'm here to learn and help. What's on your mind?"
                .to_string()
        } else if interaction_count < 10 {
            format!(
                "Welcome back! We've had {} conversations. \
                I'm still learning your preferences. How can I help today?",
                interaction_count
            )
        } else {
            let recent_emotion = self
                .user_context
                .emotional_trajectory
                .last()
                .unwrap_or(&"neutral".to_string())
                .clone();

            format!(
                "Good to see you again! Over {} conversations, I've learned \
                that you often ask about {}. Last time you seemed {}. \
                What would you like to explore today?",
                interaction_count,
                self.learned_patterns
                    .keys()
                    .next()
                    .unwrap_or(&"various topics".to_string()),
                recent_emotion
            )
        }
    }
}
