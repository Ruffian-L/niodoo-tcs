#![allow(dead_code)]
#![allow(unused_imports)]

//! Personal memory engine with semantic retrieval and emotional analytics.

use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet, VecDeque};
use uuid::Uuid;

const MEMORY_EMBEDDING_DIM: usize = 64;
const EMOTION_DIM: usize = 5; // joy, sadness, anger, fear, surprise

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonalMemory {
    pub id: Uuid,
    pub content: String,
    pub timestamp: DateTime<Utc>,
    pub emotional_weight: f32,
    pub tags: Vec<String>,
    pub embedding: Vec<f32>,
    pub emotion_profile: [f32; EMOTION_DIM],
    pub importance: f32,
}

impl Default for PersonalMemory {
    fn default() -> Self {
        Self {
            id: Uuid::nil(),
            content: String::from("default memory"),
            timestamp: Utc::now(),
            emotional_weight: 0.5,
            tags: Vec::new(),
            embedding: vec![0.0; MEMORY_EMBEDDING_DIM],
            emotion_profile: [0.2; EMOTION_DIM],
            importance: 0.5,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonalConsciousnessStats {
    pub total_memories: u64,
    pub emotional_coherence: f64,
    pub learning_rate: f64,
    pub total_insights: u64,
    pub time_span_days: u64,
    pub toroidal_nodes: u64,
}

impl Default for PersonalConsciousnessStats {
    fn default() -> Self {
        Self {
            total_memories: 0,
            emotional_coherence: 0.7,
            learning_rate: 0.12,
            total_insights: 0,
            time_span_days: 0,
            toroidal_nodes: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonalInsight {
    pub insight_type: String,
    pub confidence: f64,
    pub content: String,
    pub pattern: String,
}

impl Default for PersonalInsight {
    fn default() -> Self {
        Self {
            insight_type: "default".to_string(),
            confidence: 0.8,
            content: "default insight".to_string(),
            pattern: "default pattern".to_string(),
        }
    }
}

#[derive(Default)]
pub struct PersonalMemoryEngine {
    memories: VecDeque<PersonalMemory>,
    stats: PersonalConsciousnessStats,
    max_capacity: usize,
    rolling_sentiment: Vec<f32>,
}

impl Clone for PersonalMemoryEngine {
    fn clone(&self) -> Self {
        Self {
            memories: self.memories.clone(),
            stats: self.stats.clone(),
            max_capacity: self.max_capacity,
            rolling_sentiment: self.rolling_sentiment.clone(),
        }
    }
}

impl PersonalMemoryEngine {
    pub fn new() -> Self {
        Self {
            memories: VecDeque::new(),
            stats: PersonalConsciousnessStats::default(),
            max_capacity: 1024,
            rolling_sentiment: Vec::new(),
        }
    }

    pub fn initialize_consciousness(&mut self) -> Result<()> {
        self.stats.total_memories = self.memories.len() as u64;
        self.stats.emotional_coherence = self.compute_emotional_coherence();
        self.stats.learning_rate = self.estimate_learning_velocity();
        self.stats.toroidal_nodes = self.count_unique_tags() as u64;
        self.stats.time_span_days = self.calculate_timespan_days();
        Ok(())
    }

    pub fn create_memory_from_conversation(
        &mut self,
        content: String,
        emotional_weight: f64,
    ) -> Result<PersonalMemory> {
        if content.trim().is_empty() {
            return Err(anyhow!("cannot store empty memory content"));
        }

        let tokens = tokenize(&content);
        let embedding = build_embedding(&tokens, MEMORY_EMBEDDING_DIM);
        let emotion_profile = EMOTION_VOCAB.analyze(&tokens);
        let tags = extract_keywords(&tokens, 5);
        let importance = (emotional_weight as f32).clamp(0.05, 1.5)
            * (emotion_profile.iter().sum::<f32>() / EMOTION_DIM as f32 + 0.5);

        let memory = PersonalMemory {
            id: Uuid::new_v4(),
            content,
            timestamp: Utc::now(),
            emotional_weight: emotional_weight as f32,
            tags,
            embedding,
            emotion_profile,
            importance: importance.clamp(0.0, 2.0),
        };

        self.insert_memory(memory.clone());
        Ok(memory)
    }

    pub fn get_consciousness_stats(&self) -> &PersonalConsciousnessStats {
        &self.stats
    }

    pub fn get_personal_insights(&self) -> Vec<PersonalInsight> {
        if self.memories.is_empty() {
            return Vec::new();
        }

        let top_tag = self
            .memories
            .iter()
            .flat_map(|m| m.tags.iter().cloned())
            .fold(HashMap::new(), |mut acc: HashMap<String, usize>, tag| {
                *acc.entry(tag).or_insert(0) += 1;
                acc
            })
            .into_iter()
            .max_by(|a, b| a.1.cmp(&b.1))
            .map(|(tag, count)| (tag, count as f64 / self.memories.len() as f64));

        let mut insights = Vec::new();

        if let Some((tag, frequency)) = top_tag {
            insights.push(PersonalInsight {
                insight_type: "dominant_theme".to_string(),
                confidence: (frequency * 1.2).min(0.95),
                content: format!("Recurring motif: {tag}"),
                pattern: "tag_frequency".to_string(),
            });
        }

        let recent_intensity: f32 = self
            .memories
            .iter()
            .rev()
            .take(12)
            .map(|m| m.emotional_weight)
            .sum::<f32>()
            / 12.0_f32.max(self.memories.len() as f32);

        insights.push(PersonalInsight {
            insight_type: "emotional_pulse".to_string(),
            confidence: 0.7,
            content: format!("Recent emotional load averaging {:.2}", recent_intensity),
            pattern: "rolling_average".to_string(),
        });

        let innovation = self.estimate_learning_velocity();
        insights.push(PersonalInsight {
            insight_type: "learning_velocity".to_string(),
            confidence: (innovation * 1.4).min(0.9),
            content: format!("Adaptive rate at {:.3}", innovation),
            pattern: "velocity_estimation".to_string(),
        });

        insights
    }

    pub fn generate_personal_context(&self) -> String {
        if self.memories.is_empty() {
            return "No personal memories available".to_string();
        }

        let recent: Vec<&PersonalMemory> = self.memories.iter().rev().take(3).collect();
        let mut context_lines = Vec::new();
        context_lines.push(format!("Total memories archived: {}", self.memories.len()));
        context_lines.push(format!(
            "Emotional coherence: {:.2}",
            self.stats.emotional_coherence
        ));

        for memory in recent {
            let dominant = dominant_emotion_label(&memory.emotion_profile);
            context_lines.push(format!(
                "• [{} | {:.2}] {}",
                dominant,
                memory.emotional_weight,
                truncate(&memory.content, 160)
            ));
        }

        context_lines.join("\n")
    }

    pub fn retrieve_relevant_memories_rag(&self, query: &str) -> Vec<PersonalMemory> {
        if query.trim().is_empty() || self.memories.is_empty() {
            return self.memories.iter().cloned().collect();
        }

        let query_tokens = tokenize(query);
        let query_embedding = build_embedding(&query_tokens, MEMORY_EMBEDDING_DIM);
        let mut scored: Vec<(f32, PersonalMemory)> = self
            .memories
            .iter()
            .cloned()
            .map(|memory| {
                let similarity = cosine_similarity(&query_embedding, &memory.embedding);
                (similarity, memory)
            })
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
        scored
            .into_iter()
            .take(8)
            .map(|(_, memory)| memory)
            .collect()
    }

    pub fn get_insights_for_theme(&self, theme: &str) -> Vec<PersonalInsight> {
        if theme.trim().is_empty() {
            return Vec::new();
        }

        let theme_lower = theme.to_lowercase();
        let matching: Vec<&PersonalMemory> = self
            .memories
            .iter()
            .filter(|memory| memory.tags.iter().any(|tag| tag.contains(&theme_lower)))
            .collect();

        if matching.is_empty() {
            return Vec::new();
        }

        let average_weight: f32 = matching
            .iter()
            .map(|memory| memory.emotional_weight)
            .sum::<f32>()
            / matching.len() as f32;

        vec![PersonalInsight {
            insight_type: format!("theme::{theme_lower}"),
            confidence: (matching.len() as f64 / self.memories.len().max(1) as f64).min(0.95),
            content: format!(
                "Theme '{theme}' appears {} times with average weight {:.2}",
                matching.len(),
                average_weight
            ),
            pattern: "theme_frequency".to_string(),
        }]
    }

    pub fn get_recent_memories(&self, count: usize) -> Vec<PersonalMemory> {
        self.memories.iter().rev().take(count).cloned().collect()
    }

    pub fn get_emotional_memories(&self, emotion: &str, limit: usize) -> Vec<PersonalMemory> {
        if emotion.trim().is_empty() {
            return self.get_recent_memories(limit);
        }

        let index = match emotion.trim().to_lowercase().as_str() {
            "joy" | "happy" => 0,
            "sadness" | "grief" => 1,
            "anger" | "rage" => 2,
            "fear" | "anxiety" => 3,
            "surprise" | "wonder" => 4,
            _ => 0,
        };

        let mut ranked: Vec<PersonalMemory> = self.memories.iter().cloned().collect();
        ranked.sort_by(|a, b| {
            b.emotion_profile[index]
                .partial_cmp(&a.emotion_profile[index])
                .unwrap_or(Ordering::Equal)
        });
        ranked.into_iter().take(limit).collect()
    }

    pub fn export_knowledge_graph(&self) -> String {
        let nodes: Vec<HashMap<&str, serde_json::Value>> = self
            .memories
            .iter()
            .map(|memory| {
                let mut node = HashMap::new();
                node.insert("id", serde_json::Value::String(memory.id.to_string()));
                node.insert(
                    "label",
                    serde_json::Value::String(truncate(&memory.content, 60)),
                );
                node.insert(
                    "weight",
                    serde_json::Value::Number(
                        serde_json::Number::from_f64(memory.emotional_weight as f64).unwrap(),
                    ),
                );
                node.insert(
                    "tags",
                    serde_json::Value::Array(
                        memory
                            .tags
                            .iter()
                            .map(|tag| serde_json::Value::String(tag.clone()))
                            .collect(),
                    ),
                );
                node
            })
            .collect();

        let edges: Vec<HashMap<&str, serde_json::Value>> = self
            .memories
            .iter()
            .enumerate()
            .flat_map(|(idx, source)| {
                self.memories
                    .iter()
                    .enumerate()
                    .skip(idx + 1)
                    .filter_map(|(_, target)| {
                        let overlap = tag_overlap(source, target);
                        if overlap > 0.0 {
                            let mut edge = HashMap::new();
                            edge.insert("source", serde_json::Value::String(source.id.to_string()));
                            edge.insert("target", serde_json::Value::String(target.id.to_string()));
                            edge.insert(
                                "weight",
                                serde_json::Value::Number(
                                    serde_json::Number::from_f64(overlap as f64).unwrap(),
                                ),
                            );
                            Some(edge)
                        } else {
                            None
                        }
                    })
            })
            .collect();

        serde_json::json!({ "nodes": nodes, "edges": edges }).to_string()
    }

    fn insert_memory(&mut self, memory: PersonalMemory) {
        if self.memories.len() >= self.max_capacity {
            if let Some((index, _)) = self
                .memories
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.importance.partial_cmp(&b.importance).unwrap())
            {
                self.memories.remove(index);
            }
        }

        self.rolling_sentiment
            .push(memory.emotional_weight.clamp(0.0, 1.5));
        if self.rolling_sentiment.len() > 256 {
            self.rolling_sentiment.remove(0);
        }

        self.memories.push_back(memory);
        self.stats.total_memories = self.memories.len() as u64;
        self.stats.emotional_coherence = self.compute_emotional_coherence();
        self.stats.learning_rate = self.estimate_learning_velocity();
        self.stats.toroidal_nodes = self.count_unique_tags() as u64;
        self.stats.time_span_days = self.calculate_timespan_days();
    }

    fn compute_emotional_coherence(&self) -> f64 {
        if self.memories.len() < 2 {
            return 0.7;
        }

        let mut variance_accum = 0.0f64;
        let mut total_pairs = 0usize;

        for window in self.memories.as_slices().0.windows(2) {
            let diff = (window[0].emotional_weight - window[1].emotional_weight).abs();
            variance_accum += diff as f64;
            total_pairs += 1;
        }

        for window in self.memories.as_slices().1.windows(2) {
            let diff = (window[0].emotional_weight - window[1].emotional_weight).abs();
            variance_accum += diff as f64;
            total_pairs += 1;
        }

        if total_pairs == 0 {
            return 0.7;
        }

        (1.0 - (variance_accum / total_pairs as f64).min(1.0)).clamp(0.1, 0.98)
    }

    fn estimate_learning_velocity(&self) -> f64 {
        if self.rolling_sentiment.len() < 3 {
            return 0.1;
        }

        let recent = &self.rolling_sentiment;
        let mean: f32 = recent.iter().sum::<f32>() / recent.len() as f32;
        let variance: f32 = recent
            .iter()
            .map(|value| {
                let diff = value - mean;
                diff * diff
            })
            .sum::<f32>()
            / recent.len() as f32;

        (variance as f64 * 0.6 + self.memories.len() as f64 / 512.0).clamp(0.05, 0.9)
    }

    fn count_unique_tags(&self) -> usize {
        self.memories
            .iter()
            .flat_map(|memory| memory.tags.iter().cloned())
            .collect::<HashSet<_>>()
            .len()
    }

    fn calculate_timespan_days(&self) -> u64 {
        if self.memories.len() < 2 {
            return 0;
        }

        let first = self.memories.front().unwrap().timestamp;
        let last = self.memories.back().unwrap().timestamp;
        let duration = last.signed_duration_since(first);
        duration.num_days().max(0) as u64
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

fn build_embedding(tokens: &[String], dim: usize) -> Vec<f32> {
    if tokens.is_empty() {
        return vec![0.0; dim];
    }

    let mut embedding = vec![0.0f32; dim];
    for token in tokens {
        let hash = blake3::hash(token.as_bytes());
        let mut idx_bytes = [0u8; 4];
        idx_bytes.copy_from_slice(&hash.as_bytes()[..4]);
        let index = (u32::from_le_bytes(idx_bytes) as usize) % dim;
        embedding[index] += 1.0;
    }

    let norm = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in &mut embedding {
            *value /= norm;
        }
    }
    embedding
}

fn tokenize(text: &str) -> Vec<String> {
    text.split_whitespace()
        .filter_map(|token| {
            let cleaned = token
                .trim_matches(|c: char| !c.is_alphanumeric())
                .to_lowercase();
            if cleaned.is_empty() {
                None
            } else {
                Some(cleaned)
            }
        })
        .collect()
}

fn extract_keywords(tokens: &[String], limit: usize) -> Vec<String> {
    let mut frequencies: HashMap<String, usize> = HashMap::new();
    for token in tokens {
        if token.len() <= 2 {
            continue;
        }
        *frequencies.entry(token.clone()).or_insert(0) += 1;
    }

    let mut keyword_pairs: Vec<(String, usize)> = frequencies.into_iter().collect();
    keyword_pairs.sort_by(|a, b| b.1.cmp(&a.1));
    keyword_pairs
        .into_iter()
        .take(limit)
        .map(|(keyword, _)| keyword)
        .collect()
}

fn truncate(content: &str, limit: usize) -> String {
    if content.len() <= limit {
        content.to_string()
    } else {
        let mut truncated = content[..limit].to_string();
        truncated.push_str("…");
        truncated
    }
}

fn tag_overlap(a: &PersonalMemory, b: &PersonalMemory) -> f32 {
    if a.tags.is_empty() || b.tags.is_empty() {
        return 0.0;
    }
    let set_a: HashSet<_> = a.tags.iter().collect();
    let set_b: HashSet<_> = b.tags.iter().collect();
    let intersection = set_a.intersection(&set_b).count() as f32;
    let union = set_a.union(&set_b).count() as f32;
    if union == 0.0 {
        0.0
    } else {
        intersection / union
    }
}

fn dominant_emotion_label(profile: &[f32; EMOTION_DIM]) -> &'static str {
    let labels = ["joy", "sadness", "anger", "fear", "surprise"];
    let mut max_index = 0;
    let mut max_value = profile[0];
    for (idx, value) in profile.iter().enumerate().skip(1) {
        if value > &max_value {
            max_value = *value;
            max_index = idx;
        }
    }
    if max_value < 0.05 {
        "neutral"
    } else {
        labels[max_index]
    }
}

struct EmotionVocabulary {
    weights: HashMap<&'static str, [f32; EMOTION_DIM]>,
}

impl EmotionVocabulary {
    fn analyze(&self, tokens: &[String]) -> [f32; EMOTION_DIM] {
        if tokens.is_empty() {
            return [0.0; EMOTION_DIM];
        }

        let mut totals = [0.0f32; EMOTION_DIM];
        for token in tokens {
            if let Some(weights) = self.weights.get(token.as_str()) {
                for i in 0..EMOTION_DIM {
                    totals[i] += weights[i];
                }
            }
        }

        let sum: f32 = totals.iter().copied().sum();
        if sum > 0.0 {
            for value in &mut totals {
                *value = (*value / sum).clamp(0.0, 1.0);
            }
        }
        totals
    }
}

static EMOTION_VOCAB: Lazy<EmotionVocabulary> = Lazy::new(|| EmotionVocabulary {
    weights: {
        let mut map = HashMap::new();
        map.insert("joy", [0.9, 0.0, 0.0, 0.0, 0.3]);
        map.insert("delighted", [1.0, 0.0, 0.0, 0.0, 0.4]);
        map.insert("comfort", [0.6, 0.0, 0.0, 0.1, 0.0]);
        map.insert("grateful", [0.8, 0.0, 0.0, 0.0, 0.2]);
        map.insert("sad", [0.0, 0.95, 0.0, 0.0, 0.05]);
        map.insert("loss", [0.0, 0.9, 0.0, 0.0, 0.0]);
        map.insert("alone", [0.0, 0.85, 0.0, 0.1, 0.0]);
        map.insert("angry", [0.0, 0.0, 0.9, 0.0, 0.1]);
        map.insert("furious", [0.0, 0.0, 1.0, 0.0, 0.1]);
        map.insert("frustrated", [0.0, 0.1, 0.8, 0.0, 0.1]);
        map.insert("afraid", [0.0, 0.0, 0.0, 1.0, 0.0]);
        map.insert("anxious", [0.0, 0.1, 0.0, 0.9, 0.1]);
        map.insert("worried", [0.0, 0.2, 0.0, 0.85, 0.1]);
        map.insert("surprised", [0.2, 0.0, 0.0, 0.0, 0.9]);
        map.insert("curious", [0.3, 0.0, 0.0, 0.0, 0.8]);
        map.insert("inspired", [0.8, 0.0, 0.0, 0.0, 0.6]);
        map.insert("peaceful", [0.75, 0.0, 0.0, 0.0, 0.1]);
        map
    },
});

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedding_size_is_consistent() {
        let tokens = tokenize("Joyful connection and inspired focus");
        let embedding = build_embedding(&tokens, MEMORY_EMBEDDING_DIM);
        assert_eq!(embedding.len(), MEMORY_EMBEDDING_DIM);
    }

    #[test]
    fn memory_engine_stores_and_retrieves() {
        let mut engine = PersonalMemoryEngine::new();
        engine
            .create_memory_from_conversation(
                "Shared a breakthrough in therapy session".to_string(),
                0.9,
            )
            .unwrap();
        engine.initialize_consciousness().unwrap();

        let results = engine.retrieve_relevant_memories_rag("therapy breakthrough");
        assert!(!results.is_empty());
    }
}
