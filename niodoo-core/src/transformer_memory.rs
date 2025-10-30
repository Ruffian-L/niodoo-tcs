// Copyright 2025 Jason Van Pham (Niodoo)
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of Niodoo TCS.
// For commercial licensing, see LICENSE-COMMERCIAL.md

//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

// Transformer Memory System - Modular memory for ANY AI model
// Works with GPT, Claude, Llama, any instruction or reasoning model

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter};

/// Core memory types for real-world usage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryType {
    Goal,         // Original objectives and plans
    Context,      // Ongoing project context
    Decision,     // Key decisions made
    Divergence,   // When we strayed from plan
    Learning,     // What worked/didn't work
    Reference,    // Important information to remember
    Conversation, // Actual conversation history
    Task,         // Specific tasks and their status
}

/// A real memory entry - not philosophical, but practical
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealMemory {
    pub id: String,
    pub timestamp: DateTime<Utc>,
    pub memory_type: MemoryType,
    pub content: String,
    pub context: HashMap<String, String>,
    pub importance: f32, // 0.0 to 1.0
    pub referenced_count: usize,
    pub parent_goal: Option<String>, // Links back to original goal
}

/// Plan tracking - knows when you're going off course
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanTracker {
    pub original_goal: String,
    pub current_path: Vec<String>,
    pub divergence_points: Vec<(DateTime<Utc>, String)>,
    pub success_metrics: HashMap<String, bool>,
}

/// Universal memory interface for any AI model
pub trait AIMemoryInterface: Send + Sync {
    /// Inject relevant memories into the prompt
    fn augment_prompt(&self, base_prompt: &str, max_tokens: usize) -> String;

    /// Extract and store memories from AI response
    fn extract_memories(&mut self, response: &str, context: &HashMap<String, String>);

    /// Check if current action aligns with goals
    fn check_alignment(&self, action: &str) -> AlignmentResult;

    /// Get relevant memories for context
    fn retrieve_relevant(&self, query: &str, limit: usize) -> Vec<RealMemory>;
}

#[derive(Debug)]
pub struct AlignmentResult {
    pub aligned: bool,
    pub confidence: f32,
    pub suggestion: Option<String>,
    pub related_goal: Option<String>,
}

/// The actual transformer memory system
pub struct TransformerMemory {
    memories: VecDeque<RealMemory>,
    goals: Vec<PlanTracker>,
    active_context: HashMap<String, String>,
    max_memories: usize,
    embedding_cache: HashMap<String, Vec<f32>>, // For semantic search
}

impl TransformerMemory {
    pub fn new(max_memories: usize) -> Self {
        let mut system = Self {
            memories: VecDeque::new(),
            goals: Vec::new(),
            active_context: HashMap::new(),
            max_memories,
            embedding_cache: HashMap::new(),
        };

        // Load existing memories
        system.load_memories();
        system
    }

    /// Add a new goal/plan to track
    pub fn add_goal(&mut self, goal: String, metrics: HashMap<String, bool>) {
        let tracker = PlanTracker {
            original_goal: goal.clone(),
            current_path: vec![goal.clone()],
            divergence_points: Vec::new(),
            success_metrics: metrics,
        };

        self.goals.push(tracker);

        // Store as memory
        self.store_memory(MemoryType::Goal, goal, 1.0, None);
    }

    /// Store a real memory
    pub fn store_memory(
        &mut self,
        memory_type: MemoryType,
        content: String,
        importance: f32,
        parent_goal: Option<String>,
    ) {
        let memory = RealMemory {
            id: format!("mem_{}", Utc::now().timestamp_millis()),
            timestamp: Utc::now(),
            memory_type,
            content,
            context: self.active_context.clone(),
            importance,
            referenced_count: 0,
            parent_goal,
        };

        // Add to memory queue
        self.memories.push_back(memory);

        // Enforce memory limit (LRU with importance weighting)
        while self.memories.len() > self.max_memories {
            // Remove least important old memory
            let min_importance_idx = self
                .memories
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    let a_score = a.importance + (a.referenced_count as f32 * 0.1);
                    let b_score = b.importance + (b.referenced_count as f32 * 0.1);
                    a_score.partial_cmp(&b_score).unwrap()
                })
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            self.memories.remove(min_importance_idx);
        }

        // Auto-save every 10 memories
        if self.memories.len() % 10 == 0 {
            self.save_memories();
        }
    }

    /// Check if we're diverging from original plan
    pub fn check_divergence(&mut self, current_action: &str) -> Option<String> {
        let mut divergence_to_store = None;

        for goal in &mut self.goals {
            // Simple keyword matching for now (would use embeddings in production)
            let goal_keywords: Vec<&str> = goal.original_goal.split_whitespace().collect();

            let action_keywords: Vec<&str> = current_action.split_whitespace().collect();

            let overlap = goal_keywords
                .iter()
                .filter(|k| action_keywords.contains(k))
                .count();

            let alignment_ratio = overlap as f32 / goal_keywords.len() as f32;

            if alignment_ratio < 0.3 {
                let divergence = format!(
                    "⚠️ Divergence detected: Current action '{}' seems to be branching \
                    from original goal '{}'. Alignment: {:.0}%",
                    current_action,
                    goal.original_goal,
                    alignment_ratio * 100.0
                );

                goal.divergence_points
                    .push((Utc::now(), current_action.to_string()));

                // Store divergence info for later
                divergence_to_store = Some((divergence.clone(), goal.original_goal.clone()));

                break;
            }
        }

        // Store memory outside the loop to avoid borrow issues
        if let Some((divergence, goal)) = divergence_to_store {
            self.store_memory(MemoryType::Divergence, divergence.clone(), 0.8, Some(goal));
            return Some(divergence);
        }

        None
    }

    /// Update active context
    pub fn update_context(&mut self, key: String, value: String) {
        self.active_context.insert(key, value);
    }

    /// Save memories to disk
    fn save_memories(&self) {
        let path = "./data/transformer_memories.json";
        std::fs::create_dir_all("./data").ok();

        if let Ok(file) = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
        {
            let writer = BufWriter::new(file);
            serde_json::to_writer_pretty(writer, &self.memories).ok();
        }

        // Also save goals
        let goals_path = "./data/active_goals.json";
        if let Ok(file) = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(goals_path)
        {
            let writer = BufWriter::new(file);
            serde_json::to_writer_pretty(writer, &self.goals).ok();
        }
    }

    /// Load memories from disk
    fn load_memories(&mut self) {
        let path = "./data/transformer_memories.json";
        if let Ok(file) = File::open(path) {
            let reader = BufReader::new(file);
            if let Ok(memories) = serde_json::from_reader(reader) {
                self.memories = memories;
            }
        }

        let goals_path = "./data/active_goals.json";
        if let Ok(file) = File::open(goals_path) {
            let reader = BufReader::new(file);
            if let Ok(goals) = serde_json::from_reader(reader) {
                self.goals = goals;
            }
        }
    }
}

impl AIMemoryInterface for TransformerMemory {
    fn augment_prompt(&self, base_prompt: &str, max_tokens: usize) -> String {
        // Get relevant memories
        let relevant = self.retrieve_relevant(base_prompt, 5);

        // Build context injection
        let mut augmented = String::new();

        // Add active goals
        if !self.goals.is_empty() {
            augmented.push_str("# Active Goals:\n");
            for goal in &self.goals {
                augmented.push_str(&format!("- {}\n", goal.original_goal));
            }
            augmented.push_str("\n");
        }

        // Add relevant memories
        if !relevant.is_empty() {
            augmented.push_str("# Relevant Context:\n");
            for memory in relevant {
                augmented.push_str(&format!(
                    "- [{}] {}\n",
                    match memory.memory_type {
                        MemoryType::Goal => "GOAL",
                        MemoryType::Context => "CONTEXT",
                        MemoryType::Decision => "DECISION",
                        MemoryType::Divergence => "DIVERGENCE",
                        MemoryType::Learning => "LEARNING",
                        MemoryType::Reference => "REFERENCE",
                        MemoryType::Conversation => "PREVIOUS",
                        MemoryType::Task => "TASK",
                    },
                    memory.content
                ));
            }
            augmented.push_str("\n");
        }

        // Add the actual prompt
        augmented.push_str("# Current Request:\n");
        augmented.push_str(base_prompt);

        // Truncate if needed
        if augmented.len() > max_tokens * 4 {
            // Rough char to token estimate
            augmented.truncate(max_tokens * 4);
        }

        augmented
    }

    fn extract_memories(&mut self, response: &str, context: &HashMap<String, String>) {
        // Update active context
        for (k, v) in context {
            self.active_context.insert(k.clone(), v.clone());
        }

        // Look for decision markers
        if response.contains("decided") || response.contains("will") || response.contains("should")
        {
            self.store_memory(
                MemoryType::Decision,
                response.to_string(),
                0.7,
                self.goals.first().map(|g| g.original_goal.clone()),
            );
        }

        // Store as conversation memory
        self.store_memory(MemoryType::Conversation, response.to_string(), 0.5, None);
    }

    fn check_alignment(&self, action: &str) -> AlignmentResult {
        // Check against all goals
        for goal in &self.goals {
            let goal_keywords: Vec<&str> = goal.original_goal.split_whitespace().collect();

            let action_keywords: Vec<&str> = action.split_whitespace().collect();

            let overlap = goal_keywords
                .iter()
                .filter(|k| action_keywords.contains(k))
                .count();

            let alignment_ratio = overlap as f32 / goal_keywords.len() as f32;

            if alignment_ratio < 0.3 {
                return AlignmentResult {
                    aligned: false,
                    confidence: 1.0 - alignment_ratio,
                    suggestion: Some(format!(
                        "This seems to diverge from your goal: '{}'. \
                        Consider refocusing on the original objective.",
                        goal.original_goal
                    )),
                    related_goal: Some(goal.original_goal.clone()),
                };
            }
        }

        AlignmentResult {
            aligned: true,
            confidence: 0.8,
            suggestion: None,
            related_goal: self.goals.first().map(|g| g.original_goal.clone()),
        }
    }

    fn retrieve_relevant(&self, query: &str, limit: usize) -> Vec<RealMemory> {
        // Simple keyword matching for now
        // In production, would use embeddings and vector similarity

        let query_lower = query.to_lowercase();
        let query_keywords: Vec<&str> = query_lower.split_whitespace().collect();

        let mut scored_memories: Vec<(f32, RealMemory)> = self
            .memories
            .iter()
            .map(|memory| {
                let content_lower = memory.content.to_lowercase();
                let keyword_matches = query_keywords
                    .iter()
                    .filter(|k| content_lower.contains(*k))
                    .count() as f32;

                let recency_score = 1.0
                    / (1.0
                        + (Utc::now().timestamp() - memory.timestamp.timestamp()) as f32 / 86400.0);
                let importance_score = memory.importance;
                let reference_score = memory.referenced_count as f32 * 0.1;

                let total_score = keyword_matches * 2.0
                    + recency_score * 0.5
                    + importance_score
                    + reference_score;

                (total_score, memory.clone())
            })
            .collect();

        scored_memories.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        scored_memories
            .into_iter()
            .take(limit)
            .map(|(_, memory)| memory)
            .collect()
    }
}
