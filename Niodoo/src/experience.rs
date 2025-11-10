use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

/// Experience tuple for learning/replay buffers and curator/executor integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experience {
    pub id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub input: String,
    pub output: String,
    pub context: String, // Changed from Vec<String> to String for compatibility
    pub task_type: String,
    pub success_score: Option<i32>, // Changed to Option<i32> to match curator output
    pub rouge_l: f32, // Added for curator scoring
    pub feedback: String, // Added for curator feedback
    pub reward: f64,
    pub metadata: Option<Value>, // Added for topology/PAD/compass metadata
    pub state: Vec<f32>,
    pub action: usize,
    pub next_state: Vec<f32>,
    pub done: bool,
    pub embedding: Option<Vec<f32>>,
    pub relevance_score: f32,
}

impl Experience {
    pub fn new(
        input: String,
        output: String,
        context: String,
        task_type: String,
        success_score: Option<i32>,
        rouge_l: f32,
        feedback: String,
        reward: f64,
        metadata: Option<Value>,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            input,
            output,
            context,
            task_type,
            success_score,
            rouge_l,
            feedback,
            reward,
            metadata,
            state: Vec::new(),
            action: 0,
            next_state: Vec::new(),
            done: false,
            embedding: None,
            relevance_score: 0.0,
        }
    }

    /// Normalize embedding to unit hypersphere
    pub fn normalize_embedding(&mut self) {
        if let Some(ref mut embedding) = self.embedding {
            let norm = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for val in embedding.iter_mut() {
                    *val /= norm;
                }
            }
        }
    }

    /// Update success score
    pub fn with_success_score(mut self, score: Option<i32>) -> Self {
        self.success_score = score;
        self
    }

    /// Update task type
    pub fn with_task_type<S: Into<String>>(mut self, task_type: S) -> Self {
        self.task_type = task_type.into();
        self
    }
}

