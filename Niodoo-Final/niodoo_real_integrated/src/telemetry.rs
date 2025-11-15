//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

//! Cognitive State Telemetry
//!
//! Defines the telemetry packet structure for broadcasting AI cognitive state
//! to visualization clients via TCP.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Cognitive state packet broadcast after each pipeline iteration
/// Legacy version maintained for backward compatibility
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CognitiveStatePacket {
    /// First 3 PAD dimensions (Pleasure, Arousal, Dominance)
    pub pad_state: [f32; 3],
    /// 3D projection coordinates [x, y, z] on the torus manifold
    pub torus_projection: [f32; 3],
    /// Betti numbers (β₀, β₁, β₂) from topological analysis
    pub betti_numbers: (usize, usize, usize),
    /// Persistence entropy from topology
    pub persistence_entropy: f64,
    /// Compass quadrant: "Panic", "Persist", "Discover", or "Master"
    pub compass_quadrant: String,
    /// Compass confidence score
    pub compass_confidence: f32,
    /// Retrieved memory IDs from Qdrant
    pub retrieved_memory_ids: Vec<String>,
    /// Optional iteration counter
    pub iteration: Option<u64>,
    /// Optional prompt text (truncated if long)
    pub prompt_text: Option<String>,
    /// ISO timestamp
    pub timestamp: String,
}

/// Prompt metadata with full details
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PromptMetadata {
    /// Full prompt text (no truncation)
    pub full_text: String,
    /// Token count
    pub token_count: usize,
    /// Token IDs if available
    pub token_ids: Option<Vec<u32>>,
    /// Prompt type: user, system, few-shot, etc.
    pub prompt_type: PromptType,
    /// Unique prompt ID
    pub prompt_id: Uuid,
}

/// Prompt type classification
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum PromptType {
    User,
    System,
    FewShot,
    Augmented,
    Other(String),
}

/// Response metadata with full details
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ResponseMetadata {
    /// Full response text
    pub full_text: String,
    /// Baseline response text
    pub baseline_text: String,
    /// Token count
    pub token_count: usize,
    /// Generation tokens with metadata if available
    pub generation_tokens: Option<Vec<TokenGeneration>>,
    /// Finish reason
    pub finish_reason: FinishReason,
    /// Unique response ID
    pub response_id: Uuid,
}

/// Token generation details
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TokenGeneration {
    /// Token ID
    pub token_id: u32,
    /// Token text
    pub token_text: String,
    /// Logits if available
    pub logits: Option<Vec<f32>>,
    /// Probability
    pub probability: Option<f32>,
}

/// Finish reason for generation
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum FinishReason {
    Stop,
    Length,
    Error(String),
    Unknown,
}

/// Memory entry with full content
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MemoryEntry {
    /// Memory ID
    pub memory_id: String,
    /// Full input text
    pub input: String,
    /// Full output text
    pub output: String,
    /// Retrieval score
    pub retrieval_score: f32,
    /// Impact on generation (estimated)
    pub impact: f32,
    /// Emotional vector if available
    pub emotional_vector: Option<Vec<f32>>,
    /// Timestamp
    pub timestamp: Option<String>,
}

/// Thought node in the reasoning tree
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ThoughtNode {
    /// Node ID
    pub node_id: Uuid,
    /// Node type
    pub node_type: ThoughtNodeType,
    /// Content/description
    pub content: String,
    /// Confidence score (point estimate)
    pub confidence: f32,
    /// Confidence interval (lower, upper) for uncertainty visualization
    pub confidence_interval: Option<(f32, f32)>,
    /// Whether this node represents a pruned path (ghost branch)
    pub pruned: bool,
    /// Rationale for pruning (if pruned)
    pub pruning_rationale: Option<String>,
    /// Timestamp
    pub timestamp: String,
    /// Parent node ID (if any)
    pub parent_id: Option<Uuid>,
    /// Child node IDs
    pub children: Vec<Uuid>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Thought node type
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum ThoughtNodeType {
    Observation {
        source: String,
        raw_data: Option<String>,
    },
    Reasoning {
        reasoning_type: ReasoningType,
        chain: Vec<String>,
    },
    Decision {
        options: Vec<DecisionOption>,
        chosen: usize,
        rationale: String,
    },
    Action {
        action_type: String,
        parameters: HashMap<String, String>,
    },
    Memory {
        memory_id: String,
        retrieval_score: f32,
        impact: f32,
    },
}

/// Reasoning type
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum ReasoningType {
    Deduction,
    Induction,
    Abduction,
    Analogy,
    Other(String),
}

/// Decision option
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DecisionOption {
    /// Option index
    pub index: usize,
    /// Option description
    pub description: String,
    /// Option score/confidence
    pub score: f32,
    /// Whether this option was pruned (not chosen)
    pub pruned: bool,
    /// Rationale for pruning (if pruned)
    pub pruning_rationale: Option<String>,
}

/// Thought tree structure
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ThoughtTree {
    /// Root node
    pub root_node: Option<ThoughtNode>,
    /// All nodes in the tree
    pub nodes: Vec<ThoughtNode>,
    /// Reasoning steps
    pub reasoning_steps: Vec<ReasoningStep>,
    /// Decision points
    pub decision_points: Vec<DecisionPoint>,
    /// Confidence path through reasoning
    pub confidence_path: Vec<f32>,
}

/// Reasoning step
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ReasoningStep {
    /// Step ID
    pub step_id: Uuid,
    /// Step type
    pub step_type: ReasoningType,
    /// Premises
    pub premises: Vec<String>,
    /// Conclusion
    pub conclusion: String,
    /// Confidence
    pub confidence: f32,
}

/// Decision point
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DecisionPoint {
    /// Decision ID
    pub decision_id: Uuid,
    /// Decision description
    pub description: String,
    /// Available options (chosen + pruned)
    pub options: Vec<DecisionOption>,
    /// Chosen option index
    pub chosen: usize,
    /// Pruned options (paths not taken)
    pub pruned_options: Vec<DecisionOption>,
    /// Rationale
    pub rationale: String,
    /// Confidence
    pub confidence: f32,
}

/// Pipeline stage execution details
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct StageExecution {
    /// Stage name
    pub stage_name: String,
    /// Timing in milliseconds
    pub timing_ms: f64,
    /// Errors if any
    pub errors: Vec<StageError>,
    /// Stage-specific metrics
    pub metrics: HashMap<String, f64>,
}

/// Stage error
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct StageError {
    /// Error type
    pub error_type: String,
    /// Error message
    pub message: String,
    /// Error timestamp
    pub timestamp: String,
}

/// Performance metrics
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PerformanceMetrics {
    /// Total latency in milliseconds
    pub latency_ms: f64,
    /// Tokens per second
    pub tokens_per_second: Option<f64>,
    /// GPU utilization if available
    pub gpu_utilization: Option<f32>,
    /// Memory usage
    pub memory_usage: Option<MemoryUsage>,
    /// Cache hit rate
    pub cache_hit_rate: Option<f32>,
}

/// Memory usage metrics
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MemoryUsage {
    /// Used memory in bytes
    pub used_bytes: u64,
    /// Total memory in bytes
    pub total_bytes: u64,
    /// Peak memory in bytes
    pub peak_bytes: Option<u64>,
}

/// Test run metadata
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TestRunMetadata {
    /// Test run ID
    pub test_id: Uuid,
    /// Test name
    pub test_name: String,
    /// Test configuration
    pub test_config: HashMap<String, String>,
    /// Expected output if available
    pub expected_output: Option<String>,
    /// Evaluation metrics
    pub evaluation_metrics: HashMap<String, f64>,
}

/// Enhanced cognitive state packet with complete telemetry
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct EnhancedCognitiveStatePacket {
    /// Core state (existing fields)
    /// First 3 PAD dimensions (Pleasure, Arousal, Dominance)
    pub pad_state: [f32; 3],
    /// 3D projection coordinates [x, y, z] on the torus manifold
    pub torus_projection: [f32; 3],
    /// Betti numbers (β₀, β₁, β₂) from topological analysis
    pub betti_numbers: (usize, usize, usize),
    /// Persistence entropy from topology
    pub persistence_entropy: f64,
    /// Compass quadrant: "Panic", "Persist", "Discover", or "Master"
    pub compass_quadrant: String,
    /// Compass confidence score
    pub compass_confidence: f32,
    /// Iteration counter
    pub iteration: u64,
    /// ISO timestamp
    pub timestamp: String,

    /// Enhanced fields
    /// Prompt metadata with full text
    pub prompt: PromptMetadata,
    /// Response metadata with full text
    pub response: ResponseMetadata,
    /// Thought structure tree
    pub thought_structure: Option<ThoughtTree>,
    /// Memory retrieval with full content
    pub memory_retrieval: MemoryRetrieval,
    /// Pipeline stage execution details
    pub pipeline_stages: Vec<StageExecution>,
    /// Performance metrics
    pub performance: PerformanceMetrics,
    /// Test run metadata (optional)
    pub test_run: Option<TestRunMetadata>,
}

/// Memory retrieval details
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MemoryRetrieval {
    /// Retrieved memories with full content
    pub retrieved_memories: Vec<MemoryEntry>,
    /// Retrieval strategy used
    pub retrieval_strategy: String,
    /// Average similarity score
    pub average_similarity: f32,
}

impl EnhancedCognitiveStatePacket {
    /// Convert to legacy CognitiveStatePacket for backward compatibility
    pub fn to_legacy(&self) -> CognitiveStatePacket {
        CognitiveStatePacket {
            pad_state: self.pad_state,
            torus_projection: self.torus_projection,
            betti_numbers: self.betti_numbers,
            persistence_entropy: self.persistence_entropy,
            compass_quadrant: self.compass_quadrant.clone(),
            compass_confidence: self.compass_confidence,
            retrieved_memory_ids: self
                .memory_retrieval
                .retrieved_memories
                .iter()
                .map(|m| m.memory_id.clone())
                .collect(),
            iteration: Some(self.iteration),
            prompt_text: Some(self.prompt.full_text.chars().take(100).collect()),
            timestamp: self.timestamp.clone(),
        }
    }
}

pub mod file_logger;
pub mod replay;
pub mod server;
pub mod storage;
pub mod test_run;
pub mod thought_structure;
