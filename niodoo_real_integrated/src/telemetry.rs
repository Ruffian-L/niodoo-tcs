//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

//! Cognitive State Telemetry
//!
//! Defines the telemetry packet structure for broadcasting AI cognitive state
//! to visualization clients via TCP.

use serde::{Deserialize, Serialize};

/// Cognitive state packet broadcast after each pipeline iteration
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

pub mod server;

