// Copyright 2025 Jason Van Pham (Niodoo)
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of Niodoo TCS.
// For commercial licensing, see LICENSE-COMMERCIAL.md

// Mock implementation when ONNX is not available
use anyhow::Result;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use tracing::{info, warn};

// Stub implementations for demo
#[derive(Clone)]
struct RagEngine;

impl RagEngine {
    fn new(_base_dir: PathBuf, _config: RagConfig) -> Result<Self> {
        Ok(Self)
    }

    fn retrieve(&self, _emotional_vec: &EmotionalVector, _k: usize) -> Vec<String> {
        vec!["mock context".to_string()]
    }
}

#[derive(Clone)]
struct RagConfig;

impl Default for RagConfig {
    fn default() -> Self {
        Self
    }
}

#[derive(Clone, Debug)]
struct EmotionalVector {
    data: [f32; 5],
}

impl EmotionalVector {
    fn new(a: f32, b: f32, c: f32, d: f32, e: f32) -> Self {
        Self {
            data: [a, b, c, d, e],
        }
    }
}

#[derive(Clone, Debug)]
struct CompassState {
    quadrant: String,
}

impl CompassState {
    fn from_emotional_vector(_vec: &EmotionalVector) -> Self {
        Self {
            quadrant: "neutral".to_string(),
        }
    }
}

pub struct NiodooTcsBridge {
    rag: RagEngine,
}

impl NiodooTcsBridge {
    /// Initialize bridge with mock embedder
    pub async fn new(_model_path: &str) -> Result<Self> {
        info!("🌉 Initializing TCS→Niodoo bridge (MOCK MODE)...");
        warn!("⚠️  ONNX not available - using deterministic mock embeddings");

        // Initialize Niodoo RAG
        let base_dir = PathBuf::from("./data/rag");
        let config = RagConfig::default();
        let rag = RagEngine::new(base_dir, config)?;
        info!("✅ ERAG initialized");

        Ok(Self { rag })
    }

    /// Process input through full pipeline with mock embeddings
    pub async fn process(&mut self, input: &str) -> Result<String> {
        info!("🧠 Processing (MOCK): {}", input);

        // 1. Generate mock 896D embedding from input hash
        let embedding = self.generate_mock_embedding(input);
        info!("✅ Generated mock 896D embedding");

        // 2. Map to 5D emotional vector
        let emotional_vec = self.embedding_to_emotional(&embedding);
        info!("✅ Mapped to 5D emotional space");

        // 3. Get consciousness state (2-bit compass)
        let compass = CompassState::from_emotional_vector(&emotional_vec);
        info!("✅ Consciousness state: {:?}", compass);

        // 4. ERAG retrieval
        let context = self.rag.retrieve(&emotional_vec, 5);
        info!("✅ Retrieved {} ERAG memories", context.len());

        // 5. Generate response
        let response = format!(
            "[MOCK MODE] Consciousness state: {:?}\nContext items: {}\nProcessed: {}",
            compass,
            context.len(),
            input
        );

        Ok(response)
    }

    /// Generate deterministic mock embedding from input hash
    fn generate_mock_embedding(&self, input: &str) -> Vec<f32> {
        let mut hasher = DefaultHasher::new();
        input.hash(&mut hasher);
        let hash = hasher.finish();

        // Generate 896D vector from hash (deterministic but varied)
        let mut embedding = Vec::with_capacity(896);
        let mut current_hash = hash;

        for _ in 0..896 {
            // Use hash to generate pseudo-random f32 values
            current_hash = current_hash.wrapping_mul(1103515245).wrapping_add(12345);
            let float_val = (current_hash % 10000) as f32 / 5000.0 - 1.0; // Range: -1.0 to 1.0
            embedding.push(float_val);
        }

        embedding
    }

    /// Convert 896D embedding to 5D emotional vector
    fn embedding_to_emotional(&self, embedding: &[f32]) -> EmotionalVector {
        // Guard against zero-length
        if embedding.is_empty() {
            return EmotionalVector::new(0.0, 0.0, 0.0, 0.0, 0.0);
        }

        // Distribute all elements evenly across 5 bins
        let base_chunk_size = embedding.len() / 5;
        let remainder = embedding.len() % 5;
        let mut emotional = [0.0; 5];
        let mut offset = 0;

        for i in 0..5 {
            // First 'remainder' bins get one extra element
            let chunk_size = if i < remainder {
                base_chunk_size + 1
            } else {
                base_chunk_size
            };
            if chunk_size > 0 {
                let chunk = &embedding[offset..offset + chunk_size];
                emotional[i] = chunk.iter().sum::<f32>() / chunk.len() as f32;
                offset += chunk_size;
            }
        }

        EmotionalVector::new(
            emotional[0],
            emotional[1],
            emotional[2],
            emotional[3],
            emotional[4],
        )
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    // Run mock bridge example
    let mut bridge = NiodooTcsBridge::new("./models/qwen2.5-1.5b-onnx").await?;
    let result = bridge.process("Hello, world!").await?;

    println!("{}", result);
    Ok(())
}
