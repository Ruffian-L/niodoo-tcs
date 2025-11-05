//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

//! TCS → Niodoo Integration Bridge
//!
//! Wires TCS Qwen embedder to Niodoo consciousness pipeline

use anyhow::Result;
#[cfg(feature = "onnx")]
use tcs_ml::QwenEmbedder;
use niodoo_core::consciousness_compass::CompassState;
use niodoo_core::rag_integration::{RagEngine, RagConfig, EmotionalVector};
use tracing::{info, warn};
use std::path::PathBuf;

#[cfg(feature = "onnx")]
pub struct NiodooTcsBridge {
    embedder: QwenEmbedder,
    rag: RagEngine,
}

#[cfg(not(feature = "onnx"))]
pub struct NiodooTcsBridge {
    rag: RagEngine,
}

#[cfg(feature = "onnx")]
impl NiodooTcsBridge {
    /// Initialize bridge with model path
    pub async fn new(model_path: &str) -> Result<Self> {
        info!("🌉 Initializing TCS→Niodoo bridge...");

        // Initialize TCS embedder
        let embedder = QwenEmbedder::new(model_path)?;
        info!("✅ TCS embedder loaded");

        // Initialize Niodoo RAG
        let base_dir = PathBuf::from("./data/rag"); // TODO: make configurable
        let config = RagConfig::default();
        let rag = RagEngine::new(base_dir, config)?;
        info!("✅ ERAG initialized");

        Ok(Self { embedder, rag })
    }

    /// Process input through full pipeline
    pub async fn process(&mut self, input: &str) -> Result<String> {
        info!("🧠 Processing: {}", input);

        // 1. TCS: Generate embedding
        let embedding = self.embedder.embed(input)?;
        info!("✅ Generated 896D embedding");

        // 2. Map to 5D emotional vector
        let emotional_vec = self.embedding_to_emotional(&embedding);
        info!("✅ Mapped to 5D emotional space");

        // 3. Get consciousness state (2-bit compass)
        let compass = CompassState::from_emotional_vector(&emotional_vec);
        info!("✅ Consciousness state: {:?}", compass);

        // 4. ERAG retrieval
        let context = self.rag.retrieve(&emotional_vec, 5);
        info!("✅ Retrieved {} ERAG memories", context.len());

        // 5. Generate response (placeholder - wire to vLLM)
        let response = format!(
            "Consciousness state: {:?}\nContext items: {}\nProcessed: {}",
            compass, context.len(), input
        );

        Ok(response)
    }

    /// Convert 896D embedding to 5D emotional vector
    /// TODO: Implement proper K-Twist Möbius Torus mapping
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
            let chunk_size = if i < remainder { base_chunk_size + 1 } else { base_chunk_size };

            if chunk_size > 0 {
                let chunk = &embedding[offset..offset + chunk_size];
                emotional[i] = chunk.iter().sum::<f32>() / chunk.len() as f32;
                offset += chunk_size;
            }
        }

        EmotionalVector::new(emotional[0], emotional[1], emotional[2], emotional[3], emotional[4])
    }
}

#[cfg(not(feature = "onnx"))]
impl NiodooTcsBridge {
    /// Initialize bridge with model path (mock version without ONNX)
    pub async fn new(_model_path: &str) -> Result<Self> {
        info!("🌉 Initializing TCS→Niodoo bridge (mock mode - no ONNX)...");

        // Initialize Niodoo RAG
        let base_dir = PathBuf::from("./data/rag"); // TODO: make configurable
        let config = RagConfig::default();
        let rag = RagEngine::new(base_dir, config)?;
        info!("✅ ERAG initialized (mock embedder)");

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
            compass, context.len(), input
        );

        Ok(response)
    }

    /// Get mutable access to RAG engine for adding knowledge base data
    pub fn rag_engine(&mut self) -> &mut RagEngine {
        &mut self.rag
    }

    /// Generate mock embedding based on input text
    fn generate_mock_embedding(&self, input: &str) -> Vec<f32> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        input.hash(&mut hasher);
        let hash = hasher.finish();

        // Generate 896D embedding from hash
        (0..896).map(|i| {
            let seed = hash.wrapping_add(i as u64);
            // Simple pseudo-random generation
            ((seed % 1000) as f32 / 500.0) - 1.0
        }).collect()
    }

    /// Convert 896D embedding to 5D emotional vector
    /// TODO: Implement proper K-Twist Möbius Torus mapping
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
            let chunk_size = if i < remainder { base_chunk_size + 1 } else { base_chunk_size };

            if chunk_size > 0 {
                let chunk = &embedding[offset..offset + chunk_size];
                emotional[i] = chunk.iter().sum::<f32>() / chunk.len() as f32;
                offset += chunk_size;
            }
        }

        EmotionalVector::new(emotional[0], emotional[1], emotional[2], emotional[3], emotional[4])
    }
}