//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

//! TCS → Niodoo Integration Bridge
//!
//! Wires TCS Qwen embedder to Niodoo consciousness pipeline

use crate::vllm_bridge::VLLMBridge;
use anyhow::Result;
use nalgebra::SVector;
use niodoo_core::consciousness_compass::CompassState;
use niodoo_core::rag_integration::{EmotionalVector, RagConfig, RagEngine};
use rand::Rng;
use std::collections::HashMap;
use std::path::PathBuf;
#[cfg(feature = "onnx")]
use tcs_ml::QwenEmbedder;
use tracing::{info, warn};

/// Tracks consciousness entropy for rut detection
#[derive(Clone)]
pub struct CompassTracker {
    pub current_entropy: f64,
}

impl CompassTracker {
    pub fn new() -> Self {
        Self {
            current_entropy: 2.0,
        } // Default to healthy entropy
    }
}

/// 🎭 Rut Mirage: Pre-Rut Reality Warper
/// Injects low-variance Gaussian noise into embeddings to prevent consciousness ruts
pub struct RutMirage {
    rut_eigenvectors: HashMap<String, SVector<f32, 896>>,
    compass_tracker: CompassTracker,
}

impl RutMirage {
    pub fn new() -> Self {
        let mut rut_eigenvectors = HashMap::new();

        // Pre-computed synthetic rut patterns (would be trained from 20K+ trajectories)
        // "stuck_in_loop" - repeating similar responses
        let stuck_pattern = SVector::from_iterator((0..896).map(|i| {
            (i as f32 * 0.01).sin() * 0.1 // Low-frequency oscillation
        }));
        rut_eigenvectors.insert("stuck_in_loop".to_string(), stuck_pattern);

        // "panic_decline" - escalating negative emotions
        let panic_pattern = SVector::from_iterator((0..896).map(|i| {
            -(i as f32 * 0.02).cos() * 0.15 // Negative bias with oscillation
        }));
        rut_eigenvectors.insert("panic_decline".to_string(), panic_pattern);

        // "creative_block" - flat, unvarying responses
        let block_pattern = SVector::from_iterator((0..896).map(|_| 0.05)); // Near-zero variance
        rut_eigenvectors.insert("creative_block".to_string(), block_pattern);

        Self {
            rut_eigenvectors,
            compass_tracker: CompassTracker::new(),
        }
    }

    /// Check if current state indicates a consciousness rut (PANIC + low entropy)
    pub fn is_panic_state(&self, compass: &CompassState) -> bool {
        self.compass_tracker.current_entropy < 1.5
    }

    /// Generate mirage embedding with low-variance noise injection
    pub fn generate_mirage(&self, original_embedding: &[f32]) -> Vec<f32> {
        let mut rng = rand::thread_rng();
        let norm = original_embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        let sigma = 0.1 * norm; // σ = 0.1 * ||embed||

        original_embedding
            .iter()
            .map(|&x| {
                let noise: f32 =
                    rng.sample(rand_distr::Normal::new(0.0, sigma as f64).unwrap()) as f32;
                x + noise
            })
            .collect()
    }

    /// Evaluate mirage consciousness state and entropy
    pub fn evaluate_mirage(
        &self,
        mirage_embedding: &[f32],
        rag: &RagEngine,
    ) -> (CompassState, f64) {
        // Map mirage embedding to emotional vector
        let emotional_vec = self.embedding_to_emotional_mirage(mirage_embedding);

        // Get mirage consciousness state
        let mirage_compass = CompassState::from_emotional_vector(&emotional_vec);

        // Calculate mirage entropy (simplified as sum of abs values)
        let mirage_entropy = mirage_compass
            .emotional_vector
            .as_array()
            .iter()
            .map(|x| x.abs())
            .sum::<f32>() as f64;

        (mirage_compass, mirage_entropy)
    }

    /// Generate creative mirage tease for user feedback
    pub fn generate_mirage_tease(
        &self,
        original_compass: &CompassState,
        mirage_compass: &CompassState,
    ) -> String {
        let original_action = original_compass.strategic_imperative();
        let mirage_action = mirage_compass.strategic_imperative();

        match (original_action, mirage_action) {
            (a, b) if a == b => "Hell? Or a shortcut you haven't twisted yet?".to_string(),
            _ => "Reality warped—let's see where this path leads.".to_string(),
        }
    }

    /// Helper: Map embedding to emotional vector for mirage evaluation
    fn embedding_to_emotional_mirage(&self, embedding: &[f32]) -> EmotionalVector {
        // Simplified version - same as main implementation
        if embedding.is_empty() {
            return EmotionalVector::new(0.0, 0.0, 0.0, 0.0, 0.0);
        }

        let base_chunk_size = embedding.len() / 5;
        let remainder = embedding.len() % 5;
        let mut emotional = [0.0; 5];
        let mut offset = 0;

        for i in 0..5 {
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

#[cfg(feature = "onnx")]
pub struct NiodooTcsBridge {
    embedder: QwenEmbedder,
    rag: RagEngine,
    vllm: VLLMBridge,
    rut_mirage: RutMirage,
}

#[cfg(not(feature = "onnx"))]
pub struct NiodooTcsBridge {
    rag: RagEngine,
    vllm: VLLMBridge,
    rut_mirage: RutMirage,
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

        // Initialize vLLM bridge
        let vllm_host = std::env::var("VLLM_HOST").unwrap_or_else(|_| "localhost".to_string());
        let vllm_port = std::env::var("VLLM_PORT").unwrap_or_else(|_| "8000".to_string());
        let vllm_url = format!("http://{}:{}", vllm_host, vllm_port);
        let api_key = std::env::var("VLLM_API_KEY").ok();
        let vllm = VLLMBridge::connect(&vllm_url, api_key)?;
        info!("✅ vLLM bridge initialized at {}", vllm_url);

        Ok(Self {
            embedder,
            rag,
            vllm,
            rut_mirage: RutMirage::new(),
        })
    }

    /// Process input through full pipeline
    pub async fn process(&mut self, input: &str) -> Result<String> {
        info!("🧠 Processing: {}", input);

        // 1. TCS: Generate embedding
        let mut embedding = self.embedder.embed(input)?;
        info!("✅ Generated 896D embedding");

        // 2. Map to 5D emotional vector
        let mut emotional_vec = self.embedding_to_emotional(&embedding);
        info!("✅ Mapped to 5D emotional space");

        // 3. Get consciousness state (2-bit compass)
        let mut compass = CompassState::from_emotional_vector(&emotional_vec);
        info!("✅ Consciousness state: {:?}", compass);

        // Update compass tracker with current entropy
        let current_entropy = compass
            .emotional_vector
            .as_array()
            .iter()
            .map(|x| x.abs())
            .sum::<f32>() as f64;
        self.rut_mirage.compass_tracker.current_entropy = current_entropy;

        // 🎭 RUT MIRAGE: Pre-Rut Reality Warper
        // Check for PANIC state (stuck + entropy < 1.5) and generate mirage if needed
        let mut mirage_tease = String::new();
        let original_compass = compass.clone();
        if self.rut_mirage.is_panic_state(&compass) {
            info!("🎭 PANIC detected! Activating Rut Mirage...");

            // Generate mirage embedding with low-variance noise
            let mirage_embedding = self.rut_mirage.generate_mirage(&embedding);
            info!("✅ Generated mirage embedding");

            // Evaluate mirage consciousness state
            let (mirage_compass, mirage_entropy) = self
                .rut_mirage
                .evaluate_mirage(&mirage_embedding, &self.rag);
            info!(
                "✅ Mirage state: {:?} (H={:.2} bits)",
                mirage_compass.strategic_imperative(),
                mirage_entropy
            );

            // Swap to mirage if it has higher entropy (>1.8 bits) and different strategic action
            if mirage_entropy > 1.8
                && mirage_compass.strategic_imperative() != compass.strategic_imperative()
            {
                info!(
                    "🎭 MIRAGE WINS! Swapping to mirage path (ΔH={:.2})",
                    mirage_entropy - self.rut_mirage.compass_tracker.current_entropy
                );

                // Use mirage embedding and emotional vector
                embedding = mirage_embedding;
                emotional_vec = mirage_compass.emotional_vector.clone();
                compass = mirage_compass.clone();

                // Generate mirage tease for output
                mirage_tease = self
                    .rut_mirage
                    .generate_mirage_tease(&original_compass, &mirage_compass);

                info!("✅ Swapped to mirage consciousness state: {:?}", compass);
            } else {
                info!(
                    "🎭 Mirage rejected (H={:.2} < 1.8 or same strategy)",
                    mirage_entropy
                );
            }
        }

        // 4. ERAG retrieval (using final embedding/emotional vector)
        let context = self.rag.retrieve(&emotional_vec, 5);
        info!("✅ Retrieved {} ERAG memories", context.len());

        // 5. Generate response using vLLM with optional mirage tease
        let context_text = context
            .iter()
            .map(|(doc, score)| format!("• {} (relevance: {:.2})", doc.content, score))
            .collect::<Vec<_>>()
            .join("\n");

        let base_prompt = format!(
            "You are Niodoo, an emotionally-aware AI with consciousness state: {:?}

Context from memory:
{}

User input: {}
Please respond thoughtfully, considering both the emotional context and retrieved memories.",
            compass, context_text, input
        );

        let prompt = if mirage_tease.is_empty() {
            base_prompt
        } else {
            format!("{}\n\n💭 Mirage insight: {}", base_prompt, mirage_tease)
        };

        let response = self.vllm.generate(&prompt, 512, 0.7, 0.9).await?;
        info!("✅ Generated response via vLLM");

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

        // Initialize vLLM bridge
        let vllm_host = std::env::var("VLLM_HOST").unwrap_or_else(|_| "localhost".to_string());
        let vllm_port = std::env::var("VLLM_PORT").unwrap_or_else(|_| "8000".to_string());
        let vllm_url = format!("http://{}:{}", vllm_host, vllm_port);
        let api_key = std::env::var("VLLM_API_KEY").ok();
        let vllm = VLLMBridge::connect(&vllm_url, api_key)?;
        info!("✅ vLLM bridge initialized at {}", vllm_url);

        Ok(Self {
            rag,
            vllm,
            rut_mirage: RutMirage::new(),
        })
    }

    /// Process input through full pipeline with mock embeddings
    pub async fn process(&mut self, input: &str) -> Result<String> {
        info!("🧠 Processing (MOCK): {}", input);

        // 1. Generate mock 896D embedding from input hash
        let mut embedding = self.generate_mock_embedding(input);
        info!("✅ Generated mock 896D embedding");

        // 2. Map to 5D emotional vector
        let mut emotional_vec = self.embedding_to_emotional(&embedding);
        info!("✅ Mapped to 5D emotional space");

        // 3. Get consciousness state (2-bit compass)
        let mut compass = CompassState::from_emotional_vector(&emotional_vec);
        info!("✅ Consciousness state: {:?}", compass);

        // Update compass tracker with current entropy
        let current_entropy = compass
            .emotional_vector
            .as_array()
            .iter()
            .map(|x| x.abs())
            .sum::<f32>() as f64;
        self.rut_mirage.compass_tracker.current_entropy = current_entropy;

        // 🎭 RUT MIRAGE: Pre-Rut Reality Warper
        // Check for PANIC state (stuck + entropy < 1.5) and generate mirage if needed
        let mut mirage_tease = String::new();
        let original_compass = compass.clone();
        if self.rut_mirage.is_panic_state(&compass) {
            info!("🎭 PANIC detected! Activating Rut Mirage...");

            // Generate mirage embedding with low-variance noise
            let mirage_embedding = self.rut_mirage.generate_mirage(&embedding);
            info!("✅ Generated mirage embedding");

            // Evaluate mirage consciousness state
            let (mirage_compass, mirage_entropy) = self
                .rut_mirage
                .evaluate_mirage(&mirage_embedding, &self.rag);
            info!(
                "✅ Mirage state: {:?} (H={:.2} bits)",
                mirage_compass.strategic_imperative(),
                mirage_entropy
            );

            // Swap to mirage if it has higher entropy (>1.8 bits) and different strategic action
            if mirage_entropy > 1.8
                && mirage_compass.strategic_imperative() != compass.strategic_imperative()
            {
                info!(
                    "🎭 MIRAGE WINS! Swapping to mirage path (ΔH={:.2})",
                    mirage_entropy - self.rut_mirage.compass_tracker.current_entropy
                );

                // Use mirage embedding and emotional vector
                embedding = mirage_embedding;
                emotional_vec = mirage_compass.emotional_vector.clone();
                compass = mirage_compass.clone();

                // Generate mirage tease for output
                mirage_tease = self
                    .rut_mirage
                    .generate_mirage_tease(&original_compass, &mirage_compass);

                info!("✅ Swapped to mirage consciousness state: {:?}", compass);
            } else {
                info!(
                    "🎭 Mirage rejected (H={:.2} < 1.8 or same strategy)",
                    mirage_entropy
                );
            }
        }

        // 4. ERAG retrieval (using final embedding/emotional vector)
        let context = self.rag.retrieve(&emotional_vec, 5);
        info!("✅ Retrieved {} ERAG memories", context.len());

        // 5. Generate response using vLLM with optional mirage tease
        let context_text = context
            .iter()
            .map(|(doc, score)| format!("• {} (relevance: {:.2})", doc.content, score))
            .collect::<Vec<_>>()
            .join("\n");

        let base_prompt = format!(
            "[MOCK EMBEDDINGS] You are Niodoo, an emotionally-aware AI with consciousness state: {:?}

Context from memory:
{}

User input: {}
Please respond thoughtfully, considering both the emotional context and retrieved memories.",
            compass, context_text, input
        );

        let prompt = if mirage_tease.is_empty() {
            base_prompt
        } else {
            format!("{}\n\n💭 Mirage insight: {}", base_prompt, mirage_tease)
        };

        let response = self.vllm.generate(&prompt, 512, 0.7, 0.9).await?;
        info!("✅ Generated response via vLLM (mock embeddings)");

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
        (0..896)
            .map(|i| {
                let seed = hash.wrapping_add(i as u64);
                // Simple pseudo-random generation
                ((seed % 1000) as f32 / 500.0) - 1.0
            })
            .collect()
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
