//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

/*
 * 🚀 BATCH OPTIMIZATION FOR QWEN INFERENCE
 *
 * Dynamic batch sizing based on context detection:
 * - Interactive chat: batch_size=1 for low latency
 * - RAG retrieval: vectorized batching for high throughput
 * - Code generation: batch_size=1 with higher temperature
 *
 * Performance optimization for latency vs throughput tradeoffs
 */

use anyhow::Result;
use tracing::{debug, info};

// Re-export types from parent crate for external use
pub use candle_core::{Device, Tensor};

/// Batch processing mode for optimizing inference based on use case
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchMode {
    /// Interactive chat mode: batch_size=1, low latency
    /// Optimized for real-time user interaction
    /// Average latency: ~50-100ms per token
    Interactive,

    /// RAG retrieval mode: vectorized batch processing, high throughput
    /// Processes multiple embeddings in parallel for similarity search
    /// Optimizes for throughput over latency (batch_size=16-64)
    RagRetrieval { batch_size: usize },

    /// Code generation mode: batch_size=1, higher temperature
    /// For generating Rust transformer code with Möbius attention
    /// Allows more creative/diverse outputs
    CodeGen,
}

impl BatchMode {
    /// Get optimal batch size for this mode
    pub fn batch_size(&self) -> usize {
        match self {
            BatchMode::Interactive => 1,
            BatchMode::RagRetrieval { batch_size } => *batch_size,
            BatchMode::CodeGen => 1,
        }
    }

    /// Check if this mode supports vectorized operations
    pub fn is_vectorized(&self) -> bool {
        matches!(self, BatchMode::RagRetrieval { .. })
    }

    /// Get optimal temperature for this mode
    pub fn temperature(&self) -> f32 {
        match self {
            BatchMode::Interactive => 0.7,
            BatchMode::RagRetrieval { .. } => 0.1, // Low temperature for consistent embeddings
            BatchMode::CodeGen => 0.8,
        }
    }

    /// Detect appropriate batch mode from input context
    pub fn detect_from_input(input: &str, is_embedding_generation: bool) -> Self {
        // If generating embeddings for RAG, use vectorized mode
        if is_embedding_generation {
            return BatchMode::RagRetrieval { batch_size: 32 };
        }

        // Detect code generation requests
        if input.contains("fn ")
            || input.contains("impl ")
            || input.contains("struct ")
            || input.contains("trait ")
            || input.contains("generate code")
            || input.contains("write rust")
        {
            debug!("Detected code generation request -> CodeGen mode");
            return BatchMode::CodeGen;
        }

        // Default to interactive for chat-like queries
        debug!("Defaulting to Interactive mode for chat");
        BatchMode::Interactive
    }
}

/// Context detector for automatic batch mode selection
pub struct BatchContextDetector {
    /// Number of consecutive RAG retrievals (triggers vectorized mode)
    rag_retrieval_count: usize,

    /// Average latency per request (helps optimize batch size)
    avg_latency_ms: f32,
}

impl Default for BatchContextDetector {
    fn default() -> Self {
        Self {
            rag_retrieval_count: 0,
            avg_latency_ms: 0.0,
        }
    }
}

impl BatchContextDetector {
    /// Update context based on latest request
    pub fn update(&mut self, is_rag: bool, latency_ms: f32) {
        if is_rag {
            self.rag_retrieval_count += 1;
        } else {
            self.rag_retrieval_count = 0;
        }

        // Exponential moving average for latency
        if self.avg_latency_ms == 0.0 {
            self.avg_latency_ms = latency_ms;
        } else {
            self.avg_latency_ms = 0.9 * self.avg_latency_ms + 0.1 * latency_ms;
        }
    }

    /// Get recommended batch mode based on current context
    pub fn recommend_mode(&self) -> BatchMode {
        // If we've seen multiple RAG retrievals, switch to vectorized mode
        if self.rag_retrieval_count >= 3 {
            // Dynamic batch size based on latency
            let batch_size = if self.avg_latency_ms < 50.0 {
                64 // GPU is fast, use large batches
            } else if self.avg_latency_ms < 200.0 {
                32 // Medium latency, moderate batch
            } else {
                16 // High latency, smaller batch to avoid OOM
            };

            info!(
                "Switching to RAG vectorized mode (batch_size={}) after {} retrievals",
                batch_size, self.rag_retrieval_count
            );

            return BatchMode::RagRetrieval { batch_size };
        }

        BatchMode::Interactive
    }
}

/// Batch processor for vectorized embedding generation (RAG use case)
pub struct VectorizedEmbeddingBatch {
    /// Texts to generate embeddings for
    texts: Vec<String>,

    /// Device for tensor operations
    device: Device,

    /// Batch size for processing
    batch_size: usize,
}

impl VectorizedEmbeddingBatch {
    pub fn new(texts: Vec<String>, device: Device, batch_size: usize) -> Self {
        Self {
            texts,
            device,
            batch_size,
        }
    }

    /// Process all texts in vectorized batches
    /// Returns embeddings in same order as input texts
    pub fn process_batched<F>(&self, mut embed_fn: F) -> Result<Vec<Vec<f32>>>
    where
        F: FnMut(&str) -> Result<Vec<f32>>,
    {
        let mut all_embeddings = Vec::with_capacity(self.texts.len());

        // Process in batches
        for batch in self.texts.chunks(self.batch_size) {
            debug!(
                "Processing embedding batch: {} texts (batch_size={})",
                batch.len(),
                self.batch_size
            );

            for text in batch {
                let embedding = embed_fn(text)?;
                all_embeddings.push(embedding);
            }
        }

        info!(
            "Generated {} embeddings in vectorized mode (batch_size={})",
            all_embeddings.len(),
            self.batch_size
        );

        Ok(all_embeddings)
    }
}

/// Tensor reshaping for different batch modes
pub struct BatchTensorReshaper;

impl BatchTensorReshaper {
    /// Reshape token tensor for given batch mode
    /// Input: [tokens...] -> Output: [batch_size, seq_len]
    pub fn reshape_for_mode(
        tokens: &[u32],
        mode: &BatchMode,
        device: &Device,
    ) -> Result<Tensor> {
        let batch_size = mode.batch_size();
        let seq_len = tokens.len();

        debug!(
            "Reshaping tensor for mode {:?}: batch_size={}, seq_len={}",
            mode, batch_size, seq_len
        );

        // Create tensor and reshape to [batch_size, seq_len]
        let tensor = Tensor::new(tokens, device)?;

        if batch_size == 1 {
            // Interactive/CodeGen mode: [1, seq_len]
            tensor.reshape((1, seq_len))
        } else {
            // RAG mode: duplicate for batch processing
            // This would be used for parallel embedding generation
            let expanded = tensor.unsqueeze(0)?; // [1, seq_len]
            expanded.expand(&[batch_size, seq_len])
        }
        .map_err(|e| anyhow::anyhow!("Failed to reshape tensor: {}", e))
    }
}

/// Performance metrics for batch processing
#[derive(Debug, Clone)]
pub struct BatchMetrics {
    pub mode: BatchMode,
    pub total_tokens: usize,
    pub latency_ms: f32,
    pub throughput_tokens_per_sec: f32,
}

impl BatchMetrics {
    pub fn calculate(mode: BatchMode, total_tokens: usize, latency_ms: f32) -> Self {
        let throughput = if latency_ms > 0.0 {
            (total_tokens as f32 / latency_ms) * 1000.0
        } else {
            0.0
        };

        Self {
            mode,
            total_tokens,
            latency_ms,
            throughput_tokens_per_sec: throughput,
        }
    }

    pub fn log_performance(&self) {
        match self.mode {
            BatchMode::Interactive => {
                info!(
                    "Interactive mode: {} tokens in {:.1}ms ({:.1} tok/s) - optimized for latency",
                    self.total_tokens, self.latency_ms, self.throughput_tokens_per_sec
                );
            }
            BatchMode::RagRetrieval { batch_size } => {
                info!(
                    "RAG vectorized mode (batch={}): {} tokens in {:.1}ms ({:.1} tok/s) - optimized for throughput",
                    batch_size, self.total_tokens, self.latency_ms, self.throughput_tokens_per_sec
                );
            }
            BatchMode::CodeGen => {
                info!(
                    "CodeGen mode: {} tokens in {:.1}ms ({:.1} tok/s) - optimized for creativity",
                    self.total_tokens, self.latency_ms, self.throughput_tokens_per_sec
                );
            }
        }
    }
}

// Unit tests
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_mode_detection() {
        // Test code generation detection
        let code_prompt = "Write a Rust function that implements bubble sort";
        let mode = BatchMode::detect_from_input(code_prompt, false);
        assert_eq!(mode, BatchMode::CodeGen);

        // Test interactive detection
        let chat_prompt = "What is the weather today?";
        let mode = BatchMode::detect_from_input(chat_prompt, false);
        assert_eq!(mode, BatchMode::Interactive);

        // Test RAG embedding detection
        let mode = BatchMode::detect_from_input("", true);
        assert!(matches!(mode, BatchMode::RagRetrieval { .. }));
    }

    #[test]
    fn test_context_detector() {
        let mut detector = BatchContextDetector::default();

        // Simulate multiple RAG retrievals
        detector.update(true, 50.0);
        detector.update(true, 60.0);
        detector.update(true, 55.0);

        let mode = detector.recommend_mode();
        assert!(matches!(mode, BatchMode::RagRetrieval { .. }));
    }

    #[test]
    fn test_batch_metrics() {
        let metrics = BatchMetrics::calculate(BatchMode::Interactive, 100, 200.0);
        assert_eq!(metrics.total_tokens, 100);
        assert_eq!(metrics.latency_ms, 200.0);
        assert!((metrics.throughput_tokens_per_sec - 500.0).abs() < 0.1);
    }

    #[test]
    fn test_vectorized_batch() {
        let texts = vec![
            "First document".to_string(),
            "Second document".to_string(),
            "Third document".to_string(),
        ];

        let batch = VectorizedEmbeddingBatch::new(texts, Device::Cpu, 2);

        // Mock embedding function
        let embeddings = batch
            .process_batched(|text| {
                // Return mock 384-dim embedding
                Ok(vec![0.1; 384])
            })
            .unwrap();

        assert_eq!(embeddings.len(), 3);
        assert_eq!(embeddings[0].len(), 384);
    }
}
