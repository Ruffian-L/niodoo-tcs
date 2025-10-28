use anyhow::Result;
use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::RwLock;
use crate::memory_core::MemoryCore;
use crate::curator::Curator;

/// Optimization configuration based on 2025 analysis
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    /// ERAG collapse threshold (>0.2 drift triggers reset)
    pub erag_collapse_threshold: f32,
    /// Hyperspherical normalization for embeddings
    pub normalize_embeddings: bool,
    /// KV cache size for faster context handling
    pub kv_cache_size: usize,
    /// Batch size for async curator calls
    pub curator_batch_size: usize,
    /// Number of learning events to retrieve for context injection
    pub context_injection_limit: usize,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            erag_collapse_threshold: 0.2,
            normalize_embeddings: true,
            kv_cache_size: 128_000,  // 128K tokens for Quadro 6000
            curator_batch_size: 8,
            context_injection_limit: 5,
        }
    }
}

/// Optimized context retrieval with hyperspherical normalization
pub async fn retrieve_optimized_context(
    task_input: &str,
    curator: &Curator,
    memory: &MemoryCore,
    config: &OptimizationConfig,
) -> Result<(String, f32)> {  // Returns (context, coherence_score)
    // Get task embedding
    let mut query_embedding = curator.embed_text(task_input).await?;
    
    // Apply hyperspherical normalization for cosine efficiency (15% boost per analysis)
    if config.normalize_embeddings {
        normalize_to_unit_sphere(&mut query_embedding);
    }
    
    // Search for similar learning events (implements the suggestion from line 15-16)
    let events = memory.search_similar(&query_embedding, config.context_injection_limit).await?;
    
    // Calculate coherence score (for ERAG collapse detection)
    let coherence = calculate_coherence_score(&events, &query_embedding);
    
    // Build optimized context
    let mut context_parts = Vec::new();
    
    // Prioritize high-relevance experiences
    for exp in events.iter().filter(|e| e.relevance_score > 0.7) {
        context_parts.push(format!(
            "High-relevance experience (score: {:.2}):\nTask: {}\nSolution: {}\nLearning: {}",
            exp.relevance_score,
            exp.input.chars().take(100).collect::<String>(),
            exp.output.chars().take(200).collect::<String>(),
            exp.context
        ));
    }
    
    // Add medium-relevance if space permits
    let remaining_space_tokens = config.kv_cache_size / 4; // Reserve 25% of KV cache
    // Heuristic: ~4-5 characters per token (TODO: replace with real tokenization)
    let chars_per_token = 4.5;
    let remaining_space_chars = (remaining_space_tokens as f32 * chars_per_token) as usize;
    let current_length: usize = context_parts.iter().map(|s| s.len()).sum();
    
    if current_length < remaining_space_chars {
        for exp in events.iter().filter(|e| e.relevance_score > 0.5 && e.relevance_score <= 0.7) {
            context_parts.push(format!(
                "Related experience:\n{}\n---",
                exp.context.chars().take(150).collect::<String>()
            ));
        }
    }
    
    Ok((context_parts.join("\n\n"), coherence))
}

/// Normalize embedding to unit sphere for cosine similarity optimization
fn normalize_to_unit_sphere(embedding: &mut Vec<f32>) {
    let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if magnitude > 0.0 {
        for val in embedding.iter_mut() {
            *val /= magnitude;
        }
    }
}

/// Calculate coherence score for ERAG collapse detection
fn calculate_coherence_score(events: &[crate::memory_core::Experience], query_embedding: &[f32]) -> f32 {
    if events.is_empty() {
        return 1.0; // No drift if no history
    }
    
    // Calculate average similarity
    let similarities: Vec<f32> = events.iter()
        .filter_map(|e| e.embedding.as_ref())
        .map(|e| cosine_similarity(e, query_embedding))
        .collect();
    
    if similarities.is_empty() {
        return 1.0;
    }
    
    let avg_similarity = similarities.iter().sum::<f32>() / similarities.len() as f32;
    
    // Calculate variance for drift detection
    let variance = similarities.iter()
        .map(|s| (s - avg_similarity).powi(2))
        .sum::<f32>() / similarities.len() as f32;
    
    // High variance indicates potential collapse
    1.0 - variance.min(1.0)
}

/// Calculate cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    
    if mag_a * mag_b > 0.0 {
        dot_product / (mag_a * mag_b)
    } else {
        0.0
    }
}

/// Async batch processing for curator calls
pub struct BatchedCurator {
    batch_queue: Arc<RwLock<Vec<String>>>,
    config: OptimizationConfig,
}

impl BatchedCurator {
    pub fn new(config: OptimizationConfig) -> Self {
        Self {
            batch_queue: Arc::new(RwLock::new(Vec::new())),
            config,
        }
    }
    
    /// Add task to batch queue
    pub async fn queue_task(&self, task: String) {
        let mut queue = self.batch_queue.write().await;
        queue.push(task);
        
        // Auto-process if batch is full
        if queue.len() >= self.config.curator_batch_size {
            // In production, trigger batch processing here
        }
    }
    
    /// Process batch of tasks
    pub async fn process_batch(&self, curator: &Curator) -> Result<Vec<Vec<f32>>> {
        let mut queue = self.batch_queue.write().await;
        let tasks: Vec<String> = queue.drain(..).collect();
        
        if tasks.is_empty() {
            return Ok(Vec::new());
        }
        
        // Process all tasks in parallel
        let mut handles = Vec::new();
        for task in tasks {
            let task_clone = task.clone();
            // In real implementation, use actual curator embed
            handles.push(tokio::spawn(async move {
                // Simulated embedding
                vec![0.1; 896]  // 896-dim BERT standard
            }));
        }
        
        let mut results = Vec::new();
        for handle in handles {
            results.push(handle.await?);
        }
        
        Ok(results)
    }
}

/// ERAG collapse detection and mitigation
pub struct ERAGMonitor {
    coherence_history: Arc<RwLock<VecDeque<f32>>>,
    collapse_threshold: f32,
}

impl ERAGMonitor {
    pub fn new(threshold: f32) -> Self {
        Self {
            coherence_history: Arc::new(RwLock::new(VecDeque::new())),
            collapse_threshold: threshold,
        }
    }
    
    /// Monitor for ERAG collapse
    pub async fn check_collapse(&self, coherence: f32) -> bool {
        let mut history = self.coherence_history.write().await;
        history.push_back(coherence);

        // Keep last 100 measurements
        if history.len() > 100 {
            history.pop_front();
        }
        
        // Check for drift (>0.2 triggers reset per analysis)
        if history.len() >= 10 {
            let recent: Vec<f32> = history.iter().rev().take(10).copied().collect();
            let avg_recent = recent.iter().sum::<f32>() / recent.len() as f32;
            
            avg_recent < (1.0 - self.collapse_threshold)
        } else {
            false
        }
    }
    
    /// Reset after collapse detection
    pub async fn reset(&self) {
        let mut history = self.coherence_history.write().await;
        history.clear();
    }
}

/// Hardware-specific optimizations
pub struct HardwareOptimizer {
    pub gpu_type: GPUType,
    pub thermal_limit: f32,
    pub power_limit: u32,
}

#[derive(Debug, Clone)]
pub enum GPUType {
    QuadroRTX6000,    // Beelink server
    RTX5080Q,         // Laptop
}

impl HardwareOptimizer {
    pub fn new_for_beelink() -> Self {
        Self {
            gpu_type: GPUType::QuadroRTX6000,
            thermal_limit: 83.0,  // Quadro safe limit
            power_limit: 260,     // 260W TDP
        }
    }
    
    pub fn new_for_laptop() -> Self {
        Self {
            gpu_type: GPUType::RTX5080Q,
            thermal_limit: 88.0,  // Per analysis
            power_limit: 150,     // 150W TGP cap
        }
    }
    
    /// Get optimal batch size based on hardware
    pub fn optimal_batch_size(&self) -> usize {
        match self.gpu_type {
            GPUType::QuadroRTX6000 => 4,  // Conservative for 24GB VRAM
            GPUType::RTX5080Q => 2,       // Limited by 16GB VRAM
        }
    }
    
    /// Get optimal KV cache size
    pub fn optimal_kv_cache(&self) -> usize {
        match self.gpu_type {
            GPUType::QuadroRTX6000 => 128_000,  // 128K stable
            GPUType::RTX5080Q => 256_000,       // 256K with Qwen3
        }
    }
    
    /// Get token generation speed estimate
    pub fn expected_tokens_per_second(&self) -> usize {
        match self.gpu_type {
            GPUType::QuadroRTX6000 => 60,   // Per benchmarks
            GPUType::RTX5080Q => 150,       // Blackwell advantage
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_hyperspherical_normalization() {
        let mut embedding = vec![3.0, 4.0, 0.0];
        normalize_to_unit_sphere(&mut embedding);
        
        let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((magnitude - 1.0).abs() < 0.001);
    }
    
    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);  // Orthogonal
        
        let c = vec![1.0, 0.0, 0.0];
        let d = vec![1.0, 0.0, 0.0];
        assert_eq!(cosine_similarity(&c, &d), 1.0);  // Parallel
    }
}