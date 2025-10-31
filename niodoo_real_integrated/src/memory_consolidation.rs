//! Memory consolidation for WeightedEpisodicMem
//!
//! Implements neuroscience-inspired memory consolidation:
//! - TD learning for memory valuation
//! - Prioritized replay based on prediction error
//! - Consolidation level tracking
//! - Prediction error guidance for consolidation depth

use crate::erag::EragMemory;
use std::collections::VecDeque;
use rand::Rng;

/// TD (Temporal Difference) learning value estimator
pub struct MemoryValueEstimator {
    /// Discount factor γ
    pub gamma: f32,
    /// Learning rate α
    pub learning_rate: f32,
    /// Value cache (memory_id -> value)
    value_cache: std::collections::HashMap<String, f32>,
}

impl MemoryValueEstimator {
    /// Create new value estimator
    pub fn new(gamma: f32, learning_rate: f32) -> Self {
        Self {
            gamma,
            learning_rate,
            value_cache: std::collections::HashMap::new(),
        }
    }

    /// Compute TD error: δ = R + γV(s') - V(s)
    ///
    /// For episodic memory:
    /// - R = immediate reward (e.g., fitness score improvement)
    /// - V(s') = value of next state/memory
    /// - V(s) = current value
    pub fn compute_td_error(
        &self,
        current_value: f32,
        next_value: f32,
        reward: f32,
    ) -> f32 {
        reward + self.gamma * next_value - current_value
    }

    /// Update value estimate using TD learning
    pub fn update_value(
        &mut self,
        memory_id: &str,
        td_error: f32,
    ) {
        let current_value = self.value_cache.get(memory_id).copied().unwrap_or(0.5);
        let new_value = current_value + self.learning_rate * td_error;
        self.value_cache.insert(memory_id.to_string(), new_value.clamp(0.0, 1.0));
    }

    /// Get value estimate for memory
    pub fn get_value(&self, memory_id: &str) -> f32 {
        self.value_cache.get(memory_id).copied().unwrap_or(0.5)
    }

    /// Calculate TD error for memory sequence
    ///
    /// Processes memories in temporal order and computes TD errors
    pub fn calculate_td_errors(
        &mut self,
        memories: &[(String, f32)], // (memory_id, reward)
    ) -> Vec<f32> {
        let mut td_errors = Vec::new();

        for i in 0..memories.len() {
            let (mem_id, reward) = &memories[i];
            let current_value = self.get_value(mem_id);
            
            // Next value (if available)
            let next_value = if i + 1 < memories.len() {
                let (next_id, _) = &memories[i + 1];
                self.get_value(next_id)
            } else {
                0.0 // Terminal state
            };

            let td_error = self.compute_td_error(current_value, next_value, *reward);
            td_errors.push(td_error);

            // Update value estimate
            self.update_value(mem_id, td_error);
        }

        td_errors
    }
}

/// Prioritized replay sampler
pub struct PrioritizedReplaySampler {
    /// TD error magnitudes (for prioritization)
    td_errors: VecDeque<(String, f32)>, // (memory_id, |td_error|)
    /// Maximum replay buffer size
    max_buffer_size: usize,
    /// Priority exponent α (default 0.6)
    pub priority_exponent: f32,
}

impl PrioritizedReplaySampler {
    /// Create new replay sampler
    pub fn new(max_buffer_size: usize) -> Self {
        Self {
            td_errors: VecDeque::with_capacity(max_buffer_size),
            max_buffer_size,
            priority_exponent: 0.6,
        }
    }

    /// Add memory with TD error magnitude
    pub fn add_memory(&mut self, memory_id: String, td_error_magnitude: f32) {
        if self.td_errors.len() >= self.max_buffer_size {
            self.td_errors.pop_front();
        }
        self.td_errors.push_back((memory_id, td_error_magnitude));
    }

    /// Sample memories proportional to |td_error|^α
    pub fn sample_replay_batch(&self, batch_size: usize) -> Vec<String> {
        if self.td_errors.is_empty() {
            return Vec::new();
        }

        // Calculate priorities
        let priorities: Vec<f32> = self.td_errors
            .iter()
            .map(|(_, error)| error.abs().powf(self.priority_exponent))
            .collect();

        let sum_priorities: f32 = priorities.iter().sum();
        if sum_priorities == 0.0 {
            // Uniform sampling if all priorities are zero
            return self.td_errors
                .iter()
                .take(batch_size)
                .map(|(id, _)| id.clone())
                .collect();
        }

        // Sample proportional to priorities
        let mut rng = rand::thread_rng();
        let mut sampled = Vec::new();

        for _ in 0..batch_size.min(self.td_errors.len()) {
            let random = rng.gen::<f32>() * sum_priorities;
            let mut cumulative = 0.0;
            
            for (idx, &priority) in priorities.iter().enumerate() {
                cumulative += priority;
                if cumulative >= random {
                    sampled.push(self.td_errors[idx].0.clone());
                    break;
                }
            }
        }

        sampled
    }

    /// Get memories with highest TD error magnitudes
    pub fn get_high_priority_memories(&self, top_k: usize) -> Vec<String> {
        let mut sorted: Vec<_> = self.td_errors.iter().collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        sorted.iter()
            .take(top_k)
            .map(|(id, _)| id.clone())
            .collect()
    }
}

/// Prediction error calculator
pub struct PredictionErrorCalculator {
    /// Error accumulation window
    error_window: VecDeque<f32>,
    /// Window size
    window_size: usize,
}

impl PredictionErrorCalculator {
    /// Create new calculator
    pub fn new(window_size: usize) -> Self {
        Self {
            error_window: VecDeque::with_capacity(window_size),
            window_size,
        }
    }

    /// Calculate prediction error
    ///
    /// Error = |predicted_value - actual_value|
    pub fn calculate_error(&self, predicted: f32, actual: f32) -> f32 {
        (predicted - actual).abs()
    }

    /// Add error to window and return average
    pub fn add_error(&mut self, error: f32) -> f32 {
        if self.error_window.len() >= self.window_size {
            self.error_window.pop_front();
        }
        self.error_window.push_back(error);

        // Return average error
        if self.error_window.is_empty() {
            0.0
        } else {
            self.error_window.iter().sum::<f32>() / self.error_window.len() as f32
        }
    }

    /// Get current average prediction error
    pub fn average_error(&self) -> f32 {
        if self.error_window.is_empty() {
            0.0
        } else {
            self.error_window.iter().sum::<f32>() / self.error_window.len() as f32
        }
    }
}

/// Memory consolidation manager
pub struct MemoryConsolidationManager {
    /// Value estimator
    pub value_estimator: MemoryValueEstimator,
    /// Prioritized replay sampler
    pub replay_sampler: PrioritizedReplaySampler,
    /// Prediction error calculator
    pub error_calculator: PredictionErrorCalculator,
    /// Consolidation threshold (high error = detailed storage)
    pub high_error_threshold: f32,
}

impl MemoryConsolidationManager {
    /// Create new consolidation manager
    pub fn new() -> Self {
        Self {
            value_estimator: MemoryValueEstimator::new(0.9, 0.1), // γ=0.9, α=0.1
            replay_sampler: PrioritizedReplaySampler::new(10000),
            error_calculator: PredictionErrorCalculator::new(100),
            high_error_threshold: 0.3,
        }
    }

    /// Determine consolidation level based on prediction error
    ///
    /// High error → detailed storage (consolidation_level = 0.0)
    /// Low error → schema reconstruction (consolidation_level = 1.0)
    pub fn determine_consolidation_level(&self, prediction_error: f32) -> f32 {
        if prediction_error >= self.high_error_threshold {
            // High error: store with full detail
            0.0
        } else {
            // Low error: can be reconstructed from schema
            // Linear interpolation: error 0.0 → consolidation 1.0, error threshold → 0.0
            1.0 - (prediction_error / self.high_error_threshold)
        }
    }

    /// Process memory for consolidation
    pub fn process_memory(
        &mut self,
        memory_id: &str,
        memory: &EragMemory,
        predicted_fitness: f32,
        actual_fitness: f32,
    ) -> f32 {
        // Calculate prediction error
        let prediction_error = self.error_calculator.calculate_error(predicted_fitness, actual_fitness);
        let avg_error = self.error_calculator.add_error(prediction_error);

        // Determine consolidation level
        let consolidation_level = self.determine_consolidation_level(prediction_error);

        // Calculate TD error (using fitness as reward)
        let current_value = self.value_estimator.get_value(memory_id);
        let td_error = self.value_estimator.compute_td_error(
            current_value,
            current_value, // No next state for single memory
            actual_fitness,
        );

        // Update value estimate
        self.value_estimator.update_value(memory_id, td_error);

        // Add to replay buffer
        self.replay_sampler.add_memory(memory_id.to_string(), td_error.abs());

        consolidation_level
    }

    /// Sample memories for consolidation replay
    pub fn sample_for_replay(&self, batch_size: usize) -> Vec<String> {
        self.replay_sampler.sample_replay_batch(batch_size)
    }

    /// Get high-priority memories for consolidation
    pub fn get_high_priority_memories(&self, top_k: usize) -> Vec<String> {
        self.replay_sampler.get_high_priority_memories(top_k)
    }
}

impl Default for MemoryConsolidationManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_td_error_calculation() {
        let estimator = MemoryValueEstimator::new(0.9, 0.1);
        let error = estimator.compute_td_error(0.5, 0.7, 0.3);
        // δ = 0.3 + 0.9*0.7 - 0.5 = 0.3 + 0.63 - 0.5 = 0.43
        assert!(error > 0.4 && error < 0.45);
    }

    #[test]
    fn test_consolidation_level() {
        let manager = MemoryConsolidationManager::new();
        let high_error = manager.determine_consolidation_level(0.5);
        let low_error = manager.determine_consolidation_level(0.1);
        assert!(high_error < low_error); // High error = lower consolidation
    }

    #[test]
    fn test_prioritized_sampling() {
        let mut sampler = PrioritizedReplaySampler::new(100);
        sampler.add_memory("mem1".to_string(), 0.8);
        sampler.add_memory("mem2".to_string(), 0.2);
        
        let batch = sampler.sample_replay_batch(1);
        assert_eq!(batch.len(), 1);
    }
}

