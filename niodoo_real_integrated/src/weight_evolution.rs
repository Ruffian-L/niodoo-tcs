//! Production-optimized weight evolution for WeightedEpisodicMem
//!
//! Implements non-blocking, throttled weight optimization with:
//! - Hill-climbing fast path (80% of updates)
//! - Mini-GA exploration (20% of updates)
//! - Async discovery buffer with smart batching
//! - Thread-safe weight updates
//! - Comprehensive metrics tracking

use crate::weighted_episodic_mem::DEFAULT_FITNESS_WEIGHTS;
use parking_lot::RwLock;
use rand::Rng;
use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::Mutex as AsyncMutex;

/// Discovery event from MCTS exploration
#[derive(Debug, Clone)]
pub struct Discovery {
    /// Value score of the discovery (0.0-1.0)
    pub value: f32,
    /// Diversity score (0.0-1.0)
    pub diversity: f32,
    /// Source entropy (Shannon entropy of source memory neighborhood)
    pub source_entropy: f32,
    /// Timestamp of discovery
    pub timestamp: std::time::Instant,
}

/// Weight evolution statistics for monitoring
#[derive(Debug, Clone)]
pub struct EvolutionStats {
    /// Current fitness score from recent discoveries
    pub current_score: f32,
    /// Best score achieved so far
    pub best_score: f32,
    /// Number of updates performed
    pub updates_count: usize,
    /// Whether evolution is currently processing
    pub pending: bool,
    /// Current buffer size
    pub buffer_size: usize,
    /// Current strategy being used ("hill_climb" or "ga")
    pub strategy: String,
}

/// Production-optimized weight evolution system
///
/// Uses hybrid strategy:
/// - Hill-climbing: Fast gradient-based optimization (80% of updates)
/// - Mini-GA: Lightweight genetic algorithm for exploration (20% of updates)
pub struct SmoothWeightEvolution {
    /// Discovery buffer (maxlen=100)
    discovery_buffer: Arc<AsyncMutex<VecDeque<Discovery>>>,
    /// Minimum discoveries needed to trigger update (default: 10)
    min_discoveries_for_update: usize,
    /// Updates since last GA run
    updates_since_ga: Arc<RwLock<usize>>,
    /// Frequency of GA runs (every N hill-climbing updates)
    ga_frequency: usize,
    
    /// Current weights [temporal, pad, beta1, retrieval, consonance]
    current_weights: Arc<RwLock<[f32; 5]>>,
    /// Performance history (maxlen=50)
    weight_performance_history: Arc<RwLock<VecDeque<f32>>>,
    /// Best weights found so far
    best_weights: Arc<RwLock<[f32; 5]>>,
    /// Best score achieved
    best_score: Arc<RwLock<f32>>,
    
    /// Hill-climbing parameters
    step_size: f32,
    momentum: f32,
    velocity: Arc<RwLock<[f32; 5]>>,
    
    /// Mini-GA population (lazy init, population of 8)
    mini_population: Arc<RwLock<Option<Vec<[f32; 5]>>>>,
    
    /// Evolution lock to prevent concurrent updates
    evolution_lock: Arc<AsyncMutex<()>>,
}

impl SmoothWeightEvolution {
    /// Create new weight evolution system with default weights
    pub fn new() -> Self {
        Self {
            discovery_buffer: Arc::new(AsyncMutex::new(VecDeque::with_capacity(100))),
            min_discoveries_for_update: 10,
            updates_since_ga: Arc::new(RwLock::new(0)),
            ga_frequency: 5, // Run GA every 5 hill-climbing updates
            
            current_weights: Arc::new(RwLock::new(DEFAULT_FITNESS_WEIGHTS)),
            weight_performance_history: Arc::new(RwLock::new(VecDeque::with_capacity(50))),
            best_weights: Arc::new(RwLock::new(DEFAULT_FITNESS_WEIGHTS)),
            best_score: Arc::new(RwLock::new(0.0)),
            
            step_size: 0.02,
            momentum: 0.9,
            velocity: Arc::new(RwLock::new([0.0; 5])),
            
            mini_population: Arc::new(RwLock::new(None)),
            
            evolution_lock: Arc::new(AsyncMutex::new(())),
        }
    }

    /// Register a discovery event (non-blocking)
    pub async fn register_discovery(&self, discovery: Discovery) {
        let mut buffer = self.discovery_buffer.lock().await;
        
        // Add discovery to buffer
        if buffer.len() >= 100 {
            buffer.pop_front(); // Remove oldest if full
        }
        buffer.push_back(discovery);
        
        // Check if we should trigger evolution
        if buffer.len() >= self.min_discoveries_for_update {
            drop(buffer); // Release lock before async evolution
            
            // Fire and forget - don't await
            let evolution = self.clone();
            tokio::spawn(async move {
                evolution._async_evolve_weights().await;
            });
        }
    }

    /// Async weight evolution (internal, called from background task)
    async fn _async_evolve_weights(&self) {
        // Acquire lock to prevent concurrent evolution
        let _lock = self.evolution_lock.lock().await;
        
        // Snapshot discoveries for processing
        let mut buffer = self.discovery_buffer.lock().await;
        if buffer.len() < self.min_discoveries_for_update {
            return; // Not enough discoveries yet
        }
        
        let discoveries_snapshot: Vec<Discovery> = buffer.drain(..).collect();
        drop(buffer);
        
        // Decide evolution strategy
        let updates_since_ga = *self.updates_since_ga.read();
        let new_weights = if updates_since_ga < self.ga_frequency {
            // Fast hill-climbing
            self._hill_climb_step(&discoveries_snapshot).await
        } else {
            // Occasional GA for exploration
            self._mini_ga_evolution(&discoveries_snapshot).await
        };
        
        // Atomic weight update
        {
            *self.current_weights.write() = new_weights;
            
            if updates_since_ga < self.ga_frequency {
                *self.updates_since_ga.write() += 1;
            } else {
                *self.updates_since_ga.write() = 0;
            }
        }
        
        // Log performance
        let score = self._evaluate_discoveries(&discoveries_snapshot);
        {
            let mut history = self.weight_performance_history.write();
            if history.len() >= 50 {
                history.pop_front();
            }
            history.push_back(score);
            
            let mut best_score = self.best_score.write();
            if score > *best_score {
                *best_score = score;
                *self.best_weights.write() = new_weights;
            }
        }
    }

    /// Hill-climbing step with momentum
    async fn _hill_climb_step(&self, discoveries: &[Discovery]) -> [f32; 5] {
        let current = *self.current_weights.read();
        let base_score = self._evaluate_discoveries(discoveries);
        
        // Estimate gradient via finite differences (sample 3-4 dimensions)
        let mut gradient = [0.0; 5];
        let sampled_dims = [0, 1, 2, 3]; // Sample first 4 dimensions
        
        for &dim in &sampled_dims {
            // Perturb weight
            let mut perturbed = current;
            perturbed[dim] += self.step_size;
            
            // Normalize to maintain sum=1 constraint
            perturbed = self._project_to_simplex(perturbed);
            
            // Evaluate perturbation
            let perturbed_score = self._evaluate_weight_config(perturbed, discoveries);
            
            // Gradient estimate
            gradient[dim] = (perturbed_score - base_score) / self.step_size;
        }
        
        // Momentum update
        let mut velocity = *self.velocity.read();
        for i in 0..5 {
            velocity[i] = self.momentum * velocity[i] + (1.0 - self.momentum) * gradient[i];
        }
        *self.velocity.write() = velocity;
        
        // Apply update
        let mut new_weights = current;
        for i in 0..5 {
            new_weights[i] += self.step_size * velocity[i];
        }
        
        // Project to simplex
        self._project_to_simplex(new_weights)
    }

    /// Mini-GA evolution (population of 8)
    async fn _mini_ga_evolution(&self, discoveries: &[Discovery]) -> [f32; 5] {
        // Lazy init mini population
        let mut pop_guard = self.mini_population.write();
        let population = pop_guard.get_or_insert_with(|| {
            let mut pop = vec![*self.current_weights.read(); 8];
            // Include current best
            pop[0] = *self.best_weights.read();
            // Generate random variations
            for i in 1..8 {
                pop[i] = self._random_weight_vector();
            }
            pop
        });
        
        // Evaluate population
        let mut scores: Vec<(usize, f32)> = population
            .iter()
            .enumerate()
            .map(|(idx, w)| (idx, self._evaluate_weight_config(*w, discoveries)))
            .collect();
        
        // Tournament selection (fast)
        let mut new_population = Vec::new();
        let mut rng = rand::thread_rng();
        for _ in 0..4 {
            let idx1 = rng.gen_range(0..8);
            let idx2 = rng.gen_range(0..8);
            let winner_idx = if scores[idx1].1 > scores[idx2].1 { idx1 } else { idx2 };
            new_population.push(population[winner_idx]);
        }
        
        // Crossover and mutation
        let mut offspring = Vec::new();
        for i in 0..2 {
            let child1 = self._crossover(new_population[i * 2], new_population[i * 2 + 1]);
            let child2 = self._crossover(new_population[i * 2 + 1], new_population[i * 2]);
            offspring.push(self._mutate(child1, 0.1));
            offspring.push(self._mutate(child2, 0.1));
        }
        
        // Combine and select best 8
        let mut combined = new_population;
        combined.extend(offspring);
        
        let mut combined_scores: Vec<(usize, f32)> = combined
            .iter()
            .enumerate()
            .map(|(idx, w)| (idx, self._evaluate_weight_config(*w, discoveries)))
            .collect();
        
        combined_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Update population with best 8
        *population = combined_scores[..8]
            .iter()
            .map(|(idx, _)| combined[*idx])
            .collect();
        
        // Return best from population
        combined_scores[0].1; // Use score for logging
        combined[combined_scores[0].0]
    }

    /// Evaluate weight configuration against discoveries
    fn _evaluate_weight_config(&self, weights: [f32; 5], discoveries: &[Discovery]) -> f32 {
        if discoveries.is_empty() {
            return 0.0;
        }
        
        let values: Vec<f32> = discoveries.iter().map(|d| d.value).collect();
        let diversities: Vec<f32> = discoveries.iter().map(|d| d.diversity).collect();
        let entropies: Vec<f32> = discoveries.iter().map(|d| d.source_entropy).collect();
        
        // Score components
        let mean_value = values.iter().sum::<f32>() / values.len() as f32;
        let value_variance = values.iter()
            .map(|v| (v - mean_value).powi(2))
            .sum::<f32>() / values.len() as f32;
        let mean_diversity = diversities.iter().sum::<f32>() / diversities.len() as f32;
        let entropy_gain = entropies.iter().sum::<f32>() / entropies.len() as f32;
        
        // Weighted combination using candidate weights
        weights[0] * mean_value
            + weights[1] * (1.0 / (1.0 + value_variance)) // Reward consistency
            + weights[2] * mean_diversity
            + weights[3] * entropy_gain
            + weights[4] * (discoveries.len() as f32 / self.min_discoveries_for_update as f32)
    }

    /// Evaluate discoveries with current weights
    fn _evaluate_discoveries(&self, discoveries: &[Discovery]) -> f32 {
        let current = *self.current_weights.read();
        self._evaluate_weight_config(current, discoveries)
    }

    /// Project weights to probability simplex (sum to 1, all positive)
    fn _project_to_simplex(&self, weights: [f32; 5]) -> [f32; 5] {
        let mut weights = weights;
        
        // Clip negative values
        for w in &mut weights {
            *w = w.max(0.0);
        }
        
        // Normalize
        let sum: f32 = weights.iter().sum();
        if sum > 0.0 {
            for w in &mut weights {
                *w /= sum;
            }
        } else {
            // Uniform if all zeros
            weights = [0.2; 5];
        }
        
        weights
    }

    /// Generate random weight vector on simplex
    fn _random_weight_vector(&self) -> [f32; 5] {
        let mut rng = rand::thread_rng();
        let mut weights = [
            rng.gen(),
            rng.gen(),
            rng.gen(),
            rng.gen(),
            rng.gen(),
        ];
        self._project_to_simplex(weights)
    }

    /// Crossover two weight vectors
    fn _crossover(&self, parent1: [f32; 5], parent2: [f32; 5]) -> [f32; 5] {
        let mut rng = rand::thread_rng();
        let alpha = rng.gen::<f32>(); // Blend factor
        let mut child = [0.0; 5];
        for i in 0..5 {
            child[i] = alpha * parent1[i] + (1.0 - alpha) * parent2[i];
        }
        self._project_to_simplex(child)
    }

    /// Mutate weight vector
    fn _mutate(&self, weights: [f32; 5], rate: f32) -> [f32; 5] {
        let mut rng = rand::thread_rng();
        let mut mutated = weights;
        for w in &mut mutated {
            if rng.gen::<f32>() < rate {
                *w += (rng.gen::<f32>() - 0.5) * 0.1; // Small perturbation
            }
        }
        self._project_to_simplex(mutated)
    }

    /// Get current weights (thread-safe)
    pub fn get_current_weights(&self) -> [f32; 5] {
        *self.current_weights.read()
    }

    /// Get evolution statistics
    pub async fn get_evolution_stats(&self) -> EvolutionStats {
        let history = self.weight_performance_history.read();
        let updates_since_ga = *self.updates_since_ga.read();
        let buffer = self.discovery_buffer.lock().await;
        
        EvolutionStats {
            current_score: history.back().copied().unwrap_or(0.0),
            best_score: *self.best_score.read(),
            updates_count: history.len(),
            pending: self.evolution_lock.try_lock().is_err(),
            buffer_size: buffer.len(),
            strategy: if updates_since_ga < self.ga_frequency {
                "hill_climb".to_string()
            } else {
                "ga".to_string()
            },
        }
    }
}

impl Clone for SmoothWeightEvolution {
    fn clone(&self) -> Self {
        Self {
            discovery_buffer: Arc::clone(&self.discovery_buffer),
            min_discoveries_for_update: self.min_discoveries_for_update,
            updates_since_ga: Arc::clone(&self.updates_since_ga),
            ga_frequency: self.ga_frequency,
            current_weights: Arc::clone(&self.current_weights),
            weight_performance_history: Arc::clone(&self.weight_performance_history),
            best_weights: Arc::clone(&self.best_weights),
            best_score: Arc::clone(&self.best_score),
            step_size: self.step_size,
            momentum: self.momentum,
            velocity: Arc::clone(&self.velocity),
            mini_population: Arc::clone(&self.mini_population),
            evolution_lock: Arc::clone(&self.evolution_lock),
        }
    }
}

impl Default for SmoothWeightEvolution {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_weight_evolution_creation() {
        let evolution = SmoothWeightEvolution::new();
        let weights = evolution.get_current_weights();
        assert_eq!(weights.len(), 5);
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 0.001); // Should sum to ~1.0
    }

    #[tokio::test]
    async fn test_discovery_registration() {
        let evolution = SmoothWeightEvolution::new();
        
        for _ in 0..5 {
            let discovery = Discovery {
                value: 0.7,
                diversity: 0.5,
                source_entropy: 0.6,
                timestamp: std::time::Instant::now(),
            };
            evolution.register_discovery(discovery).await;
        }
        
        let stats = evolution.get_evolution_stats().await;
        assert_eq!(stats.buffer_size, 5);
    }
}

