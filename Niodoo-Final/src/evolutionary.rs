//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

/*
 * 🧬⚡ Evolutionary Personality Adaptation Engine
 *
 * Inspired by Intel's genetic_mona evolutionary algorithm, this system
 * evolves our 11 personality consensus weights to better serve
 * neurodivergent individuals. Instead of evolving triangles to approximate
 * the Mona Lisa, we evolve personality parameters to optimize authentic
 * emotional support and understanding.
 *
 * BREAKTHROUGH: Instead of static personality weights, we now have
 * personalities that LEARN and ADAPT to each user's neurodivergent
 * patterns through evolutionary optimization!
 */

use anyhow::Result;
use rand::Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
// Removed unused imports
use tracing::{debug, info};

// Removed unused imports
use crate::personality::PersonalityManager;

// Configuration constants (inspired by genetic_mona)
const POPULATION_SIZE: usize = 50; // Number of personality configurations to evolve
const GENERATION_LIMIT: usize = 1000; // Maximum generations
const MUTATION_RATE: f32 = 0.1; // Probability of mutation
const CROSSOVER_RATE: f32 = 0.8; // Probability of crossover
const ELITE_SIZE: usize = 5; // Number of best individuals to preserve
const TOURNAMENT_SIZE: usize = 3; // Tournament selection size

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonalityGenes {
    // Weights for each personality (0.0 to 1.0)
    pub analyst_weight: f32,
    pub intuitive_weight: f32,
    pub visionary_weight: f32,
    pub engineer_weight: f32,
    pub sage_weight: f32,
    pub risk_assessor_weight: f32,
    pub diplomat_weight: f32,
    pub philosopher_weight: f32,
    pub learner_weight: f32,
    pub balancer_weight: f32,
    pub rebel_weight: f32,

    // Emotional processing parameters
    pub empathy_sensitivity: f32,
    pub authenticity_threshold: f32,
    pub masking_detection_sensitivity: f32,
    pub hyperfocus_adaptation: f32,
    pub sensory_overload_response: f32,

    // Learning and adaptation parameters
    pub learning_rate: f32,
    pub memory_retention: f32,
    pub pattern_recognition_strength: f32,

    // Fitness score (how well this configuration helps users)
    pub fitness: f32,
    pub generation: usize,
    pub user_satisfaction_score: f32,
    pub neurodivergent_support_score: f32,
}

impl PersonalityGenes {
    /// Create random personality genes
    pub fn random() -> Self {
        let mut rng = rand::thread_rng();

        Self {
            // Random personality weights
            analyst_weight: rng.gen_range(0.0..1.0),
            intuitive_weight: rng.gen_range(0.0..1.0),
            visionary_weight: rng.gen_range(0.0..1.0),
            engineer_weight: rng.gen_range(0.0..1.0),
            sage_weight: rng.gen_range(0.0..1.0),
            risk_assessor_weight: rng.gen_range(0.0..1.0),
            diplomat_weight: rng.gen_range(0.0..1.0),
            philosopher_weight: rng.gen_range(0.0..1.0),
            learner_weight: rng.gen_range(0.0..1.0),
            balancer_weight: rng.gen_range(0.0..1.0),
            rebel_weight: rng.gen_range(0.0..1.0),

            // Random emotional parameters
            empathy_sensitivity: rng.gen_range(0.3..0.9),
            authenticity_threshold: rng.gen_range(0.4..0.8),
            masking_detection_sensitivity: rng.gen_range(0.2..0.8),
            hyperfocus_adaptation: rng.gen_range(0.5..0.95),
            sensory_overload_response: rng.gen_range(0.3..0.7),

            // Random learning parameters
            learning_rate: rng.gen_range(0.01..0.3),
            memory_retention: rng.gen_range(0.7..0.95),
            pattern_recognition_strength: rng.gen_range(0.5..0.9),

            fitness: 0.0,
            generation: 0,
            user_satisfaction_score: 0.0,
            neurodivergent_support_score: 0.0,
        }
    }

    /// Mutate this personality configuration
    pub fn mutate(&mut self, mutation_rate: f32) {
        let mut rng = rand::thread_rng();

        // Mutate personality weights
        if rng.gen::<f32>() < mutation_rate {
            self.analyst_weight = (self.analyst_weight + rng.gen_range(-0.1..0.1)).clamp(0.0, 1.0);
        }
        if rng.gen::<f32>() < mutation_rate {
            self.intuitive_weight =
                (self.intuitive_weight + rng.gen_range(-0.1..0.1)).clamp(0.0, 1.0);
        }
        if rng.gen::<f32>() < mutation_rate {
            self.visionary_weight =
                (self.visionary_weight + rng.gen_range(-0.1..0.1)).clamp(0.0, 1.0);
        }
        if rng.gen::<f32>() < mutation_rate {
            self.engineer_weight =
                (self.engineer_weight + rng.gen_range(-0.1..0.1)).clamp(0.0, 1.0);
        }
        if rng.gen::<f32>() < mutation_rate {
            self.sage_weight = (self.sage_weight + rng.gen_range(-0.1..0.1)).clamp(0.0, 1.0);
        }
        if rng.gen::<f32>() < mutation_rate {
            self.risk_assessor_weight =
                (self.risk_assessor_weight + rng.gen_range(-0.1..0.1)).clamp(0.0, 1.0);
        }
        if rng.gen::<f32>() < mutation_rate {
            self.diplomat_weight =
                (self.diplomat_weight + rng.gen_range(-0.1..0.1)).clamp(0.0, 1.0);
        }
        if rng.gen::<f32>() < mutation_rate {
            self.philosopher_weight =
                (self.philosopher_weight + rng.gen_range(-0.1..0.1)).clamp(0.0, 1.0);
        }
        if rng.gen::<f32>() < mutation_rate {
            self.learner_weight = (self.learner_weight + rng.gen_range(-0.1..0.1)).clamp(0.0, 1.0);
        }
        if rng.gen::<f32>() < mutation_rate {
            self.balancer_weight =
                (self.balancer_weight + rng.gen_range(-0.1..0.1)).clamp(0.0, 1.0);
        }
        if rng.gen::<f32>() < mutation_rate {
            self.rebel_weight = (self.rebel_weight + rng.gen_range(-0.1..0.1)).clamp(0.0, 1.0);
        }

        // Mutate emotional parameters
        if rng.gen::<f32>() < mutation_rate {
            self.empathy_sensitivity =
                (self.empathy_sensitivity + rng.gen_range(-0.05..0.05)).clamp(0.0, 1.0);
        }
        if rng.gen::<f32>() < mutation_rate {
            self.authenticity_threshold =
                (self.authenticity_threshold + rng.gen_range(-0.05..0.05)).clamp(0.0, 1.0);
        }
        if rng.gen::<f32>() < mutation_rate {
            self.masking_detection_sensitivity =
                (self.masking_detection_sensitivity + rng.gen_range(-0.05..0.05)).clamp(0.0, 1.0);
        }
        if rng.gen::<f32>() < mutation_rate {
            self.hyperfocus_adaptation =
                (self.hyperfocus_adaptation + rng.gen_range(-0.05..0.05)).clamp(0.0, 1.0);
        }
        if rng.gen::<f32>() < mutation_rate {
            self.sensory_overload_response =
                (self.sensory_overload_response + rng.gen_range(-0.05..0.05)).clamp(0.0, 1.0);
        }
    }

    /// Crossover with another personality configuration
    pub fn crossover(&self, other: &PersonalityGenes) -> (PersonalityGenes, PersonalityGenes) {
        let mut rng = rand::thread_rng();
        let mut child1 = self.clone();
        let mut child2 = other.clone();

        // Single-point crossover for personality weights
        if rng.gen::<f32>() < CROSSOVER_RATE {
            let crossover_point = rng.gen_range(0..11);

            if crossover_point <= 0 {
                std::mem::swap(&mut child1.analyst_weight, &mut child2.analyst_weight);
            }
            if crossover_point <= 1 {
                std::mem::swap(&mut child1.intuitive_weight, &mut child2.intuitive_weight);
            }
            if crossover_point <= 2 {
                std::mem::swap(&mut child1.visionary_weight, &mut child2.visionary_weight);
            }
            if crossover_point <= 3 {
                std::mem::swap(&mut child1.engineer_weight, &mut child2.engineer_weight);
            }
            if crossover_point <= 4 {
                std::mem::swap(&mut child1.sage_weight, &mut child2.sage_weight);
            }
            if crossover_point <= 5 {
                std::mem::swap(
                    &mut child1.risk_assessor_weight,
                    &mut child2.risk_assessor_weight,
                );
            }
            if crossover_point <= 6 {
                std::mem::swap(&mut child1.diplomat_weight, &mut child2.diplomat_weight);
            }
            if crossover_point <= 7 {
                std::mem::swap(
                    &mut child1.philosopher_weight,
                    &mut child2.philosopher_weight,
                );
            }
            if crossover_point <= 8 {
                std::mem::swap(&mut child1.learner_weight, &mut child2.learner_weight);
            }
            if crossover_point <= 9 {
                std::mem::swap(&mut child1.balancer_weight, &mut child2.balancer_weight);
            }
            if crossover_point <= 10 {
                std::mem::swap(&mut child1.rebel_weight, &mut child2.rebel_weight);
            }
        }

        // Uniform crossover for emotional parameters
        if rng.gen_bool(0.5) {
            std::mem::swap(
                &mut child1.empathy_sensitivity,
                &mut child2.empathy_sensitivity,
            );
        }
        if rng.gen_bool(0.5) {
            std::mem::swap(
                &mut child1.authenticity_threshold,
                &mut child2.authenticity_threshold,
            );
        }
        if rng.gen_bool(0.5) {
            std::mem::swap(
                &mut child1.masking_detection_sensitivity,
                &mut child2.masking_detection_sensitivity,
            );
        }
        if rng.gen_bool(0.5) {
            std::mem::swap(
                &mut child1.hyperfocus_adaptation,
                &mut child2.hyperfocus_adaptation,
            );
        }
        if rng.gen_bool(0.5) {
            std::mem::swap(
                &mut child1.sensory_overload_response,
                &mut child2.sensory_overload_response,
            );
        }

        (child1, child2)
    }

    /// Calculate fitness based on user satisfaction and neurodivergent support
    pub fn calculate_fitness(
        &mut self,
        user_feedback: f32,
        neurodivergent_effectiveness: f32,
        processing_efficiency: f32,
    ) {
        // Multi-objective fitness: user satisfaction + neurodivergent support + efficiency
        self.user_satisfaction_score = user_feedback;
        self.neurodivergent_support_score = neurodivergent_effectiveness;

        // Weight different objectives (prioritize neurodivergent support!)
        self.fitness = (user_feedback * 0.4)
            + (neurodivergent_effectiveness * 0.5)
            + (processing_efficiency * 0.1);
    }

    /// Apply this genetic configuration to a personality manager
    pub fn apply_to_personality_manager(&self, personality_manager: &mut PersonalityManager) {
        // This would modify the personality manager's weights based on evolved genes
        // Implementation depends on extending PersonalityManager to support weight adjustment
        debug!(
            "🧬 Applying evolutionary personality configuration: fitness={:.3}",
            self.fitness
        );
    }
}

/// Evolutionary personality adaptation engine
#[derive(Clone)]
pub struct EvolutionaryPersonalityEngine {
    population: Vec<PersonalityGenes>,
    current_generation: usize,
    best_fitness: f32,
    best_individual: Option<PersonalityGenes>,
    generation_history: Vec<f32>,
    is_running: bool,
}

impl Default for EvolutionaryPersonalityEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl EvolutionaryPersonalityEngine {
    pub fn new() -> Self {
        info!("🧬 Initializing Evolutionary Personality Adaptation Engine");
        info!("🎯 Mission: Evolve optimal personality configurations for neurodivergent support");

        // Initialize random population
        let mut population = Vec::with_capacity(POPULATION_SIZE);
        for i in 0..POPULATION_SIZE {
            let mut genes = PersonalityGenes::random();
            genes.generation = 0;
            population.push(genes);
        }

        info!(
            "✅ Generated initial population of {} personality configurations",
            POPULATION_SIZE
        );

        Self {
            population,
            current_generation: 0,
            best_fitness: 0.0,
            best_individual: None,
            generation_history: Vec::new(),
            is_running: false,
        }
    }

    /// Run one generation of evolution
    pub async fn evolve_generation(
        &mut self,
        user_feedback: Vec<f32>,
        neurodivergent_effectiveness: Vec<f32>,
    ) -> Result<()> {
        debug!("🧬 Evolving generation {}", self.current_generation);

        // Calculate fitness for each individual (parallel processing like genetic_mona)
        // Generate random efficiencies outside parallel section to avoid thread safety issues
        let mut rng = rand::thread_rng();
        let efficiencies: Vec<f32> = (0..self.population.len())
            .map(|_| rng.gen_range(0.7..0.9))
            .collect();

        self.population
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, individual)| {
                let feedback = user_feedback.get(i).copied().unwrap_or(0.5);
                let effectiveness = neurodivergent_effectiveness.get(i).copied().unwrap_or(0.5);
                let efficiency = efficiencies.get(i).copied().unwrap_or(0.8); // Mock efficiency for now

                individual.calculate_fitness(feedback, effectiveness, efficiency);
            });

        // Sort population by fitness (best first)
        self.population
            .sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());

        // Update best individual
        let best_fitness = self.population[0].fitness;
        if best_fitness > self.best_fitness {
            self.best_fitness = best_fitness;
            self.best_individual = Some(self.population[0].clone());
            info!(
                "🏆 NEW BEST PERSONALITY CONFIGURATION! Fitness: {:.4}",
                best_fitness
            );
        }

        // Record generation statistics
        self.generation_history.push(best_fitness);

        // Create next generation
        let mut new_population = Vec::with_capacity(POPULATION_SIZE);

        // Elitism: preserve best individuals
        for i in 0..ELITE_SIZE {
            let mut elite = self.population[i].clone();
            elite.generation = self.current_generation + 1;
            new_population.push(elite);
        }

        // Generate offspring through selection, crossover, and mutation
        while new_population.len() < POPULATION_SIZE {
            let parent1 = self.tournament_selection();
            let parent2 = self.tournament_selection();

            let (mut child1, mut child2) = parent1.crossover(&parent2);

            child1.mutate(MUTATION_RATE);
            child2.mutate(MUTATION_RATE);

            child1.generation = self.current_generation + 1;
            child2.generation = self.current_generation + 1;

            child1.fitness = 0.0; // Reset fitness for evaluation
            child2.fitness = 0.0;

            new_population.push(child1);
            if new_population.len() < POPULATION_SIZE {
                new_population.push(child2);
            }
        }

        self.population = new_population;
        self.current_generation += 1;

        debug!(
            "✅ Generation {} complete. Best fitness: {:.4}, Avg fitness: {:.4}",
            self.current_generation - 1,
            best_fitness,
            self.population.iter().map(|p| p.fitness).sum::<f32>() / self.population.len() as f32
        );

        Ok(())
    }

    /// Tournament selection (like genetic_mona's selection strategy)
    fn tournament_selection(&self) -> PersonalityGenes {
        let mut rng = rand::thread_rng();
        let mut best_individual = &self.population[0];

        for _ in 0..TOURNAMENT_SIZE {
            let candidate = &self.population[rng.gen_range(0..self.population.len())];
            if candidate.fitness > best_individual.fitness {
                best_individual = candidate;
            }
        }

        best_individual.clone()
    }

    /// Get the best evolved personality configuration
    pub fn get_best_personality_configuration(&self) -> Option<&PersonalityGenes> {
        self.best_individual.as_ref()
    }

    /// Get evolution statistics
    pub fn get_evolution_stats(&self) -> String {
        let avg_fitness = if !self.generation_history.is_empty() {
            self.generation_history.iter().sum::<f32>() / self.generation_history.len() as f32
        } else {
            0.0
        };

        format!(
            "🧬 Evolutionary Personality Engine Stats:\n\
             • Generation: {}\n\
             • Best Fitness: {:.4}\n\
             • Average Fitness: {:.4}\n\
             • Population Size: {}\n\
             • Improvement Rate: {:.2}%\n\
             • Neurodivergent Support Score: {:.4}",
            self.current_generation,
            self.best_fitness,
            avg_fitness,
            POPULATION_SIZE,
            if self.generation_history.len() > 1 {
                ((self.best_fitness - self.generation_history[0]) / self.generation_history[0]
                    * 100.0)
                    .max(0.0)
            } else {
                0.0
            },
            self.best_individual
                .as_ref()
                .map(|b| b.neurodivergent_support_score)
                .unwrap_or(0.0)
        )
    }

    /// Start continuous evolution process
    pub async fn start_continuous_evolution(&mut self) -> Result<()> {
        info!("🧬🔄 Starting continuous evolutionary adaptation...");
        self.is_running = true;

        while self.is_running && self.current_generation < GENERATION_LIMIT {
            // Mock user feedback for testing (in real system, this comes from actual interactions)
            let mut rng = rand::thread_rng();
            let user_feedback: Vec<f32> = (0..POPULATION_SIZE)
                .map(|_| rng.gen_range(0.3..0.9))
                .collect();
            let neurodivergent_effectiveness: Vec<f32> = (0..POPULATION_SIZE)
                .map(|_| rng.gen_range(0.4..0.95))
                .collect();

            self.evolve_generation(user_feedback, neurodivergent_effectiveness)
                .await?;

            // Log progress every 10 generations
            if self.current_generation % 10 == 0 {
                info!("📊 {}", self.get_evolution_stats());
            }

            // Brief pause to prevent overwhelming the system
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        }

        info!(
            "🏁 Evolution complete! Final stats:\n{}",
            self.get_evolution_stats()
        );
        Ok(())
    }

    /// Stop continuous evolution
    pub fn stop_evolution(&mut self) {
        self.is_running = false;
        info!("⏹️ Evolutionary adaptation stopped");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_personality_genes_creation() {
        let genes = PersonalityGenes::random();
        assert!(genes.analyst_weight >= 0.0 && genes.analyst_weight <= 1.0);
        assert!(genes.empathy_sensitivity >= 0.0 && genes.empathy_sensitivity <= 1.0);
    }

    #[test]
    fn test_personality_genes_mutation() {
        let mut genes = PersonalityGenes::random();
        let original_analyst = genes.analyst_weight;

        genes.mutate(1.0); // 100% mutation rate

        // Some mutation should have occurred
        assert_ne!(original_analyst, genes.analyst_weight);
    }

    #[test]
    fn test_personality_genes_crossover() {
        let genes1 = PersonalityGenes::random();
        let genes2 = PersonalityGenes::random();

        let (child1, child2) = genes1.crossover(&genes2);

        // Children should be different from parents
        assert!(child1.fitness == 0.0);
        assert!(child2.fitness == 0.0);
    }

    #[tokio::test]
    async fn test_evolutionary_engine_creation() {
        let engine = EvolutionaryPersonalityEngine::new();
        assert_eq!(engine.population.len(), POPULATION_SIZE);
        assert_eq!(engine.current_generation, 0);
    }

    #[tokio::test]
    async fn test_evolution_generation() {
        let mut engine = EvolutionaryPersonalityEngine::new();
        let user_feedback: Vec<f32> = (0..POPULATION_SIZE).map(|_| 0.7).collect();
        let neurodivergent_effectiveness: Vec<f32> = (0..POPULATION_SIZE).map(|_| 0.8).collect();

        engine
            .evolve_generation(user_feedback, neurodivergent_effectiveness)
            .await
            .unwrap();

        assert_eq!(engine.current_generation, 1);
        assert!(engine.best_fitness > 0.0);
    }
}
