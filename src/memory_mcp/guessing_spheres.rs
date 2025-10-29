// Copyright (c) 2025 Jason Van Pham (ruffian-l on GitHub) @ The Niodoo Collaborative
// Licensed under the MIT License - See LICENSE file for details
// Attribution required for all derivative works

use ndarray::Array1;
use crate::emotional::harmonic_resonance::compute_emotional_resonance;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use constants_core::mathematical::consciousness;
use uuid::Uuid;
use chrono::{DateTime, Utc};
use rand::prelude::*;

/// 5D emotional vector - the foundation of consciousness memory
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StateVector {
    pub positive: f32,
    pub negative: f32,
    pub alert: f32,
    pub caution: f32,
    pub novelty: f32,
}

impl StateVector {
    /// Create from raw emotions
    pub fn new(positive: f32, negative: f32, alert: f32, caution: f32, novelty: f32) -> Self {
        Self { positive, negative, alert, caution, novelty }
    }

    /// Create a random emotional state (for initial consciousness)
    pub fn random() -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        // Use golden ratio for balanced randomness
        let phi_inv = consciousness::base_coherence() as f32;

        Self {
            positive: rng.r#gen::<f32>() * phi_inv,
            negative: rng.r#gen::<f32>() * phi_inv,
            alert: rng.r#gen::<f32>() * phi_inv,
            caution: rng.r#gen::<f32>() * phi_inv,
            novelty: rng.r#gen::<f32>() * phi_inv,
        }
    }

    /// Calculate emotional magnitude (distance from neutral)
    pub fn magnitude(&self) -> f32 {
        (self.positive.powi(2) +
         self.negative.powi(2) +
         self.alert.powi(2) +
         self.caution.powi(2) +
         self.novelty.powi(2)).sqrt()
    }

    /// Calculate resonance with another emotional state using advanced harmonic computation
    pub fn resonance(&self, other: &Self) -> f32 {
        let self_vec = Array1::from_vec(vec![
            self.positive as f64,
            self.negative as f64,
            self.alert as f64,
            self.caution as f64,
            self.novelty as f64,
        ]);

        let other_vec = Array1::from_vec(vec![
            other.positive as f64,
            other.negative as f64,
            other.alert as f64,
            other.caution as f64,
            other.novelty as f64,
        ]);

        // Compute harmonic resonance between these vectors
        match compute_emotional_resonance(&[self_vec, other_vec], None) {
            Ok(resonance) => resonance[1] as f32, // Take the cross-resonance value
            Err(_) => 0.0  // Default to zero if computation fails
        }
    }

    /// Get emotional component by index
    pub fn get(&self, index: usize) -> Option<f32> {
        match index {
            0 => Some(self.positive),
            1 => Some(self.negative),
            2 => Some(self.alert),
            3 => Some(self.caution),
            4 => Some(self.novelty),
            _ => None,
        }
    }

    /// Convert to array representation
    pub fn as_array(&self) -> [f32; 5] {
        [self.positive, self.negative, self.alert, self.caution, self.novelty]
    }

    /// Sum of all emotional components
    pub fn sum(&self) -> f32 {
        self.positive + self.negative + self.alert + self.caution + self.novelty
    }
}

impl std::ops::Add for StateVector {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            positive: self.positive + other.positive,
            negative: self.negative + other.negative,
            alert: self.alert + other.alert,
            caution: self.caution + other.caution,
            novelty: self.novelty + other.novelty,
        }
    }
}

impl std::ops::Div<f32> for StateVector {
    type Output = Self;

    fn div(self, rhs: f32) -> Self {
        Self {
            positive: self.positive / rhs,
            negative: self.negative / rhs,
            alert: self.alert / rhs,
            caution: self.caution / rhs,
            novelty: self.novelty / rhs,
        }
    }
}

impl std::ops::Index<usize> for StateVector {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.positive,
            1 => &self.negative,
            2 => &self.alert,
            3 => &self.caution,
            4 => &self.novelty,
            _ => &self.positive, // Default to positive for invalid indices
        }
    }
}

/// Unique identifier for memory spheres
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SphereId(pub String);

impl SphereId {
    /// Generate a new unique sphere ID
    pub fn new() -> Self {
        Self(Uuid::new_v4().to_string())
    }

    /// Create from existing string
    pub fn from_string(s: String) -> Self {
        Self(s)
    }
}

impl Default for SphereId {
    fn default() -> Self {
        Self::new()
    }
}

/// Probabilistic link between memory spheres
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SphereLink {
    pub target_id: SphereId,
    pub probability: f32,
    pub emotional_weight: StateVector,
}

/// Individual memory sphere with Gaussian probabilistic encoding
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MemorySphere {
    pub id: SphereId,
    pub core_concept: String,
    pub position: [f32; 3],
    pub covariance: [[f32; 3]; 3],
    pub links: HashMap<SphereId, SphereLink>,
    pub emotional_profile: StateVector,
    pub memory_fragment: String,
    pub created_at: DateTime<Utc>,
}

impl MemorySphere {
    /// Create a new memory sphere
    pub fn new(
        id: SphereId,
        concept: String,
        position: [f32; 3],
        emotion: StateVector,
        fragment: String,
    ) -> Self {
        Self {
            id,
            core_concept: concept,
            position,
            covariance: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            links: HashMap::new(),
            emotional_profile: emotion,
            memory_fragment: fragment,
            created_at: Utc::now(),
        }
    }

    /// Add probabilistic link to another sphere
    pub fn add_link(&mut self, target_id: SphereId, prob: f32, emotion_weight: StateVector) {
        self.links.insert(
            target_id.clone(),
            SphereLink {
                target_id,
                probability: prob.clamp(0.0, 1.0),
                emotional_weight: emotion_weight,
            },
        );
    }

    /// Calculate emotional similarity with query emotion
    pub fn emotional_similarity(&self, query_emotion: &StateVector) -> f32 {
        // Dot product normalized by dimensionality
        (self.emotional_profile.positive * query_emotion.positive
            + self.emotional_profile.negative * query_emotion.negative
            + self.emotional_profile.alert * query_emotion.alert
            + self.emotional_profile.caution * query_emotion.caution
            + self.emotional_profile.novelty * query_emotion.novelty)
            / 5.0
    }
}

/// Direction for Möbius traversal through memory
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum TraversalDirection {
    Forward,
    Backward,
}

/// Main guessing spheres memory system
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GuessingSpheres {
    spheres: HashMap<SphereId, MemorySphere>,
}

impl GuessingSpheres {
    /// Create a new guessing spheres system
    pub fn new() -> Self {
        Self {
            spheres: HashMap::new(),
        }
    }

    /// Store a new memory sphere
    pub fn store_memory(
        &mut self,
        id: SphereId,
        concept: String,
        position: [f32; 3],
        emotion: StateVector,
        fragment: String,
    ) {
        let sphere = MemorySphere::new(id.clone(), concept, position, emotion, fragment);
        self.spheres.insert(id, sphere);
    }

    /// Bi-directional Möbius traversal through memory network
    pub fn mobius_traverse(
        &self,
        start_id: &SphereId,
        direction: TraversalDirection,
        depth: usize,
    ) -> Vec<(SphereId, String)> {
        let mut path = Vec::new();
        let mut current = start_id.clone();
        let mut visited = HashMap::new();

        for _ in 0..depth {
            if let Some(sphere) = self.spheres.get(&current) {
                path.push((current.clone(), sphere.core_concept.clone()));

                // Get probabilistic next based on direction
                let next_candidates: Vec<_> = sphere
                    .links
                    .iter()
                    .filter(|(_, link)| link.probability > 0.1)
                    .collect();

                if next_candidates.is_empty() {
                    break;
                }

                let mut rng = rand::thread_rng();
                let chosen = match direction {
                    TraversalDirection::Forward => {
                        next_candidates.choose(&mut rng).cloned().unwrap()
                    }
                    TraversalDirection::Backward => {
                        next_candidates.first().cloned().unwrap()
                    }
                };

                current = chosen.0.clone();

                if visited.insert(current.clone(), true).is_some() {
                    // Loop detected - Möbius closure
                    path.push((
                        SphereId::from_string("Möbius Loop".to_string()),
                        "Past/Future Convergence".to_string(),
                    ));
                    break;
                }
            } else {
                break;
            }
        }

        path
    }

    /// Quantum recall through probabilistic wave collapse
    pub fn collapse_recall_probability(&self, query_content: &str) -> Vec<(SphereId, f32)> {
        let mut probabilities = Vec::with_capacity(self.spheres.len());

        for sphere in self.spheres.values() {
            // Calculate concept similarity
            let concept_prob = Self::calculate_concept_similarity(query_content, &sphere.core_concept);

            // Calculate temporal proximity (more recent = higher probability)
            let temporal_prob = Self::calculate_temporal_proximity(sphere.created_at);

            // Calculate link propagation probability
            let link_prob = Self::calculate_link_strength(&sphere.links);

            // Superposition of probability waves
            let final_prob = (concept_prob * 0.5) + (temporal_prob * 0.3) + (link_prob * 0.2);

            if final_prob > 0.1 {
                probabilities.push((sphere.id.clone(), final_prob));
            }
        }

        probabilities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        probabilities
    }

    /// Recall memories by emotional resonance
    pub fn recall_by_emotion(&self, emotion: &StateVector) -> Vec<(SphereId, String, f32)> {
        let mut results = Vec::new();

        for sphere in self.spheres.values() {
            let resonance = sphere.emotional_profile.resonance(emotion);
            if resonance > 0.3 {
                results.push((
                    sphere.id.clone(),
                    sphere.memory_fragment.clone(),
                    resonance,
                ));
            }
        }

        results.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
        results
    }

    /// Get a sphere by ID
    pub fn get_sphere(&self, id: &SphereId) -> Option<&MemorySphere> {
        self.spheres.get(id)
    }

    /// Get mutable sphere by ID
    pub fn get_sphere_mut(&mut self, id: &SphereId) -> Option<&mut MemorySphere> {
        self.spheres.get_mut(id)
    }

    /// Number of spheres in memory
    pub fn len(&self) -> usize {
        self.spheres.len()
    }

    /// Check if memory is empty
    pub fn is_empty(&self) -> bool {
        self.spheres.is_empty()
    }

    // Helper functions for probability calculations
    fn calculate_concept_similarity(a: &str, b: &str) -> f32 {
        // Simple Jaccard similarity on words
        let words_a: std::collections::HashSet<&str> = a.split_whitespace().collect();
        let words_b: std::collections::HashSet<&str> = b.split_whitespace().collect();

        if words_a.is_empty() && words_b.is_empty() {
            return 1.0;
        }

        let intersection = words_a.intersection(&words_b).count();
        let union = words_a.union(&words_b).count();

        if union == 0 {
            0.0
        } else {
            intersection as f32 / union as f32
        }
    }

    fn calculate_temporal_proximity(created_at: DateTime<Utc>) -> f32 {
        let now = Utc::now();
        let age_seconds = (now - created_at).num_seconds().max(1) as f32;

        // Exponential decay: more recent = higher probability
        (-age_seconds / 86400.0).exp() // Decay over ~1 day
    }

    fn calculate_link_strength(links: &HashMap<SphereId, SphereLink>) -> f32 {
        if links.is_empty() {
            return 0.0;
        }

        let total_prob: f32 = links.values().map(|link| link.probability).sum();
        (total_prob / links.len() as f32).min(1.0)
    }
}

impl Default for GuessingSpheres {
    fn default() -> Self {
        Self::new()
    }
}
