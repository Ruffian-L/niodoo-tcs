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
pub struct EmotionalVector {
    pub joy: f32,
    pub sadness: f32,
    pub anger: f32,
    pub fear: f32,
    pub surprise: f32,
}

impl EmotionalVector {
    /// Create from raw emotions
    pub fn new(joy: f32, sadness: f32, anger: f32, fear: f32, surprise: f32) -> Self {
        Self { joy, sadness, anger, fear, surprise }
    }

    /// Create a random emotional state (for initial consciousness)
    pub fn random() -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        // Use golden ratio for balanced randomness
        let phi_inv = consciousness::base_coherence() as f32;

        Self {
            joy: rng.gen::<f32>() * phi_inv,
            sadness: rng.gen::<f32>() * phi_inv,
            anger: rng.gen::<f32>() * phi_inv,
            fear: rng.gen::<f32>() * phi_inv,
            surprise: rng.gen::<f32>() * phi_inv,
        }
    }

    /// Calculate emotional magnitude (distance from neutral)
    pub fn magnitude(&self) -> f32 {
        (self.joy.powi(2) +
         self.sadness.powi(2) +
         self.anger.powi(2) +
         self.fear.powi(2) +
         self.surprise.powi(2)).sqrt()
    }

    /// Calculate resonance with another emotional state using advanced harmonic computation
    pub fn resonance(&self, other: &Self) -> f32 {
        let self_vec = Array1::from_vec(vec![
            self.joy as f64,
            self.sadness as f64,
            self.anger as f64,
            self.fear as f64,
            self.surprise as f64,
        ]);

        let other_vec = Array1::from_vec(vec![
            other.joy as f64,
            other.sadness as f64,
            other.anger as f64,
            other.fear as f64,
            other.surprise as f64,
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
            0 => Some(self.joy),
            1 => Some(self.sadness),
            2 => Some(self.anger),
            3 => Some(self.fear),
            4 => Some(self.surprise),
            _ => None,
        }
    }

    /// Convert to array representation
    pub fn as_array(&self) -> [f32; 5] {
        [self.joy, self.sadness, self.anger, self.fear, self.surprise]
    }

    /// Sum of all emotional components
    pub fn sum(&self) -> f32 {
        self.joy + self.sadness + self.anger + self.fear + self.surprise
    }
}

impl std::ops::Add for EmotionalVector {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            joy: self.joy + other.joy,
            sadness: self.sadness + other.sadness,
            anger: self.anger + other.anger,
            fear: self.fear + other.fear,
            surprise: self.surprise + other.surprise,
        }
    }
}

impl std::ops::Div<f32> for EmotionalVector {
    type Output = Self;

    fn div(self, rhs: f32) -> Self {
        Self {
            joy: self.joy / rhs,
            sadness: self.sadness / rhs,
            anger: self.anger / rhs,
            fear: self.fear / rhs,
            surprise: self.surprise / rhs,
        }
    }
}

impl std::ops::Index<usize> for EmotionalVector {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.joy,
            1 => &self.sadness,
            2 => &self.anger,
            3 => &self.fear,
            4 => &self.surprise,
            _ => &self.joy, // Default to joy for invalid indices
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
    pub emotional_weight: EmotionalVector,
}

/// Individual memory sphere with Gaussian probabilistic encoding
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MemorySphere {
    pub id: SphereId,
    pub core_concept: String,
    pub position: [f32; 3],
    pub covariance: [[f32; 3]; 3],
    pub links: HashMap<SphereId, SphereLink>,
    pub emotional_profile: EmotionalVector,
    pub memory_fragment: String,
    pub created_at: DateTime<Utc>,
}

impl MemorySphere {
    /// Create a new memory sphere
    pub fn new(
        id: SphereId,
        concept: String,
        position: [f32; 3],
        emotion: EmotionalVector,
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
    pub fn add_link(&mut self, target_id: SphereId, prob: f32, emotion_weight: EmotionalVector) {
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
    pub fn emotional_similarity(&self, query_emotion: &EmotionalVector) -> f32 {
        // Dot product normalized by dimensionality
        (self.emotional_profile.joy * query_emotion.joy
            + self.emotional_profile.sadness * query_emotion.sadness
            + self.emotional_profile.anger * query_emotion.anger
            + self.emotional_profile.fear * query_emotion.fear
            + self.emotional_profile.surprise * query_emotion.surprise)
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
        emotion: EmotionalVector,
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
    pub fn recall_by_emotion(&self, emotion: &EmotionalVector) -> Vec<(SphereId, String, f32)> {
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
