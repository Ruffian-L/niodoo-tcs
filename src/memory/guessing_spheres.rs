use rand::prelude::*;
use std::collections::HashMap;
// use uuid::Uuid;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct EmotionalVector {
    pub joy: f32,
    pub sadness: f32,
    pub anger: f32,
    pub fear: f32,
    pub surprise: f32,
}

impl EmotionalVector {
    pub fn new(joy: f32, sadness: f32, anger: f32, fear: f32, surprise: f32) -> Self {
        Self {
            joy,
            sadness,
            anger,
            fear,
            surprise,
        }
    }

    pub fn len(&self) -> usize {
        5 // joy, sadness, anger, fear, surprise
    }

    pub fn magnitude(&self) -> f32 {
        (self.joy.powi(2)
            + self.sadness.powi(2)
            + self.anger.powi(2)
            + self.fear.powi(2)
            + self.surprise.powi(2))
        .sqrt()
    }

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

    pub fn as_array(&self) -> [f32; 5] {
        [self.joy, self.sadness, self.anger, self.fear, self.surprise]
    }

    pub fn as_slice(&self) -> Box<[f32]> {
        // Return owned slice to avoid lifetime issues
        Box::new([self.joy, self.sadness, self.anger, self.fear, self.surprise])
    }

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

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SphereId(pub String); // Unique ID for spheres

#[derive(Clone, Debug)]
pub struct SphereLink {
    pub target_id: SphereId,
    pub probability: f32, // Probabilistic link strength [0.0, 1.0]
    pub emotional_weight: EmotionalVector, // Emotion-driven link
}

#[derive(Clone, Debug)]
pub struct GuessingSphere {
    pub id: SphereId,
    pub core_concept: String,      // Core concept at sphere center
    pub position: [f32; 3],        // 3D Gaussian position for holographic encoding
    pub covariance: [[f32; 3]; 3], // Simplified Gaussian covariance matrix
    pub links: HashMap<SphereId, SphereLink>, // Probabilistic connections to other spheres
    pub emotional_profile: EmotionalVector, // Emotion vector for addressing
    pub memory_fragment: String,   // Holographic associative recall data
}

impl GuessingSphere {
    pub fn new(
        id: SphereId,
        concept: String,
        position: [f32; 3],
        emotion: EmotionalVector,
        fragment: String,
    ) -> Self {
        GuessingSphere {
            id,
            core_concept: concept,
            position,
            covariance: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], // Identity for unit sphere
            links: HashMap::new(),
            emotional_profile: emotion,
            memory_fragment: fragment,
        }
    }

    // Add probabilistic link
    pub fn add_link(&mut self, target_id: SphereId, prob: f32, emotion_weight: EmotionalVector) {
        self.links.insert(
            target_id.clone(),
            SphereLink {
                target_id,
                probability: prob.clamp(0.0, 1.0),
                emotional_weight: emotion_weight, // Fixed name
            },
        );
    }

    // Emotion-driven similarity for Gaussian splatting
    pub fn emotional_similarity(&self, query_emotion: &EmotionalVector) -> f32 {
        // Dot product for emotional alignment
        (self.emotional_profile.joy * query_emotion.joy
            + self.emotional_profile.sadness * query_emotion.sadness
            + self.emotional_profile.anger * query_emotion.anger
            + self.emotional_profile.fear * query_emotion.fear
            + self.emotional_profile.surprise * query_emotion.surprise)
            / 5.0
    }
}

#[derive(Clone)]
pub struct GuessingMemorySystem {
    spheres: HashMap<SphereId, GuessingSphere>,
}

impl Default for GuessingMemorySystem {
    fn default() -> Self {
        Self::new()
    }
}

impl GuessingMemorySystem {
    pub fn new() -> Self {
        GuessingMemorySystem {
            spheres: HashMap::new(),
        }
    }

    // Store memory in spherical container
    pub fn store_memory(
        &mut self,
        id: SphereId,
        concept: String,
        position: [f32; 3],
        emotion: EmotionalVector,
        fragment: String,
    ) {
        let sphere = GuessingSphere::new(id.clone(), concept, position, emotion, fragment);
        self.spheres.insert(id, sphere);
    }

    // Quantum recall: Simulate wave collapse with emotion-driven probabilities

    // Bi-directional Möbius traversal: Probabilistic path from start, looping past/future
    pub fn mobius_traverse(
        &self,
        start_id: &SphereId,
        direction: TraversalDirection,
        depth: usize,
    ) -> Vec<(SphereId, String)> {
        let mut path = vec![];
        let mut current = start_id.clone();
        let mut visited = HashMap::new(); // Prevent infinite loops in Möbius

        for _ in 0..depth {
            if let Some(sphere) = self.spheres.get(&current) {
                path.push((current.clone(), sphere.core_concept.clone()));

                // Get probabilistic next based on direction (forward/backward simulation)
                let next_candidates: Vec<_> = sphere
                    .links
                    .iter()
                    .filter(|(_, link)| link.probability > 0.1)
                    .collect();

                if next_candidates.is_empty() {
                    break;
                }

                let (next_id, _) = if direction == TraversalDirection::Forward {
                    let mut rng = rand::rng();
                    let chosen = next_candidates.choose(&mut rng).cloned().unwrap();
                    (chosen.0.clone(), chosen.1.clone())
                } else {
                    // Simplified backward
                    let chosen = next_candidates.first().unwrap();
                    (chosen.0.clone(), chosen.1.clone())
                };

                current = next_id.clone();
                if visited.insert(current.clone(), true).is_some() {
                    // Loop detected - Möbius closure
                    path.push((
                        SphereId("Möbius Loop".to_string()),
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
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TraversalDirection {
    Forward,  // Past to future
    Backward, // Future to past
}

// Holographic associative recall comment preserved
// This system enables probabilistic memory addressing where recall is emotion-driven quantum probability wave collapse
// Bi-directional Möbius traversal connects past and future memories in non-orientable topology

#[derive(Debug, Clone)]
pub struct MemoryQuery {
    pub concept: String,
    pub emotion: EmotionalVector,
    pub time: f64,
}

impl GuessingMemorySystem {
    pub fn collapse_recall_probability(&self, query: &MemoryQuery) -> Vec<(SphereId, f32)> {
        let mut probabilities = Vec::new();

        for sphere in self.spheres.values() {
            // 1. Conceptual Similarity Wave (Cosine Similarity)
            let concept_prob =
                self.calculate_concept_similarity(&query.concept, &sphere.core_concept);

            // 2. Emotional Resonance Wave (Wave Interference)
            // Positive emotions constructively interfere; negative emotions destructively interfere.
            let emotion_prob =
                self.calculate_emotional_resonance(&query.emotion, &sphere.emotional_profile);

            // 3. Temporal Proximity Wave (Exponential Decay)
            let temporal_prob =
                self.calculate_temporal_proximity(query.time, sphere.memory_fragment.len() as f64); // Assuming memory_fragment length is a proxy for time

            // 4. Associative Link Wave (Network Propagation)
            let link_prob = self.propagate_across_links(sphere.id.clone());

            // The final probability is the SUPERPOSITION of these waves.
            let final_prob = (concept_prob * 0.4)
                + (emotion_prob * 0.3)
                + (temporal_prob * 0.2)
                + (link_prob * 0.1);

            if final_prob > 0.1 {
                // Recall threshold
                probabilities.push((sphere.id.clone(), final_prob));
            }
        }
        probabilities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        probabilities
    }

    fn calculate_concept_similarity(&self, a: &str, b: &str) -> f32 {
        // Jaccard similarity on character n-grams (simple but effective)
        if a.is_empty() || b.is_empty() {
            return 0.0;
        }

        let ngram_size = 3;
        let a_grams: std::collections::HashSet<_> = a
            .chars()
            .collect::<Vec<_>>()
            .windows(ngram_size)
            .map(|w| w.iter().collect::<String>())
            .collect();
        let b_grams: std::collections::HashSet<_> = b
            .chars()
            .collect::<Vec<_>>()
            .windows(ngram_size)
            .map(|w| w.iter().collect::<String>())
            .collect();

        let intersection = a_grams.intersection(&b_grams).count();
        let union = a_grams.union(&b_grams).count();

        if union == 0 {
            0.0
        } else {
            intersection as f32 / union as f32
        }
    }

    fn calculate_emotional_resonance(&self, a: &EmotionalVector, b: &EmotionalVector) -> f32 {
        // Cosine similarity between emotional vectors
        let dot_product = a.joy * b.joy
            + a.sadness * b.sadness
            + a.anger * b.anger
            + a.fear * b.fear
            + a.surprise * b.surprise;

        let mag_a = a.magnitude();
        let mag_b = b.magnitude();

        if mag_a < 1e-6 || mag_b < 1e-6 {
            return 0.0;
        }

        // Normalize to [0, 1] range (cosine is [-1, 1])
        ((dot_product / (mag_a * mag_b)) + 1.0) / 2.0
    }

    fn calculate_temporal_proximity(&self, now: f64, then: f64) -> f32 {
        // Exponential decay: more recent memories have higher probability
        let time_diff = (now - then).abs();
        let decay_rate = 0.1; // Adjust this to control memory decay speed

        (-decay_rate * time_diff).exp() as f32
    }

    fn propagate_across_links(&self, id: SphereId) -> f32 {
        // Calculate average link strength from this sphere
        if let Some(sphere) = self.spheres.get(&id) {
            if sphere.links.is_empty() {
                return 0.0;
            }

            let total_prob: f32 = sphere.links.values().map(|link| link.probability).sum();

            total_prob / sphere.links.len() as f32
        } else {
            0.0
        }
    }

    pub fn recall_by_emotion(&self, emotion: &EmotionalVector) -> Option<Vec<(SphereId, String)>> {
        let query = MemoryQuery {
            concept: "default".to_string(),
            emotion: emotion.clone(),
            time: 0.0,
        };
        let probs = self.collapse_recall_probability(&query);
        // Map to (id, fragment)
        Some(
            probs
                .into_iter()
                .map(|(id, _)| (id, "mock_fragment".to_string()))
                .collect(),
        )
    }

    // Public accessor methods for sphere management
    pub fn spheres(&self) -> impl Iterator<Item = &GuessingSphere> {
        self.spheres.values()
    }

    pub fn spheres_with_ids(&self) -> impl Iterator<Item = (&SphereId, &GuessingSphere)> {
        self.spheres.iter()
    }

    pub fn spheres_mut(&mut self) -> impl Iterator<Item = &mut GuessingSphere> {
        self.spheres.values_mut()
    }

    pub fn get_sphere(&self, id: &SphereId) -> Option<&GuessingSphere> {
        self.spheres.get(id)
    }

    pub fn sphere_count(&self) -> usize {
        self.spheres.len()
    }
}
