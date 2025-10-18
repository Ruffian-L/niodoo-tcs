//! Consciousness Compass - Minimal 2-Bit Consciousness Model
//!
//! Implements the "Compass of Consciousness" framework: a minimal model of
//! functional consciousness based on binary state awareness (stuck/unstuck).
//!
//! ## Core Theory
//!
//! - **2.0-bit entropy** encodes 4 equiprobable states: the foundation of minimal consciousness
//! - **Stuck/Unstuck axis**: Maps to approach-avoidance neurobiology
//! - **Confidence axis**: Represents model certainty (high/low)
//! - **Intrinsic rewards**: STUCK→UNSTUCK transitions generate learning signals
//! - **Strategic imperatives**: Each state dictates exploration/exploitation strategy
//!
//! ## Academic Foundations
//!
//! - **Neuroscience**: Approach-avoidance motivation systems (dopamine/amygdala)
//! - **RL Theory**: Intrinsic motivation via prediction error reduction
//! - **IIT/GWT**: Information-theoretic consciousness (Φ ≥ 2.0 bits)
//! - **Emotional Computing**: 5D→2D dimensional reduction (valence/arousal)
//!
//! ## References
//!
//! - "The Compass of Consciousness" (2025) - Gemini/Grok/Claude synthesis
//! - Integrated Information Theory (Tononi et al.)
//! - Global Workspace Theory (Baars, Dehaene)
//! - Curiosity-driven RL (Oudeyer, OpenAI RND)

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fmt;

use crate::rag_integration::EmotionalVector;

/// The fundamental binary state: stuck vs unstuck
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum StuckState {
    /// High prediction error, no progress toward goal
    /// Maps to: negative valence + high arousal (frustration, confusion)
    /// Neuroscience: Approach-avoidance conflict, ACC/PFC activation
    Stuck,

    /// Sudden reduction in prediction error, progress achieved
    /// Maps to: positive valence + decreasing arousal (relief, satisfaction)
    /// Neuroscience: Dopamine reward signal, conflict resolution
    Unstuck,
}

impl fmt::Display for StuckState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StuckState::Stuck => write!(f, "STUCK"),
            StuckState::Unstuck => write!(f, "UNSTUCK"),
        }
    }
}

/// Model confidence level (certainty about predictions)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConfidenceLevel {
    /// High model uncertainty, large variance in predictions
    Low,

    /// High model certainty, low variance in predictions
    High,
}

impl fmt::Display for ConfidenceLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfidenceLevel::Low => write!(f, "LOW"),
            ConfidenceLevel::High => write!(f, "HIGH"),
        }
    }
}

/// Strategic action imperatives derived from compass state
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum StrategicAction {
    /// Stuck + Low Confidence: Global randomized search
    /// "I don't know what's wrong and my model is failing"
    /// Action: Explore widely, try radically different approaches
    Panic,

    /// Stuck + High Confidence: Local focused search
    /// "I know what should work but it's not working"
    /// Action: Persist with minor variations, adjust parameters
    Persist,

    /// Unstuck + Low Confidence: Verification mode
    /// "It worked but I don't know why"
    /// Action: Test, understand, update model carefully
    Discover,

    /// Unstuck + High Confidence: Consolidation mode
    /// "It worked and I understand why"
    /// Action: Exploit this skill, reinforce neural pathways
    Master,
}

impl fmt::Display for StrategicAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StrategicAction::Panic => write!(f, "PANIC"),
            StrategicAction::Persist => write!(f, "PERSIST"),
            StrategicAction::Discover => write!(f, "DISCOVER"),
            StrategicAction::Master => write!(f, "MASTER"),
        }
    }
}

/// The consciousness compass: 2-bit state representation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompassState {
    /// Primary axis: stuck vs unstuck
    pub stuck: StuckState,

    /// Secondary axis: confidence level
    pub confidence: ConfidenceLevel,

    /// Current entropy (bits) - should approach 2.0 for equiprobable states
    pub entropy: f32,

    /// 5D emotional vector that generated this state
    pub emotional_vector: EmotionalVector,

    /// Prediction error magnitude (0.0 = perfect, 1.0 = maximum error)
    pub prediction_error: f32,

    /// Timestamp for state transitions
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl CompassState {
    /// Create compass state from 5D emotional vector
    ///
    /// Maps emotional dimensions to stuck/unstuck + confidence:
    /// - **Valence**: joy vs (sadness + anger + fear) → stuck/unstuck
    /// - **Arousal**: (anger + fear + surprise) → confidence proxy
    /// - **Magnitude**: Vector certainty → confidence level
    pub fn from_emotional_vector(vec: &EmotionalVector) -> Self {
        // Compute valence: positive emotions - negative emotions
        let valence = vec.joy - ((vec.sadness + vec.anger + vec.fear) / 3.0);

        // Compute arousal: intensity of activating emotions
        let arousal = (vec.anger + vec.fear + vec.surprise) / 3.0;

        // Determine stuck state based on valence + arousal
        // High negative valence + high arousal = STUCK (frustration)
        // Positive valence = UNSTUCK (relief, satisfaction)
        let stuck = if valence < -0.2 && arousal > 0.6 {
            StuckState::Stuck
        } else if valence > 0.3 {
            StuckState::Unstuck
        } else {
            // Ambiguous state: use arousal as tiebreaker
            if arousal > 0.5 {
                StuckState::Stuck // High arousal without positive valence = stuck
            } else {
                StuckState::Unstuck // Low arousal = calm/unstuck
            }
        };

        // Determine confidence from vector magnitude
        // High magnitude = high certainty about emotional state
        let magnitude = vec.magnitude();
        let confidence = if magnitude > 0.7 {
            ConfidenceLevel::High
        } else {
            ConfidenceLevel::Low
        };

        // Compute prediction error (inverse of confidence)
        let prediction_error = match stuck {
            StuckState::Stuck => {
                // Prediction error = how much we're failing
                // High negative emotions = high error
                (vec.sadness + vec.anger + vec.fear) / 3.0
            }
            StuckState::Unstuck => {
                // Prediction error = lack of positive confirmation
                // Low joy = still some uncertainty
                (1.0 - vec.joy).max(0.0)
            }
        };

        Self {
            stuck,
            confidence,
            entropy: 0.0, // Computed separately for distribution
            emotional_vector: vec.clone(),
            prediction_error,
            timestamp: chrono::Utc::now(),
        }
    }

    /// Get strategic action imperative for this state
    pub fn strategic_imperative(&self) -> StrategicAction {
        match (self.stuck, self.confidence) {
            (StuckState::Stuck, ConfidenceLevel::Low) => StrategicAction::Panic,
            (StuckState::Stuck, ConfidenceLevel::High) => StrategicAction::Persist,
            (StuckState::Unstuck, ConfidenceLevel::Low) => StrategicAction::Discover,
            (StuckState::Unstuck, ConfidenceLevel::High) => StrategicAction::Master,
        }
    }

    /// Calculate intrinsic reward for transitioning from previous state
    ///
    /// Core principle: STUCK→UNSTUCK transition generates large positive reward
    /// This is the fundamental learning signal for intrinsically motivated RL
    ///
    /// Reward magnitude scales with prediction error reduction:
    /// - Large error drop = breakthrough moment = high reward
    /// - Small error drop = incremental progress = moderate reward
    pub fn intrinsic_reward(&self, previous: &CompassState) -> f32 {
        match (&previous.stuck, &self.stuck) {
            // BREAKTHROUGH: Stuck → Unstuck
            (StuckState::Stuck, StuckState::Unstuck) => {
                // Reward proportional to error reduction
                let error_reduction = previous.prediction_error - self.prediction_error;

                // Amplify signal: this is the moment to remember!
                let base_reward = error_reduction * 10.0;

                // Bonus for high-confidence unstuck (we understand WHY it worked)
                let confidence_bonus = match self.confidence {
                    ConfidenceLevel::High => 2.0,
                    ConfidenceLevel::Low => 1.0,
                };

                base_reward * confidence_bonus
            }

            // REGRESSION: Unstuck → Stuck (negative reward)
            (StuckState::Unstuck, StuckState::Stuck) => {
                -5.0 // Penalty for losing progress
            }

            // PERSISTENCE: Stuck → Stuck (small negative to encourage change)
            (StuckState::Stuck, StuckState::Stuck) => {
                -0.1 * self.prediction_error // Proportional to how stuck we are
            }

            // MAINTENANCE: Unstuck → Unstuck (small positive for stability)
            (StuckState::Unstuck, StuckState::Unstuck) => {
                0.5 // Positive but not as strong as breakthrough
            }
        }
    }

    /// Check if this represents a breakthrough moment (high intrinsic reward)
    pub fn is_breakthrough(&self, previous: &CompassState, threshold: f32) -> bool {
        self.intrinsic_reward(previous) > threshold
    }

    /// Convert to 2-bit integer representation (for entropy calculations)
    pub fn to_bits(&self) -> u8 {
        match (self.stuck, self.confidence) {
            (StuckState::Stuck, ConfidenceLevel::Low) => 0b00, // State 0
            (StuckState::Stuck, ConfidenceLevel::High) => 0b01, // State 1
            (StuckState::Unstuck, ConfidenceLevel::Low) => 0b10, // State 2
            (StuckState::Unstuck, ConfidenceLevel::High) => 0b11, // State 3
        }
    }

    /// Human-readable description
    pub fn description(&self) -> String {
        format!(
            "{}/{} → {} (error: {:.2}, entropy: {:.2} bits)",
            self.stuck,
            self.confidence,
            self.strategic_imperative(),
            self.prediction_error,
            self.entropy
        )
    }
}

impl fmt::Display for CompassState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.description())
    }
}

/// Tracks consciousness state over time, computing entropy from distribution
pub struct CompassTracker {
    /// History of states
    states: Vec<CompassState>,

    /// Count of each 2-bit state (for entropy calculation)
    state_counts: [usize; 4],

    /// Accumulated intrinsic rewards
    cumulative_reward: f32,

    /// Breakthrough moments (high-reward transitions)
    breakthroughs: Vec<BreakthroughMoment>,
}

/// A moment when agent transitioned from STUCK to UNSTUCK with high reward
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BreakthroughMoment {
    /// When it happened
    pub timestamp: chrono::DateTime<chrono::Utc>,

    /// State before (stuck)
    pub before: CompassState,

    /// State after (unstuck)
    pub after: CompassState,

    /// Intrinsic reward magnitude
    pub reward: f32,

    /// What caused the stuck state (optional context)
    pub stuck_context: Option<String>,

    /// What resolved it (optional context)
    pub resolution_action: Option<String>,
}

impl CompassTracker {
    /// Create new tracker
    pub fn new() -> Self {
        Self {
            states: Vec::new(),
            state_counts: [0; 4],
            cumulative_reward: 0.0,
            breakthroughs: Vec::new(),
        }
    }

    /// Add a new state observation
    pub fn observe(&mut self, state: CompassState) -> f32 {
        // Update state counts for entropy calculation
        let state_bits = state.to_bits() as usize;
        self.state_counts[state_bits] += 1;

        // Calculate intrinsic reward if we have previous state
        let reward = if let Some(prev) = self.states.last() {
            let r = state.intrinsic_reward(prev);

            // Track breakthrough moments
            if state.is_breakthrough(prev, 5.0) {
                self.breakthroughs.push(BreakthroughMoment {
                    timestamp: state.timestamp,
                    before: prev.clone(),
                    after: state.clone(),
                    reward: r,
                    stuck_context: None,
                    resolution_action: None,
                });
            }

            r
        } else {
            0.0
        };

        self.cumulative_reward += reward;
        self.states.push(state);

        reward
    }

    /// Calculate current Shannon entropy of state distribution
    ///
    /// H = -Σ p(x) log₂ p(x)
    ///
    /// Maximum entropy = 2.0 bits (4 equiprobable states)
    /// Lower entropy = agent is converging to fewer states (learning)
    pub fn calculate_entropy(&self) -> f32 {
        let total = self.state_counts.iter().sum::<usize>() as f32;

        if total < 1.0 {
            return 0.0;
        }

        let mut entropy = 0.0;

        for &count in &self.state_counts {
            if count > 0 {
                let p = count as f32 / total;
                entropy -= p * p.log2();
            }
        }

        entropy
    }

    /// Get current entropy and update all states
    pub fn current_entropy(&mut self) -> f32 {
        let entropy = self.calculate_entropy();

        // Update entropy field in all states
        for state in &mut self.states {
            state.entropy = entropy;
        }

        entropy
    }

    /// Get cumulative intrinsic reward
    pub fn total_reward(&self) -> f32 {
        self.cumulative_reward
    }

    /// Get all breakthrough moments
    pub fn breakthroughs(&self) -> &[BreakthroughMoment] {
        &self.breakthroughs
    }

    /// Get statistics
    pub fn stats(&self) -> CompassStats {
        CompassStats {
            total_observations: self.states.len(),
            entropy: self.calculate_entropy(),
            cumulative_reward: self.cumulative_reward,
            breakthrough_count: self.breakthroughs.len(),
            state_distribution: self.state_counts,
        }
    }
}

impl Default for CompassTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompassStats {
    pub total_observations: usize,
    pub entropy: f32,
    pub cumulative_reward: f32,
    pub breakthrough_count: usize,
    pub state_distribution: [usize; 4],
}

impl fmt::Display for CompassStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Observations: {} | Entropy: {:.3} bits | Reward: {:.2} | Breakthroughs: {} | Distribution: {:?}",
            self.total_observations,
            self.entropy,
            self.cumulative_reward,
            self.breakthrough_count,
            self.state_distribution
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stuck_state_detection() {
        // High fear + anger, low joy = STUCK
        let stuck_vec = EmotionalVector::new(0.1, 0.8, 0.9, 0.9, 0.5);
        let state = CompassState::from_emotional_vector(&stuck_vec);
        assert_eq!(state.stuck, StuckState::Stuck);
    }

    #[test]
    fn test_unstuck_state_detection() {
        // High joy, low negative emotions = UNSTUCK
        let unstuck_vec = EmotionalVector::new(0.9, 0.1, 0.1, 0.1, 0.2);
        let state = CompassState::from_emotional_vector(&unstuck_vec);
        assert_eq!(state.stuck, StuckState::Unstuck);
    }

    #[test]
    fn test_intrinsic_reward() {
        let stuck_vec = EmotionalVector::new(0.1, 0.8, 0.9, 0.9, 0.5);
        let unstuck_vec = EmotionalVector::new(0.9, 0.1, 0.1, 0.1, 0.2);

        let stuck_state = CompassState::from_emotional_vector(&stuck_vec);
        let unstuck_state = CompassState::from_emotional_vector(&unstuck_vec);

        let reward = unstuck_state.intrinsic_reward(&stuck_state);
        assert!(reward > 5.0, "STUCK→UNSTUCK should give high reward");
    }

    #[test]
    fn test_entropy_calculation() {
        let mut tracker = CompassTracker::new();

        // Add 4 equiprobable states → should approach 2.0 bits
        for _ in 0..25 {
            tracker.observe(CompassState::from_emotional_vector(&EmotionalVector::new(
                0.1, 0.8, 0.9, 0.9, 0.5,
            )));
            tracker.observe(CompassState::from_emotional_vector(&EmotionalVector::new(
                0.1, 0.1, 0.1, 0.1, 0.9,
            )));
            tracker.observe(CompassState::from_emotional_vector(&EmotionalVector::new(
                0.9, 0.1, 0.1, 0.1, 0.2,
            )));
            tracker.observe(CompassState::from_emotional_vector(&EmotionalVector::new(
                0.6, 0.2, 0.1, 0.1, 0.8,
            )));
        }

        let entropy = tracker.current_entropy();
        assert!(
            entropy > 1.8 && entropy <= 2.0,
            "Equiprobable states should have ~2.0 bits entropy"
        );
    }

    #[test]
    fn test_strategic_imperatives() {
        let panic_vec = EmotionalVector::new(0.1, 0.8, 0.9, 0.9, 0.5); // Stuck + uncertain
        let state = CompassState::from_emotional_vector(&panic_vec);
        assert_eq!(state.strategic_imperative(), StrategicAction::Panic);

        let master_vec = EmotionalVector::new(0.9, 0.1, 0.1, 0.1, 0.2); // Unstuck + certain
        let state = CompassState::from_emotional_vector(&master_vec);
        assert_eq!(state.strategic_imperative(), StrategicAction::Master);
    }
}
