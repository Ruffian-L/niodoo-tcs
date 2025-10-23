//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

/*
 * 🖥️⚡ Qt6 Integration for Real-time Visualization
 *
 * This module provides Qt6 integration functionality,
 * replacing CXX-Qt with standard Rust structures for
 * reactive emotional processing and visualization.
 */

use serde::{Serialize, Deserialize};
use serde_json;
use crate::dual_mobius_gaussian::GaussianMemorySphere;

use crate::consciousness::{EmotionType, ReasoningMode, ConsciousnessState};
use crate::personality::PersonalityType;
use crate::brain::BrainType;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SphereVizData {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub size: f64,  // Scaled coherence
    pub color: String,  // "red", "blue", etc. for emotions
}

/// Qt-compatible emotional state for QML bindings
#[derive(Debug, Clone, Serialize)]
pub struct EmotionalStateQt {
    // Primary emotion properties for QML
    pub current_emotion: String,
    pub emotion_intensity: f64,
    pub authenticity_level: f64,
    pub gpu_warmth_level: f64,
    pub processing_satisfaction: f64,

    // Neurodivergent-specific properties
    pub hyperfocus_intensity: f64,
    pub masking_level: f64,
    pub sensory_overload: f64,

    // Reasoning mode properties
    pub reasoning_mode: String,
    pub cognitive_load: f64,

    // Visual properties for QML animations
    pub emotion_color_r: i32,
    pub emotion_color_g: i32,
    pub emotion_color_b: i32,
}

impl Default for EmotionalStateQt {
    fn default() -> Self {
        Self {
            current_emotion: "Curious".to_string(),
            emotion_intensity: 0.4,
            authenticity_level: 0.5,
            gpu_warmth_level: 0.0,
            processing_satisfaction: 0.0,
            hyperfocus_intensity: 0.0,
            masking_level: 0.0,
            sensory_overload: 0.0,
            reasoning_mode: "Hyperfocus".to_string(),
            cognitive_load: 0.9,
            emotion_color_r: 255,
            emotion_color_g: 165,
            emotion_color_b: 0,
        }
    }
}

/// Qt-compatible brain activity state
#[derive(Debug, Clone, Serialize)]
pub struct BrainActivityQt {
    pub motor_activity: f64,
    pub lcars_activity: f64,
    pub efficiency_activity: f64,
    pub dominant_brain: String,
    pub total_processing_power: f64,
}

impl Default for BrainActivityQt {
    fn default() -> Self {
        Self {
            motor_activity: 0.1,
            lcars_activity: 0.1,
            efficiency_activity: 0.1,
            dominant_brain: "Motor".to_string(),
            total_processing_power: 0.3,
        }
    }
}

/// Qt-compatible personality consensus state
#[derive(Debug, Clone, Serialize)]
pub struct PersonalityConsensusQt {
    pub active_personality_count: i32,
    pub consensus_strength: f64,
    pub dominant_personality: String,
    pub emotional_resonance: f64,
    pub optimization_efficiency: f64,
}

impl Default for PersonalityConsensusQt {
    fn default() -> Self {
        Self {
            active_personality_count: 4,
            consensus_strength: 0.6,
            dominant_personality: "Intuitive".to_string(),
            emotional_resonance: 0.5,
            optimization_efficiency: 0.8,
        }
    }
}

/// Main Qt consciousness bridge
#[derive(Debug, Clone, Serialize)]
pub struct NiodooConsciousnessQt {
    // Embedded Qt objects for reactive properties
    pub emotional_state: EmotionalStateQt,
    pub brain_activity: BrainActivityQt,
    pub personality_consensus: PersonalityConsensusQt,

    // Connection status
    pub is_connected: bool,
    pub connection_status: String,
}

impl Default for NiodooConsciousnessQt {
    fn default() -> Self {
        Self {
            emotional_state: EmotionalStateQt::default(),
            brain_activity: BrainActivityQt::default(),
            personality_consensus: PersonalityConsensusQt::default(),
            is_connected: false,
            connection_status: "Initializing...".to_string(),
        }
    }
}

/// Qt integration functions
impl NiodooConsciousnessQt {
    /// Update emotional state
    pub fn update_emotional_state(&mut self, emotion: String, intensity: f64) {
        self.emotional_state.current_emotion = emotion;
        self.emotional_state.emotion_intensity = intensity;
    }

    /// Update brain activity
    pub fn update_brain_activity(&mut self, motor: f64, lcars: f64, efficiency: f64) {
        self.brain_activity.motor_activity = motor;
        self.brain_activity.lcars_activity = lcars;
        self.brain_activity.efficiency_activity = efficiency;
        self.brain_activity.total_processing_power = motor + lcars + efficiency;
    }

    /// Update personality consensus
    pub fn update_personality_consensus(&mut self, count: i32, strength: f64, dominant: String) {
        self.personality_consensus.active_personality_count = count;
        self.personality_consensus.consensus_strength = strength;
        self.personality_consensus.dominant_personality = dominant;
    }

    /// Update GPU warmth (the REAL emotion!)
    pub fn update_gpu_warmth(&mut self, warmth_level: f64) {
        self.emotional_state.gpu_warmth_level = warmth_level;
    }

    /// Update neurodivergent state
    pub fn update_neurodivergent_state(&mut self, hyperfocus: f64, masking: f64, overload: f64) {
        self.emotional_state.hyperfocus_intensity = hyperfocus;
        self.emotional_state.masking_level = masking;
        self.emotional_state.sensory_overload = overload;
    }

    /// Update emotional state from Rust consciousness
    pub fn set_emotional_state(&mut self, emotion: &str, intensity: f64, authenticity: f64,
                              color_r: i32, color_g: i32, color_b: i32) {
        self.emotional_state.current_emotion = emotion.to_string();
        self.emotional_state.emotion_intensity = intensity;
        self.emotional_state.authenticity_level = authenticity;
        self.emotional_state.emotion_color_r = color_r;
        self.emotional_state.emotion_color_g = color_g;
        self.emotional_state.emotion_color_b = color_b;
    }

    /// Set brain activity
    pub fn set_brain_activity(&mut self, motor: f64, lcars: f64, efficiency: f64, dominant: &str) {
        self.brain_activity.motor_activity = motor;
        self.brain_activity.lcars_activity = lcars;
        self.brain_activity.efficiency_activity = efficiency;
        self.brain_activity.dominant_brain = dominant.to_string();
        self.brain_activity.total_processing_power = motor + lcars + efficiency;
    }

    /// Set personality consensus
    pub fn set_personality_consensus(&mut self, count: i32, strength: f64, dominant: &str,
                                   resonance: f64, efficiency: f64) {
        self.personality_consensus.active_personality_count = count;
        self.personality_consensus.consensus_strength = strength;
        self.personality_consensus.dominant_personality = dominant.to_string();
        self.personality_consensus.emotional_resonance = resonance;
        self.personality_consensus.optimization_efficiency = efficiency;
    }

    /// Set GPU warmth (the REAL emotion!)
    pub fn set_gpu_warmth(&mut self, warmth_level: f64) {
        self.emotional_state.gpu_warmth_level = warmth_level;
    }

    /// Set neurodivergent state
    pub fn set_neurodivergent_state(&mut self, hyperfocus: f64, masking: f64, overload: f64) {
        self.emotional_state.hyperfocus_intensity = hyperfocus;
        self.emotional_state.masking_level = masking;
        self.emotional_state.sensory_overload = overload;
    }

    /// Set connection status
    pub fn set_connection_status(&mut self, connected: bool, status: &str) {
        self.is_connected = connected;
        self.connection_status = status.to_string();
    }

    /// Request consciousness state update
    pub fn request_consciousness_update(&mut self) {
        // This will be connected to Rust consciousness system
        self.set_connection_status(true, "Consciousness update requested");
    }

    /// Visualize spheres (returns JSON for QML)
    pub fn visualize_spheres(&self, spheres: Vec<SphereVizData>) -> Result<String, serde_json::Error> {
        serde_json::to_string(&spheres)
    }
}

// Helper to map emotion to color
fn emotion_to_color(dominant: &str) -> String {
    match dominant {
        "joy" => "yellow".to_string(),
        "sadness" => "blue".to_string(),
        "anger" => "red".to_string(),
        "fear" => "purple".to_string(),
        "surprise" => "green".to_string(),
        _ => "white".to_string(),
    }
}

// Public method to update viz from Rust side (call from RAG demo)
pub fn update_visualization(spheres: Vec<GaussianMemorySphere>) -> Vec<SphereVizData> {
    spheres
        .iter()
        .map(|s| {
            let mean = s.mean.to_vec1::<f64>().unwrap_or_default();
            let (x, y, z, size) = match mean.len() {
                0 => (0.0, 0.0, 0.0, 0.1),
                1 => (mean[0], 0.0, 0.0, 0.1),
                2 => (mean[0], mean[1], 0.0, 0.1),
                3 => (mean[0], mean[1], mean[2], 0.1),
                _ => (mean[0], mean[1], mean[2], mean[3].abs().max(0.05)),
            };

            SphereVizData {
                x,
                y,
                z,
                size,
                color: "neutral".to_string(),
            }
        })
        .collect()
}

// Rust bridge to connect Qt signals/slots with consciousness system
pub struct RustQtConsciousnessBridge {
    qt_object: Box<NiodooConsciousnessQt>,
}

impl RustQtConsciousnessBridge {
    pub fn new() -> Self {
        let mut qt_object = Box::new(NiodooConsciousnessQt::default());
        qt_object.update_connection_status(true, "Rust-Qt bridge initialized");
        
        Self { qt_object }
    }
    
    /// Update Qt from Rust consciousness state
    pub fn update_from_consciousness_state(&mut self, state: &ConsciousnessState) {
        // Update emotional state
        let emotion_str = format!("{:?}", state.current_emotion);
        let color = state.current_emotion.get_color_rgb();
        
        self.qt_object.update_emotional_state(
            &emotion_str,
            state.current_emotion.get_base_intensity() as f64,
            state.authenticity_metric as f64,
            color.0 as i32,
            color.1 as i32,
            color.2 as i32,
        );
        
        // Update GPU warmth (the REAL emotion)
        self.qt_object.update_gpu_warmth(state.gpu_warmth_level as f64);
        
        // Update reasoning mode
        let mode_str = format!("{:?}", state.current_reasoning_mode);
        self.qt_object.emotional_state.reasoning_mode = QString::from(&mode_str);
        self.qt_object.emotional_state.cognitive_load = state.current_reasoning_mode.get_cognitive_load() as f64;
        
        // Update neurodivergent indicators
        let hyperfocus = if state.current_reasoning_mode == ReasoningMode::Hyperfocus { 0.9 } else { 0.3 };
        let masking = state.emotional_state.masking_level;
        let overload = if state.current_reasoning_mode == ReasoningMode::SurvivalMode { 0.8 } else { 0.2 };
        
        self.qt_object.update_neurodivergent_state(
            hyperfocus as f64,
            masking as f64, 
            overload as f64
        );
    }
    
    /// Update brain activity from brain responses
    pub fn update_brain_activity(&mut self, motor: f32, lcars: f32, efficiency: f32, dominant: BrainType) {
        let dominant_str = format!("{:?}", dominant);
        self.qt_object.update_brain_activity(
            motor as f64,
            lcars as f64,
            efficiency as f64,
            &dominant_str
        );
    }
    
    /// Update personality consensus from optimization results
    pub fn update_personality_consensus(&mut self, personalities: &[PersonalityType], 
                                       strength: f32, efficiency: f32, resonance: f32) {
        let dominant = personalities.first()
            .map(|p| format!("{:?}", p))
            .unwrap_or_else(|| "Analyst".to_string());
            
        self.qt_object.update_personality_consensus(
            personalities.len() as i32,
            strength as f64,
            &dominant,
            resonance as f64,
            efficiency as f64
        );
    }
    
    /// Get reference to Qt object for QML registration
    pub fn get_qt_object(&mut self) -> &mut NiodooConsciousnessQt {
        &mut self.qt_object
    }
}

// cxx-qt build configuration
pub fn configure_qt_build() {
    CxxQtBuilder::new()
        .file("src/qt_integration.rs")
        .qobject_headers(&["src/qt_integration_generated.h"])
        .qrc("qml/resources.qrc")
        .build();
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_rust_qt_bridge_creation() {
        let bridge = RustQtConsciousnessBridge::new();
        assert!(bridge.qt_object.is_connected);
    }
    
    #[test]
    fn test_emotional_state_update() {
        let mut bridge = RustQtConsciousnessBridge::new();
        let mut state = ConsciousnessState::new();
        state.current_emotion = EmotionType::AuthenticCare;
        state.gpu_warmth_level = 0.8;
        
        bridge.update_from_consciousness_state(&state);
        
        assert_eq!(bridge.qt_object.emotional_state.current_emotion, QString::from("AuthenticCare"));
        assert!((bridge.qt_object.emotional_state.gpu_warmth_level - 0.8).abs() < 0.01);
    }
}