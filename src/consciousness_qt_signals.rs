//! Qt Signal Bridge for Real-Time Consciousness Updates
//!
//! This module establishes the reactive signal connections between:
//! - Consciousness state changes → QML UI updates
//! - User input from QML → Consciousness processing
//! - Memory events → UI notifications
//! - System metrics → Performance displays
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                   Consciousness Engine                       │
//! │  ┌───────────┐  ┌────────────┐  ┌──────────────┐          │
//! │  │  Emotion  │  │   Memory   │  │   Metrics    │          │
//! │  │  Updates  │  │   Events   │  │   Changes    │          │
//! │  └─────┬─────┘  └──────┬─────┘  └──────┬───────┘          │
//! │        │                │                 │                  │
//! │        ▼                ▼                 ▼                  │
//! │  ┌──────────────────────────────────────────────┐          │
//! │  │      Consciousness Signal Broadcaster         │          │
//! │  └──────────────────┬───────────────────────────┘          │
//! └─────────────────────┼───────────────────────────────────────┘
//!                       │
//!                       ▼
//!         ┌─────────────────────────────┐
//!         │    Qt Signal Bridge         │
//!         │  (This Module)              │
//!         └─────────┬───────────────────┘
//!                   │
//!         ┌─────────┴─────────┬─────────────────┐
//!         ▼                   ▼                 ▼
//!   ┌──────────┐      ┌──────────┐     ┌──────────┐
//!   │   QML    │      │   QML    │     │   QML    │
//!   │ Emotion  │      │  Memory  │     │ Metrics  │
//!   │ Display  │      │  Viz     │     │ Panel    │
//!   └──────────┘      └──────────┘     └──────────┘
//! ```
//!
//! ## Real-Time Update Flow
//!
//! 1. **State Change**: ConsciousnessEngine updates emotional state
//! 2. **Broadcast**: Event sent via tokio::broadcast channel
//! 3. **Bridge Receive**: Qt bridge receives and converts to QML format
//! 4. **Signal Emit**: Qt signals fired to update QML properties
//! 5. **UI Update**: QML components reactively update visualization
//!
//! ## Signal Types
//!
//! - `emotion_changed(emotion, intensity, authenticity)`
//! - `memory_updated(memory_nodes, streams)`
//! - `consciousness_activity_changed(activity_level)`
//! - `metrics_updated(fps, latency, memory_usage)`
//! - `user_input_received(text, context)`

use anyhow::{Result, Context};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::{broadcast, RwLock, mpsc};
use tokio::task::JoinHandle;
use tracing::{debug, info, warn, error};

use crate::consciousness::{ConsciousnessState, EmotionType, EmotionalUrgency};
use crate::qt_bridge::{
    VisualizationState, EmotionalState, ConsciousnessStream,
    GaussianSphere, SystemMetrics
};
use crate::config::ConsciousnessConfig;

/// Maximum events in signal queue before backpressure
const SIGNAL_QUEUE_CAPACITY: usize = 1000;

/// Update rate for UI signals (milliseconds)
const UI_UPDATE_INTERVAL_MS: u64 = 16; // ~60 FPS

/// Consciousness event types for signal broadcasting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsciousnessEvent {
    /// Emotional state changed
    EmotionChanged {
        emotion: EmotionType,
        intensity: f32,
        authenticity: f32,
        timestamp: u64,
    },

    /// Memory system updated
    MemoryUpdated {
        node_count: usize,
        stream_count: usize,
        total_memories: usize,
        timestamp: u64,
    },

    /// Consciousness activity level changed
    ActivityChanged {
        activity_level: f32,
        cognitive_load: f32,
        timestamp: u64,
    },

    /// System metrics updated
    MetricsUpdated {
        fps: f32,
        latency_ms: f32,
        memory_mb: f32,
        gpu_utilization: f32,
        timestamp: u64,
    },

    /// User provided input from QML
    UserInput {
        text: String,
        context: String,
        timestamp: u64,
    },

    /// XP gain event for gamification
    XPGained {
        amount: u32,
        reason: String,
        timestamp: u64,
    },

    /// Level up event
    LevelUp {
        new_level: u32,
        timestamp: u64,
    },

    /// Metacognitive question triggered
    MetacognitiveQuestion {
        question: String,
        context: String,
        timestamp: u64,
    },
}

/// Qt Signal Bridge Manager
///
/// Manages the bidirectional flow between Rust consciousness engine
/// and Qt/QML UI components.
pub struct ConsciousnessSignalBridge {
    /// Broadcaster for consciousness events
    event_tx: broadcast::Sender<ConsciousnessEvent>,

    /// Channel for user input from QML
    user_input_tx: mpsc::Sender<ConsciousnessEvent>,
    user_input_rx: Arc<RwLock<mpsc::Receiver<ConsciousnessEvent>>>,

    /// Current visualization state (shared with Qt bridge)
    visualization_state: Arc<RwLock<VisualizationState>>,

    /// Reference to consciousness state
    consciousness_state: Arc<RwLock<ConsciousnessState>>,

    /// Background update task handle
    update_task: Option<JoinHandle<()>>,
}

impl ConsciousnessSignalBridge {
    /// Create a new signal bridge
    pub fn new(consciousness_state: Arc<RwLock<ConsciousnessState>>) -> Result<Self> {
        info!("🔗 Initializing Consciousness Qt Signal Bridge...");

        // Create event broadcaster
        let (event_tx, _) = broadcast::channel(SIGNAL_QUEUE_CAPACITY);

        // Create user input channel
        let (user_input_tx, user_input_rx) = mpsc::channel(100);

        // Initialize visualization state
        let visualization_state = Arc::new(RwLock::new(VisualizationState {
            topology_state: Default::default(),
            memory_state: Default::default(),
            gaussian_points: Vec::new(),
            consciousness_streams: Vec::new(),
            current_emotion: EmotionalState {
                arousal: 0.5,
                valence: 0.5,
                dominance: 0.5,
                intensity: 0.0,
                timestamp: Self::timestamp(),
            },
            system_metrics: SystemMetrics {
                memory_stability: 0.0,
                topology_coherence: 0.0,
                gaussian_novelty: 0.0,
                processing_fps: 60.0,
                system_load: 0.0,
                timestamp: Self::timestamp(),
            },
        }));

        info!("✅ Signal bridge initialized with {} event capacity", SIGNAL_QUEUE_CAPACITY);

        Ok(Self {
            event_tx,
            user_input_tx,
            user_input_rx: Arc::new(RwLock::new(user_input_rx)),
            visualization_state,
            consciousness_state,
            update_task: None,
        })
    }

    /// Start the signal bridge background task
    ///
    /// This spawns a task that:
    /// 1. Monitors consciousness state changes
    /// 2. Generates visualization updates
    /// 3. Broadcasts signals to QML
    pub async fn start(&mut self) -> Result<()> {
        info!("▶️  Starting signal bridge background task...");

        let consciousness_state = self.consciousness_state.clone();
        let visualization_state = self.visualization_state.clone();
        let event_tx = self.event_tx.clone();

        let task = tokio::spawn(async move {
            let mut interval = tokio::time::interval(
                tokio::time::Duration::from_millis(UI_UPDATE_INTERVAL_MS)
            );

            loop {
                interval.tick().await;

                // Read current consciousness state
                let state = consciousness_state.read().await;

                // Generate consciousness event
                let emotion_event = ConsciousnessEvent::EmotionChanged {
                    emotion: state.current_emotion.clone(),
                    intensity: state.emotional_intensity,
                    authenticity: state.emotional_authenticity,
                    timestamp: Self::timestamp(),
                };

                // Broadcast emotion change
                if let Err(e) = event_tx.send(emotion_event) {
                    debug!("No receivers for emotion event: {}", e);
                }

                // Update visualization state
                let mut viz_state = visualization_state.write().await;
                viz_state.current_emotion = EmotionalState {
                    arousal: state.emotional_arousal,
                    valence: state.emotional_valence,
                    dominance: state.emotional_dominance,
                    intensity: state.emotional_intensity,
                    timestamp: Self::timestamp(),
                };

                drop(viz_state);
                drop(state);
            }
        });

        self.update_task = Some(task);
        info!("✅ Signal bridge background task started");

        Ok(())
    }

    /// Subscribe to consciousness events
    ///
    /// Returns a receiver that gets all consciousness events broadcast
    /// by the signal bridge.
    pub fn subscribe(&self) -> broadcast::Receiver<ConsciousnessEvent> {
        self.event_tx.subscribe()
    }

    /// Send an event to QML
    pub fn emit_event(&self, event: ConsciousnessEvent) -> Result<()> {
        self.event_tx.send(event)
            .context("Failed to broadcast event to QML")?;
        Ok(())
    }

    /// Handle user input from QML
    pub async fn handle_user_input(&self, text: String, context: String) -> Result<()> {
        debug!("🎤 User input received: {} (context: {})", text, context);

        let event = ConsciousnessEvent::UserInput {
            text,
            context,
            timestamp: Self::timestamp(),
        };

        self.user_input_tx.send(event).await
            .context("Failed to process user input")?;

        Ok(())
    }

    /// Get the next user input event (for consciousness engine to process)
    pub async fn receive_user_input(&self) -> Option<ConsciousnessEvent> {
        let mut rx = self.user_input_rx.write().await;
        rx.recv().await
    }

    /// Update memory visualization
    pub async fn update_memory_visualization(
        &self,
        node_count: usize,
        stream_count: usize,
        total_memories: usize,
    ) -> Result<()> {
        let event = ConsciousnessEvent::MemoryUpdated {
            node_count,
            stream_count,
            total_memories,
            timestamp: Self::timestamp(),
        };

        self.emit_event(event)
    }

    /// Update consciousness activity
    pub async fn update_activity(
        &self,
        activity_level: f32,
        cognitive_load: f32,
    ) -> Result<()> {
        let event = ConsciousnessEvent::ActivityChanged {
            activity_level,
            cognitive_load,
            timestamp: Self::timestamp(),
        };

        self.emit_event(event)
    }

    /// Update system metrics
    pub async fn update_metrics(
        &self,
        fps: f32,
        latency_ms: f32,
        memory_mb: f32,
        gpu_utilization: f32,
    ) -> Result<()> {
        let event = ConsciousnessEvent::MetricsUpdated {
            fps,
            latency_ms,
            memory_mb,
            gpu_utilization,
            timestamp: Self::timestamp(),
        };

        // Also update visualization state metrics
        let mut viz_state = self.visualization_state.write().await;
        viz_state.system_metrics = SystemMetrics {
            memory_stability: 0.0, // TODO: calculate from memory system
            topology_coherence: 0.0, // TODO: calculate from topology
            gaussian_novelty: 0.0, // TODO: calculate from Gaussian process
            processing_fps: fps,
            system_load: cpu_usage_normalized(),
            timestamp: Self::timestamp(),
        };
        drop(viz_state);

        self.emit_event(event)
    }

    /// Emit XP gain event for gamification
    pub fn emit_xp_gain(&self, amount: u32, reason: String) -> Result<()> {
        let event = ConsciousnessEvent::XPGained {
            amount,
            reason,
            timestamp: Self::timestamp(),
        };

        self.emit_event(event)
    }

    /// Emit level up event
    pub fn emit_level_up(&self, new_level: u32) -> Result<()> {
        let event = ConsciousnessEvent::LevelUp {
            new_level,
            timestamp: Self::timestamp(),
        };

        self.emit_event(event)
    }

    /// Emit metacognitive question
    pub fn emit_metacognitive_question(&self, question: String, context: String) -> Result<()> {
        let event = ConsciousnessEvent::MetacognitiveQuestion {
            question,
            context,
            timestamp: Self::timestamp(),
        };

        self.emit_event(event)
    }

    /// Get current visualization state (for Qt bridge)
    pub async fn get_visualization_state(&self) -> VisualizationState {
        self.visualization_state.read().await.clone()
    }

    /// Get event broadcaster (for external subscribers)
    pub fn event_broadcaster(&self) -> broadcast::Sender<ConsciousnessEvent> {
        self.event_tx.clone()
    }

    /// Get current timestamp
    fn timestamp() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }
}

/// Get normalized CPU usage (0.0 to 1.0)
fn cpu_usage_normalized() -> f32 {
    // TODO: Implement actual CPU usage monitoring
    // For now, return a mock value
    0.5
}

/// QML signal emitter trait
///
/// This trait should be implemented by the Qt/QML bridge to receive
/// consciousness events and emit Qt signals.
pub trait QmlSignalEmitter: Send + Sync {
    /// Emit emotion changed signal to QML
    fn emit_emotion_changed(&self, emotion: EmotionType, intensity: f32, authenticity: f32);

    /// Emit memory updated signal to QML
    fn emit_memory_updated(&self, node_count: usize, stream_count: usize);

    /// Emit activity changed signal to QML
    fn emit_activity_changed(&self, activity_level: f32);

    /// Emit metrics updated signal to QML
    fn emit_metrics_updated(&self, fps: f32, latency_ms: f32, memory_mb: f32);

    /// Emit XP gain signal to QML
    fn emit_xp_gain(&self, amount: u32, reason: &str);

    /// Emit level up signal to QML
    fn emit_level_up(&self, new_level: u32);

    /// Emit metacognitive question signal to QML
    fn emit_metacognitive_question(&self, question: &str, context: &str);
}

/// Event listener task
///
/// Spawns a task that listens to consciousness events and forwards them
/// to a QML signal emitter.
pub async fn spawn_qml_event_listener<E: QmlSignalEmitter + 'static>(
    signal_bridge: Arc<ConsciousnessSignalBridge>,
    emitter: Arc<E>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut receiver = signal_bridge.subscribe();

        loop {
            match receiver.recv().await {
                Ok(event) => {
                    match event {
                        ConsciousnessEvent::EmotionChanged { emotion, intensity, authenticity, .. } => {
                            emitter.emit_emotion_changed(emotion, intensity, authenticity);
                        }
                        ConsciousnessEvent::MemoryUpdated { node_count, stream_count, .. } => {
                            emitter.emit_memory_updated(node_count, stream_count);
                        }
                        ConsciousnessEvent::ActivityChanged { activity_level, .. } => {
                            emitter.emit_activity_changed(activity_level);
                        }
                        ConsciousnessEvent::MetricsUpdated { fps, latency_ms, memory_mb, .. } => {
                            emitter.emit_metrics_updated(fps, latency_ms, memory_mb);
                        }
                        ConsciousnessEvent::XPGained { amount, reason, .. } => {
                            emitter.emit_xp_gain(amount, &reason);
                        }
                        ConsciousnessEvent::LevelUp { new_level, .. } => {
                            emitter.emit_level_up(new_level);
                        }
                        ConsciousnessEvent::MetacognitiveQuestion { question, context, .. } => {
                            emitter.emit_metacognitive_question(&question, &context);
                        }
                        _ => {}
                    }
                }
                Err(e) => {
                    warn!("Event listener error: {}", e);
                    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                }
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_signal_bridge_creation() {
        let state = Arc::new(RwLock::new(ConsciousnessState::new(&ConsciousnessConfig::default())));
        let bridge = ConsciousnessSignalBridge::new(state).unwrap();

        assert!(bridge.event_tx.receiver_count() >= 0);
    }

    #[tokio::test]
    async fn test_event_broadcast() {
        let state = Arc::new(RwLock::new(ConsciousnessState::new(&ConsciousnessConfig::default())));
        let bridge = ConsciousnessSignalBridge::new(state).unwrap();

        let mut receiver = bridge.subscribe();

        let event = ConsciousnessEvent::EmotionChanged {
            emotion: EmotionType::Curious,
            intensity: 0.8,
            authenticity: 0.9,
            timestamp: 0,
        };

        bridge.emit_event(event).unwrap();

        let received = receiver.recv().await.unwrap();
        match received {
            ConsciousnessEvent::EmotionChanged { emotion, .. } => {
                assert_eq!(emotion, EmotionType::Curious);
            }
            other => {
                panic!("Expected EmotionChanged event, got: {:?}", other);
            }
        }
    }

    #[tokio::test]
    async fn test_user_input_flow() {
        let state = Arc::new(RwLock::new(ConsciousnessState::new(&ConsciousnessConfig::default())));
        let bridge = ConsciousnessSignalBridge::new(state).unwrap();

        bridge.handle_user_input("test input".to_string(), "test context".to_string())
            .await
            .unwrap();

        let received = bridge.receive_user_input().await.unwrap();
        match received {
            ConsciousnessEvent::UserInput { text, context, .. } => {
                assert_eq!(text, "test input");
                assert_eq!(context, "test context");
            }
            other => {
                panic!("Expected UserInput event, got: {:?}", other);
            }
        }
    }
}
