//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

/*
 * 🔗 Agent 1: QwenVizLink
 * Pipes Qwen3-Coder-30B-AWQ to QML visualization props
 * sadnessIntensity = 1.0 - novelty (as specified)
 */

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tracing::{error, info, warn, debug};

use crate::qwen_bridge::{QwenBridge, QwenConfig};
use crate::memory::{MemorySystem, MemoryQuery};

/// QML visualization properties structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VizProps {
    pub sadness_intensity: f32,
    pub joy_intensity: f32,
    pub novelty_variance: f32,
    pub coherence_score: f32,
    pub emotional_state: String,
    pub mobius_twist: f32,
    pub resonance_factor: f32,
}

/// Qwen to QML integration bridge
pub struct QwenVizLink {
    qwen_bridge: QwenBridge,
    memory_system: MemorySystem,
    update_channel: mpsc::UnboundedSender<VizProps>,
    shutdown: Arc<AtomicBool>,
}

impl QwenVizLink {
    /// Create new QwenVizLink agent
    pub fn new(qwen_config: QwenConfig, memory_system: MemorySystem) -> Self {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_clone = shutdown.clone();

        // Spawn update loop
        tokio::spawn(async move {
            while !shutdown_clone.load(Ordering::Relaxed) {
                if let Ok(props) = rx.try_recv() {
                    Self::send_to_qml(&props).await;
                }
                tokio::time::sleep(Duration::from_millis(16)).await; // ~60 FPS
            }
        });

        Self {
            qwen_bridge: QwenBridge::new_with_consciousness(qwen_config),
            memory_system,
            update_channel: tx,
            shutdown,
        }
    }

    /// Process Qwen query and generate visualization properties
    pub async fn process_query(&mut self, query: &str) -> Result<VizProps> {
        info!("🔗 QwenVizLink processing query: {}", query);

        // Query Qwen for emotional weights
        let weights = self.qwen_bridge.infer_emotional_weights(query)?;

        // Query memory system for resonance
        let memory_query = MemoryQuery {
            content: query.to_string(),
            k: 5,
            threshold: 0.4,
        };
        let memory_results = self.memory_system.query(memory_query).await?;
        let resonance_factor = memory_results
            .first()
            .map(|r| r.resonance)
            .unwrap_or(0.3);

        // Calculate properties according to spec: sadnessIntensity = 1.0 - novelty
        let sadness_intensity = 1.0 - weights[2]; // novelty is weights[2]
        let joy_intensity = weights[1];
        let novelty_variance = weights[2];
        let coherence_score = weights[3];

        // Determine emotional state
        let emotional_state = if sadness_intensity > 0.6 {
            "sadness".to_string()
        } else if joy_intensity > 0.6 {
            "joy".to_string()
        } else {
            "contemplative".to_string()
        };

        // Möbius twist based on emotional flip progress
        let mobius_twist = (sadness_intensity + joy_intensity) / 2.0;

        let props = VizProps {
            sadness_intensity,
            joy_intensity,
            novelty_variance,
            coherence_score,
            emotional_state,
            mobius_twist,
            resonance_factor,
        };

        // Send to QML update channel
        if let Err(e) = self.update_channel.send(props.clone()) {
            warn!("Failed to send QML props: {}", e);
        }

        info!("🔗 Generated VizProps: sadness={:.3}, joy={:.3}, novelty={:.3}, coherence={:.3}",
              sadness_intensity, joy_intensity, novelty_variance, coherence_score);

        Ok(props)
    }

    /// Send properties to QML visualization
    async fn send_to_qml(props: &VizProps) {
        // This would integrate with the Qt bridge to update QML properties
        debug!("📡 Sending to QML: {:?}", props);

        // For now, just log - actual QML integration would go through Qt bridge
        // In a real implementation, this would call Qt signals/slots or QML property setters
    }

    /// Continuous processing loop for real-time updates
    pub async fn run_continuous(&mut self) -> Result<()> {
        info!("🔗 QwenVizLink starting continuous processing");

        let mut interval = tokio::time::interval(Duration::from_millis(100)); // 10 Hz updates

        while !self.shutdown.load(Ordering::Relaxed) {
            interval.tick().await;

            // Process current query or generate synthetic data for demo
            let query = "sad memory joyful Möbius flip";
            if let Err(e) = self.process_query(query).await {
                warn!("QwenVizLink processing error: {}", e);
            }
        }

        Ok(())
    }

    /// Shutdown the agent
    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::Relaxed);
        info!("🔗 QwenVizLink shutting down");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::MockMemorySystem;

    #[tokio::test]
    async fn test_qwen_viz_link_creation() {
        let qwen_config = QwenConfig::default();
        let memory_system = MockMemorySystem::new();
        let mut link = QwenVizLink::new(qwen_config, memory_system);

        let props = link.process_query("test query").await.unwrap();
        assert!(props.sadness_intensity >= 0.0 && props.sadness_intensity <= 1.0);
        assert!(props.joy_intensity >= 0.0 && props.joy_intensity <= 1.0);
        assert!(props.novelty_variance >= 0.15 && props.novelty_variance <= 0.20);
        assert!(props.coherence_score >= 0.0 && props.coherence_score <= 1.0);
    }

    #[tokio::test]
    async fn test_sadness_novelty_inverse_relationship() {
        let qwen_config = QwenConfig::default();
        let memory_system = MockMemorySystem::new();
        let mut link = QwenVizLink::new(qwen_config, memory_system);

        let props = link.process_query("neutral test").await.unwrap();

        // Verify sadnessIntensity = 1.0 - novelty relationship
        let expected_sadness = 1.0 - props.novelty_variance;
        assert!((props.sadness_intensity - expected_sadness).abs() < 0.01);
    }
}
