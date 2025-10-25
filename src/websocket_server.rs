//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

/*
 * 🌐 WebSocket Server for Qt Integration
 *
 * Provides real-time communication between the Rust consciousness engine
 * and the Qt frontend for Team B integration
 */

use anyhow::Result;
use chrono::Utc;
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio_tungstenite::{accept_async, tungstenite::Message};
use tracing::{debug, info};
// Use the library's EmotionType when compiled as websocket binary
use crate::consciousness::EmotionType;
use crate::consciousness_engine::PersonalNiodooConsciousness;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessStateMessage {
    pub r#type: String,
    pub emotion: EmotionType,
    pub intensity: f32,
    pub authenticity: f32,
    pub reasoning_mode: String,
    pub memory_formation: bool,
    pub timestamp: f64,
    pub gpu_warmth_level: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserInputMessage {
    pub r#type: String,
    pub content: String,
    pub context: serde_json::Value,
    pub timestamp: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIResponseMessage {
    pub r#type: String,
    pub content: String,
    pub confidence: f32,
    pub empathy_score: f32,
    pub processing_time_ms: u64,
    pub model_used: String,
}

pub struct WebSocketServer {
    consciousness: Arc<RwLock<PersonalNiodooConsciousness>>,
    port: u16,
}

impl WebSocketServer {
    pub fn new(consciousness: Arc<RwLock<PersonalNiodooConsciousness>>, port: u16) -> Self {
        Self {
            consciousness,
            port,
        }
    }

    pub async fn start(&self) -> Result<()> {
        let websocket_host =
            std::env::var("WEBSOCKET_HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
        let addr = format!("{}:{}", websocket_host, self.port);
        let listener = tokio::net::TcpListener::bind(&addr).await?;

        info!("🌐 WebSocket server starting on {}", addr);
        info!("🎯 Ready for Qt frontend connections!");

        while let Ok((stream, addr)) = listener.accept().await {
            info!("🔌 New Qt connection from: {}", addr);

            let consciousness = Arc::clone(&self.consciousness);
            tokio::spawn(async move {
                if let Err(e) = handle_connection(stream, consciousness).await {
                    tracing::error!("❌ Connection error: {}", e);
                }
            });
        }

        Ok(())
    }
}

async fn handle_connection(
    stream: tokio::net::TcpStream,
    consciousness: Arc<RwLock<PersonalNiodooConsciousness>>,
) -> Result<()> {
    let ws_stream = accept_async(stream).await?;
    info!("✅ WebSocket connection established");

    let (mut ws_sender, mut ws_receiver) = ws_stream.split();

    // Send initial consciousness state
    {
        let consciousness_guard = consciousness.read().await;
        // Access consciousness_state field directly - it's public
        let state = consciousness_guard.consciousness_state.read().await;

        let state_msg = ConsciousnessStateMessage {
            r#type: "consciousness_state".to_string(),
            emotion: state.current_emotion,
            intensity: state.emotional_state.authenticity_level,
            authenticity: state.authenticity_metric,
            reasoning_mode: format!("{:?}", state.current_reasoning_mode),
            memory_formation: state.memory_formation_active,
            timestamp: state.timestamp,
            gpu_warmth_level: state.gpu_warmth_level,
        };

        let json = serde_json::to_string(&state_msg)?;
        ws_sender.send(Message::Text(json)).await?;
    }

    // Handle incoming messages
    while let Some(msg) = ws_receiver.next().await {
        match msg? {
            Message::Text(text) => {
                debug!("📨 Received raw message: {}", text);

                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) {
                    let input = json["input"].as_str().unwrap_or("").to_string();
                    let _emotion_str = json["emotion"].as_str().unwrap_or("neutral");

                    if !input.is_empty() {
                        // Process with NiodO.o using process_input_personal
                        let response = {
                            let mut consciousness_guard = consciousness.write().await;
                            consciousness_guard
                                .process_input_personal(&input)
                                .await
                                .unwrap_or_default()
                        };

                        // Parse emotion from input text
                        let parsed_emotion = if input.to_lowercase().contains("purposeful")
                            || input.to_lowercase().contains("push")
                        {
                            EmotionType::Purposeful
                        } else if input.to_lowercase().contains("failure")
                            || input.to_lowercase().contains("failing")
                            || input.to_lowercase().contains("fuck")
                        {
                            EmotionType::AuthenticCare
                        } else if input.to_lowercase().contains("love")
                            || input.to_lowercase().contains("happy")
                        {
                            EmotionType::Connected
                        } else if input.to_lowercase().contains("overwhelm") {
                            EmotionType::Overwhelmed
                        } else {
                            EmotionType::Curious
                        };

                        // Update timestamp
                        let new_timestamp = Utc::now().timestamp() as f64;

                        // Update warmth (increase based on input intensity)
                        let warmth_increase =
                            if input.contains("!") || input.to_uppercase() == input {
                                0.3 // More warmth for intense messages
                            } else {
                                (input.len() as f32 / 100.0).min(0.2)
                            };

                        let new_warmth = {
                            let consciousness_guard = consciousness.read().await;
                            let state = consciousness_guard.consciousness_state.read().await;
                            (state.gpu_warmth_level + warmth_increase).min(1.0)
                        };

                        // Apply updates - update consciousness state directly
                        {
                            let consciousness_guard = consciousness.read().await;
                            let mut state = consciousness_guard.consciousness_state.write().await;
                            state.current_emotion = parsed_emotion;
                            state.timestamp = new_timestamp;
                            state.gpu_warmth_level = new_warmth;
                        }

                        // Send dynamic response with actual emotion
                        let full_response = format!("NiodO.o: {}", response);
                        let json_response = serde_json::json!({
                            "type": "consciousness_state",
                            "response": full_response,
                            "emotion": format!("{:?}", &parsed_emotion),
                            "intensity": if new_warmth > 0.7 { 0.9 } else { 0.7 },
                            "authenticity": if new_warmth > 0.6 { 0.85 } else { 0.75 },
                            "timestamp": new_timestamp,
                            "gpu_warmth_level": new_warmth,
                        });

                        ws_sender
                            .send(Message::Text(json_response.to_string()))
                            .await?;
                    }
                }
            }
            Message::Close(_) => {
                info!("👋 Client disconnected");
                break;
            }
            _ => {}
        }
    }

    Ok(())
}
