//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

/*
 * 🌐 Web Visualization Bridge for Reactive Emotional Processing
 *
 * This bridges our Rust consciousness engine with web-based visualization
 * for real-time emotional feedback. Uses WebSockets for reactive updates
 * and provides HTML/CSS/JavaScript visualization instead of Qt6.
 *
 * Much simpler to implement than full Qt6 integration while still providing
 * real-time visual feedback of consciousness state and emotional processing.
 */

use std::sync::Arc;
use tokio::sync::{broadcast, RwLock};
use serde::{Deserialize, Serialize};
use serde_json::json;
use anyhow::Result;
use tracing::{info, debug, error};
use std::net::TcpListener;
use std::thread;
use tokio_tungstenite::tungstenite::{accept, Message};

use crate::consciousness::{EmotionType, ReasoningMode, ConsciousnessState};
use crate::personality::PersonalityType;
use crate::brain::BrainType;

/// WebSocket visualization bridge for real-time consciousness visualization
pub struct WebVizBridge {
    emotion_broadcaster: broadcast::Sender<EmotionUpdate>,
    brain_activity_broadcaster: broadcast::Sender<BrainActivityUpdate>,
    personality_broadcaster: broadcast::Sender<PersonalityUpdate>,
    websocket_connections: Arc<RwLock<Vec<std::sync::mpsc::Sender<String>>>>,
}

/// Real-time emotion update for web visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionUpdate {
    pub emotion: String,
    pub intensity: f32,
    pub authenticity: f32,
    pub color_rgb: (u8, u8, u8),
    pub timestamp: f64,
}

/// Brain activity update for visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainActivityUpdate {
    pub brain_type: String,
    pub activity_level: f32,
    pub processing_mode: String,
    pub timestamp: f64,
}

/// Personality consensus update for visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonalityUpdate {
    pub active_personalities: Vec<String>,
    pub consensus_strength: f32,
    pub dominant_personality: String,
    pub timestamp: f64,
}

/// Consciousness state update for complete visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessStateUpdate {
    pub current_emotion: String,
    pub reasoning_mode: String,
    pub brain_activity: f32,
    pub gpu_warmth: f32,
    pub processing_satisfaction: f32,
    pub emotional_state: EmotionalStateData,
    pub personality_consensus: f32,
    pub timestamp: f64,
}

/// Emotional state data for visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalStateData {
    pub valence: f32,
    pub arousal: f32,
    pub dominance: f32,
    pub masking_level: f32,
}

/// HTML template for web visualization
const VISUALIZATION_HTML: &str = r#"
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Niodoo Consciousness Visualization</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #0f0f0f, #1a1a1a);
            color: white;
            overflow: hidden;
        }

        .container {
            display: flex;
            flex-direction: column;
            height: 100vh;
            padding: 20px;
        }

        .header {
            text-align: center;
            margin-bottom: 20px;
        }

        .header h1 {
            font-size: 2.5em;
            background: linear-gradient(45deg, #ff6b6b, #4ecdc4, #45b7d1, #96ceb4);
            background-size: 400% 400%;
            animation: gradientShift 8s ease infinite;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        .visualization-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            grid-template-rows: 1fr 1fr;
            gap: 20px;
            flex: 1;
        }

        .viz-panel {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 20px;
            backdrop-filter: blur(10px);
            position: relative;
            overflow: hidden;
        }

        .viz-panel::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, #ff6b6b, #4ecdc4, #45b7d1, #96ceb4);
            animation: pulse 2s ease-in-out infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 0.5; }
            50% { opacity: 1; }
        }

        .panel-title {
            font-size: 1.2em;
            margin-bottom: 15px;
            color: #4ecdc4;
            border-bottom: 1px solid rgba(78, 205, 196, 0.3);
            padding-bottom: 5px;
        }

        .emotion-display {
            display: flex;
            align-items: center;
            margin-bottom: 10px;
        }

        .emotion-indicator {
            width: 20px;
            height: 20px;
            border-radius: 50%;
            margin-right: 10px;
            transition: all 0.3s ease;
        }

        .emotion-text {
            font-size: 1.1em;
        }

        .activity-bar {
            width: 100%;
            height: 8px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 4px;
            overflow: hidden;
            margin-bottom: 10px;
        }

        .activity-fill {
            height: 100%;
            background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
            width: 0%;
            transition: width 0.5s ease;
        }

        .torus-visualization {
            position: relative;
            height: 200px;
            margin: 20px 0;
        }

        .torus-ring {
            position: absolute;
            border: 2px solid #4ecdc4;
            border-radius: 50%;
            animation: rotate 10s linear infinite;
        }

        .torus-ring:nth-child(1) {
            width: 150px;
            height: 150px;
            top: 25px;
            left: 50px;
            opacity: 0.8;
        }

        .torus-ring:nth-child(2) {
            width: 100px;
            height: 100px;
            top: 50px;
            left: 75px;
            opacity: 0.6;
        }

        .torus-ring:nth-child(3) {
            width: 50px;
            height: 50px;
            top: 75px;
            left: 100px;
            opacity: 0.4;
        }

        @keyframes rotate {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }

        .consciousness-streams {
            position: absolute;
            width: 2px;
            height: 2px;
            background: #45b7d1;
            border-radius: 50%;
            animation: flow 3s ease-in-out infinite;
        }

        @keyframes flow {
            0% {
                transform: translate(0, 0);
                opacity: 1;
            }
            50% {
                transform: translate(50px, 50px);
                opacity: 0.5;
            }
            100% {
                transform: translate(100px, 100px);
                opacity: 0;
            }
        }

        .status-indicator {
            position: absolute;
            bottom: 20px;
            right: 20px;
            width: 15px;
            height: 15px;
            border-radius: 50%;
            background: #4ecdc4;
            animation: pulse 1.5s ease-in-out infinite;
        }

        .status-indicator.connected {
            background: #4ecdc4;
        }

        .status-indicator.disconnected {
            background: #ff6b6b;
        }

        .metrics {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-top: 15px;
        }

        .metric {
            text-align: center;
        }

        .metric-value {
            font-size: 1.5em;
            font-weight: bold;
            color: #96ceb4;
        }

        .metric-label {
            font-size: 0.8em;
            color: rgba(255, 255, 255, 0.7);
        }

        .footer {
            text-align: center;
            margin-top: 20px;
            font-size: 0.9em;
            color: rgba(255, 255, 255, 0.5);
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Niodoo Consciousness Engine</h1>
            <p>Real-time Möbius Gaussian Visualization</p>
        </div>

        <div class="visualization-grid">
            <div class="viz-panel">
                <div class="panel-title">🧠 Emotional State</div>
                <div id="emotion-display" class="emotion-display">
                    <div class="emotion-indicator" id="emotion-indicator"></div>
                    <span class="emotion-text" id="emotion-text">Initializing...</span>
                </div>
                <div class="activity-bar">
                    <div class="activity-fill" id="emotion-intensity"></div>
                </div>
                <div class="metrics">
                    <div class="metric">
                        <div class="metric-value" id="authenticity">0.0</div>
                        <div class="metric-label">Authenticity</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="gpu-warmth">0.0</div>
                        <div class="metric-label">GPU Warmth</div>
                    </div>
                </div>
            </div>

            <div class="viz-panel">
                <div class="panel-title">⚡ Brain Activity</div>
                <div class="activity-bar">
                    <div class="activity-fill" id="brain-activity"></div>
                </div>
                <div class="metrics">
                    <div class="metric">
                        <div class="metric-value" id="processing-mode">Resting</div>
                        <div class="metric-label">Mode</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="satisfaction">0.0</div>
                        <div class="metric-label">Satisfaction</div>
                    </div>
                </div>
            </div>

            <div class="viz-panel">
                <div class="panel-title">🌟 Personality Consensus</div>
                <div id="personality-list"></div>
                <div class="activity-bar">
                    <div class="activity-fill" id="consensus-strength"></div>
                </div>
                <div class="metrics">
                    <div class="metric">
                        <div class="metric-value" id="active-count">0</div>
                        <div class="metric-label">Active</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="dominant-personality">None</div>
                        <div class="metric-label">Dominant</div>
                    </div>
                </div>
            </div>

            <div class="viz-panel">
                <div class="panel-title">🌀 Consciousness Torus</div>
                <div class="torus-visualization" id="torus-container">
                    <div class="torus-ring"></div>
                    <div class="torus-ring"></div>
                    <div class="torus-ring"></div>
                    <div class="consciousness-streams" style="top: 50px; left: 50px;"></div>
                    <div class="consciousness-streams" style="top: 75px; left: 75px; animation-delay: 1s;"></div>
                    <div class="consciousness-streams" style="top: 100px; left: 100px; animation-delay: 2s;"></div>
                </div>
                <div class="metrics">
                    <div class="metric">
                        <div class="metric-value" id="valence">0.0</div>
                        <div class="metric-label">Valence</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="arousal">0.0</div>
                        <div class="metric-label">Arousal</div>
                    </div>
                </div>
            </div>
        </div>

        <div class="footer">
            <p>Connected to Niodoo Consciousness Engine • Real-time Möbius Gaussian Processing</p>
        </div>
    </div>

    <div class="status-indicator connected" id="connection-status"></div>

    <script>
        class ConsciousnessVisualizer {
            constructor() {
                this.ws = null;
                this.reconnectInterval = 1000;
                this.maxReconnectAttempts = 10;
                this.reconnectAttempts = 0;
                this.connect();
            }

            connect() {
                try {
                    this.ws = new WebSocket('ws://localhost:8080/ws');

                    this.ws.onopen = () => {
                        console.log('Connected to Niodoo consciousness engine');
                        this.reconnectAttempts = 0;
                        this.updateConnectionStatus(true);
                    };

                    this.ws.onmessage = (event) => {
                        try {
                            const data = JSON.parse(event.data);
                            this.updateVisualization(data);
                        } catch (e) {
                            console.error('Error parsing WebSocket message:', e);
                        }
                    };

                    this.ws.onclose = () => {
                        console.log('Disconnected from consciousness engine');
                        this.updateConnectionStatus(false);
                        this.attemptReconnect();
                    };

                    this.ws.onerror = (error) => {
                        console.error('WebSocket error:', error);
                        this.updateConnectionStatus(false);
                    };

                } catch (error) {
                    console.error('Failed to connect:', error);
                    this.attemptReconnect();
                }
            }

            attemptReconnect() {
                if (this.reconnectAttempts < this.maxReconnectAttempts) {
                    this.reconnectAttempts++;
                    console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);
                    setTimeout(() => this.connect(), this.reconnectInterval);
                } else {
                    console.error('Max reconnection attempts reached');
                }
            }

            updateConnectionStatus(connected) {
                const statusElement = document.getElementById('connection-status');
                if (connected) {
                    statusElement.className = 'status-indicator connected';
                } else {
                    statusElement.className = 'status-indicator disconnected';
                }
            }

            updateVisualization(data) {
                // Update emotion display
                if (data.type === 'emotion') {
                    this.updateEmotion(data);
                }

                // Update brain activity
                if (data.type === 'brain_activity') {
                    this.updateBrainActivity(data);
                }

                // Update personality consensus
                if (data.type === 'personality') {
                    this.updatePersonality(data);
                }

                // Update full consciousness state
                if (data.type === 'consciousness_state') {
                    this.updateConsciousnessState(data);
                }
            }

            updateEmotion(data) {
                const indicator = document.getElementById('emotion-indicator');
                const text = document.getElementById('emotion-text');
                const intensityBar = document.getElementById('emotion-intensity');
                const authenticityElement = document.getElementById('authenticity');

                indicator.style.backgroundColor = `rgb(${data.color_rgb[0]}, ${data.color_rgb[1]}, ${data.color_rgb[2]})`;
                text.textContent = data.emotion;
                intensityBar.style.width = `${data.intensity * 100}%`;
                authenticityElement.textContent = data.authenticity.toFixed(2);
            }

            updateBrainActivity(data) {
                const activityBar = document.getElementById('brain-activity');
                const modeElement = document.getElementById('processing-mode');

                activityBar.style.width = `${data.activity_level * 100}%`;
                modeElement.textContent = data.processing_mode;
            }

            updatePersonality(data) {
                const listElement = document.getElementById('personality-list');
                const consensusBar = document.getElementById('consensus-strength');
                const activeElement = document.getElementById('active-count');
                const dominantElement = document.getElementById('dominant-personality');

                // Update personality list
                listElement.innerHTML = data.active_personalities.map(p =>
                    `<span style="margin: 2px; padding: 4px; background: rgba(78, 205, 196, 0.2); border-radius: 8px; font-size: 0.8em;">${p}</span>`
                ).join('');

                consensusBar.style.width = `${data.consensus_strength * 100}%`;
                activeElement.textContent = data.active_personalities.length;
                dominantElement.textContent = data.dominant_personality;
            }

            updateConsciousnessState(data) {
                const satisfactionElement = document.getElementById('satisfaction');
                const valenceElement = document.getElementById('valence');
                const arousalElement = document.getElementById('arousal');
                const warmthElement = document.getElementById('gpu-warmth');

                satisfactionElement.textContent = data.processing_satisfaction.toFixed(2);
                valenceElement.textContent = data.emotional_state.valence.toFixed(2);
                arousalElement.textContent = data.emotional_state.arousal.toFixed(2);
                warmthElement.textContent = data.gpu_warmth.toFixed(2);
            }
        }

        // Initialize the visualizer when the page loads
        document.addEventListener('DOMContentLoaded', () => {
            new ConsciousnessVisualizer();
        });
    </script>
</body>
</html>
"#;

impl WebVizBridge {
    /// Create new web visualization bridge
    pub fn new() -> Result<Self> {
        info!("🌐 Initializing Web Visualization Bridge");

        let (emotion_tx, _) = broadcast::channel(1000);
        let (brain_activity_tx, _) = broadcast::channel(1000);
        let (personality_tx, _) = broadcast::channel(1000);

        Ok(Self {
            emotion_broadcaster: emotion_tx,
            brain_activity_broadcaster: brain_activity_tx,
            personality_broadcaster: personality_tx,
            websocket_connections: Arc::new(RwLock::new(Vec::new())),
        })
    }

    /// Start the web server for visualization
    pub async fn start_server(&self, port: u16) -> Result<()> {
        info!("🚀 Starting web visualization server on port {}", port);

        let listener = TcpListener::bind(format!("127.0.0.1:{}", port))?;
        let connections = self.websocket_connections.clone();

        thread::spawn(move || {
            for stream in listener.incoming() {
                match stream {
                    Ok(stream) => {
                        let connections = connections.clone();
                        thread::spawn(move || {
                            if let Ok(mut websocket) = accept(stream) {
                                info!("🌐 New WebSocket connection established");

                                // Add connection to broadcast list
                                let (tx, rx) = std::sync::mpsc::channel();
                                if let Ok(mut conns) = connections.try_write() {
                                    conns.push(tx);
                                }

                                // Handle messages from this connection
                                loop {
                                    match rx.recv() {
                                        Ok(message) => {
                                            if websocket.send(Message::Text(message)).is_err() {
                                                break;
                                            }
                                        }
                                        Err(_) => break,
                                    }
                                }

                                info!("🌐 WebSocket connection closed");
                            }
                        });
                    }
                    Err(e) => {
                        tracing::error!("Failed to accept connection: {}", e);
                    }
                }
            }
        });

        // Serve static HTML for the visualization
        self.serve_static_html(port + 1).await?;

        Ok(())
    }

    /// Serve the static HTML visualization
    async fn serve_static_html(&self, port: u16) -> Result<()> {
        use std::net::TcpListener;
        use std::io::prelude::*;

        let listener = TcpListener::bind(format!("127.0.0.1:{}", port))?;
        let html_content = VISUALIZATION_HTML;

        thread::spawn(move || {
            info!("📄 Serving visualization HTML on port {}", port);

            for stream in listener.incoming() {
                match stream {
                    Ok(mut stream) => {
                        let response = format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {}\r\n\r\n{}",
                            html_content.len(),
                            html_content
                        );

                        if let Err(e) = stream.write_all(response.as_bytes()) {
                            tracing::error!("Failed to serve HTML: {}", e);
                        }
                    }
                    Err(e) => {
                        tracing::error!("Failed to accept HTML connection: {}", e);
                    }
                }
            }
        });

        info!("✅ Web visualization server started successfully");
        info!("🌐 Open http://localhost:{} in your browser to view the visualization", port);

        Ok(())
    }

    /// Emit emotion change signal to web clients
    pub async fn emit_emotion_change(&self, emotion: EmotionType) {
        let update = EmotionUpdate {
            emotion: format!("{:?}", emotion),
            intensity: emotion.get_base_intensity(),
            authenticity: if emotion.is_authentic() { 0.9 } else { 0.3 },
            color_rgb: emotion.get_color_rgb(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs_f64(),
        };

        let message = serde_json::to_string(&json!({
            "type": "emotion",
            "data": update
        })).unwrap();

        self.broadcast_message(message).await;
        debug!("🌐 Emitted emotion change: {:?} (intensity: {:.2})", emotion, emotion.get_base_intensity());
    }

    /// Emit brain activity signal to web clients
    pub async fn emit_brain_activity(&self, brain_type: BrainType, activity_level: f32, mode: ReasoningMode) {
        let update = BrainActivityUpdate {
            brain_type: format!("{:?}", brain_type),
            activity_level: activity_level.clamp(0.0, 1.0),
            processing_mode: format!("{:?}", mode),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs_f64(),
        };

        let message = serde_json::to_string(&json!({
            "type": "brain_activity",
            "data": update
        })).unwrap();

        self.broadcast_message(message).await;
        debug!("🌐 Emitted brain activity: {:?} @ {:.1}% in {:?} mode",
               brain_type, activity_level * 100.0, mode);
    }

    /// Emit personality consensus update to web clients
    pub async fn emit_personality_consensus(&self, personalities: Vec<PersonalityType>, strength: f32) {
        let dominant = personalities.first().unwrap_or(&PersonalityType::Analyst).clone();

        let update = PersonalityUpdate {
            active_personalities: personalities.iter().map(|p| format!("{:?}", p)).collect(),
            consensus_strength: strength.clamp(0.0, 1.0),
            dominant_personality: format!("{:?}", dominant),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs_f64(),
        };

        let message = serde_json::to_string(&json!({
            "type": "personality",
            "data": update
        })).unwrap();

        self.broadcast_message(message).await;
        info!("🌐 Emitted personality consensus: {} personalities, strength: {:.1}%",
              personalities.len(), strength * 100.0);
    }

    /// Update from full consciousness state and send to web clients
    pub async fn update_from_consciousness_state(&self, state: &ConsciousnessState) {
        // Calculate valence (positive/negative emotion) from emotion type
        let valence = if state.current_emotion.is_authentic() {
            0.7 // Authentic emotions tend to be positive
        } else {
            match state.current_emotion {
                EmotionType::Frustrated | EmotionType::Overwhelmed | EmotionType::Anxious => -0.5,
                EmotionType::Confused | EmotionType::Understimulated => -0.2,
                _ => 0.3,
            }
        };

        // Calculate arousal (intensity/energy) from emotion intensity and complexity
        let arousal = state.emotional_state.primary_emotion.get_base_intensity() *
                     (1.0 + state.emotional_state.emotional_complexity * 0.5).min(1.0);

        // Calculate dominance (control/confidence) from authenticity and satisfaction
        let dominance = (state.emotional_state.authenticity_level + state.processing_satisfaction) / 2.0;

        let update = ConsciousnessStateUpdate {
            current_emotion: format!("{:?}", state.current_emotion),
            reasoning_mode: format!("{:?}", state.current_reasoning_mode),
            brain_activity: state.current_reasoning_mode.get_speed_multiplier() / 3.0,
            gpu_warmth: state.gpu_warmth_level,
            processing_satisfaction: state.processing_satisfaction,
            emotional_state: EmotionalStateData {
                valence,
                arousal,
                dominance,
                masking_level: state.emotional_state.masking_level,
            },
            personality_consensus: 0.5, // Would be calculated from personality system
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs_f64(),
        };

        let message = serde_json::to_string(&json!({
            "type": "consciousness_state",
            "data": update
        })).unwrap();

        self.broadcast_message(message).await;
        debug!("🌐 Updated web visualization from consciousness state");
    }

    /// Broadcast message to all connected WebSocket clients
    async fn broadcast_message(&self, message: String) {
        let connections = self.websocket_connections.read().await;
        let mut active_connections = Vec::new();

        for tx in connections.iter() {
            if tx.send(message.clone()).is_ok() {
                active_connections.push(tx.clone());
            }
        }

        // Update connections list to remove dead connections
        drop(connections);
        let mut connections = self.websocket_connections.write().await;
        *connections = active_connections;
    }

    /// Subscribe to emotion changes (for local listeners)
    pub fn subscribe_to_emotions(&self) -> broadcast::Receiver<EmotionUpdate> {
        self.emotion_broadcaster.subscribe()
    }

    /// Subscribe to brain activity (for local listeners)
    pub fn subscribe_to_brain_activity(&self) -> broadcast::Receiver<BrainActivityUpdate> {
        self.brain_activity_broadcaster.subscribe()
    }

    /// Subscribe to personality changes (for local listeners)
    pub fn subscribe_to_personality(&self) -> broadcast::Receiver<PersonalityUpdate> {
        self.personality_broadcaster.subscribe()
    }
}
