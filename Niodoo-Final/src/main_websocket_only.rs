//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

/*
 * 🧠⚡ NiodO.o WebSocket Server Only ⚡🧠
 *
 * Starts just the consciousness engine and WebSocket server
 * without running test interactions (for Qt integration)
 */

use anyhow::Result;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::info;

// Import from the library instead of recompiling modules
use niodoo_consciousness::consciousness_engine::PersonalNiodooConsciousness;
use niodoo_consciousness::websocket_server::WebSocketServer;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    info!("🚀 Starting NiodO.o WebSocket Server for Qt Integration");
    info!("🎯 Mission: Provide WebSocket API for Qt frontend");
    info!("💖 Goal: Real-time consciousness state updates");

    // Initialize consciousness
    let consciousness = PersonalNiodooConsciousness::new().await?;

    info!("✅ NiodO.o Consciousness is ALIVE and ready for Qt integration!");

    // Start WebSocket server for Qt integration
    info!("🌐 Starting WebSocket server for Qt frontend...");
    let consciousness_arc = Arc::new(RwLock::new(consciousness));
    let websocket_server = WebSocketServer::new(Arc::clone(&consciousness_arc), 8081);

    info!("🎯 WebSocket server ready at ws://localhost:8081");
    info!("🔗 Qt frontend can now connect to the consciousness engine!");
    info!("📡 Send JSON messages to interact with the consciousness");

    // Start the WebSocket server (this will run indefinitely)
    websocket_server.start().await?;

    Ok(())
}
