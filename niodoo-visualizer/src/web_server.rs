//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

//! Web-based telemetry visualizer server
//!
//! Serves an HTML page with Three.js 3D visualization that connects to
//! NIODOO telemetry stream. Perfect for SSH/RunPod scenarios.

use axum::{
    extract::ws::{Message, WebSocket, WebSocketUpgrade},
    extract::State,
    response::{Html, Response},
    routing::get,
    Router,
};
use serde::{Deserialize, Serialize};
use tokio::net::TcpStream;
use tokio::sync::broadcast;
use tokio_util::codec::{FramedRead, LinesCodec};
use futures::StreamExt;
use tower::ServiceBuilder;
use tower_http::cors::CorsLayer;

#[derive(Serialize, Deserialize, Debug, Clone)]
struct CognitiveStatePacket {
    pad_state: [f32; 3],
    torus_projection: [f32; 3],
    betti_numbers: (usize, usize, usize),
    persistence_entropy: f64,
    compass_quadrant: String,
    compass_confidence: f32,
    retrieved_memory_ids: Vec<String>,
    iteration: Option<u64>,
    prompt_text: Option<String>,
    timestamp: String,
}

pub async fn start_web_server(port: u16) -> anyhow::Result<()> {
    let (tx, _rx) = broadcast::channel::<CognitiveStatePacket>(1000);
    
    // Spawn telemetry client task
    let tx_clone = tx.clone();
    tokio::spawn(async move {
        telemetry_client_task(tx_clone).await;
    });

    let app = Router::new()
        .route("/", get(index_handler))
        .route("/ws", get(websocket_handler))
        .layer(
            ServiceBuilder::new()
                .layer(CorsLayer::permissive())
                .into_inner(),
        )
        .with_state(tx);

    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{}", port)).await?;
    println!("🌐 Mind's Eye Web Visualizer running on http://0.0.0.0:{}", port);
    println!("   Open this URL in your browser (use port forwarding if SSH'd in)");
    
    axum::serve(listener, app).await?;
    Ok(())
}

async fn telemetry_client_task(tx: broadcast::Sender<CognitiveStatePacket>) {
    loop {
        match TcpStream::connect("127.0.0.1:9999").await {
            Ok(stream) => {
                println!("✅ Connected to NIODOO telemetry server");
                let mut reader = FramedRead::new(stream, LinesCodec::new());

                while let Some(Ok(line)) = reader.next().await {
                    if let Ok(packet) = serde_json::from_str::<CognitiveStatePacket>(&line) {
                        let _ = tx.send(packet);
                    }
                }
                println!("⚠️  Connection lost, reconnecting...");
            }
            Err(e) => {
                eprintln!("❌ Failed to connect: {}, retrying in 2s...", e);
                tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
            }
        }
    }
}

async fn index_handler() -> Html<&'static str> {
    Html(include_str!("visualization.html"))
}

async fn websocket_handler(
    ws: WebSocketUpgrade,
    State(tx): State<broadcast::Sender<CognitiveStatePacket>>,
) -> Response {
    ws.on_upgrade(|socket| handle_websocket(socket, tx))
}

async fn handle_websocket(socket: WebSocket, tx: broadcast::Sender<CognitiveStatePacket>) {
    let mut rx = tx.subscribe();
    
    let (mut sender, mut receiver) = socket.split();
    
    // Spawn task to send telemetry packets
    let send_task = tokio::spawn(async move {
        while let Ok(packet) = rx.recv().await {
            let json = serde_json::to_string(&packet).unwrap();
            if sender.send(Message::Text(json)).await.is_err() {
                break;
            }
        }
    });
    
    // Spawn task to receive messages (for ping/pong)
    let recv_task = tokio::spawn(async move {
        while let Some(Ok(_msg)) = receiver.next().await {
            // Just keep connection alive
        }
    });
    
    tokio::select! {
        _ = send_task => recv_task.abort(),
        _ = recv_task => send_task.abort(),
    };
}

