//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

//! Web-based telemetry visualizer server
//!
//! Serves an HTML page with Three.js 3D visualization that connects to
//! NIODOO telemetry stream. Perfect for SSH/RunPod scenarios.

use axum::extract::ws::WebSocket;
use axum::{
    extract::ws::{Message, WebSocketUpgrade},
    extract::State,
    response::{Html, Response},
    routing::get,
    Router,
};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use tokio::net::TcpStream;
use tokio::sync::broadcast;
use tokio_util::codec::{FramedRead, LinesCodec};
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
    println!(
        "🌐 Mind's Eye Web Visualizer running on http://0.0.0.0:{}",
        port
    );
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

                while let Some(result) = reader.next().await {
                    match result {
                        Ok(line) => {
                            if line.trim().is_empty() {
                                continue;
                            }
                            match serde_json::from_str::<CognitiveStatePacket>(&line) {
                                Ok(packet) => {
                                    let _ = tx.send(packet);
                                }
                                Err(e) => {
                                    eprintln!("⚠️  Failed to parse legacy packet: {} (first 200 chars: {})", e, line.chars().take(200).collect::<String>());
                                    // Try to parse as enhanced packet and convert
                                    if let Ok(enhanced) =
                                        serde_json::from_str::<serde_json::Value>(&line)
                                    {
                                        // Check if it has enhanced fields but try to extract legacy fields
                                        if let (
                                            Some(pad_state),
                                            Some(torus_projection),
                                            Some(betti_numbers),
                                            Some(persistence_entropy),
                                            Some(compass_quadrant),
                                            Some(compass_confidence),
                                            Some(timestamp),
                                        ) = (
                                            enhanced.get("pad_state").and_then(|v| v.as_array()),
                                            enhanced
                                                .get("torus_projection")
                                                .and_then(|v| v.as_array()),
                                            enhanced.get("betti_numbers"),
                                            enhanced
                                                .get("persistence_entropy")
                                                .and_then(|v| v.as_f64()),
                                            enhanced
                                                .get("compass_quadrant")
                                                .and_then(|v| v.as_str()),
                                            enhanced
                                                .get("compass_confidence")
                                                .and_then(|v| v.as_f64()),
                                            enhanced.get("timestamp").and_then(|v| v.as_str()),
                                        ) {
                                            // Try to extract memory IDs
                                            let memory_ids = if let Some(mem_retrieval) =
                                                enhanced.get("memory_retrieval")
                                            {
                                                if let Some(memories) = mem_retrieval
                                                    .get("retrieved_memories")
                                                    .and_then(|v| v.as_array())
                                                {
                                                    memories
                                                        .iter()
                                                        .filter_map(|m| {
                                                            m.get("memory_id")
                                                                .and_then(|v| v.as_str())
                                                                .map(|s| s.to_string())
                                                        })
                                                        .collect()
                                                } else {
                                                    Vec::new()
                                                }
                                            } else {
                                                enhanced
                                                    .get("retrieved_memory_ids")
                                                    .and_then(|v| v.as_array())
                                                    .map(|arr| {
                                                        arr.iter()
                                                            .filter_map(|v| {
                                                                v.as_str().map(|s| s.to_string())
                                                            })
                                                            .collect()
                                                    })
                                                    .unwrap_or_default()
                                            };

                                            // Extract iteration
                                            let iteration = enhanced
                                                .get("iteration")
                                                .and_then(|v| v.as_u64())
                                                .or_else(|| {
                                                    enhanced
                                                        .get("iteration")
                                                        .and_then(|v| v.as_i64())
                                                        .map(|i| i as u64)
                                                });

                                            // Extract prompt text
                                            let prompt_text = enhanced
                                                .get("prompt")
                                                .and_then(|p| {
                                                    p.get("full_text")
                                                        .and_then(|v| v.as_str())
                                                        .map(|s| s.to_string())
                                                })
                                                .or_else(|| {
                                                    enhanced
                                                        .get("prompt_text")
                                                        .and_then(|v| v.as_str())
                                                        .map(|s| s.to_string())
                                                });

                                            // Convert arrays to proper types
                                            if pad_state.len() >= 3 && torus_projection.len() >= 3 {
                                                let pad: [f32; 3] = [
                                                    pad_state[0].as_f64().unwrap_or(0.0) as f32,
                                                    pad_state[1].as_f64().unwrap_or(0.0) as f32,
                                                    pad_state[2].as_f64().unwrap_or(0.0) as f32,
                                                ];
                                                let torus: [f32; 3] = [
                                                    torus_projection[0].as_f64().unwrap_or(0.0)
                                                        as f32,
                                                    torus_projection[1].as_f64().unwrap_or(0.0)
                                                        as f32,
                                                    torus_projection[2].as_f64().unwrap_or(0.0)
                                                        as f32,
                                                ];

                                                // Extract betti numbers (can be array or tuple format)
                                                let betti = if let Some(betti_arr) =
                                                    betti_numbers.as_array()
                                                {
                                                    if betti_arr.len() >= 3 {
                                                        (
                                                            betti_arr[0]
                                                                .as_u64()
                                                                .or_else(|| {
                                                                    betti_arr[0]
                                                                        .as_i64()
                                                                        .map(|i| i as u64)
                                                                })
                                                                .unwrap_or(0)
                                                                as usize,
                                                            betti_arr[1]
                                                                .as_u64()
                                                                .or_else(|| {
                                                                    betti_arr[1]
                                                                        .as_i64()
                                                                        .map(|i| i as u64)
                                                                })
                                                                .unwrap_or(0)
                                                                as usize,
                                                            betti_arr[2]
                                                                .as_u64()
                                                                .or_else(|| {
                                                                    betti_arr[2]
                                                                        .as_i64()
                                                                        .map(|i| i as u64)
                                                                })
                                                                .unwrap_or(0)
                                                                as usize,
                                                        )
                                                    } else {
                                                        (0, 0, 0)
                                                    }
                                                } else if let Some(betti_obj) =
                                                    betti_numbers.as_object()
                                                {
                                                    (
                                                        betti_obj
                                                            .get("0")
                                                            .or_else(|| betti_obj.get("b0"))
                                                            .and_then(|v| {
                                                                v.as_u64().or_else(|| {
                                                                    v.as_i64().map(|i| i as u64)
                                                                })
                                                            })
                                                            .unwrap_or(0)
                                                            as usize,
                                                        betti_obj
                                                            .get("1")
                                                            .or_else(|| betti_obj.get("b1"))
                                                            .and_then(|v| {
                                                                v.as_u64().or_else(|| {
                                                                    v.as_i64().map(|i| i as u64)
                                                                })
                                                            })
                                                            .unwrap_or(0)
                                                            as usize,
                                                        betti_obj
                                                            .get("2")
                                                            .or_else(|| betti_obj.get("b2"))
                                                            .and_then(|v| {
                                                                v.as_u64().or_else(|| {
                                                                    v.as_i64().map(|i| i as u64)
                                                                })
                                                            })
                                                            .unwrap_or(0)
                                                            as usize,
                                                    )
                                                } else {
                                                    // Try to parse as tuple string like "(1, 2, 3)"
                                                    if let Some(betti_str) = betti_numbers.as_str()
                                                    {
                                                        // Simple regex-like parsing
                                                        let nums: Vec<usize> = betti_str
                                                            .trim_matches(|c| c == '(' || c == ')')
                                                            .split(',')
                                                            .filter_map(|s| {
                                                                s.trim().parse::<usize>().ok()
                                                            })
                                                            .collect();
                                                        if nums.len() >= 3 {
                                                            (nums[0], nums[1], nums[2])
                                                        } else {
                                                            (0, 0, 0)
                                                        }
                                                    } else {
                                                        (0, 0, 0)
                                                    }
                                                };

                                                let packet = CognitiveStatePacket {
                                                    pad_state: pad,
                                                    torus_projection: torus,
                                                    betti_numbers: betti,
                                                    persistence_entropy,
                                                    compass_quadrant: compass_quadrant.to_string(),
                                                    compass_confidence: compass_confidence as f32,
                                                    retrieved_memory_ids: memory_ids,
                                                    iteration,
                                                    prompt_text,
                                                    timestamp: timestamp.to_string(),
                                                };
                                                let _ = tx.send(packet);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            eprintln!("⚠️  Error reading line: {}", e);
                        }
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
