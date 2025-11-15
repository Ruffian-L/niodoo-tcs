//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

//! TCP Telemetry Server
//!
//! Simple TCP server that broadcasts cognitive state packets as newline-delimited JSON
//! to connected visualization clients.

use crate::telemetry::CognitiveStatePacket;
use anyhow::Result;
use std::net::SocketAddr;
use tokio::io::AsyncWriteExt;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::broadcast;
use tracing::{error, info, warn};

/// Start the TCP telemetry server
///
/// Listens on the specified address and broadcasts cognitive state packets
/// to all connected clients as newline-delimited JSON.
pub async fn start_telemetry_server(
    addr: SocketAddr,
    mut rx: broadcast::Receiver<CognitiveStatePacket>,
) -> Result<()> {
    let listener = TcpListener::bind(addr).await?;
    info!(addr = %addr, "Mind's Eye Telemetry Server listening");

    loop {
        match listener.accept().await {
            Ok((socket, client_addr)) => {
                info!(client = %client_addr, "New telemetry client connected");
                let mut client_rx = rx.resubscribe();
                tokio::spawn(async move {
                    if let Err(e) = handle_client(socket, client_addr, &mut client_rx).await {
                        warn!(client = %client_addr, error = %e, "Client disconnected");
                    }
                });
            }
            Err(e) => {
                error!(error = %e, "Failed to accept connection");
            }
        }
    }
}

/// Handle a single client connection
async fn handle_client(
    mut socket: TcpStream,
    client_addr: SocketAddr,
    rx: &mut broadcast::Receiver<CognitiveStatePacket>,
) -> Result<()> {
    while let Ok(packet) = rx.recv().await {
        let json_string = match serde_json::to_string(&packet) {
            Ok(json) => json + "\n",
            Err(e) => {
                warn!(error = %e, "Failed to serialize telemetry packet");
                continue;
            }
        };

        if socket.write_all(json_string.as_bytes()).await.is_err() {
            // Client disconnected
            break;
        }
    }

    info!(client = %client_addr, "Client disconnected");
    Ok(())
}
