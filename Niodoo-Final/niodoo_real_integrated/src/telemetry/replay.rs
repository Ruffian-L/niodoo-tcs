//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

//! Telemetry Replay System
//!
//! Reads saved telemetry logs and replays them for visualization or analysis.

use crate::telemetry::EnhancedCognitiveStatePacket;
use anyhow::{Context, Result};
use serde_json;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use tokio::sync::broadcast;
use tokio::time::{Duration, Instant};

/// Replay telemetry logs from JSONL files
pub struct TelemetryReplayer {
    log_file: PathBuf,
    speed_multiplier: f64, // 1.0 = real-time, 2.0 = 2x speed, etc.
}

impl TelemetryReplayer {
    /// Create a new replayer for a log file
    pub fn new(log_file: impl AsRef<Path>, speed_multiplier: f64) -> Self {
        Self {
            log_file: log_file.as_ref().to_path_buf(),
            speed_multiplier: speed_multiplier.max(0.01), // Minimum 0.01x speed
        }
    }

    /// Replay logs to a broadcast channel (for visualization)
    pub async fn replay_to_channel(
        &self,
        tx: broadcast::Sender<crate::telemetry::CognitiveStatePacket>,
    ) -> Result<()> {
        let file = File::open(&self.log_file)
            .with_context(|| format!("Failed to open log file: {:?}", self.log_file))?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();
        let mut last_timestamp: Option<chrono::DateTime<chrono::Utc>> = None;

        while let Some(Ok(line)) = lines.next() {
            if line.trim().is_empty() {
                continue;
            }

            let packet: EnhancedCognitiveStatePacket = serde_json::from_str(&line)
                .with_context(|| format!("Failed to parse packet from line: {}", line))?;

            // Calculate delay based on timestamps
            if let Some(last_ts) = last_timestamp {
                let current_ts = packet
                    .timestamp
                    .parse::<chrono::DateTime<chrono::Utc>>()
                    .with_context(|| format!("Failed to parse timestamp: {}", packet.timestamp))?;
                let delay = current_ts - last_ts;
                let delay_ms = delay.num_milliseconds() as u64;
                let adjusted_delay = (delay_ms as f64 / self.speed_multiplier) as u64;

                if adjusted_delay > 0 {
                    tokio::time::sleep(Duration::from_millis(adjusted_delay)).await;
                }
            }

            // Convert to legacy packet and send
            let legacy_packet = packet.to_legacy();
            let _ = tx.send(legacy_packet);

            last_timestamp = Some(
                packet
                    .timestamp
                    .parse::<chrono::DateTime<chrono::Utc>>()
                    .with_context(|| format!("Failed to parse timestamp: {}", packet.timestamp))?,
            );
        }

        Ok(())
    }

    /// Replay logs and collect all packets (for analysis)
    pub fn replay_to_vec(&self) -> Result<Vec<EnhancedCognitiveStatePacket>> {
        let file = File::open(&self.log_file)
            .with_context(|| format!("Failed to open log file: {:?}", self.log_file))?;
        let reader = BufReader::new(file);
        let mut packets = Vec::new();

        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }

            let packet: EnhancedCognitiveStatePacket = serde_json::from_str(&line)
                .with_context(|| format!("Failed to parse packet from line: {}", line))?;
            packets.push(packet);
        }

        Ok(packets)
    }

    /// List all available log files in the logs directory
    pub fn list_log_files(log_dir: impl AsRef<Path>) -> Result<Vec<PathBuf>> {
        let log_dir = log_dir.as_ref();
        let mut log_files = Vec::new();

        if !log_dir.exists() {
            return Ok(log_files);
        }

        for entry in std::fs::read_dir(log_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("jsonl") {
                log_files.push(path);
            }
        }

        log_files.sort();
        Ok(log_files)
    }
}

/// Replay server that serves replayed telemetry to visualization clients
pub struct ReplayServer {
    log_file: PathBuf,
    speed_multiplier: f64,
}

impl ReplayServer {
    /// Create a new replay server
    pub fn new(log_file: impl AsRef<Path>, speed_multiplier: f64) -> Self {
        Self {
            log_file: log_file.as_ref().to_path_buf(),
            speed_multiplier: speed_multiplier,
        }
    }

    /// Start replay server on specified address
    pub async fn start(&self, addr: std::net::SocketAddr) -> Result<()> {
        let (tx, _rx) = broadcast::channel::<crate::telemetry::CognitiveStatePacket>(1000);

        // Spawn replay task
        let replayer = TelemetryReplayer::new(&self.log_file, self.speed_multiplier);
        let tx_clone = tx.clone();
        tokio::spawn(async move {
            if let Err(e) = replayer.replay_to_channel(tx_clone).await {
                tracing::error!(error = %e, "Replay failed");
            }
        });

        // Start telemetry server
        crate::telemetry::server::start_telemetry_server(addr, _rx).await?;

        Ok(())
    }
}
