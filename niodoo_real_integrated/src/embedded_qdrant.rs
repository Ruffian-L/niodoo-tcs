use anyhow::{Context, Result};
use tokio::process::Command;
use tracing::{info, warn};

/// Spawn embedded Qdrant as a managed child process
#[cfg(feature = "embedded-qdrant")]
pub async fn spawn_embedded_qdrant() -> Result<tokio::process::Child> {
    let storage_path =
        std::env::var("QDRANT_STORAGE_PATH").unwrap_or_else(|_| "./qdrant_storage".to_string());

    // Ensure storage directory exists
    std::fs::create_dir_all(&storage_path).with_context(|| {
        format!(
            "failed to create Qdrant storage directory: {}",
            storage_path
        )
    })?;

    // Try to find Qdrant binary in common locations
    let qdrant_binary = std::env::var("QDRANT_BINARY")
        .or_else(|_| {
            // Check common paths
            let paths = [
                "/usr/local/bin/qdrant",
                "/usr/bin/qdrant",
                "./qdrant",
                "./target/release/qdrant",
            ];
            for path in &paths {
                if std::path::Path::new(path).exists() {
                    return Ok(path.to_string());
                }
            }
            Err(std::env::VarError::NotPresent)
        })
        .or_else(|_| {
            // Try which/where
            let output = std::process::Command::new("which").arg("qdrant").output();
            if let Ok(output) = output {
                if output.status.success() {
                    let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
                    if !path.is_empty() {
                        return Ok(path);
                    }
                }
            }
            Err(std::env::VarError::NotPresent)
        })
        .with_context(|| "Qdrant binary not found. Set QDRANT_BINARY env var or install qdrant")?;

    info!(binary = %qdrant_binary, storage = %storage_path, "Spawning embedded Qdrant");

    // Spawn Qdrant with storage path
    let mut cmd = Command::new(&qdrant_binary);
    cmd.args(&[
        "--storage-path",
        &storage_path,
        "--grpc-port",
        "6334", // Use different port for embedded
        "--http-port",
        "6333", // Standard HTTP port
    ]);
    cmd.env("QDRANT_STORAGE_PATH", &storage_path);
    cmd.stdout(std::process::Stdio::piped());
    cmd.stderr(std::process::Stdio::piped());
    cmd.kill_on_drop(true);

    let child = cmd
        .spawn()
        .with_context(|| format!("failed to spawn Qdrant binary at {}", qdrant_binary))?;

    // Wait a moment for Qdrant to start
    tokio::time::sleep(std::time::Duration::from_secs(2)).await;

    // Verify Qdrant is responding
    let client = reqwest::Client::new();
    for attempt in 0..10 {
        if let Ok(resp) = client
            .get("http://127.0.0.1:6333/health")
            .timeout(std::time::Duration::from_secs(1))
            .send()
            .await
        {
            if resp.status().is_success() {
                info!("Embedded Qdrant started successfully");
                return Ok(child);
            }
        }
        if attempt < 9 {
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        }
    }

    warn!("Embedded Qdrant failed to start or respond on http://127.0.0.1:6333/health");
    anyhow::bail!("Embedded Qdrant did not become healthy after startup");
}
