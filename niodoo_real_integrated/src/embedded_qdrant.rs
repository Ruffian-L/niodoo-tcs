use anyhow::{Context, Result};
use std::path::PathBuf;

#[cfg(feature = "embedded-qdrant")]
use tokio::fs::{self, OpenOptions};
#[cfg(feature = "embedded-qdrant")]
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
#[cfg(feature = "embedded-qdrant")]
use tokio::process::Command;
#[cfg(feature = "embedded-qdrant")]
use tokio::time::{sleep, Duration};
#[cfg(feature = "embedded-qdrant")]
use tracing::{info, warn};

/// Spawn embedded Qdrant as a managed child process
#[cfg(feature = "embedded-qdrant")]
pub async fn spawn_embedded_qdrant() -> Result<tokio::process::Child> {
    let raw_storage_path = std::env::var("QDRANT_STORAGE_PATH")
        .unwrap_or_else(|_| "/var/lib/niodoo/qdrant_storage".to_string());

    let storage_path = normalize_storage_path(&raw_storage_path)?;

    if storage_path.starts_with("/tmp") || storage_path.starts_with("/var/tmp") {
        anyhow::bail!(
            "Qdrant storage path resolves to temporary directory {:?}. Set QDRANT_STORAGE_PATH to a persistent location.",
            storage_path
        );
    }

    // Ensure storage directory exists
    std::fs::create_dir_all(&storage_path).with_context(|| {
        format!(
            "failed to create Qdrant storage directory: {}",
            storage_path
        )
    })?;

    let snapshots_path = PathBuf::from(&storage_path).join("snapshots");
    std::fs::create_dir_all(&snapshots_path).with_context(|| {
        format!(
            "failed to create Qdrant snapshots directory: {}",
            snapshots_path.display()
        )
    })?;

    let logs_dir = PathBuf::from(&storage_path).join("logs");
    fs::create_dir_all(&logs_dir).await.with_context(|| {
        format!(
            "failed to create Qdrant logs directory: {}",
            logs_dir.display()
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

    let config_path = PathBuf::from(&storage_path).join("embedded_qdrant_config.yaml");
    let config_contents = format!(
        "log_level: INFO\nstorage:\n  path: {path}\n  snapshots_path: {snapshots}\nservice:\n  grpc_port: 6334\n  http_port: 6333\n",
        path = storage_path,
        snapshots = snapshots_path.display(),
    );
    fs::write(&config_path, config_contents)
        .await
        .with_context(|| {
            format!(
                "failed to write embedded Qdrant config at {}",
                config_path.display()
            )
        })?;

    info!(binary = %qdrant_binary, storage = %storage_path, config = %config_path.display(), "Spawning embedded Qdrant");

    // Spawn Qdrant with generated config
    let mut cmd = Command::new(&qdrant_binary);
    cmd.arg("--disable-telemetry");
    if let Some(config) = config_path.to_str() {
        cmd.args(["--config-path", config]);
    } else {
        warn!(path = %config_path.display(), "Embedded Qdrant config path contained invalid UTF-8; falling back to workspace config");
        cmd.args(["--config-path", "/workspace/qdrant_config/config.yaml"]);
    }
    cmd.env("QDRANT_STORAGE_PATH", &storage_path);
    cmd.env("QDRANT__STORAGE__PATH", &storage_path);
    cmd.env("QDRANT__STORAGE__STORAGE_PATH", &storage_path);
    cmd.env("QDRANT__STORAGE__SNAPSHOTS_PATH", &snapshots_path);
    cmd.env("QDRANT__SERVICE__HOST", "127.0.0.1");
    cmd.env("QDRANT__SERVICE__HTTP_PORT", "6333");
    cmd.env("QDRANT__SERVICE__GRPC_PORT", "6334");
    cmd.env("QDRANT__STORAGE__DISABLE_FUSE_CHECK", "true");
    cmd.stdout(std::process::Stdio::piped());
    cmd.stderr(std::process::Stdio::piped());
    cmd.kill_on_drop(true);

    let mut child = cmd
        .spawn()
        .with_context(|| format!("failed to spawn Qdrant binary at {}", qdrant_binary))?;

    let stdout_log = logs_dir.join("embedded_qdrant_stdout.log");
    let stderr_log = logs_dir.join("embedded_qdrant_stderr.log");

    if let Some(stdout) = child.stdout.take() {
        let stdout_log = stdout_log.clone();
        tokio::spawn(async move {
            let mut lines = BufReader::new(stdout).lines();
            let mut file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&stdout_log)
                .await
                .ok();
            while let Ok(Some(line)) = lines.next_line().await {
                info!(target = "embedded_qdrant::stdout", "{line}");
                if let Some(f) = &mut file {
                    let _ = f.write_all(line.as_bytes()).await;
                    let _ = f.write_all(b"\n").await;
                }
            }
        });
    }

    if let Some(stderr) = child.stderr.take() {
        let stderr_log = stderr_log.clone();
        tokio::spawn(async move {
            let mut lines = BufReader::new(stderr).lines();
            let mut file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&stderr_log)
                .await
                .ok();
            while let Ok(Some(line)) = lines.next_line().await {
                warn!(target = "embedded_qdrant::stderr", "{line}");
                if let Some(f) = &mut file {
                    let _ = f.write_all(line.as_bytes()).await;
                    let _ = f.write_all(b"\n").await;
                }
            }
        });
    }

    // Wait a moment for Qdrant to start
    sleep(Duration::from_secs(2)).await;

    // Verify Qdrant is responding
    let client = reqwest::Client::new();
    for attempt in 0..10 {
        if let Ok(resp) = client
            .get("http://127.0.0.1:6333/health")
            .timeout(Duration::from_secs(1))
            .send()
            .await
        {
            let status = resp.status();
            if status.is_success() || status == reqwest::StatusCode::NOT_FOUND {
                info!(status = %status, "Embedded Qdrant responded to health probe");
                return Ok(child);
            }
        }
        if attempt < 9 {
            sleep(Duration::from_millis(500)).await;
        }
    }

    warn!("Embedded Qdrant failed to start or respond on http://127.0.0.1:6333/health");
    anyhow::bail!("Embedded Qdrant did not become healthy after startup");
}

fn normalize_storage_path(raw: &str) -> Result<String> {
    let path = PathBuf::from(raw);
    let absolute = if path.is_relative() {
        std::env::current_dir()
            .context("failed to resolve current working directory")?
            .join(&path)
    } else {
        path
    };

    Ok(absolute
        .to_str()
        .context("Qdrant storage path contains invalid UTF-8")?
        .to_string())
}
