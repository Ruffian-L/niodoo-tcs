//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

/*
 * 🎯 Agent 10: RunCommander
 * Full run script for Beelink (Ubuntu 25.04, RTX 6000)
 */

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tracing::{error, info, warn, debug};

/// Beelink system configuration
#[derive(Debug, Clone)]
pub struct BeelinkConfig {
    pub ubuntu_version: String,
    pub gpu_model: String,
    pub memory_gb: usize,
    pub cuda_version: String,
    pub qt_version: String,
}

impl Default for BeelinkConfig {
    fn default() -> Self {
        Self {
            ubuntu_version: "25.04".to_string(),
            gpu_model: "RTX 6000".to_string(),
            memory_gb: 32,
            cuda_version: "12.2".to_string(),
            qt_version: "6.7".to_string(),
        }
    }
}

/// System command execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandResult {
    pub command: String,
    pub success: bool,
    pub execution_time_ms: u64,
    pub exit_code: Option<i32>,
    pub output: String,
    pub timestamp: u64,
}

/// RunCommander agent for Beelink system management
pub struct RunCommander {
    config: BeelinkConfig,
    result_channel: mpsc::UnboundedSender<CommandResult>,
    command_history: Vec<CommandResult>,
    shutdown: Arc<AtomicBool>,
}

impl RunCommander {
    /// Create new RunCommander agent
    pub fn new(config: BeelinkConfig) -> Self {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_clone = shutdown.clone();

        tokio::spawn(async move {
            while !shutdown_clone.load(Ordering::Relaxed) {
                if let Ok(result) = rx.try_recv() {
                    Self::process_command_result(&result).await;
                }
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        });

        Self {
            config,
            result_channel: tx,
            command_history: Vec::new(),
            shutdown,
        }
    }

    /// Execute cargo run command for Beelink
    pub async fn run_beelink_demo(&mut self) -> Result<CommandResult> {
        info!("🎯 RunCommander executing Beelink demo");

        let command_str = "cargo run --bin niodo_full_demo --features cuda --release".to_string();
        let start_time = Instant::now();

        // Build and execute cargo command
        let execution_result = tokio::task::spawn_blocking(move || {
            let mut cmd = Command::new("cargo");
            cmd.arg("run")
                .arg("--bin")
                .arg("niodo_full_demo")
                .arg("--features")
                .arg("cuda")
                .arg("--release")
                .stdout(Stdio::piped())
                .stderr(Stdio::piped());

            cmd.output()
        }).await;

        let execution_time = start_time.elapsed().as_millis() as u64;

        match execution_result {
            Ok(Ok(output)) => {
                let success = output.status.success();
                let exit_code = output.status.code();
                let stdout = String::from_utf8_lossy(&output.stdout);
                let stderr = String::from_utf8_lossy(&output.stderr);
                let full_output = format!("STDOUT: {}\nSTDERR: {}", stdout, stderr);

                let result = CommandResult {
                    command: command_str,
                    success,
                    execution_time_ms: execution_time,
                    exit_code,
                    output: full_output,
                    timestamp: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs(),
                };

                if let Err(e) = self.result_channel.send(result.clone()) {
                    warn!("Failed to send command result: {}", e);
                }

                self.command_history.push(result.clone());
                info!("🎯 Beelink demo completed: success={}, time={}ms", success, execution_time);

                Ok(result)
            }
            Ok(Err(e)) => {
                let result = CommandResult {
                    command: command_str,
                    success: false,
                    execution_time_ms: execution_time,
                    exit_code: None,
                    output: format!("Command execution failed: {}", e),
                    timestamp: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs(),
                };

                if let Err(e) = self.result_channel.send(result.clone()) {
                    warn!("Failed to send command result: {}", e);
                }

                Err(anyhow!("Beelink demo execution failed: {}", e))
            }
            Err(e) => {
                let result = CommandResult {
                    command: command_str,
                    success: false,
                    execution_time_ms: execution_time,
                    exit_code: None,
                    output: format!("Task join error: {}", e),
                    timestamp: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs(),
                };

                if let Err(e) = self.result_channel.send(result.clone()) {
                    warn!("Failed to send command result: {}", e);
                }

                Err(anyhow!("Beelink demo task failed: {}", e))
            }
        }
    }

    /// Process command result for monitoring
    async fn process_command_result(result: &CommandResult) {
        debug!("📊 Processing command result: {:?}", result);

        if result.success {
            info!("✅ Beelink command succeeded: {}", result.command);
        } else {
            warn!("❌ Beelink command failed: {}", result.command);
        }
    }

    /// Get system statistics
    pub fn get_system_stats(&self) -> SystemStats {
        let total_commands = self.command_history.len();
        let successful_commands = self.command_history.iter().filter(|r| r.success).count();
        let total_execution_time: u64 = self.command_history.iter().map(|r| r.execution_time_ms).sum();

        SystemStats {
            total_commands,
            successful_commands,
            total_execution_time_ms: total_execution_time,
            success_rate: if total_commands > 0 {
                (successful_commands as f32 / total_commands as f32) * 100.0
            } else {
                0.0
            },
            beelink_config: self.config.clone(),
        }
    }

    /// Continuous monitoring loop
    pub async fn run_monitoring(&mut self) -> Result<()> {
        info!("🎯 RunCommander starting Beelink monitoring");

        let mut interval = tokio::time::interval(Duration::from_secs(30));

        while !self.shutdown.load(Ordering::Relaxed) {
            interval.tick().await;

            let stats = self.get_system_stats();
            info!("🎯 Beelink stats: {} commands, {:.1}% success rate, {}ms total time",
                  stats.total_commands, stats.success_rate, stats.total_execution_time_ms);
        }

        Ok(())
    }

    /// Shutdown the agent
    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::Relaxed);
        info!("🎯 RunCommander shutting down");
    }
}

/// System statistics for Beelink monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemStats {
    pub total_commands: usize,
    pub successful_commands: usize,
    pub total_execution_time_ms: u64,
    pub success_rate: f32,
    pub beelink_config: BeelinkConfig,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_run_commander_creation() {
        let config = BeelinkConfig::default();
        let mut commander = RunCommander::new(config);

        // Note: This would actually run cargo in a real environment
        let stats = commander.get_system_stats();
        assert_eq!(stats.total_commands, 0);
        assert_eq!(stats.successful_commands, 0);
        assert_eq!(stats.success_rate, 0.0);
    }

    #[test]
    fn test_system_stats() {
        let config = BeelinkConfig::default();
        let commander = RunCommander::new(config);

        let stats = commander.get_system_stats();
        assert_eq!(stats.beelink_config.ubuntu_version, "25.04");
        assert_eq!(stats.beelink_config.gpu_model, "RTX 6000");
    }
}
