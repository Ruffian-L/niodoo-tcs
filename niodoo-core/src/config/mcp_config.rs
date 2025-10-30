// Copyright 2025 Jason Van Pham (Niodoo)
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of Niodoo TCS.
// For commercial licensing, see LICENSE-COMMERCIAL.md

//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

/// MCP (Model Context Protocol) Configuration
///
/// Converted from configure_mcp.py to idiomatic Rust
use super::{ConfigError, ConfigResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use tracing::{info, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServer {
    pub command: String,
    pub args: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub env: Option<HashMap<String, String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpProjectConfig {
    #[serde(rename = "mcpServers")]
    pub mcp_servers: HashMap<String, McpServer>,
    #[serde(
        rename = "enabledMcpjsonServers",
        skip_serializing_if = "Option::is_none"
    )]
    pub enabled_servers: Option<Vec<String>>,
    #[serde(
        rename = "disabledMcpjsonServers",
        skip_serializing_if = "Option::is_none"
    )]
    pub disabled_servers: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaudeConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub projects: Option<HashMap<String, McpProjectConfig>>,
}

pub struct McpConfig {
    config_path: PathBuf,
    backup_path: PathBuf,
}

impl McpConfig {
    pub fn new() -> Self {
        let home = std::env::var("HOME")
            .unwrap_or_else(|_| std::env::temp_dir().to_string_lossy().to_string());
        let config_path = PathBuf::from(&home).join(".claude.json");
        let backup_path = PathBuf::from(&home).join(".claude.json.backup");

        Self {
            config_path,
            backup_path,
        }
    }

    pub fn with_path(config_path: PathBuf) -> Self {
        let backup_path = config_path.with_extension("json.backup");

        Self {
            config_path,
            backup_path,
        }
    }

    fn backup_config(&self) -> ConfigResult<()> {
        if self.config_path.exists() {
            fs::copy(&self.config_path, &self.backup_path)?;
            info!("Backed up config to: {:?}", self.backup_path);
        } else {
            warn!("No existing config to backup at: {:?}", self.config_path);
        }
        Ok(())
    }

    fn load_config(&self) -> ConfigResult<ClaudeConfig> {
        if !self.config_path.exists() {
            return Ok(ClaudeConfig { projects: None });
        }

        let content = fs::read_to_string(&self.config_path)?;
        let config: ClaudeConfig = serde_json::from_str(&content)?;
        Ok(config)
    }

    fn save_config(&self, config: &ClaudeConfig) -> ConfigResult<()> {
        let json = serde_json::to_string_pretty(config)?;
        fs::write(&self.config_path, json)?;
        info!("Updated Claude Code config at: {:?}", self.config_path);
        Ok(())
    }

    pub fn configure_project_servers(
        &self,
        project_path: &str,
        servers: HashMap<String, McpServer>,
    ) -> ConfigResult<()> {
        // Backup existing config
        self.backup_config()?;

        // Load current config
        let mut config = self.load_config()?;

        // Ensure projects structure exists
        if config.projects.is_none() {
            config.projects = Some(HashMap::new());
        }

        let projects = config
            .projects
            .as_mut()
            .expect("Projects field should be present");

        // Add or update project configuration
        let project_config =
            projects
                .entry(project_path.to_string())
                .or_insert_with(|| McpProjectConfig {
                    mcp_servers: HashMap::new(),
                    enabled_servers: Some(Vec::new()),
                    disabled_servers: Some(Vec::new()),
                });

        project_config.mcp_servers = servers;

        // Save updated config
        self.save_config(&config)?;

        Ok(())
    }
}

impl Default for McpConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Configure Claude Code MCP servers for the Niodoo-Feeling project
pub fn configure_claude_mcp(project_path: &Path) -> ConfigResult<()> {
    let project_path_str = project_path
        .to_str()
        .ok_or_else(|| ConfigError::Invalid("Invalid project path".to_string()))?;

    let mut servers = HashMap::new();

    // Local RAG server
    servers.insert(
        "local-rag".to_string(),
        McpServer {
            command: "python3".to_string(),
            args: vec![format!("{}/mcp_rag_wrapper.py", project_path_str)],
            env: Some({
                let mut env = HashMap::new();
                env.insert("PYTHONUNBUFFERED".to_string(), "1".to_string());
                env
            }),
        },
    );

    // El Chapo v3.2 server
    servers.insert(
        "el-chapo-v3.2".to_string(),
        McpServer {
            command: "python3".to_string(),
            args: vec![format!("{}/mcp_stdio_wrapper.py", project_path_str)],
            env: Some({
                let mut env = HashMap::new();
                env.insert("REPO_PATH".to_string(), project_path_str.to_string());
                env.insert(
                    "FAISS_INDEX_PATH".to_string(),
                    format!("{}/data/faiss", project_path_str),
                );
                env.insert("DASK_WORKERS".to_string(), "8".to_string());
                env.insert("ENVIRONMENT".to_string(), "development".to_string());
                env.insert("PYTHONUNBUFFERED".to_string(), "1".to_string());
                env
            }),
        },
    );

    // Pinecone server
    servers.insert(
        "niodoo-pinecone".to_string(),
        McpServer {
            command: "python3".to_string(),
            args: vec![format!("{}/mcp_pinecone_wrapper.py", project_path_str)],
            env: Some({
                let mut env = HashMap::new();
                env.insert("REPO_PATH".to_string(), project_path_str.to_string());
                env.insert("PYTHONUNBUFFERED".to_string(), "1".to_string());
                env
            }),
        },
    );

    let mcp_config = McpConfig::new();
    mcp_config.configure_project_servers(project_path_str, servers)?;

    info!("MCP servers configured successfully");
    info!("Next steps:");
    info!("1. Restart Claude Code (or reload the window)");
    info!("2. Check for MCP tools with: /mcp");
    info!("3. Tools should appear as: mcp__query_embeddings, mcp__embed_repo, etc.");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_mcp_config_creation() {
        let config = McpConfig::new();
        assert!(config.config_path.ends_with(".claude.json"));
        assert!(config.backup_path.ends_with(".claude.json.backup"));
    }

    #[test]
    fn test_configure_project_servers() {
        let temp_dir = tempdir().expect("Failed to create temp directory in test");
        let config_path = temp_dir.path().join("test_config.json");
        let mcp_config = McpConfig::with_path(config_path.clone());

        let mut servers = HashMap::new();
        servers.insert(
            "test-server".to_string(),
            McpServer {
                command: "python3".to_string(),
                args: vec!["test.py".to_string()],
                env: None,
            },
        );

        let result = mcp_config.configure_project_servers("/test/path", servers);
        assert!(result.is_ok());
        assert!(config_path.exists());
    }

    #[test]
    fn test_server_serialization() {
        let server = McpServer {
            command: "python3".to_string(),
            args: vec!["script.py".to_string()],
            env: Some({
                let mut env = HashMap::new();
                env.insert("TEST".to_string(), "value".to_string());
                env
            }),
        };

        let json = serde_json::to_string(&server).expect("Failed to serialize McpServer in test");
        let deserialized: McpServer =
            serde_json::from_str(&json).expect("Failed to deserialize McpServer in test");

        assert_eq!(server.command, deserialized.command);
        assert_eq!(server.args, deserialized.args);
        assert_eq!(server.env, deserialized.env);
    }
}
