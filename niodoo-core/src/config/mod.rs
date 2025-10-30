// Copyright 2025 Jason Van Pham (Niodoo)
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of Niodoo TCS.
// For commercial licensing, see LICENSE-COMMERCIAL.md

//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

pub mod mcp_config;
/// Configuration management module
///
/// Handles all configuration loading, validation, and management for the Niodoo-Feeling project.
pub mod system_config;

pub use mcp_config::{configure_claude_mcp, McpConfig, McpServer};
pub use system_config::*;

use std::path::PathBuf;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("YAML error: {0}")]
    Yaml(#[from] serde_yaml::Error),

    #[error("TOML error: {0}")]
    Toml(#[from] toml::de::Error),

    #[error("Invalid configuration: {0}")]
    Invalid(String),

    #[error("Missing configuration file: {0}")]
    MissingFile(PathBuf),
}

/// Result type alias for config errors
/// Note: Use ConfigResult to avoid conflicts with other Result aliases
pub type ConfigResult<T> = std::result::Result<T, ConfigError>;
