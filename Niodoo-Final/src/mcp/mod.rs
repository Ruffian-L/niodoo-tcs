//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

/// MCP (Model Context Protocol) integration module
///
/// Provides stdio bridges and RPC communication for MCP servers
pub mod stdio_bridge;

pub use stdio_bridge::{RpcError, RpcRequest, RpcResponse, StdioBridge};

use thiserror::Error;

#[derive(Debug, Error)]
pub enum McpError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("RPC error: {0}")]
    Rpc(String),

    #[error("Server not healthy: {0}")]
    ServerNotHealthy(String),

    #[error("Communication error: {0}")]
    Communication(String),
}

/// Result type alias for MCP errors
/// Note: Use McpResult to avoid conflicts with other Result aliases
pub type McpResult<T> = std::result::Result<T, McpError>;
