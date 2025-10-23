//! MCP Notifier for real-time embedding updates
//! 
//! This module provides MCP integration to notify the AI transformer
//! about embedding operations in real-time.

use std::sync::Arc;
use anyhow::Result;
use serde_json::json;
use tokio::net::TcpStream;
use tokio::io::{AsyncWriteExt, AsyncReadExt};

/// MCP notifier implementation for embedding events
pub struct McpEmbeddingNotifier {
    mcp_server_url: String,
    client: reqwest::Client,
}

impl McpEmbeddingNotifier {
    /// Create a new MCP notifier
    pub fn new(mcp_server_url: String) -> Self {
        Self {
            mcp_server_url,
            client: reqwest::Client::new(),
        }
    }

    /// Send notification to MCP server
    async fn send_notification(&self, method: &str, params: serde_json::Value) -> Result<()> {
        let notification = json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params
        });

        let response = self.client
            .post(&format!("{}/notify", self.mcp_server_url))
            .json(&notification)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(anyhow::anyhow!("MCP notification failed: {}", response.status()));
        }

        Ok(())
    }
}

impl super::McpNotifier for McpEmbeddingNotifier {
    async fn notify_embedding_created(&self, file_path: &str, embedding_id: &str) -> Result<()> {
        let params = json!({
            "type": "embedding_created",
            "file_path": file_path,
            "embedding_id": embedding_id,
            "timestamp": chrono::Utc::now().timestamp()
        });

        self.send_notification("embeddings/created", params).await
    }

    async fn notify_embedding_updated(&self, file_path: &str, embedding_id: &str) -> Result<()> {
        let params = json!({
            "type": "embedding_updated",
            "file_path": file_path,
            "embedding_id": embedding_id,
            "timestamp": chrono::Utc::now().timestamp()
        });

        self.send_notification("embeddings/updated", params).await
    }

    async fn notify_batch_complete(&self, total_files: usize, success_count: usize) -> Result<()> {
        let params = json!({
            "type": "batch_complete",
            "total_files": total_files,
            "success_count": success_count,
            "failure_count": total_files - success_count,
            "timestamp": chrono::Utc::now().timestamp()
        });

        self.send_notification("embeddings/batch_complete", params).await
    }

    async fn notify_error(&self, error: &str) -> Result<()> {
        let params = json!({
            "type": "error",
            "error": error,
            "timestamp": chrono::Utc::now().timestamp()
        });

        self.send_notification("embeddings/error", params).await
    }
}

/// Mock MCP notifier for testing
pub struct MockMcpNotifier {
    pub notifications: Arc<tokio::sync::RwLock<Vec<String>>>,
}

impl MockMcpNotifier {
    pub fn new() -> Self {
        Self {
            notifications: Arc::new(tokio::sync::RwLock::new(Vec::new())),
        }
    }

    pub async fn get_notifications(&self) -> Vec<String> {
        self.notifications.read().await.clone()
    }

    pub async fn clear_notifications(&self) {
        self.notifications.write().await.clear();
    }
}

impl super::McpNotifier for MockMcpNotifier {
    async fn notify_embedding_created(&self, file_path: &str, embedding_id: &str) -> Result<()> {
        let notification = format!("embedding_created: {} -> {}", file_path, embedding_id);
        self.notifications.write().await.push(notification);
        Ok(())
    }

    async fn notify_embedding_updated(&self, file_path: &str, embedding_id: &str) -> Result<()> {
        let notification = format!("embedding_updated: {} -> {}", file_path, embedding_id);
        self.notifications.write().await.push(notification);
        Ok(())
    }

    async fn notify_batch_complete(&self, total_files: usize, success_count: usize) -> Result<()> {
        let notification = format!("batch_complete: {}/{}", success_count, total_files);
        self.notifications.write().await.push(notification);
        Ok(())
    }

    async fn notify_error(&self, error: &str) -> Result<()> {
        let notification = format!("error: {}", error);
        self.notifications.write().await.push(notification);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mock_notifier() {
        let notifier = MockMcpNotifier::new();
        
        notifier.notify_embedding_created("test.rs", "embed123").await.unwrap();
        notifier.notify_batch_complete(10, 8).await.unwrap();
        notifier.notify_error("Test error").await.unwrap();
        
        let notifications = notifier.get_notifications().await;
        assert_eq!(notifications.len(), 3);
        assert!(notifications[0].contains("embedding_created"));
        assert!(notifications[1].contains("batch_complete"));
        assert!(notifications[2].contains("error"));
    }
}

