/// MCP STDIO Bridge - Bridges stdio communication with HTTP servers
///
/// Converted from mcp_stdio_wrapper.py to idiomatic Rust with async support
use super::{McpError, McpResult};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::io::{self, BufRead, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::signal;
use tracing::{debug, info};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RpcRequest {
    pub jsonrpc: String,
    pub method: Option<String>,
    pub params: Option<Value>,
    pub id: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RpcResponse {
    pub jsonrpc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<RpcError>,
    pub id: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RpcError {
    pub code: i32,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

impl RpcError {
    fn parse_error(id: Option<Value>) -> RpcResponse {
        RpcResponse {
            jsonrpc: "2.0".to_string(),
            result: None,
            error: Some(RpcError {
                code: -32700,
                message: "Parse error".to_string(),
                data: None,
            }),
            id,
        }
    }

    fn internal_error(message: String, id: Option<Value>) -> RpcResponse {
        RpcResponse {
            jsonrpc: "2.0".to_string(),
            result: None,
            error: Some(RpcError {
                code: -32603,
                message,
                data: None,
            }),
            id,
        }
    }

    fn server_communication_error(message: String, id: Option<Value>) -> RpcResponse {
        RpcResponse {
            jsonrpc: "2.0".to_string(),
            result: None,
            error: Some(RpcError {
                code: -32603,
                message: format!("Server communication error: {}", message),
                data: None,
            }),
            id,
        }
    }
}

pub struct StdioBridge {
    server_url: String,
    client: Client,
    shutdown: Arc<AtomicBool>,
}

impl StdioBridge {
    pub fn new(server_url: Option<String>) -> Self {
        let server_url = server_url.unwrap_or_else(|| {
            std::env::var("MCP_SERVER_URL").unwrap_or_else(|_| "http://localhost:8002".to_string())
        });

        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            server_url,
            client,
            shutdown: Arc::new(AtomicBool::new(false)),
        }
    }

    async fn check_server_health(&self) -> McpResult<()> {
        let health_url = format!("{}/health", self.server_url);

        match self.client.get(&health_url).send().await {
            Ok(response) => {
                let status = response.status();
                if status.is_success() || status.as_u16() == 503 {
                    // 503 is acceptable during startup
                    Ok(())
                } else {
                    Err(McpError::ServerNotHealthy(format!(
                        "Health check failed with status: {}",
                        status
                    )))
                }
            }
            Err(e) => Err(McpError::Http(e)),
        }
    }

    async fn handle_request(&self, request: RpcRequest) -> RpcResponse {
        let rpc_url = format!("{}/rpc", self.server_url);

        match self.client.post(&rpc_url).json(&request).send().await {
            Ok(response) => match response.json::<RpcResponse>().await {
                Ok(rpc_response) => rpc_response,
                Err(e) => {
                    tracing::error!("Failed to parse RPC response: {}", e);
                    RpcError::internal_error(format!("Failed to parse response: {}", e), request.id)
                }
            },
            Err(e) => {
                tracing::error!("HTTP request failed: {}", e);
                RpcError::server_communication_error(e.to_string(), request.id)
            }
        }
    }

    fn send_response(response: &RpcResponse) -> io::Result<()> {
        let json = serde_json::to_string(response)?;
        let mut stdout = io::stdout();
        writeln!(stdout, "{}", json)?;
        stdout.flush()?;
        debug!("Sent response: {}", json);
        Ok(())
    }

    pub async fn run(&self) -> McpResult<()> {
        info!("MCP STDIO Bridge started");
        info!("Connecting to server at: {}", self.server_url);

        // Check server health
        if let Err(e) = self.check_server_health().await {
            tracing::error!("Server health check failed: {}", e);
            tracing::info!("Error: MCP server not healthy. Please start the MCP server first.");
            return Err(e);
        }

        info!("Server is healthy, starting stdio loop");

        // Setup signal handler
        let shutdown = Arc::clone(&self.shutdown);
        tokio::spawn(async move {
            if let Err(e) = signal::ctrl_c().await {
                tracing::error!("Failed to install signal handler: {}", e);
            } else {
                info!("Shutdown signal received");
                shutdown.store(true, Ordering::Relaxed);
            }
        });

        // Main stdio loop
        let stdin = io::stdin();
        let mut lines = stdin.lock().lines();

        while !self.shutdown.load(Ordering::Relaxed) {
            match lines.next() {
                Some(Ok(line)) => {
                    if line.is_empty() {
                        continue;
                    }

                    debug!("Received: {}", line);

                    // Parse JSON-RPC request
                    let request: RpcRequest = match serde_json::from_str(&line) {
                        Ok(req) => req,
                        Err(e) => {
                            tracing::error!("Invalid JSON: {}", e);
                            let error_response = RpcError::parse_error(None);
                            if let Err(e) = Self::send_response(&error_response) {
                                tracing::error!("Failed to send error response: {}", e);
                            }
                            continue;
                        }
                    };

                    // Handle the request
                    let response = self.handle_request(request).await;

                    // Send response
                    if let Err(e) = Self::send_response(&response) {
                        tracing::error!("Failed to send response: {}", e);
                        break;
                    }
                }
                Some(Err(e)) => {
                    tracing::error!("Error reading from stdin: {}", e);
                    break;
                }
                None => {
                    info!("EOF received, exiting");
                    break;
                }
            }
        }

        info!("MCP STDIO Bridge shutdown complete");
        Ok(())
    }

    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rpc_request_serialization() {
        let request = RpcRequest {
            jsonrpc: "2.0".to_string(),
            method: Some("test_method".to_string()),
            params: Some(serde_json::json!({"key": "value"})),
            id: Some(serde_json::json!(1)),
        };

        let json = serde_json::to_string(&request).unwrap();
        let deserialized: RpcRequest = serde_json::from_str(&json).unwrap();

        assert_eq!(request.jsonrpc, deserialized.jsonrpc);
        assert_eq!(request.method, deserialized.method);
        assert_eq!(request.id, deserialized.id);
    }

    #[test]
    fn test_rpc_error_response() {
        let response = RpcError::parse_error(Some(serde_json::json!(1)));

        assert_eq!(response.jsonrpc, "2.0");
        assert!(response.error.is_some());
        assert_eq!(response.error.unwrap().code, -32700);
    }

    #[test]
    fn test_bridge_creation() {
        let bridge = StdioBridge::new(None);
        assert!(bridge.server_url.contains("localhost"));
    }

    #[tokio::test]
    async fn test_shutdown() {
        let bridge = StdioBridge::new(Some("http://localhost:8002".to_string()));
        assert!(!bridge.shutdown.load(Ordering::Relaxed));

        bridge.shutdown();
        assert!(bridge.shutdown.load(Ordering::Relaxed));
    }
}
