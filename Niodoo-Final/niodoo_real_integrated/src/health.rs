//! Health Check Endpoints for Production Monitoring
//!
//! Provides /health, /ready, and /metrics endpoints for Kubernetes liveness/readiness probes
//! and Prometheus metrics scraping.
//!
//! Requires `svc` feature for HTTP server functionality.

#[cfg(feature = "svc")]
use anyhow::{Context, Result};
#[cfg(feature = "svc")]
use axum::{extract::State, http::StatusCode, response::Json, routing::get, Router};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
#[cfg(feature = "svc")]
use tracing::{info, warn};

/// Health check status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HealthStatus {
    Healthy,
    Unhealthy,
    Degraded,
}

/// Component health information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentHealth {
    pub name: String,
    pub status: HealthStatus,
    pub message: Option<String>,
    pub last_checked_at: Option<chrono::DateTime<chrono::Utc>>,
    #[serde(skip)]
    pub last_check_instant: Option<Instant>,
}

/// Overall system health
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemHealth {
    pub status: HealthStatus,
    pub timestamp: String,
    pub version: String,
    pub components: Vec<ComponentHealth>,
    pub uptime_secs: u64,
}

/// Health check registry
pub struct HealthRegistry {
    components: Arc<RwLock<Vec<ComponentHealth>>>,
    start_time: Instant,
}

impl HealthRegistry {
    pub fn new() -> Self {
        Self {
            components: Arc::new(RwLock::new(Vec::new())),
            start_time: Instant::now(),
        }
    }

    pub async fn register_component(
        &self,
        name: String,
        status: HealthStatus,
        message: Option<String>,
    ) {
        let mut components = self.components.write().await;
        if let Some(component) = components.iter_mut().find(|c| c.name == name) {
            component.status = status;
            component.message = message;
            component.last_check_instant = Some(Instant::now());
            component.last_checked_at = Some(chrono::Utc::now());
        } else {
            components.push(ComponentHealth {
                name,
                status,
                message,
                last_checked_at: Some(chrono::Utc::now()),
                last_check_instant: Some(Instant::now()),
            });
        }
    }

    pub async fn get_health(&self) -> SystemHealth {
        let components = self.components.read().await.clone();
        let overall_status = if components
            .iter()
            .any(|c| c.status == HealthStatus::Unhealthy)
        {
            HealthStatus::Unhealthy
        } else if components
            .iter()
            .any(|c| c.status == HealthStatus::Degraded)
        {
            HealthStatus::Degraded
        } else {
            HealthStatus::Healthy
        };

        SystemHealth {
            status: overall_status,
            timestamp: chrono::Utc::now().to_rfc3339(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            components,
            uptime_secs: self.start_time.elapsed().as_secs(),
        }
    }

    pub async fn is_ready(&self) -> bool {
        let health = self.get_health().await;
        matches!(
            health.status,
            HealthStatus::Healthy | HealthStatus::Degraded
        )
    }

    pub async fn is_healthy(&self) -> bool {
        let health = self.get_health().await;
        health.status == HealthStatus::Healthy
    }
}

impl Default for HealthRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Health check HTTP server (requires svc feature)
#[cfg(feature = "svc")]
pub struct HealthServer {
    registry: Arc<HealthRegistry>,
    port: u16,
}

#[cfg(feature = "svc")]
impl HealthServer {
    pub fn new(port: u16) -> Self {
        Self {
            registry: Arc::new(HealthRegistry::new()),
            port,
        }
    }

    pub fn registry(&self) -> Arc<HealthRegistry> {
        self.registry.clone()
    }

    /// Start the health check HTTP server
    pub async fn start(&self) -> Result<()> {
        let app = Router::new()
            .route("/health", get(health_handler))
            .route("/ready", get(ready_handler))
            .route("/metrics", get(metrics_handler))
            .with_state(self.registry.clone());

        let addr = format!("0.0.0.0:{}", self.port);
        info!(port = self.port, "Starting health check server on {}", addr);

        let listener = tokio::net::TcpListener::bind(&addr)
            .await
            .with_context(|| format!("Failed to bind health check server to {}", addr))?;

        axum::serve(listener, app)
            .await
            .context("Health check server error")?;

        Ok(())
    }
}

#[cfg(feature = "svc")]
async fn health_handler(
    State(registry): axum::extract::State<Arc<HealthRegistry>>,
) -> (StatusCode, Json<SystemHealth>) {
    let health = registry.get_health().await;
    let status = match health.status {
        HealthStatus::Healthy => StatusCode::OK,
        HealthStatus::Degraded => StatusCode::OK, // Still accept requests
        HealthStatus::Unhealthy => StatusCode::SERVICE_UNAVAILABLE,
    };
    (status, Json(health))
}

#[cfg(feature = "svc")]
async fn ready_handler(
    State(registry): axum::extract::State<Arc<HealthRegistry>>,
) -> (StatusCode, Json<serde_json::Value>) {
    let ready = registry.is_ready().await;
    let status = if ready {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (
        status,
        Json(serde_json::json!({
            "ready": ready,
            "timestamp": chrono::Utc::now().to_rfc3339(),
        })),
    )
}

#[cfg(feature = "svc")]
async fn metrics_handler() -> (StatusCode, String) {
    use crate::metrics::metrics;
    match metrics().gather() {
        Ok(metrics_text) => (StatusCode::OK, metrics_text),
        Err(e) => {
            warn!(error = %e, "Failed to gather metrics");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to gather metrics: {}", e),
            )
        }
    }
}
