use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use reqwest::Client;
use serde::Deserialize;
use serde_json::Value;
use tracing::{info, warn};

use crate::config::{env_value, BackendType, RuntimeConfig};

const DEFAULT_TIMEOUT_SECS: u64 = 5;

/// Performs live endpoint smoke verification for critical services.
///
/// This is intentionally strict: each endpoint must respond successfully within the timeout
/// window and without relying on any mock implementations.
pub struct ServiceSmokeVerifier {
    client: Client,
    timeout: Duration,
}

impl ServiceSmokeVerifier {
    /// Creates a verifier with the default timeout.
    pub fn new() -> Result<Self> {
        let timeout = Duration::from_secs(DEFAULT_TIMEOUT_SECS);
        let client = Client::builder()
            .timeout(timeout)
            .build()
            .context("failed to build reqwest client for smoke verification")?;

        Ok(Self { client, timeout })
    }

    /// Runs the smoke checks against the live services specified in the runtime config.
    pub async fn verify(&self, config: &RuntimeConfig) -> Result<()> {
        if std::env::var("NIODOO_SKIP_SMOKE")
            .map(|value| matches!(value.to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
            .unwrap_or(false)
        {
            warn!("NIODOO_SKIP_SMOKE set—skipping live service verification");
            return Ok(());
        }

        let mut failures: Vec<String> = Vec::new();

        if let Err(err) = self.check_qdrant(config).await {
            failures.push(format!("Qdrant: {err:#}"));
        } else {
            info!("✅ Qdrant endpoint responded within {:?}.", self.timeout);
        }

        if let Err(err) = self.check_generation_backend(config).await {
            failures.push(format!("Generation endpoint: {err:#}"));
        } else {
            info!(
                "✅ Generation endpoint responded within {:?}.",
                self.timeout
            );
        }

        if let Err(err) = self.check_training_service(config).await {
            failures.push(format!("Training service: {err:#}"));
        } else {
            info!(
                "✅ Training service endpoint responded within {:?}.",
                self.timeout
            );
        }

        if let Err(err) = self.check_ollama(config).await {
            failures.push(format!("Ollama curator: {err:#}"));
        } else {
            info!(
                "✅ Ollama curator endpoint responded within {:?}.",
                self.timeout
            );
        }

        if failures.is_empty() {
            Ok(())
        } else {
            Err(anyhow!(
                "Service smoke verification failed:\n{}",
                failures.join("\n")
            ))
        }
    }

    async fn check_qdrant(&self, config: &RuntimeConfig) -> Result<()> {
        let base = config.qdrant_url.trim_end_matches('/');
        let paths = ["health", "healthz"];

        let mut last_error: Option<anyhow::Error> = None;

        for path in paths {
            let endpoint = format!("{base}/{path}");
            let mut request = self.client.get(&endpoint);

            if let Some(api_key) = env_value("QDRANT_API_KEY") {
                if !api_key.is_empty() {
                    request = request.header("api-key", api_key);
                }
            } else {
                warn!("QDRANT_API_KEY not set; assuming public access for Qdrant smoke check");
            }

            let response = match request.send().await {
                Ok(resp) => resp,
                Err(err) => {
                    last_error = Some(anyhow!(
                        "failed to reach Qdrant health endpoint at {endpoint}: {err:#}"
                    ));
                    continue;
                }
            };

            if response.status().is_success() {
                let body = response
                    .text()
                    .await
                    .context("unable to read Qdrant health payload")?;

                let trimmed = body.trim();
                if trimmed.is_empty()
                    || trimmed.eq_ignore_ascii_case("ok")
                    || trimmed.eq_ignore_ascii_case("healthz check passed")
                {
                    return Ok(());
                }

                let payload: Value = serde_json::from_str(trimmed)
                    .context("unable to parse Qdrant health payload")?;

                return match payload.get("status") {
                    Some(Value::String(status)) if status.eq_ignore_ascii_case("ok") => Ok(()),
                    other => Err(anyhow!(
                        "unexpected Qdrant health payload: {:?}",
                        other.unwrap_or(&Value::Null)
                    )),
                };
            }

            if response.status().as_u16() == 404 && path == "health" {
                // Retry with /healthz on next iteration
                last_error = Some(anyhow!("Qdrant /health returned HTTP 404"));
                continue;
            }

            return Err(anyhow!("Qdrant {path} returned HTTP {}", response.status()));
        }

        Err(last_error.unwrap_or_else(|| {
            anyhow!("Qdrant health check failed for /health and /healthz endpoints")
        }))
    }

    async fn check_generation_backend(&self, config: &RuntimeConfig) -> Result<()> {
        match config.generation_backend {
            BackendType::OllamaCpu => self.check_ollama_generation(config).await,
            _ => self.check_vllm_generation(config).await,
        }
    }

    async fn check_vllm_generation(&self, config: &RuntimeConfig) -> Result<()> {
        let base = config.vllm_endpoint.trim_end_matches('/');
        let health_endpoint = format!("{base}/health");
        let response = self
            .client
            .get(&health_endpoint)
            .send()
            .await
            .with_context(|| {
                format!("failed to reach generation health endpoint at {health_endpoint}")
            })?;

        if response.status().is_success() {
            return Ok(());
        }

        // Some deployments may not expose /health; try /v1/models as a fallback.
        let models_endpoint = format!("{base}/v1/models");
        let response = self
            .client
            .get(&models_endpoint)
            .send()
            .await
            .with_context(|| {
                format!("failed to reach generation models endpoint at {models_endpoint}")
            })?;

        if !response.status().is_success() {
            return Err(anyhow!(
                "generation endpoint returned HTTP {} for /v1/models",
                response.status()
            ));
        }

        #[derive(Deserialize)]
        struct ModelList {
            data: Vec<Value>,
        }

        let body: ModelList = response
            .json()
            .await
            .context("unable to parse generation /v1/models payload")?;

        if body.data.is_empty() {
            return Err(anyhow!("generation endpoint returned an empty model list"));
        }

        Ok(())
    }

    async fn check_ollama_generation(&self, config: &RuntimeConfig) -> Result<()> {
        let base = config.ollama_endpoint.trim_end_matches('/');
        let health_endpoint = format!("{base}/api/health");
        if let Ok(response) = self.client.get(&health_endpoint).send().await {
            if response.status().is_success() {
                return Ok(());
            }
        }

        // Fallback to OpenAI-compatible /v1/models probe.
        let models_endpoint = format!("{base}/v1/models");
        let response = self
            .client
            .get(&models_endpoint)
            .send()
            .await
            .with_context(|| format!("failed to reach Ollama OpenAI shim at {models_endpoint}"))?;

        if !response.status().is_success() {
            return Err(anyhow!(
                "Ollama generation endpoint returned HTTP {} for /v1/models",
                response.status()
            ));
        }

        #[derive(Deserialize)]
        struct ModelList {
            data: Vec<Value>,
        }

        let body: ModelList = response
            .json()
            .await
            .context("unable to parse Ollama /v1/models payload")?;

        if body.data.is_empty() {
            return Err(anyhow!(
                "Ollama generation endpoint returned an empty model list"
            ));
        }

        Ok(())
    }

    async fn check_training_service(&self, config: &RuntimeConfig) -> Result<()> {
        let endpoint = format!(
            "{}/health",
            config.training_service_url.trim_end_matches('/')
        );
        let response = self
            .client
            .get(&endpoint)
            .send()
            .await
            .with_context(|| format!("failed to reach training service at {endpoint}"))?;

        if !response.status().is_success() {
            return Err(anyhow!(
                "training service health returned HTTP {}",
                response.status()
            ));
        }

        let payload: Value = response
            .json()
            .await
            .context("unable to parse training service health payload")?;

        match payload.get("status") {
            Some(Value::String(status)) if status.eq_ignore_ascii_case("healthy") => Ok(()),
            Some(Value::String(status)) if status.eq_ignore_ascii_case("ok") => Ok(()),
            other => Err(anyhow!(
                "unexpected training service health payload: {:?}",
                other.unwrap_or(&Value::Null)
            )),
        }
    }

    async fn check_ollama(&self, config: &RuntimeConfig) -> Result<()> {
        let endpoint = format!("{}/api/tags", config.ollama_endpoint.trim_end_matches('/'));
        let response = self
            .client
            .get(&endpoint)
            .send()
            .await
            .with_context(|| format!("failed to reach Ollama curator at {endpoint}"))?;

        if !response.status().is_success() {
            return Err(anyhow!(
                "Ollama curator returned HTTP {}",
                response.status()
            ));
        }

        #[derive(Deserialize)]
        struct OllamaTagList {
            models: Vec<Value>,
        }

        let payload: OllamaTagList = response
            .json()
            .await
            .context("unable to parse Ollama tag list payload")?;

        if payload.models.is_empty() {
            return Err(anyhow!(
                "Ollama curator responded but no models are available in tag list"
            ));
        }

        Ok(())
    }
}
