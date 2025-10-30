use std::cmp::Ordering;
use std::time::Duration;

use bincode::deserialize;
use chrono::{DateTime, TimeZone, Utc};
use prost::Message;
use thiserror::Error;
use tokio::time;
use tracing::instrument;

use crate::pipeline::Pipeline;
use crate::tcs_analysis::TopologicalSignature;

pub mod proto {
    tonic::include_proto!("niodoo.federated");
}

pub type NodeSig = TopologicalSignature;
type FluxCoeff = f64;

/// Maximum number of telemetry samples retained to bound memory while
/// keeping an adequate rolling horizon for resilience calculations.
const MAX_TRACE_MEMORY: usize = 512;

#[derive(Debug, Clone)]
struct FluxObservation {
    value: f64,
    timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Default)]
pub struct FluxMetrics {
    pub fused_flux: f64,
    pub shard_count: u32,
    pub interquartile_range: f64,
    pub median_gap: f64,
    pub shard_id: Option<String>,
    pub latest_timestamp: Option<DateTime<Utc>>,
}

#[derive(Debug, Error)]
pub enum FederatedError {
    #[error("failed to decode flux telemetry: {0}")]
    FluxDecode(#[from] prost::DecodeError),
    #[error("invalid telemetry timestamp: {0}")]
    InvalidTimestamp(i64),
    #[error("failed to deserialize shard signature: {0}")]
    SignatureDecode(#[from] bincode::Error),
    #[error("transport setup failed: {0}")]
    Transport(#[from] tonic::transport::Error),
    #[error("gRPC request failed: {0}")]
    Rpc(#[from] tonic::Status),
    #[error("fetch_shard_signatures exceeded timeout of {0:?}")]
    RpcTimeout(Duration),
}

pub type FederatedResult<T> = Result<T, FederatedError>;

#[derive(Debug, Clone, Default)]
pub struct NodalDiagnostics {
    flux_traces: Vec<FluxObservation>,
}

impl NodalDiagnostics {
    pub fn new() -> Self {
        Self::default()
    }

    #[instrument(skip(self, proto_flux, interquartile_gaps))]
    pub fn merge_shard_metrics(
        &mut self,
        proto_flux: &[u8],
        interquartile_gaps: &[f64],
    ) -> FederatedResult<FluxMetrics> {
        let proto_batch = proto::FluxTraceBatch::decode(proto_flux)?;
        let shard_identifier =
            (!proto_batch.shard_id.is_empty()).then(|| proto_batch.shard_id.clone());

        let mut shard_samples = 0u32;
        for trace in proto_batch.traces {
            let timestamp = Utc
                .timestamp_millis_opt(trace.timestamp_ms)
                .single()
                .ok_or(FederatedError::InvalidTimestamp(trace.timestamp_ms))?;

            self.flux_traces.push(FluxObservation {
                value: trace.value,
                timestamp,
            });
            shard_samples += 1;
        }

        if self.flux_traces.len() > MAX_TRACE_MEMORY {
            let excess = self.flux_traces.len() - MAX_TRACE_MEMORY;
            self.flux_traces.drain(0..excess);
        }

        let (median_gap, interquartile_range) = match Self::compute_quartiles(interquartile_gaps) {
            Some((q1, median_value, q3)) => {
                let iqr = (q3 - q1).max(f64::EPSILON);
                (median_value, iqr)
            }
            None => (1.0, 1.0),
        };
        let gap_weight = 1.0 + median_gap / interquartile_range;

        let latest_timestamp = self.flux_traces.last().map(|obs| obs.timestamp);
        let timespan_ms = match (self.flux_traces.first(), latest_timestamp) {
            (Some(first), Some(last)) => last
                .signed_duration_since(first.timestamp)
                .num_milliseconds()
                .abs()
                .max(1) as f64,
            _ => 1.0,
        };

        let mut weighted_sum = 0.0;
        let mut weight_total = 0.0;

        for observation in &self.flux_traces {
            let age_ms = latest_timestamp
                .map(|latest| {
                    latest
                        .signed_duration_since(observation.timestamp)
                        .num_milliseconds()
                        .abs() as f64
                })
                .unwrap_or_default();
            let recency_weight = 1.0 - (age_ms / timespan_ms).min(1.0);
            let weight = gap_weight * (recency_weight + f64::EPSILON);
            weighted_sum += observation.value * weight;
            weight_total += weight;
        }

        let fused_flux = if weight_total > f64::EPSILON {
            weighted_sum / weight_total
        } else {
            0.0
        };

        Ok(FluxMetrics {
            fused_flux,
            shard_count: shard_samples,
            interquartile_range,
            median_gap,
            shard_id: shard_identifier,
            latest_timestamp,
        })
    }

    fn compute_quartiles(samples: &[f64]) -> Option<(f64, f64, f64)> {
        if samples.is_empty() {
            return None;
        }
        let mut sorted = samples.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

        let median_value = median(&sorted);
        let mid = sorted.len() / 2;
        let (lower_half, upper_half) = if sorted.len() % 2 == 0 {
            (&sorted[..mid], &sorted[mid..])
        } else {
            (&sorted[..mid], &sorted[mid + 1..])
        };

        let q1 = if lower_half.is_empty() {
            median_value
        } else {
            median(lower_half)
        };

        let q3 = if upper_half.is_empty() {
            median_value
        } else {
            median(upper_half)
        };

        Some((q1, median_value, q3))
    }
}

fn median(sorted: &[f64]) -> f64 {
    debug_assert!(!sorted.is_empty());
    let len = sorted.len();
    if len % 2 == 1 {
        sorted[len / 2]
    } else {
        let upper = len / 2;
        let lower = upper - 1;
        (sorted[lower] + sorted[upper]) * 0.5
    }
}

// FederatedResilienceOrchestrator trait for distributed recovery
pub trait FederatedResilienceOrchestrator {
    fn aggregate_topology(&self, shards: &[NodeSig]) -> FluxCoeff;
}

// Impl for Pipeline: Aggregate via mean spectral gap
impl FederatedResilienceOrchestrator for Pipeline {
    fn aggregate_topology(&self, shards: &[NodeSig]) -> FluxCoeff {
        if shards.is_empty() {
            return 1.0;
        }
        let sum_gaps = shards.iter().map(|sig| sig.spectral_gap).sum::<f64>();
        sum_gaps / shards.len() as f64
    }
}

#[derive(Debug, Clone)]
pub struct ShardClient {
    client: proto::shard_telemetry_client::ShardTelemetryClient<tonic::transport::Channel>,
    request_timeout: Duration,
}

impl ShardClient {
    const DEFAULT_TIMEOUT: Duration = Duration::from_secs(1);

    pub async fn connect(endpoint: &str) -> FederatedResult<Self> {
        Self::connect_with_timeout(endpoint, Self::DEFAULT_TIMEOUT).await
    }

    pub async fn connect_with_timeout(
        endpoint: &str,
        request_timeout: Duration,
    ) -> FederatedResult<Self> {
        let channel = tonic::transport::Endpoint::from_shared(endpoint.to_string())?
            .connect_timeout(request_timeout)
            .concurrency_limit(32)
            .tcp_nodelay(true)
            .connect()
            .await?;
        let client = proto::shard_telemetry_client::ShardTelemetryClient::new(channel);

        Ok(Self {
            client,
            request_timeout,
        })
    }

    pub async fn fetch_shard_signatures(
        &self,
        endpoints: &[String],
    ) -> FederatedResult<Vec<NodeSig>> {
        if endpoints.is_empty() {
            return Ok(Vec::new());
        }

        let mut client = self.client.clone();
        let request = proto::ShardSignatureRequest {
            endpoints: endpoints.to_vec(),
        };

        let response = time::timeout(self.request_timeout, client.pull_signatures(request))
            .await
            .map_err(|_| FederatedError::RpcTimeout(self.request_timeout))??;

        let payload = response.into_inner();

        let mut result = Vec::with_capacity(payload.signatures.len());
        for blob in payload.signatures {
            let signature: NodeSig = deserialize(&blob.payload)?;
            result.push(signature);
        }

        Ok(result)
    }
}

#[instrument(skip(orchestrator, flux_batches, interquartile_gaps))]
pub async fn orchestrate_federated_topology<O: FederatedResilienceOrchestrator + ?Sized>(
    orchestrator: &O,
    telemetry_endpoint: &str,
    shard_ids: &[String],
    flux_batches: &[Vec<u8>],
    interquartile_gaps: &[f64],
) -> FederatedResult<(FluxCoeff, FluxMetrics, Vec<NodeSig>)> {
    if shard_ids.is_empty() {
        return Ok((1.0, FluxMetrics::default(), Vec::new()));
    }

    let client = ShardClient::connect(telemetry_endpoint).await?;
    let signatures = client.fetch_shard_signatures(shard_ids).await?;
    let flux_coeff = orchestrator.aggregate_topology(&signatures);

    let mut diagnostics = NodalDiagnostics::new();
    let mut metrics = FluxMetrics::default();
    for payload in flux_batches {
        metrics = diagnostics.merge_shard_metrics(payload, interquartile_gaps)?;
    }

    Ok((flux_coeff, metrics, signatures))
}

impl Pipeline {
    #[instrument(skip(self, flux_batches, interquartile_gaps))]
    pub async fn synchronize_federated_topology(
        &self,
        telemetry_endpoint: &str,
        shard_ids: &[String],
        flux_batches: &[Vec<u8>],
        interquartile_gaps: &[f64],
    ) -> FederatedResult<(FluxCoeff, FluxMetrics)> {
        let (coeff, metrics, _) = orchestrate_federated_topology(
            self,
            telemetry_endpoint,
            shard_ids,
            flux_batches,
            interquartile_gaps,
        )
        .await?;
        Ok((coeff, metrics))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, Utc};
    use prost::Message;

    #[test]
    fn test_merge_shard_metrics() {
        let mut diagnostics = NodalDiagnostics::new();
        let now = Utc::now();

        let batch = proto::FluxTraceBatch {
            shard_id: "alpha".into(),
            traces: vec![
                proto::FluxTrace {
                    value: 0.72,
                    timestamp_ms: now.timestamp_millis(),
                },
                proto::FluxTrace {
                    value: 0.64,
                    timestamp_ms: (now + Duration::milliseconds(8)).timestamp_millis(),
                },
                proto::FluxTrace {
                    value: 0.91,
                    timestamp_ms: (now + Duration::milliseconds(16)).timestamp_millis(),
                },
            ],
        };

        let payload = batch.encode_to_vec();
        let gaps = vec![0.12, 0.18, 0.22, 0.34, 0.40];

        let metrics = diagnostics
            .merge_shard_metrics(&payload, &gaps)
            .expect("telemetry decode must succeed");

        assert_eq!(metrics.shard_count, 3);
        assert!(metrics.fused_flux.is_finite());
        assert!(metrics.fused_flux > 0.0);
        assert_eq!(metrics.shard_id.as_deref(), Some("alpha"));
        assert!(metrics.interquartile_range >= f64::EPSILON);
        assert!(metrics.median_gap > 0.0);
    }

    #[test]
    fn quartile_matches_expected() {
        let samples = [1.0, 3.0, 5.0, 7.0];
        let (q1, median_value, q3) =
            NodalDiagnostics::compute_quartiles(&samples).expect("quartiles must exist");
        assert!((q1 - 2.0).abs() < 1e-9);
        assert!((median_value - 4.0).abs() < 1e-9);
        assert!((q3 - 6.0).abs() < 1e-9);
    }
}
