//! Mock Shard Telemetry Server for testing federated resilience.
//!
//! This server implements the ShardTelemetry gRPC service to provide
//! synthetic topology signatures for testing the federated resilience system.

use anyhow::Result;
use niodoo_real_integrated::federated::proto::{
    shard_telemetry_server::{ShardTelemetry, ShardTelemetryServer},
    NodeSignatureBlob, ShardSignatureRequest, ShardSignatureResponse,
};
use niodoo_real_integrated::tcs_analysis::TopologicalSignature;
use tcs_tqft::Cobordism;
use tonic::{transport::Server, Request, Response, Status};
use tracing::{info, warn};

#[derive(Debug, Default)]
struct ShardTelemetryService {
    shard_id: String,
}

#[tonic::async_trait]
impl ShardTelemetry for ShardTelemetryService {
    async fn pull_signatures(
        &self,
        request: Request<ShardSignatureRequest>,
    ) -> Result<Response<ShardSignatureResponse>, Status> {
        let req = request.into_inner();
        info!(
            shard_id = %self.shard_id,
            endpoint_count = req.endpoints.len(),
            "Received pull_signatures request"
        );

        // Generate synthetic signatures for each requested endpoint
        let mut signatures = Vec::new();
        for (idx, endpoint) in req.endpoints.iter().enumerate() {
            let signature = generate_synthetic_signature(&self.shard_id, idx, endpoint);
            let payload = match serialize_signature(&signature) {
                Ok(bytes) => bytes,
                Err(err) => {
                    warn!(error = %err, "Failed to serialize shard signature");
                    return Err(Status::internal("signature serialization failed"));
                }
            };
            signatures.push(NodeSignatureBlob {
                shard_id: format!("{}-{}", self.shard_id, idx),
                payload,
            });
        }

        Ok(Response::new(ShardSignatureResponse { signatures }))
    }
}

fn generate_synthetic_signature(
    shard_id: &str,
    idx: usize,
    endpoint: &str,
) -> TopologicalSignature {
    use rand::Rng;

    let mut rng = rand::thread_rng();
    let spectral_gap = 0.6 + rng.gen_range(0.0..1.0) + idx as f64 * 0.05;
    let knot_complexity = 0.2 + idx as f64 * 0.03;
    let persistence_entropy = 0.15 + rng.gen_range(0.0..0.2);
    let cobordism = if idx % 2 == 0 {
        Some(Cobordism::Identity)
    } else {
        Some(Cobordism::Split)
    };
    let endpoint_tag = endpoint.replace(':', "-");

    TopologicalSignature::new(
        Vec::new(),
        [1, 1 + idx, 0],
        knot_complexity,
        format!("poly_{}_{}_{}", shard_id, endpoint_tag, idx),
        2,
        cobordism,
        3.0,
        persistence_entropy,
        spectral_gap,
    )
}

fn serialize_signature(signature: &TopologicalSignature) -> Result<Vec<u8>> {
    Ok(bincode::serialize(signature)?)
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let shard_id = std::env::var("SHARD_ID").unwrap_or_else(|_| "mock-shard".to_string());
    let addr = std::env::var("TELEMETRY_ADDR")
        .unwrap_or_else(|_| "0.0.0.0:50051".to_string())
        .parse()?;

    let service = ShardTelemetryService { shard_id };

    info!(addr = %addr, shard_id = %service.shard_id, "Starting Shard Telemetry Server");

    Server::builder()
        .add_service(ShardTelemetryServer::new(service))
        .serve(addr)
        .await?;

    Ok(())
}
