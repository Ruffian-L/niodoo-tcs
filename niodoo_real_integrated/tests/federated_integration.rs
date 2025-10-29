use std::net::SocketAddr;
use std::time::Duration;

use anyhow::Result;
use chrono::Utc;
use niodoo_real_integrated::federated::proto::{
    FluxTrace, FluxTraceBatch, NodeSignatureBlob, ShardSignatureRequest, ShardSignatureResponse,
    shard_telemetry_server::{ShardTelemetry, ShardTelemetryServer},
};
use niodoo_real_integrated::federated::{
    FederatedResilienceOrchestrator, NodalDiagnostics, ShardClient, orchestrate_federated_topology,
};
use niodoo_real_integrated::tcs_analysis::TopologicalSignature;
use prost::Message;
use tokio::net::TcpListener;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::{Request, Response, Status, transport::Server};

#[derive(Default)]
struct TestTelemetry;

#[tonic::async_trait]
impl ShardTelemetry for TestTelemetry {
    async fn pull_signatures(
        &self,
        request: Request<ShardSignatureRequest>,
    ) -> Result<Response<ShardSignatureResponse>, Status> {
        let req = request.into_inner();
        let mut signatures = Vec::new();
        for (idx, endpoint) in req.endpoints.iter().enumerate() {
            let topo = make_signature(endpoint, idx);
            let payload = bincode::serialize(&topo)
                .map_err(|err| Status::internal(format!("serialize signature failed: {err}")))?;
            signatures.push(NodeSignatureBlob {
                shard_id: format!("test-{}", idx),
                payload,
            });
        }

        Ok(Response::new(ShardSignatureResponse { signatures }))
    }
}

fn make_signature(endpoint: &str, idx: usize) -> TopologicalSignature {
    let spectral_gap = 0.8 + idx as f64 * 0.1;
    let knot_complexity = 0.25 + idx as f64 * 0.05;
    let entropy = 0.2 + idx as f64 * 0.02;
    TopologicalSignature::new(
        Vec::new(),
        [1, 2 + idx, 0],
        knot_complexity,
        format!("poly_{}_{}", endpoint.replace(':', "-"), idx),
        2,
        None,
        2.5,
        entropy,
        spectral_gap,
    )
}

struct MeanAggregator;

impl FederatedResilienceOrchestrator for MeanAggregator {
    fn aggregate_topology(&self, shards: &[TopologicalSignature]) -> f64 {
        if shards.is_empty() {
            return 1.0;
        }
        shards.iter().map(|sig| sig.spectral_gap).sum::<f64>() / shards.len() as f64
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn shard_client_round_trip() -> Result<()> {
    let addr = spawn_test_server().await?;
    let endpoint = format!("http://{}", addr);

    let client = ShardClient::connect_with_timeout(&endpoint, Duration::from_secs(2)).await?;
    let shard_ids = vec!["alpha".to_string(), "beta".to_string()];
    let signatures = client.fetch_shard_signatures(&shard_ids).await?;
    assert_eq!(signatures.len(), shard_ids.len());
    assert!(signatures.iter().all(|sig| sig.spectral_gap > 0.0));

    let mut diagnostics = NodalDiagnostics::new();
    let gaps: Vec<f64> = signatures.iter().map(|sig| sig.spectral_gap).collect();
    let flux_batch = FluxTraceBatch {
        shard_id: "alpha".into(),
        traces: vec![
            FluxTrace {
                value: 0.72,
                timestamp_ms: Utc::now().timestamp_millis(),
            },
            FluxTrace {
                value: 0.88,
                timestamp_ms: Utc::now().timestamp_millis() + 16,
            },
            FluxTrace {
                value: 0.91,
                timestamp_ms: Utc::now().timestamp_millis() + 24,
            },
        ],
    };
    let metrics = diagnostics.merge_shard_metrics(&flux_batch.encode_to_vec(), &gaps)?;
    assert!(metrics.fused_flux > 0.0);
    assert_eq!(metrics.shard_count, 3);

    let orchestrator = MeanAggregator;
    let (flux_coeff, synced_metrics, synced_signatures) = orchestrate_federated_topology(
        &orchestrator,
        &endpoint,
        &shard_ids,
        &[flux_batch.encode_to_vec()],
        &gaps,
    )
    .await?;

    assert_eq!(synced_signatures.len(), shard_ids.len());
    assert!(flux_coeff > 0.0);
    assert_eq!(synced_metrics.shard_count, 3);
    assert!(synced_metrics.fused_flux > 0.0);

    Ok(())
}

async fn spawn_test_server() -> Result<SocketAddr> {
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let addr = listener.local_addr()?;
    let incoming = TcpListenerStream::new(listener);
    tokio::spawn(async move {
        let service = TestTelemetry::default();
        Server::builder()
            .add_service(ShardTelemetryServer::new(service))
            .serve_with_incoming(incoming)
            .await
            .ok();
    });
    // Allow server a moment to start
    tokio::time::sleep(Duration::from_millis(50)).await;
    Ok(addr)
}
