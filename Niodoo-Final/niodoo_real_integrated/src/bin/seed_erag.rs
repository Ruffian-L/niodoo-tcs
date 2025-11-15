#[cfg(feature = "cli_bins")]
mod cli {
    use std::fs::File;
    use std::io::{BufRead, BufReader};
    use std::path::{Path, PathBuf};
    use std::time::Instant;

    use anyhow::{anyhow, Context, Result};
    use clap::Parser;
    use qdrant_client::qdrant::vectors_config::Config as VectorsConfigEnum;
    use qdrant_client::Payload;
    use qdrant_client::{
        config::QdrantConfig,
        qdrant::{CreateCollection, PointStruct, UpsertPoints, VectorParams, VectorsConfig},
        Qdrant,
    };
    use rand::distributions::Uniform;
    use rand::rngs::ThreadRng;
    use rand::Rng;
    use serde_json::Value;
    use tracing::{info, warn};
    use uuid::Uuid;

    /// Default HTTP endpoint fallback if no environment override is provided.
    const DEFAULT_QDRANT_ENDPOINT: &str = "http://127.0.0.1:6333";
    /// Safe fallback for embedding dimension when environment overrides are absent.
    const DEFAULT_VECTOR_DIMENSION: usize = 2560;

    #[derive(Parser, Debug)]
    #[command(
        name = "seed_erag",
        about = "Seed Qdrant ERAG collections with Euler-style experiences"
    )]
    struct SeedArgs {
        /// Name of the Qdrant collection to populate
        #[arg(long)]
        collection: String,

        /// Number of points to ingest from the JSONL payload
        #[arg(long)]
        points: usize,

        /// Path to JSONL file containing ERAG payloads
        #[arg(long = "memory-file")]
        memory_file: PathBuf,

        /// Qdrant HTTP endpoint (defaults to QDRANT_URL/QDRANT_ENDPOINT env or http://127.0.0.1:6333)
        #[arg(long)]
        endpoint: Option<String>,

        /// Vector dimension override (defaults to QDRANT_VECTOR_DIM/QDRANT_VECTOR_SIZE env or 2560)
        #[arg(long = "dim")]
        vector_dim: Option<usize>,
    }

    pub async fn run() -> Result<()> {
        tracing_subscriber::fmt()
            .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
            .with_target(false)
            .try_init()
            .ok();

        let args = SeedArgs::parse();

        if !args.memory_file.exists() {
            return Err(anyhow!(
                "memory file '{}' does not exist",
                args.memory_file.display()
            ));
        }
        if args.points == 0 {
            return Err(anyhow!("points must be greater than zero"));
        }

        let endpoint = resolve_endpoint(args.endpoint.as_deref());
        let vector_dim = resolve_vector_dim(args.vector_dim);

        info!(
            collection = %args.collection,
            %endpoint,
            vector_dim,
            target_points = args.points,
            "starting ERAG seeding run"
        );

        let config = QdrantConfig::from_url(&endpoint);
        let client = Qdrant::new(config)?;

        ensure_collection(&client, &args.collection, vector_dim).await?;
        let points = load_points(&args.memory_file, args.points, vector_dim)?;
        if points.is_empty() {
            warn!("no payloads were loaded; nothing to insert");
            return Ok(());
        }

        ingest_points(&client, &args.collection, &points).await?;

        info!(
            inserted = points.len(),
            collection = %args.collection,
            "completed ERAG seeding run"
        );
        Ok(())
    }

    fn resolve_endpoint(override_endpoint: Option<&str>) -> String {
        let env_keys = ["QDRANT_URL", "QDRANT_ENDPOINT", "TEST_ENDPOINT_QDRANT"];
        if let Some(explicit) = override_endpoint {
            return explicit.trim_end_matches('/').to_string();
        }
        for key in env_keys {
            if let Ok(value) = std::env::var(key) {
                if !value.trim().is_empty() {
                    return value.trim().trim_end_matches('/').to_string();
                }
            }
        }
        DEFAULT_QDRANT_ENDPOINT.to_string()
    }

    fn resolve_vector_dim(override_dim: Option<usize>) -> usize {
        if let Some(dim) = override_dim {
            return dim;
        }
        let env_keys = [
            "QDRANT_VECTOR_DIM",
            "QDRANT_VECTOR_SIZE",
            "NIODOO_EMBED_DIM",
            "EMBEDDING_DIMENSION",
        ];
        for key in env_keys {
            if let Ok(value) = std::env::var(key) {
                if let Ok(parsed) = value.trim().parse::<usize>() {
                    return parsed;
                }
            }
        }
        DEFAULT_VECTOR_DIMENSION
    }

    async fn ensure_collection(client: &Qdrant, collection: &str, vector_dim: usize) -> Result<()> {
        if client.collection_exists(collection).await? {
            info!(%collection, "collection already exists; proceeding with upsert");
            return Ok(());
        }

        let vector_params = VectorParams {
            size: vector_dim as u64,
            distance: 3,       // Cosine similarity
            datatype: Some(0), // Float32
            hnsw_config: None,
            quantization_config: None,
            multivector_config: None,
            on_disk: None,
        };

        let vectors_config = VectorsConfig {
            config: Some(VectorsConfigEnum::Params(vector_params)),
        };
        let create = CreateCollection {
            collection_name: collection.to_string(),
            vectors_config: Some(vectors_config),
            ..Default::default()
        };

        client.create_collection(create).await?;
        info!(%collection, vector_dim, "created new collection");
        Ok(())
    }

    fn load_points(
        path: &Path,
        target_points: usize,
        vector_dim: usize,
    ) -> Result<Vec<PointStruct>> {
        let file = File::open(path).context("failed to open memory file")?;
        let reader = BufReader::new(file);

        let mut rng = rand::thread_rng();
        let mut points = Vec::with_capacity(target_points.min(1024));
        let mut count = 0usize;

        let started_at = Instant::now();
        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }

            let payload_value: Value = serde_json::from_str(&line)
                .with_context(|| format!("failed to parse JSON payload at line {}", count + 1))?;

            let payload = value_to_payload(payload_value)?;
            let vector = random_embedding(&mut rng, vector_dim);
            let point_id = Uuid::new_v4().to_string();
            points.push(PointStruct::new(point_id, vector, payload));
            count += 1;

            if count >= target_points {
                break;
            }
        }

        info!(
            inserted = points.len(),
            elapsed_ms = started_at.elapsed().as_millis(),
            vector_dim,
            "loaded payloads from JSONL"
        );

        Ok(points)
    }

    fn value_to_payload(value: Value) -> Result<Payload> {
        Payload::try_from(value).map_err(|err| anyhow!("invalid payload JSON: {err}"))
    }

    fn random_embedding(rng: &mut ThreadRng, vector_dim: usize) -> Vec<f32> {
        let distribution = Uniform::new(-1.0f32, 1.0f32);
        (0..vector_dim).map(|_| rng.sample(distribution)).collect()
    }

    async fn ingest_points(
        client: &Qdrant,
        collection: &str,
        points: &[PointStruct],
    ) -> Result<()> {
        const BATCH_SIZE: usize = 128;
        let mut inserted_total = 0usize;
        for chunk in points.chunks(BATCH_SIZE) {
            let batch = chunk.to_vec();
            let upsert = UpsertPoints {
                collection_name: collection.to_string(),
                points: batch,
                wait: Some(false),
                ordering: None,
                ..Default::default()
            };
            client.upsert_points(upsert).await?;
            inserted_total += chunk.len();
            info!(inserted_total, collection, "seeded ERAG points");
        }
        Ok(())
    }
}

#[cfg(feature = "cli_bins")]
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    cli::run().await
}

#[cfg(not(feature = "cli_bins"))]
fn main() {
    eprintln!("Enable the `cli_bins` feature to run `seed_erag`.");
}
