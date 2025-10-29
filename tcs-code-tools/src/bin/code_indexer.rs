use anyhow::{Context, Result};
use qdrant_client::prelude::*;
use qdrant_client::qdrant::{vectors_config, CreateCollection, Distance, PointStruct, VectorsConfig, VectorParams};
use std::env;
use std::fs;
use std::path::Path;
use tcs_ml::InferenceModelBackend;
use walkdir::WalkDir;

const DIMS: usize = 512;

fn should_index(path: &Path) -> bool {
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        matches!(ext, "rs" | "toml" | "md")
    } else {
        false
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let root = args.get(1).map(|s| s.as_str()).unwrap_or(".");
    let collection = env::var("QDRANT_COLLECTION").unwrap_or_else(|_| "code_index".to_string());
    let qdrant_url = env::var("QDRANT_URL").unwrap_or_else(|_| "http://127.0.0.1:6333".to_string());

    let client = QdrantClient::from_url(qdrant_url).build()?;

    // Ensure collection exists
    let _ = client
        .create_collection(&CreateCollection {
            collection_name: collection.clone(),
            vectors_config: Some(VectorsConfig {
                config: Some(vectors_config::Config::Params(VectorParams {
                    size: DIMS as u64,
                    distance: Distance::Cosine as i32,
                    hnsw_config: None,
                    quantization_config: None,
                    on_disk: None,
                })),
            }),
            ..Default::default()
        })
        .await;

    let backend = InferenceModelBackend::new("code-indexer")?;

    let mut points: Vec<PointStruct> = Vec::new();
    let mut idx: u64 = 1;

    for entry in WalkDir::new(root).into_iter().filter_map(Result::ok) {
        let path = entry.into_path();
        if path.is_file() && should_index(&path) {
            let content = fs::read_to_string(&path)
                .with_context(|| format!("failed to read {}", path.display()))?;
            let text = if content.len() > 16 * 1024 { content[..16 * 1024].to_string() } else { content };
            let vector = backend.extract_embeddings(&text)?;
            if vector.len() != DIMS { continue; }
            let id = idx;
            idx += 1;
            let payload = serde_json::json!({
                "path": path.display().to_string(),
            });
            points.push(PointStruct::new(id.into(), vector, payload));
            if points.len() >= 64 {
                client.upsert_points_blocking(collection.clone(), None, points.drain(..).collect()).await?;
            }
        }
    }

    if !points.is_empty() {
        client.upsert_points_blocking(collection, None, points).await?;
    }

    Ok(())
}

//

