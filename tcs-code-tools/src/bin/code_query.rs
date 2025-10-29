use anyhow::Result;
use qdrant_client::prelude::*;
use qdrant_client::qdrant::{QueryPoints, WithPayloadSelector, with_payload_selector::SelectorOptions};
use std::env;
use tcs_ml::InferenceModelBackend;

const DIMS: usize = 512;

#[tokio::main]
async fn main() -> Result<()> {
    let mut args = std::env::args();
    let _ = args.next();
    let query = args.next().unwrap_or_else(|| "find topology engine".to_string());
    let k: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(10);

    let collection = env::var("QDRANT_COLLECTION").unwrap_or_else(|_| "code_index".to_string());
    let qdrant_url = env::var("QDRANT_URL").unwrap_or_else(|_| "http://127.0.0.1:6333".to_string());
    let client = QdrantClient::from_url(qdrant_url).build()?;

    let backend = InferenceModelBackend::new("code-query")?;
    let embedding = backend.extract_embeddings(&query)?;
    if embedding.len() != DIMS { return Ok(()); }

    let result = client
        .query_points(&QueryPoints {
            collection_name: collection,
            limit: k,
            params: None,
            vector: embedding,
            with_payload: Some(WithPayloadSelector { selector_options: Some(SelectorOptions::Enable(true)) }),
            ..Default::default()
        })
        .await?;

    for point in result.result { 
        let score = point.score.unwrap_or_default();
        let path = point.payload.get("path").and_then(|v| v.as_str()).unwrap_or("<unknown>");
        println!("{score:.4}  {path}");
    }

    Ok(())
}

