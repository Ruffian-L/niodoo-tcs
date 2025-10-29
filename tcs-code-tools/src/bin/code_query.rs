use anyhow::Result;
use qdrant_client::qdrant::{
    SearchPoints, WithPayloadSelector, with_payload_selector::SelectorOptions,
};
use std::env;
use tcs_ml::InferenceModelBackend;

const DIMS: usize = 512;

#[tokio::main]
async fn main() -> Result<()> {
    let mut args = std::env::args();
    let _ = args.next();
    let query = args
        .next()
        .unwrap_or_else(|| "find topology engine".to_string());
    let k: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(10);

    let collection = env::var("QDRANT_COLLECTION").unwrap_or_else(|_| "code_index".to_string());
    let qdrant_url = env::var("QDRANT_URL").unwrap_or_else(|_| "http://127.0.0.1:6333".to_string());
    let client = qdrant_client::Qdrant::from_url(&qdrant_url).build()?;

    let backend = InferenceModelBackend::new("code-query")?;
    let embedding = backend.extract_embeddings(&query)?;
    if embedding.len() != DIMS {
        return Ok(());
    }

    let result = client
        .search_points(SearchPoints {
            collection_name: collection,
            vector: embedding,
            limit: k,
            with_payload: Some(WithPayloadSelector {
                selector_options: Some(SelectorOptions::Enable(true)),
            }),
            ..Default::default()
        })
        .await?;

    for point in result.result {
        let score = point.score;
        let path_opt = point.payload.get("path").and_then(|v| v.as_str());
        match path_opt {
            Some(p) => println!("{score:.4}  {}", p),
            None => println!("{score:.4}  <unknown>"),
        }
    }

    Ok(())
}
