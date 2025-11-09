use std::cmp::Ordering;

use anyhow::{Context, Result};
use chrono::Utc;
use reqwest::Client;
use serde_json::json;

#[tokio::main]
async fn main() -> Result<()> {
    let config = niodoo_cli::erag::EragConfig::from_file("config/erag.toml")?;
    let embedder = niodoo_cli::embedding::LocalEmbedder::from_env()?;

    let memories: Vec<(&str, &str)> = vec![
        (
            "Discover",
            "During convergence spikes, hyperfocus detection watches the beta_meta slope and captures payload snapshots of the prompt that triggered the spike.",
        ),
        (
            "Persist",
            "When the system drifts into cooldown, ERAG retrieval biases toward memories tagged Persist so the cognitive loop keeps enough friction to stay engaged.",
        ),
        (
            "Master",
            "Whenever hyperfocus sustainment lasts longer than 90 seconds, we store the control inputs and latency curve so future requests can reuse the cadence.",
        ),
    ];

    let mut points = Vec::new();
    let expected_dim = config.qdrant.vector_size;

    for (idx, (compass, text)) in memories.iter().enumerate() {
        let mut vector = embedder.embed(text)?;
        match vector.len().cmp(&expected_dim) {
            Ordering::Less => vector.resize(expected_dim, 0.0),
            Ordering::Greater => vector.truncate(expected_dim),
            Ordering::Equal => {}
        }

        points.push(json!({
            "id": idx as u64 + 1,
            "vector": vector,
            "payload": {
                "compass_quadrant": *compass,
                "timestamp": Utc::now().timestamp(),
                "text": *text,
            }
        }));
    }

    let client = Client::builder()
        .timeout(std::time::Duration::from_secs(15))
        .build()
        .context("failed to build reqwest client")?;

    let base_url = config.qdrant.http_url.trim_end_matches('/');
    let url = format!(
        "{}/collections/{}/points?wait=true",
        base_url, config.qdrant.collection
    );

    let response = client
        .put(url)
        .json(&json!({ "points": points }))
        .send()
        .await
        .context("failed to upsert memories into Qdrant")?
        .text()
        .await?;

    println!("Seeding response: {}", response);
    Ok(())
}
