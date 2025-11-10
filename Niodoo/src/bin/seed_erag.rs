use anyhow::Result;
use niodoo_cli::embedding::LocalEmbedder;
use serde_json::json;

#[tokio::main]
async fn main() -> Result<()> {
    let embedder = LocalEmbedder::from_env()?;
    
    let memories = vec![
        (1, "During convergence spikes, hyperfocus detection watches the beta_meta slope and captures payload snapshots of the prompt that triggered the spike.", "Discover", 1762667859),
        (2, "When the system drifts into cooldown, ERAG retrieval biases toward memories tagged Persist so the cognitive loop keeps enough friction to stay engaged.", "Persist", 1762667860),
        (3, "Whenever hyperfocus sustainment lasts longer than 90 seconds, we store the control inputs and latency curve so future requests can reuse the cadence.", "Master", 1762667861),
    ];
    
    let client = reqwest::Client::new();
    let qdrant_url = "http://127.0.0.1:6333";
    
    for (id, text, compass, timestamp) in memories {
        println!("Embedding memory {}: {} ({})", id, compass, &text[..60]);
        let embedding = embedder.embed(text)?;
        
        let point = json!({
            "id": id,
            "vector": embedding,
            "payload": {
                "text": text,
                "compass_quadrant": compass,
                "timestamp": timestamp
            }
        });
        
        let points = json!({ "points": [point] });
        
        let response = client
            .put(format!("{}/collections/niodoo_erag_memories/points?wait=true", qdrant_url))
            .json(&points)
            .send()
            .await?;
        
        if response.status().is_success() {
            println!("  ✓ Upserted memory {} with {} dims", id, embedding.len());
        } else {
            eprintln!("  ✗ Failed: {}", response.status());
            eprintln!("  {}", response.text().await?);
        }
    }
    
    println!("\n✓ Seeded 3 ERAG memories with embeddings");
    Ok(())
}
