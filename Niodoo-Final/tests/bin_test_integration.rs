use anyhow::Result;
use niodoo_consciousness::rag::local_embeddings::Document as LocalDocument;
use niodoo_consciousness::rag::RetrievalEngine;

fn main() -> Result<()> {
    let mut engine = RetrievalEngine::new()?;

    let mut doc = LocalDocument::new("doc1", "Consciousness explores Möbius empathy vectors");
    doc.metadata
        .insert("source".into(), "integration_smoke".into());
    engine.add_document(doc)?;

    let results = engine.retrieve("Möbius empathy")?;
    println!(
        "integration smoke test retrieved {} documents",
        results.len()
    );

    Ok(())
}
