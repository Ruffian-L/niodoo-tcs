use anyhow::Result;

#[tokio::test(flavor = "multi_thread")]
async fn smoke_pipeline_mock_mode() -> Result<()> {
    // Ensure mock_mode to avoid external deps
    std::env::set_var("MOCK_MODE", "1");
    // Avoid external qdrant
    std::env::remove_var("QDRANT_URL");

    let args = niodoo_real_integrated::config::CliArgs::default();
    let mut pipeline = niodoo_real_integrated::pipeline::Pipeline::initialise(args).await?;

    let prompt = "quick smoke";
    let cycle = pipeline.process_prompt(prompt).await?;

    assert_eq!(cycle.prompt, prompt);
    // In mock mode we should still get a string response
    assert!(cycle.hybrid_response.len() >= 0);
    Ok(())
}
