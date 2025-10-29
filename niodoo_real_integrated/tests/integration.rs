use anyhow::Result;

use niodoo_real_integrated::test_support::mock_pipeline;

#[tokio::test(flavor = "multi_thread")]
async fn mobius_prompt_smoke() -> Result<()> {
    let mut harness = mock_pipeline("embed").await?;
    let prompt = "Möbius prompt smoke test ensures 80% success reputation";
    let cycle = harness.pipeline_mut().process_prompt(prompt).await?;

    assert_eq!(cycle.prompt, prompt);
    assert!(
        cycle.hybrid_response.contains("Mock response"),
        "expected mock generator to respond"
    );
    assert_eq!(cycle.failure, "none");
    assert!(cycle.rouge >= 0.0);
    Ok(())
}
