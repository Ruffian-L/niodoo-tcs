use anyhow::Result;

use niodoo_real_integrated::config::{CliArgs, HardwareProfile, TopologyMode};
use niodoo_real_integrated::pipeline::Pipeline;
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
    assert!(
        cycle.failure == "none" || cycle.failure == "soft",
        "expected non-hard outcome, got {}",
        cycle.failure
    );
    assert!(cycle.rouge >= 0.0);
    Ok(())
}

#[tokio::test]
async fn smoke_pipeline() {
    let args = CliArgs {
        hardware: HardwareProfile::Beelink, // CPU
        ..Default::default()
    };
    let mut pipeline = Pipeline::initialise_with_mode(args, TopologyMode::Hybrid)
        .await
        .unwrap();
    let prompt = "test query";
    let cycle = pipeline.process_prompt(prompt).await.unwrap();
    assert!(!cycle.hybrid_response.is_empty());
    assert!(cycle.latency_ms > 0.0);
    assert!(cycle.entropy.is_finite());
}
