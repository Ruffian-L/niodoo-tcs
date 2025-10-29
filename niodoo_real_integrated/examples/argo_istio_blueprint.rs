use anyhow::Result;

/// Simple orchestration experiment that stitches together several prompts and
/// inspects the pipeline response quality. This runs in mock mode so it is
/// deterministic and CI-friendly while still exercising the full pipeline
/// surface.
#[tokio::main]
async fn main() -> Result<()> {
    let mut harness = niodoo_real_integrated::test_support::mock_pipeline("embed").await?;

    let blueprint_steps = [
        "Argo mesh gateway handshake with Möbius resilience",
        "Istio traffic policy to stabilise emotional variance",
        "Federated healing loop verifying consensus across pods",
    ];

    let mut mesh_scores = Vec::with_capacity(blueprint_steps.len());

    for step in blueprint_steps {
        let cycle = harness.pipeline_mut().process_prompt(step).await?;
        let compass = &cycle.compass;
        let score = compass.ucb1_score.unwrap_or_else(|| {
            compass
                .mcts_branches
                .iter()
                .map(|b| b.ucb_score)
                .sum::<f64>()
                / compass.mcts_branches.len().max(1) as f64
        });
        mesh_scores.push((step, score, cycle.rouge));
    }

    let average_score =
        mesh_scores.iter().map(|(_, score, _)| score).sum::<f64>() / mesh_scores.len() as f64;

    println!("Argo-Istio mesh synthesis summary:\n");
    for (step, score, rouge) in mesh_scores {
        println!("- {step}: ucb1={score:.3}, rouge={rouge:.3}");
    }
    println!("\nCohesion score: {:.3}", average_score);

    Ok(())
}
