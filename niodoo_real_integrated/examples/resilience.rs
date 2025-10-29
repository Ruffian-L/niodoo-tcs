// future
use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    let mut harness = niodoo_real_integrated::test_support::mock_pipeline("embed").await?;

    let scenarios = [
        "Stabilise Möbius drift when resilience falters",
        "Escalate regenerative response during entropy spike",
        "Diffuse recursive loop with compassionate debrief",
    ];

    let mut outcomes = Vec::with_capacity(scenarios.len());
    let mut recovery_successes = 0usize;

    for scenario in scenarios {
        let cycle = harness.pipeline_mut().process_prompt(scenario).await?;
        let failure = cycle.failure != "none";
        let resilience_gain = 1.0 - cycle.pad_state.entropy;
        if !failure {
            recovery_successes += 1;
        }
        outcomes.push((scenario, failure, resilience_gain));
    }

    println!("Resilience audit across {} scenarios:\n", outcomes.len());
    for (scenario, failure, resilience_gain) in &outcomes {
        println!(
            "- {scenario}: status={}, resilience_gain={:.3}",
            if *failure { "retry" } else { "stable" },
            resilience_gain
        );
    }

    let stability_ratio = recovery_successes as f64 / outcomes.len() as f64;
    println!("\nStability ratio: {:.3}", stability_ratio);

    Ok(())
}
