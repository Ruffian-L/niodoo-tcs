use anyhow::Result;
use niodoo_real_integrated::config::CliArgs;
use niodoo_real_integrated::eval::metrics::{pearson, spearman};
use niodoo_real_integrated::eval::synthetic::{generate_prompts, reference_for};
use niodoo_real_integrated::pipeline::Pipeline;
use niodoo_real_integrated::tcs_analysis::TCSAnalyzer;
use tracing::info;

#[tokio::test]
#[ignore]
async fn prove_topology_correlates_with_quality() -> Result<()> {
    // Fixed seed for reproducibility
    let seed = 42u64;
    let prompts = generate_prompts(100, seed);

    let mut pipeline = Pipeline::initialise(CliArgs::default()).await?;

    let mut rouge_scores: Vec<f64> = Vec::with_capacity(prompts.len());
    let mut betti_1: Vec<f64> = Vec::with_capacity(prompts.len());
    let mut spectral_gap: Vec<f64> = Vec::with_capacity(prompts.len());
    let mut persistence_entropy: Vec<f64> = Vec::with_capacity(prompts.len());

    let mut analyzer = TCSAnalyzer::new()?;
    for (i, prompt) in prompts.iter().enumerate() {
        let cycle = pipeline.process_prompt(prompt).await?;
        let reference = reference_for(prompt);
        let rouge = niodoo_real_integrated::util::rouge_l(&cycle.generation.hybrid_response, &reference);

        // Recompute topology from pad_state to avoid relying on internal fields
        let topo = analyzer.analyze_state(&cycle.pad_state).expect("topology compute");

        rouge_scores.push(rouge);
        betti_1.push(topo.betti_numbers[1] as f64);
        spectral_gap.push(topo.spectral_gap);
        persistence_entropy.push(topo.persistence_entropy);

        if (i + 1) % 10 == 0 {
            info!(i = i + 1, "progress: processed prompts");
        }
    }

    let r_betti = pearson(&rouge_scores, &betti_1)?;
    let r_gap = pearson(&rouge_scores, &spectral_gap)?;
    let r_entropy = pearson(&rouge_scores, &persistence_entropy)?;
    let rho_betti = spearman(&rouge_scores, &betti_1)?;

    println!(
        "metric,pearson,spearman\nBetti-1,{:.4},{:.4}\nSpectralGap,{:.4},\nPersistenceEntropy,{:.4},",
        r_betti, rho_betti, r_gap, r_entropy
    );

    // Minimal threshold to flag signal; adjust if synthetic references are too strict
    assert!(r_betti.abs() >= 0.30 || r_gap >= 0.30, "Topology signal too weak: r_betti={:.3}, r_gap={:.3}", r_betti, r_gap);

    Ok(())
}


