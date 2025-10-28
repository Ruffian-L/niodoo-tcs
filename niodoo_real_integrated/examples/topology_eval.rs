use anyhow::Result;
use clap::Parser;
use csv::Writer;
use niodoo_real_integrated::config::CliArgs;
use niodoo_real_integrated::eval::synthetic::{generate_prompts, reference_for};
use niodoo_real_integrated::pipeline::Pipeline;
use niodoo_real_integrated::tcs_analysis::TCSAnalyzer;
use std::fs;

#[derive(Debug, Clone, Parser)]
struct Args {
    #[arg(long, default_value_t = 100)]
    num_prompts: usize,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(
        long,
        default_value = "/workspace/Niodoo-Final/results/topology_eval.csv"
    )]
    out: String,
    /// Modes: erag, erag+lora, full
    #[arg(long, num_args = 1.., value_delimiter = ' ')]
    modes: Vec<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    if let Some(parent) = std::path::Path::new(&args.out).parent() {
        fs::create_dir_all(parent)?;
    }
    let mut wtr = Writer::from_path(&args.out)?;
    wtr.write_record([
        "id",
        "mode",
        "prompt",
        "candidate",
        "reference",
        "rouge_l",
        "betti_1",
        "spectral_gap",
        "persistence_entropy",
    ])?;

    let prompts = generate_prompts(args.num_prompts, args.seed);

    for mode in args.modes.iter() {
        // Configure ablation toggles via env vars used internally BEFORE init
        std::env::set_var(
            "ENABLE_CURATOR",
            if mode == "erag" { "false" } else { "true" },
        );
        std::env::set_var(
            "ENABLE_CONSISTENCY_VOTING",
            if mode == "full" { "true" } else { "false" },
        );
        match mode.as_str() {
            "erag" => {
                std::env::set_var("DISABLE_LORA", "true");
                std::env::set_var("DISABLE_TOPOLOGY_AUGMENTATION", "true");
            }
            "erag+lora" => {
                std::env::set_var("DISABLE_LORA", "false");
                std::env::set_var("DISABLE_TOPOLOGY_AUGMENTATION", "true");
            }
            _ => {
                std::env::set_var("DISABLE_LORA", "false");
                std::env::set_var("DISABLE_TOPOLOGY_AUGMENTATION", "false");
            }
        }

        let mut pipeline = Pipeline::initialise(CliArgs::default()).await?;
        let mut analyzer = TCSAnalyzer::new()?;

        for (i, prompt) in prompts.iter().enumerate() {
            let cycle = pipeline.process_prompt(prompt).await?;
            let reference = reference_for(prompt);
            let candidate = &cycle.generation.hybrid_response;
            let rouge = niodoo_real_integrated::util::rouge_l(candidate, &reference);

            let topo_sig = analyzer.analyze_state(&cycle.pad_state)?;
            let betti1 = topo_sig.betti_numbers[1] as f64;
            let spectral_gap = topo_sig.spectral_gap;
            let persistence_entropy = topo_sig.persistence_entropy;

            wtr.write_record([
                (i + 1).to_string(),
                mode.clone(),
                prompt.clone(),
                candidate.clone(),
                reference,
                format!("{:.6}", rouge),
                format!("{:.6}", betti1),
                format!("{:.6}", spectral_gap),
                format!("{:.6}", persistence_entropy),
            ])?;
        }
    }

    wtr.flush()?;
    println!("Wrote {}", args.out);
    Ok(())
}
