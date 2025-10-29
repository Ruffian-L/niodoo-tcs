use anyhow::Result;
use niodoo_real_integrated::config::{CliArgs, HardwareProfile, OutputFormat};
use niodoo_real_integrated::pipeline::{self, Pipeline};
use prost::Message;
use tracing::info;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    info!("Starting NIODOO pipeline smoke test");

    // Initialize REAL pipeline with default config
    let args = CliArgs {
        hardware: HardwareProfile::H200,
        prompt: None,
        prompt_file: None,
        swarm: 1,
        iterations: 1,
        output: OutputFormat::Csv,
        config: None,
        rng_seed_override: None,
    };

    let mut pipeline = Pipeline::initialise(args).await?;
    info!("Pipeline initialized");

    // Sample prompt from Oct 27 log
    let prompt = "When I feel overwhelmed by deadlines, how do I manage stress?";
    info!("Processing prompt: {}", prompt);

    let cycle = pipeline.process_prompt(prompt).await?;
    let topology = &cycle.topology;

    info!("Processed prompt: {}", cycle.prompt);
    info!("Baseline: {}", cycle.baseline_response);
    info!("Hybrid: {}", cycle.hybrid_response);
    info!("Latency: {}ms", cycle.latency_ms);
    info!("Entropy: {}", cycle.entropy);
    info!("ROUGE: {}", cycle.rouge);

    // Protobuf serialization test - create ConsciousnessState from real topology
    use pipeline::proto::{ConsciousnessState, PadGhostState, TopologyState};

    let pad = cycle
        .pad_state
        .pad
        .iter()
        .map(|value| *value as f32)
        .collect::<Vec<_>>();
    let mu = cycle
        .pad_state
        .mu
        .iter()
        .map(|value| *value as f32)
        .collect::<Vec<_>>();
    let sigma = cycle
        .pad_state
        .sigma
        .iter()
        .map(|value| *value as f32)
        .collect::<Vec<_>>();
    let raw_stds = cycle
        .pad_state
        .raw_stds
        .iter()
        .map(|value| *value as f32)
        .collect::<Vec<_>>();

    let consciousness_state = ConsciousnessState {
        topology: Some(TopologyState {
            entropy: cycle.entropy as f32,
            iit_phi: topology.persistence_entropy as f32,
            knots: topology
                .knots
                .iter()
                .map(|value| *value as f32)
                .collect::<Vec<_>>(),
            betti_numbers: topology
                .betti_numbers
                .iter()
                .map(|value| *value as i32)
                .collect::<Vec<_>>(),
            spectral_gap: topology.spectral_gap as f32,
            persistent_entropy: topology.persistence_entropy as f32,
        }),
        pad_ghost: Some(PadGhostState {
            pad,
            mu,
            sigma,
            raw_stds,
        }),
        quadrant: format!("{:?}", cycle.compass.quadrant),
        threat: cycle.failure == "hard",
        healing: cycle.failure == "none",
        rouge_score: cycle.rouge as f32,
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64,
    };

    let bytes = consciousness_state.encode_to_vec();
    info!("Serialized size: {} bytes", bytes.len());

    let decoded = ConsciousnessState::decode(&bytes[..])?;
    info!(
        "Decoded ROUGE: {}, Quadrant: {}",
        decoded.rouge_score, decoded.quadrant
    );
    info!(
        "Pad sample: {:?}",
        decoded.pad_ghost.as_ref().and_then(|pad| pad.pad.first())
    );

    info!("✅ Proto round-trip complete");
    Ok(())
}
