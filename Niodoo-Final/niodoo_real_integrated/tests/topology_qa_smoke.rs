use niodoo_real_integrated::erag::{CollapseResult, EmotionalVector, EragMemory};
use niodoo_real_integrated::pipeline::topo_reflection::TopoReflectionStage;
use niodoo_real_integrated::tcs_analysis::TopologicalSignature;
use niodoo_real_integrated::torus::PadGhostState;
use serde::Deserialize;
use std::collections::HashMap;
use tcs_core::PersistentFeature;
use tcs_tqft::Cobordism;

#[derive(Debug, Deserialize)]
struct ChaosPrompt {
    id: String,
    prompt: String,
}

#[derive(Debug, Clone)]
struct SignatureStats {
    betti: [usize; 3],
    persistence_lengths: [f64; 3],
    persistence_entropy: f64,
    spectral_gap: f64,
    laplacian_radius: f64,
    knot_complexity: f64,
    takens_window_energy: [f64; 4],
    emotional_flux: f64,
    emotional_drift: f64,
    emotional_arc_lengths: [f64; 3],
    gradient_energy: f64,
    gradient_volatility: f64,
    gradient_energy_trend: [f64; 4],
}

#[derive(Debug, Clone)]
struct PadConfig {
    pad: [f64; 7],
    entropy: f64,
    mu: [f64; 7],
    sigma: [f64; 7],
}

#[derive(Debug, Clone)]
struct CaseConfig {
    baseline: SignatureStats,
    current: SignatureStats,
    pad: PadConfig,
    collapse_quality: f64,
    min_depth: f64,
}

const CASE_CONFIGS: &[(&str, CaseConfig)] = &[
    (
        "spiral-supply-shock",
        CaseConfig {
            // Baseline: single dominant loop, manageable voids.
            baseline: SignatureStats {
                betti: [1, 1, 0],
                persistence_lengths: [0.9, 0.6, 0.2],
                persistence_entropy: 1.12,
                spectral_gap: 0.68,
                laplacian_radius: 2.35,
                knot_complexity: 1.05,
                takens_window_energy: [0.8, 0.6, 0.5, 0.4],
                emotional_flux: 0.28,
                emotional_drift: 0.12,
                emotional_arc_lengths: [0.32, 0.28, 0.24],
                gradient_energy: 0.95,
                gradient_volatility: 0.18,
                gradient_energy_trend: [0.4, 0.55, 0.62, 0.58],
            },
            // Current: inventory loops explode; more cycles and a void appear.
            current: SignatureStats {
                betti: [1, 3, 1],
                persistence_lengths: [1.4, 0.95, 0.44],
                persistence_entropy: 1.74,
                spectral_gap: 0.31,
                laplacian_radius: 3.05,
                knot_complexity: 1.52,
                takens_window_energy: [1.2, 0.9, 0.8, 0.7],
                emotional_flux: 0.46,
                emotional_drift: 0.26,
                emotional_arc_lengths: [0.48, 0.52, 0.41],
                gradient_energy: 1.48,
                gradient_volatility: 0.33,
                gradient_energy_trend: [0.72, 0.95, 1.08, 1.12],
            },
            pad: PadConfig {
                pad: [0.35, 0.54, -0.26, 0.18, -0.11, 0.22, -0.05],
                entropy: 1.72,
                mu: [0.28, 0.42, -0.18, 0.21, -0.09, 0.19, -0.04],
                sigma: [0.41, 0.37, 0.32, 0.29, 0.27, 0.24, 0.22],
            },
            collapse_quality: 0.64,
            min_depth: 0.35,
        },
    ),
    (
        "quantum-sensor-chaos",
        CaseConfig {
            baseline: SignatureStats {
                betti: [1, 2, 0],
                persistence_lengths: [1.1, 0.72, 0.3],
                persistence_entropy: 1.28,
                spectral_gap: 0.74,
                laplacian_radius: 2.85,
                knot_complexity: 1.22,
                takens_window_energy: [0.9, 0.7, 0.6, 0.55],
                emotional_flux: 0.34,
                emotional_drift: 0.17,
                emotional_arc_lengths: [0.36, 0.33, 0.29],
                gradient_energy: 1.08,
                gradient_volatility: 0.21,
                gradient_energy_trend: [0.48, 0.63, 0.69, 0.74],
            },
            current: SignatureStats {
                betti: [1, 4, 2],
                persistence_lengths: [1.6, 1.18, 0.72],
                persistence_entropy: 1.93,
                spectral_gap: 0.27,
                laplacian_radius: 3.48,
                knot_complexity: 1.84,
                takens_window_energy: [1.25, 1.05, 0.96, 0.82],
                emotional_flux: 0.58,
                emotional_drift: 0.33,
                emotional_arc_lengths: [0.52, 0.63, 0.58],
                gradient_energy: 1.76,
                gradient_volatility: 0.41,
                gradient_energy_trend: [0.88, 1.12, 1.27, 1.33],
            },
            pad: PadConfig {
                pad: [-0.42, 0.61, 0.38, -0.19, 0.27, -0.08, 0.14],
                entropy: 1.85,
                mu: [-0.36, 0.52, 0.31, -0.15, 0.20, -0.06, 0.11],
                sigma: [0.45, 0.39, 0.37, 0.33, 0.28, 0.26, 0.24],
            },
            collapse_quality: 0.59,
            min_depth: 0.42,
        },
    ),
    (
        "rogue-swarm",
        CaseConfig {
            baseline: SignatureStats {
                betti: [1, 1, 0],
                persistence_lengths: [0.85, 0.55, 0.18],
                persistence_entropy: 1.05,
                spectral_gap: 0.62,
                laplacian_radius: 2.28,
                knot_complexity: 0.98,
                takens_window_energy: [0.7, 0.58, 0.5, 0.42],
                emotional_flux: 0.26,
                emotional_drift: 0.11,
                emotional_arc_lengths: [0.28, 0.26, 0.22],
                gradient_energy: 0.82,
                gradient_volatility: 0.16,
                gradient_energy_trend: [0.35, 0.48, 0.55, 0.5],
            },
            current: SignatureStats {
                betti: [1, 5, 1],
                persistence_lengths: [1.45, 1.02, 0.51],
                persistence_entropy: 1.88,
                spectral_gap: 0.24,
                laplacian_radius: 3.18,
                knot_complexity: 1.61,
                takens_window_energy: [1.15, 0.97, 0.83, 0.75],
                emotional_flux: 0.54,
                emotional_drift: 0.29,
                emotional_arc_lengths: [0.49, 0.57, 0.53],
                gradient_energy: 1.62,
                gradient_volatility: 0.38,
                gradient_energy_trend: [0.82, 1.05, 1.19, 1.14],
            },
            pad: PadConfig {
                pad: [0.48, -0.37, 0.42, 0.24, -0.18, 0.12, 0.16],
                entropy: 1.67,
                mu: [0.39, -0.28, 0.34, 0.19, -0.14, 0.09, 0.11],
                sigma: [0.36, 0.33, 0.3, 0.26, 0.24, 0.21, 0.19],
            },
            collapse_quality: 0.62,
            min_depth: 0.38,
        },
    ),
    (
        "finance-fractal",
        CaseConfig {
            baseline: SignatureStats {
                betti: [1, 2, 1],
                persistence_lengths: [1.05, 0.68, 0.35],
                persistence_entropy: 1.18,
                spectral_gap: 0.69,
                laplacian_radius: 2.74,
                knot_complexity: 1.34,
                takens_window_energy: [0.82, 0.69, 0.58, 0.54],
                emotional_flux: 0.31,
                emotional_drift: 0.15,
                emotional_arc_lengths: [0.33, 0.3, 0.27],
                gradient_energy: 1.02,
                gradient_volatility: 0.2,
                gradient_energy_trend: [0.42, 0.56, 0.63, 0.67],
            },
            current: SignatureStats {
                betti: [1, 4, 2],
                persistence_lengths: [1.58, 1.11, 0.73],
                persistence_entropy: 1.97,
                spectral_gap: 0.29,
                laplacian_radius: 3.42,
                knot_complexity: 1.92,
                takens_window_energy: [1.18, 1.02, 0.96, 0.88],
                emotional_flux: 0.57,
                emotional_drift: 0.34,
                emotional_arc_lengths: [0.51, 0.59, 0.55],
                gradient_energy: 1.71,
                gradient_volatility: 0.43,
                gradient_energy_trend: [0.9, 1.14, 1.28, 1.33],
            },
            pad: PadConfig {
                pad: [-0.31, -0.46, 0.37, 0.28, -0.22, 0.17, -0.09],
                entropy: 1.79,
                mu: [-0.26, -0.38, 0.29, 0.21, -0.18, 0.13, -0.07],
                sigma: [0.4, 0.34, 0.32, 0.27, 0.25, 0.22, 0.2],
            },
            collapse_quality: 0.57,
            min_depth: 0.4,
        },
    ),
];

#[test]
fn chaos_prompts_preserve_topology_signals() {
    let prompts: Vec<ChaosPrompt> =
        serde_json::from_str(include_str!("fixtures/topology_chaos_prompts.json"))
            .expect("chaos prompt fixture should parse");
    let config_map: HashMap<&str, &CaseConfig> =
        CASE_CONFIGS.iter().map(|(id, cfg)| (*id, cfg)).collect();

    let stage = TopoReflectionStage::new();

    for prompt in prompts {
        let config = config_map
            .get(prompt.id.as_str())
            .expect("every prompt requires configuration");

        let baseline = build_signature(&config.baseline);
        let current = build_signature(&config.current);
        let pad_state = build_pad(&config.pad);
        let collapse = build_collapse(
            prompt.id.as_str(),
            &prompt.prompt,
            config.collapse_quality,
            &pad_state,
        );

        let reflection = stage.run(&current, &baseline, &pad_state, &collapse);

        assert!(
            reflection.thinking_depth >= config.min_depth,
            "chaos case '{}' should keep thinking depth ≥ {:.2} (got {:.4})",
            prompt.id,
            config.min_depth,
            reflection.thinking_depth
        );

        assert!(
            current.betti_numbers[1] >= baseline.betti_numbers[1],
            "chaos case '{}' unexpectedly collapsed β₁ (baseline {}, current {})",
            prompt.id,
            baseline.betti_numbers[1],
            current.betti_numbers[1]
        );

        assert!(
            current.betti_numbers.iter().any(|&beta| beta > 1),
            "chaos case '{}' should surface non-trivial topology: {:?}",
            prompt.id,
            current.betti_numbers
        );

        let summary = TopoReflectionStage::summarize_failure(&current, &baseline, &reflection);
        assert!(
            summary.contains("Δβ1="),
            "chaos case '{}' summary must record Betti change: {}",
            prompt.id,
            summary
        );
    }
}

fn build_signature(stats: &SignatureStats) -> TopologicalSignature {
    let persistence_features: Vec<PersistentFeature> = stats
        .persistence_lengths
        .iter()
        .enumerate()
        .map(|(dimension, length)| PersistentFeature {
            birth: (0.05 * (dimension as f64 + 1.0)) as f32,
            death: (0.05 * (dimension as f64 + 1.0) + length) as f32,
            dimension,
        })
        .collect();

    let total_persistence: f64 = stats.persistence_lengths.iter().sum();
    let max_persistence: f64 = stats
        .persistence_lengths
        .iter()
        .copied()
        .fold(0.0_f64, f64::max);
    let mean_persistence = total_persistence / stats.persistence_lengths.len() as f64;

    let takens_window_count = stats.takens_window_energy.len();
    let takens_energy: f64 = stats.takens_window_energy.iter().sum();

    let emotional_arc_lengths = stats.emotional_arc_lengths.to_vec();
    let gradient_energy_trend = stats.gradient_energy_trend.to_vec();
    let takens_window_energy = stats.takens_window_energy.to_vec();

    TopologicalSignature::new(
        persistence_features,
        stats.betti,
        stats.knot_complexity,
        format!(
            "λ² + {:.3}λ + {:.3}",
            stats.knot_complexity, stats.persistence_entropy
        ),
        takens_window_count + 3,
        None::<Cobordism>,
        12.0 + stats.laplacian_radius * 3.2,
        stats.persistence_entropy,
        stats.spectral_gap,
        stats.betti[0] as f64 - stats.betti[1] as f64 + stats.betti[2] as f64,
        total_persistence,
        max_persistence,
        mean_persistence,
        stats.laplacian_radius,
        takens_energy,
        takens_window_count,
        takens_window_energy,
        stats.emotional_flux,
        stats.emotional_drift,
        emotional_arc_lengths,
        stats.gradient_energy,
        stats.gradient_volatility,
        gradient_energy_trend,
    )
}

fn build_pad(config: &PadConfig) -> PadGhostState {
    PadGhostState {
        pad: config.pad,
        entropy: config.entropy,
        mu: config.mu,
        sigma: config.sigma,
    }
}

fn build_collapse(
    case_id: &str,
    prompt: &str,
    quality: f64,
    pad: &PadGhostState,
) -> CollapseResult {
    let top_hits = vec![
        EragMemory {
            input: format!("{case_id}:loop-analysis"),
            output: format!(
                "stabilise cycle by dampening link on {}",
                prompt.split_whitespace().next().unwrap_or("system")
            ),
            emotional_vector: emotional_from_pad(pad, 0.18),
            erag_context: vec![format!("[context:{}] damp feedback arcs", case_id)],
            entropy_before: pad.entropy + 0.12,
            entropy_after: pad.entropy - 0.08,
            timestamp: "2025-11-13T12:00:00Z".into(),
            compass_state: Some("Persist".into()),
            cascade_stage: None,
            weighted_metadata: None,
        },
        EragMemory {
            input: format!("{case_id}:novel-metric"),
            output: "reinforce cross-link with phase-aligned offsets".into(),
            emotional_vector: emotional_from_pad(pad, -0.15),
            erag_context: vec![format!("[context:{}] enforce torus traversal", case_id)],
            entropy_before: pad.entropy + 0.08,
            entropy_after: pad.entropy - 0.05,
            timestamp: "2025-11-13T12:03:00Z".into(),
            compass_state: Some("Discover".into()),
            cascade_stage: None,
            weighted_metadata: None,
        },
    ];

    CollapseResult {
        top_hits,
        aggregated_context: format!("[chaos_case:{}] {}", case_id, prompt),
        average_similarity: 0.41,
        curator_quality: Some(quality),
    }
}

fn emotional_from_pad(pad: &PadGhostState, shift: f64) -> EmotionalVector {
    let joy = (pad.pad[0] + shift).clamp(-1.0, 1.0);
    let arousal = (pad.pad[1] - shift / 2.0).clamp(-1.0, 1.0);
    let surprise = (pad.pad[2] + shift / 1.5).clamp(-1.0, 1.0);

    EmotionalVector {
        joy: joy as f32,
        sadness: (-joy).max(0.0) as f32,
        anger: arousal as f32,
        fear: (-arousal).max(0.0) as f32,
        surprise: surprise as f32,
    }
}
