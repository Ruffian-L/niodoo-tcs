use std::collections::HashMap;
use std::env;

use chrono::Utc;
use niodoo_real_integrated::compass::{CompassOutcome, CompassQuadrant, MctsBranch};
use niodoo_real_integrated::config::{self, CliArgs, HardwareProfile, OutputFormat};
use niodoo_real_integrated::erag::{CollapseResult, EmotionalVector, EragMemory};
use niodoo_real_integrated::generation::{GenerationResult, LensEcho};
use niodoo_real_integrated::learning::LearningOutcome;
use niodoo_real_integrated::pipeline::{Pipeline, StageTimings};
use niodoo_real_integrated::torus::PadGhostState;
use niodoo_real_integrated::tcs_analysis::TopologicalSignature;

const TRACKED_ENV_VARS: [&str; 3] = ["VLLM_ENDPOINT", "QDRANT_URL", "OLLAMA_ENDPOINT"];

struct TestEnvGuard {
    saved: Vec<(&'static str, Option<String>)>,
}

impl TestEnvGuard {
    fn new() -> Self {
        let saved = TRACKED_ENV_VARS
            .iter()
            .map(|key| (*key, env::var(key).ok()))
            .collect();
        Self { saved }
    }
}

impl Drop for TestEnvGuard {
    fn drop(&mut self) {
        for (key, value) in self.saved.drain(..) {
            match value {
                Some(v) => env::set_var(key, v),
                None => env::remove_var(key),
            }
        }
    }
}

fn prime_test_environment() -> TestEnvGuard {
    let guard = TestEnvGuard::new();
    config::prime_environment();
    guard
}

struct MockGenerationEngine;

impl MockGenerationEngine {
    async fn generate(&self, prompt: &str) -> GenerationResult {
        GenerationResult {
            baseline_response: format!("baseline::{prompt}"),
            hybrid_response: format!("hybrid::{prompt}"),
            echoes: vec![LensEcho {
                lens: "discover".into(),
                response: format!("lens::{prompt}"),
            }],
            rouge_to_baseline: 0.82,
            latency_ms: 128.0,
            rouge_score: 0.84,
            entropy_delta: -0.06,
            source: "mock-generator".into(),
            ucb1_score: 0.5,
            curator_quality: 0.72,
            failure_type: None,
            failure_details: None,
        }
    }
}

struct MockEragClient;

impl MockEragClient {
    async fn collapse(&self, prompt: &str) -> CollapseResult {
        let memory = EragMemory {
            input: prompt.to_string(),
            output: format!("memory::{prompt}"),
            emotional_vector: EmotionalVector::default(),
            erag_context: vec!["mock context".into()],
            entropy_before: 1.96,
            entropy_after: 1.80,
            timestamp: Utc::now().to_rfc3339(),
            compass_state: Some("Discover".into()),
            quality_score: Some(0.71),
            topology_betti: Some([1, 0, 0]),
            topology_knot_complexity: Some(0.12),
            solution_path: Some("mock-solution".into()),
            conversation_history: vec![prompt.to_string()],
            iteration_count: 1,
        };

        CollapseResult {
            top_hits: vec![memory],
            aggregated_context: format!("aggregated::{prompt}"),
            average_similarity: 0.64,
            curator_quality: Some(0.70),
            failure_type: None,
            failure_details: None,
        }
    }
}

#[derive(Debug)]
struct MockCycle {
    prompt: String,
    generation: GenerationResult,
    collapse: CollapseResult,
    pad_state: PadGhostState,
    learning: LearningOutcome,
    compass: CompassOutcome,
    timings: StageTimings,
    topology: TopologicalSignature,
}

#[derive(Default)]
struct MockPipeline {
    generation: MockGenerationEngine,
    erag: MockEragClient,
}

impl MockPipeline {
    async fn process_prompt(&self, prompt: &str) -> MockCycle {
        let generation = self.generation.generate(prompt).await;
        let collapse = self.erag.collapse(prompt).await;

        let pad_state = PadGhostState {
            pad: [0.1; 7],
            entropy: 1.95,
            mu: [0.0; 7],
            sigma: [0.1; 7],
            raw_stds: vec![0.1; 7],
        };

        let mut adjusted_params = HashMap::new();
        adjusted_params.insert("temperature".into(), 0.6);

        let learning = LearningOutcome {
            events: vec!["entropy_drop".into()],
            breakthroughs: vec!["qlora-adapted".into()],
            qlora_updates: vec!["mock-adapter".into()],
            entropy_delta: generation.entropy_delta,
            adjusted_params,
        };

        let compass = CompassOutcome {
            quadrant: CompassQuadrant::Discover,
            is_threat: false,
            is_healing: true,
            mcts_branches: vec![MctsBranch {
                label: "mock-branch".into(),
                ucb_score: 0.42,
                entropy_projection: 0.08,
            }],
            intrinsic_reward: 0.8,
            ucb1_score: Some(0.42),
        };

        let timings = StageTimings {
            embedding_ms: 4.0,
            torus_ms: 1.2,
            tcs_ms: 0.8,
            compass_ms: 0.4,
            erag_ms: 0.6,
            tokenizer_ms: 0.5,
            generation_ms: 12.0,
            learning_ms: 3.5,
            threat_cycle_ms: 0.2,
        };

        let topology = TopologicalSignature::new(
            Vec::new(),
            [1, 0, 0],
            0.1,
            "unknot".into(),
            1,
            None,
            0.5,
            0.05,
            0.8,
        );

        MockCycle {
            prompt: prompt.to_string(),
            generation,
            collapse,
            pad_state,
            learning,
            compass,
            timings,
            topology,
        }
    }
}

// NOTE: #[tokio::test] is the Tokio async test macro; each test runs on its own lightweight runtime.

#[tokio::test]
async fn mock_pipeline_cycle_has_expected_shape() {
    let _guard = prime_test_environment();
    let pipeline = MockPipeline::default();
    let cycle = pipeline.process_prompt("test prompt").await;

    assert_eq!(cycle.prompt, "test prompt");
    assert!(cycle.generation.hybrid_response.contains("hybrid"));
    assert!(cycle.collapse.aggregated_context.contains("test prompt"));
    assert_eq!(cycle.pad_state.raw_stds.len(), 7);
    assert!(cycle.learning.entropy_delta < 0.0);
    assert!(cycle.timings.generation_ms > 0.0);
    assert_eq!(cycle.topology.betti_numbers[0], 1);
}

#[tokio::test]
async fn mock_learning_outcome_tracks_entropy_delta() {
    let _guard = prime_test_environment();
    let pipeline = MockPipeline::default();
    let cycle = pipeline.process_prompt("entropy prompt").await;

    assert!(cycle
        .learning
        .adjusted_params
        .contains_key("temperature"));
    assert!(cycle
        .learning
        .events
        .iter()
        .any(|event| event.contains("entropy")));
    assert!(cycle.learning.entropy_delta < 0.0);
}

#[tokio::test]
#[ignore = "requires live vLLM/Qdrant backends"]
async fn real_pipeline_initialises_with_backends() {
    let _guard = prime_test_environment();
    let args = CliArgs {
        hardware: HardwareProfile::Beelink,
        prompt: None,
        prompt_file: None,
        swarm: 1,
        output: OutputFormat::Json,
        config: None,
    };

    Pipeline::initialise(args)
        .await
        .expect("pipeline should initialise when backends are available");
}
 
