//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

//! AI inference module powered by the real NIODOO topological pipeline.
//!
//! This module wraps the `niodoo_real_integrated` runtime, providing a thin,
//! topology-aware interface for legacy callers inside the monolithic crate.

use anyhow::{anyhow, Result};
use niodoo_real_integrated::config::{CliArgs, TopologyMode};
use niodoo_real_integrated::pipeline::{Pipeline, PipelineCycle};
use niodoo_real_integrated::tcs_analysis::TopologicalSignature;
use niodoo_real_integrated::torus::PadGhostState;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::feeling_model::EmotionalAnalysis;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageTimingSnapshot {
    pub embedding_ms: f64,
    pub torus_ms: f64,
    pub tcs_ms: f64,
    pub compass_ms: f64,
    pub erag_ms: f64,
    pub tokenizer_ms: f64,
    pub generation_ms: f64,
    pub learning_ms: f64,
    pub threat_cycle_ms: f64,
}

impl StageTimingSnapshot {
    fn from_cycle(cycle: &PipelineCycle) -> Self {
        let timings = &cycle.stage_timings;
        Self {
            embedding_ms: timings.embedding_ms,
            torus_ms: timings.torus_ms,
            tcs_ms: timings.tcs_ms,
            compass_ms: timings.compass_ms,
            erag_ms: timings.erag_ms,
            tokenizer_ms: timings.tokenizer_ms,
            generation_ms: timings.generation_ms,
            learning_ms: timings.learning_ms,
            threat_cycle_ms: timings.threat_cycle_ms,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIInferenceResult {
    pub prompt: String,
    pub response: String,
    pub baseline_response: String,
    pub confidence: f64,
    pub entropy: f64,
    pub rouge: f64,
    pub latency_ms: f64,
    pub topology_mode: TopologyMode,
    pub topology_signature: TopologicalSignature,
    pub timings: StageTimingSnapshot,
}

#[derive(Clone)]
pub struct AIInferenceEngine {
    pipeline: Arc<Mutex<Pipeline>>,
    topology_mode: TopologyMode,
    model_name: Arc<String>,
}

impl AIInferenceEngine {
    pub async fn new_default() -> Result<Self> {
        Self::with_args(CliArgs::default(), TopologyMode::Hybrid).await
    }

    pub async fn with_args(args: CliArgs, mode: TopologyMode) -> Result<Self> {
        let mut pipeline = Pipeline::initialise_with_mode(args, mode).await?;
        pipeline.set_topology_mode(mode)?;
        let model_name = pipeline.config.vllm_model.clone();
        Ok(Self {
            pipeline: Arc::new(Mutex::new(pipeline)),
            topology_mode: mode,
            model_name: Arc::new(model_name),
        })
    }

    pub fn topology_mode(&self) -> TopologyMode {
        self.topology_mode
    }

    pub async fn set_topology_mode(&mut self, mode: TopologyMode) -> Result<()> {
        if self.topology_mode == mode {
            return Ok(());
        }

        let mut pipeline = self.pipeline.lock().await;
        pipeline.set_topology_mode(mode)?;
        self.topology_mode = mode;
        Ok(())
    }

    pub async fn generate(&self, input: &str) -> Result<AIInferenceResult> {
        let cycle = self.run_cycle(input).await?;
        Ok(Self::cycle_to_result(cycle))
    }

    pub async fn detect_emotion(
        &self,
        input: &str,
    ) -> Result<EmotionalAnalysis, Box<dyn std::error::Error>> {
        let cycle = self
            .run_cycle(input)
            .await
            .map_err(|err| -> Box<dyn std::error::Error> { Box::new(err) })?;
        Ok(Self::pad_to_emotional_analysis(&cycle.pad_state))
    }

    fn cycle_to_result(cycle: PipelineCycle) -> AIInferenceResult {
        let response = match cycle.topology_mode {
            TopologyMode::Hybrid => cycle.hybrid_response.clone(),
            TopologyMode::Baseline => cycle.baseline_response.clone(),
        };

        let confidence = 1.0 - (cycle.entropy / 3.0).clamp(0.0, 1.0);

        AIInferenceResult {
            prompt: cycle.prompt.clone(),
            response,
            baseline_response: cycle.baseline_response.clone(),
            confidence,
            entropy: cycle.entropy,
            rouge: cycle.rouge,
            latency_ms: cycle.latency_ms,
            topology_mode: cycle.topology_mode,
            topology_signature: cycle.topology.clone(),
            timings: StageTimingSnapshot::from_cycle(&cycle),
        }
    }

    fn pad_to_emotional_analysis(pad_state: &PadGhostState) -> EmotionalAnalysis {
        let joy = ((pad_state.pad[0] + 1.0) / 2.0).clamp(0.0, 1.0) as f32;
        let sadness = ((-pad_state.pad[0]).max(0.0)).clamp(0.0, 1.0) as f32;
        let anger = pad_state.pad[1].max(0.0).clamp(0.0, 1.0) as f32;
        let fear = (-pad_state.pad[1]).max(0.0).clamp(0.0, 1.0) as f32;
        let surprise = pad_state.pad[2].abs().clamp(0.0, 1.0) as f32;
        let intensity = (pad_state.entropy / 2.5).clamp(0.0, 1.0) as f32;

        let mut scores = vec![
            ("joy", joy),
            ("sadness", sadness),
            ("anger", anger),
            ("fear", fear),
            ("surprise", surprise),
        ];
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let dominant_emotion = scores
            .first()
            .map(|(label, _)| (*label).to_string())
            .unwrap_or_else(|| "neutral".to_string());

        EmotionalAnalysis {
            joy,
            sadness,
            anger,
            fear,
            surprise,
            emotional_intensity: intensity,
            dominant_emotion,
        }
    }

    async fn run_cycle(&self, prompt: &str) -> Result<PipelineCycle> {
        let mut pipeline = self.pipeline.lock().await;
        let cycle = pipeline
            .process_prompt(prompt)
            .await
            .map_err(|err| anyhow!("pipeline execution failed: {err}"))?;
        Ok(cycle)
    }

    pub fn model_name(&self) -> &str {
        self.model_name.as_str()
    }
}
