use std::cmp::Ordering;
use std::sync::atomic::Ordering as AtomicOrdering;
use std::time::{Duration, Instant};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use crate::compass::{CompassOutcome, CompassQuadrant};
use crate::config::{env_value, TopologyMode};
use crate::consonance::compute_consonance;
#[cfg(feature = "gpu")]
use crate::consonance::{
    compute_compass_transition, compute_confidence, compute_erag_relevance,
    compute_topological_consistency,
};
use crate::data::Experience;
use crate::erag::{CollapseResult, EragMemory};
use crate::generation::GenerationResult;
use crate::learning::LearningOutcome;
use crate::metrics::metrics;
use crate::ntoken_client;
use crate::pipeline::generation::topo_reasoning::{TopoCoT, TopoCotEvaluation};
use crate::pipeline::topo_executor::{ExecutionError, ExecutionResult};
use crate::pipeline::topo_reflection::{TopoReflection, TopoReflectionStage};
use crate::signals::FailureSignals;
use crate::tcs_analysis::{baseline_topological_signature, TopologicalSignature};
use crate::telemetry::storage::{append_topocot_log, TopoCotLogEntry};
use crate::token_manager::TokenizerOutput;
use crate::torus::PadGhostState;
use crate::util::rouge_l;
use blake3;
use chrono;

use super::cache::cache_key;
use super::core::Pipeline;
use super::metrics::StageTimings;
use super::state::{CuratedExperience, PipelineCycle, TopoCotTelemetry};

/// Failure sample for Learning Gate processing
#[derive(Serialize, Deserialize, Debug, Clone)]
struct FailureSample {
    prompt: String,
    bad_response: String,
    quality_score: u8,
    pad_context: String,
    compass_context: String,
}

/// Gemini feedback for failure corrections
#[derive(Serialize, Deserialize, Debug, Clone)]
struct GeminiFeedback {
    corrected_response: String,
    explanation: String,
}

#[derive(Clone, Copy, Debug)]
struct MemoryAffinityDetail {
    weight: f32,
    persistence_norm: f32,
    betti_alignment: f32,
    entropy_delta: f32,
    anomaly_penalty: f32,
}

#[derive(Clone, Copy, Debug)]
struct TopologyRetrievalStats {
    best_weight: f32,
    mean_weight: f32,
    best_persistence_norm: f32,
    best_betti_alignment: f32,
    best_entropy_delta: f32,
}

impl Pipeline {
    pub async fn process_prompt(&mut self, prompt: &str) -> Result<PipelineCycle> {
        let overall_start = Instant::now();
        let mut timings = StageTimings::default();
        let cache_key = cache_key(prompt);
        let now = Instant::now();

        // Initialize thought tree builder for capturing reasoning structure
        let mut thought_builder = crate::telemetry::thought_structure::ThoughtTreeBuilder::new();

        // Stage 1: Embedding (with cache)
        let embedding_start = Instant::now();
        let embedding_hit = self.embedding_cache.get(&cache_key, now).await;
        let embedding = if let Some(hit) = embedding_hit {
            // Observation: Cache hit
            thought_builder.add_observation(
                "embedding_cache".to_string(),
                format!("Cache hit for prompt ({} chars)", prompt.len()),
                1.0,
                None,
                None,
            );
            hit
        } else {
            // Observation: Cache miss, computing embedding
            let obs_id = thought_builder.add_observation(
                "embedding".to_string(),
                format!("Computing embedding for prompt ({} chars)", prompt.len()),
                0.9,
                None,
                None,
            );
            let emb = self.embedder.embed(prompt).await?;
            self.embedding_cache
                .insert(cache_key, emb.clone(), now)
                .await;
            // Action: Embedding computed and cached
            thought_builder.add_action(
                "embedding_complete".to_string(),
                format!("Embedding computed: {} dimensions", emb.len()),
                std::collections::HashMap::new(),
                1.0,
                Some(obs_id),
            );
            emb
        };
        timings.embedding_ms = embedding_start.elapsed().as_secs_f64() * 1000.0;
        info!(
            "Pipeline stage: embedding completed in {:.2}ms",
            timings.embedding_ms
        );

        // Stage 2: Torus projection
        let torus_start = Instant::now();
        let mut torus_mapper = self.next_torus_mapper();
        let pad_state = torus_mapper.project(&embedding)?;
        timings.torus_ms = torus_start.elapsed().as_secs_f64() * 1000.0;

        let (emotional_snapshot, gradient_snapshot) = self.drain_pending_signals();
        if let Some(analyzer) = self.tcs_analyzer.as_mut() {
            analyzer.ingest_signals(
                &pad_state,
                if emotional_snapshot.is_empty() {
                    None
                } else {
                    Some(emotional_snapshot.as_slice())
                },
                if gradient_snapshot.is_empty() {
                    None
                } else {
                    Some(gradient_snapshot.as_slice())
                },
            );
        }

        // Reasoning: Torus projection maps embedding to PAD space
        thought_builder.add_reasoning(
            crate::telemetry::ReasoningType::Deduction,
            vec![
                format!("Embedding vector: {} dimensions", embedding.len()),
                "Torus mapper projects to 3D PAD space".to_string(),
            ],
            format!(
                "PAD state: [{:.3}, {:.3}, {:.3}]",
                pad_state.pad[0], pad_state.pad[1], pad_state.pad[2]
            ),
            0.95,
            None,
        );

        let tcs_start = Instant::now();
        let (topology, analysis_label) = match self.config.topology_mode {
            TopologyMode::Hybrid => match self.tcs_analyzer.as_mut() {
                Some(analyzer) => match analyzer.analyze_state(&pad_state) {
                    Ok(signature) => (signature, "hybrid"),
                    Err(error) => {
                        if self.config.tcs.robust_mode {
                            warn!(
                                %error,
                                "TCS analyzer failed while robust mode enabled; aborting pipeline"
                            );
                            return Err(error);
                        }
                        warn!(%error, "TCS analyzer failed; using analytic baseline signature");
                        if let Err(log_error) = crate::util::append_robust_log(
                            "tcs_analyzer",
                            &format!("hybrid fallback triggered: {error:?}"),
                        ) {
                            warn!(%log_error, "Failed to append robust audit log entry");
                        }
                        (
                            baseline_topological_signature(&pad_state, &embedding),
                            "hybrid_fallback",
                        )
                    }
                },
                None => {
                    warn!(
                        "Hybrid mode requested but TCS analyzer unavailable; using analytic baseline signature"
                    );
                    (
                        baseline_topological_signature(&pad_state, &embedding),
                        "hybrid_fallback",
                    )
                }
            },
            TopologyMode::Baseline => (
                baseline_topological_signature(&pad_state, &embedding),
                "baseline",
            ),
        };
        timings.tcs_ms = tcs_start.elapsed().as_secs_f64() * 1000.0;
        info!(
            "Pipeline stage: topology analysis completed in {:.2}ms ({analysis_label})",
            timings.tcs_ms
        );

        // Reasoning: Topological analysis
        thought_builder.add_reasoning(
            crate::telemetry::ReasoningType::Induction,
            vec![
                format!(
                    "PAD state: [{:.3}, {:.3}, {:.3}]",
                    pad_state.pad[0], pad_state.pad[1], pad_state.pad[2]
                ),
                format!("Topology mode: {:?}", self.config.topology_mode),
            ],
            format!(
                "Betti numbers: {:?}, Persistence entropy: {:.3}",
                topology.betti_numbers, topology.persistence_entropy
            ),
            0.9,
            None,
        );

        // Phase 5.3: Check if predictor should trigger (knot > 0.4)
        let _topology_json = match serde_json::to_string(&topology) {
            Ok(json) => json,
            Err(e) => {
                warn!(error = %e, "Failed to serialize topology to JSON");
                String::new()
            }
        };
        info!(
            "Topological signature: knot={:.3}, betti={:?}, pe={:.3}, gap={:.3}",
            topology.knot_complexity,
            topology.betti_numbers,
            topology.persistence_entropy,
            topology.spectral_gap
        );

        // Fetch nToken features early (with prompt only) for compass integration
        // This allows PAD state to update automatically based on H₁ persistence and sheaf energy
        let ntoken_features_for_compass = if self.config.n_tokens_bypass {
            None
        } else if let Ok(endpoint) = std::env::var("NTOKEN_ENDPOINT") {
            // Fetch with prompt only (context not available yet)
            match ntoken_client::fetch_features(&endpoint, prompt, None).await {
                Ok(Some(features)) => {
                    info!(
                        "nToken features (compass): h1_persistence={:.4}, sheaf_energy={:.4}",
                        features.h1_total_persistence, features.sheaf_energy
                    );
                    Some(features)
                }
                Ok(None) => None,
                Err(error) => {
                    warn!(%error, "nToken service call failed for compass; continuing without nToken integration");
                    None
                }
            }
        } else {
            None
        };

        // Evaluate compass on blocking thread without locking inside closure
        let pad_state_for_compass = pad_state.clone();
        let topology_for_compass = topology.clone();
        let ntoken_for_compass = ntoken_features_for_compass.clone();
        let compass_guard = self.compass.clone().lock_owned().await;
        let compass_scope = format!("compass/{}", cache_key);
        let compass_task = tokio::task::spawn_blocking(move || {
            let mut engine = compass_guard;
            let mut rng = crate::util::seed_manager().get_rng(&compass_scope);
            engine.evaluate_with_rng(
                &pad_state_for_compass,
                Some(&topology_for_compass),
                &mut rng,
                ntoken_for_compass.as_ref(),
            )
        });

        let embedding_for_collapse = embedding.clone();
        let collapse_cache = self.collapse_cache.clone();
        let erag_client = self.erag.clone();
        let retrieval_top_k_increment = self.config.phase2_retrieval_top_k_increment;
        let erag_bypass = self.config.erag_bypass;
        let base_retrieval_top_k = self.config.base_retrieval_top_k;
        let pipeline_retrieval_top_k_min = self.config.pipeline_retrieval_top_k_min;
        let pipeline_retrieval_top_k_max = self.config.pipeline_retrieval_top_k_max;

        // Start timing BEFORE the parallel work begins
        let compass_erag_start = Instant::now();
        let (compass, collapse) = tokio::try_join!(
            async {
                match compass_task.await {
                    Ok(inner) => inner,
                    Err(e) => Err(anyhow::anyhow!(format!(
                        "compass evaluation panicked: {}",
                        e
                    ))),
                }
            },
            async move {
                // Ablation testing: ERAG bypass (zero-shot mode)
                if erag_bypass {
                    info!("ERAG bypass enabled (zero-shot mode); returning empty collapse");
                    return Ok(crate::erag::CollapseResult {
                        top_hits: Vec::new(),
                        aggregated_context: String::new(),
                        average_similarity: 0.0,
                        curator_quality: None,
                    });
                }

                if let Some(hit) = collapse_cache.get(&cache_key, now).await {
                    Ok(hit)
                } else {
                    // Dynamic top_k based on config knobs (reuses retrieval_top_k_increment as delta)
                    let top_k = (base_retrieval_top_k + retrieval_top_k_increment).clamp(
                        pipeline_retrieval_top_k_min as i32,
                        pipeline_retrieval_top_k_max as i32,
                    ) as usize;

                    // Try Golden Memory first, fallback to regular ERAG
                    let collapse = match erag_client
                        .search_golden_memory(&embedding_for_collapse, top_k / 2, 0.8)
                        .await
                    {
                        Ok(golden_memories) if !golden_memories.is_empty() => {
                            info!("🌟 Using {} Golden Memories", golden_memories.len());
                            let remaining = top_k.saturating_sub(golden_memories.len());
                            let mut all_memories = golden_memories;
                            if remaining > 0 {
                                if let Ok(regular_collapse) = erag_client
                                    .collapse_with_limit(&embedding_for_collapse, remaining)
                                    .await
                                {
                                    all_memories.extend(regular_collapse.top_hits);
                                }
                            }
                            crate::erag::CollapseResult {
                                top_hits: all_memories,
                                aggregated_context: "Golden Memory + ERAG".to_string(),
                                average_similarity: 0.8,
                                curator_quality: Some(0.9),
                            }
                        }
                        _ => {
                            erag_client
                                .collapse_with_limit(&embedding_for_collapse, top_k)
                                .await?
                        }
                    };
                    collapse_cache
                        .insert(cache_key, collapse.clone(), now)
                        .await;
                    Ok(collapse)
                }
            }
        )?;
        let mut collapse = collapse;
        // Measure elapsed time AFTER the work completes
        let compass_erag_elapsed = compass_erag_start.elapsed().as_secs_f64() * 1000.0;
        let split_ratio = self.config.pipeline_timing_split_ratio;
        timings.compass_ms = compass_erag_elapsed * split_ratio;
        timings.erag_ms = compass_erag_elapsed * (1.0 - split_ratio);
        info!(
            "Pipeline stage: compass completed in {:.2}ms",
            timings.compass_ms
        );
        info!("Pipeline stage: erag completed in {:.2}ms", timings.erag_ms);

        let reweight_stats = self.apply_topological_reweighting(&topology, &mut collapse);
        if let Some(stats) = reweight_stats {
            thought_builder.add_reasoning(
                crate::telemetry::ReasoningType::Other("topology_retrieval".to_string()),
                vec![
                    format!("Mean affinity {:.3}", stats.mean_weight),
                    format!("Best β1 alignment {:.3}", stats.best_betti_alignment),
                ],
                "Reordered ERAG memories using homology-guided weighting".to_string(),
                0.82,
                None,
            );
        }

        self.set_pending_topology_emotions(
            collapse
                .top_hits
                .iter()
                .map(|mem| mem.emotional_vector.clone())
                .collect(),
        );

        let consensus_threshold = f64::from(self.config.tcs.persistence_threshold.max(0.8f32));
        let mut reflective_prompt: Option<String> = None;
        if topology.max_persistence < consensus_threshold {
            reflective_prompt = Some(format!(
                "Reflective mode: reconsider topology. max_persistence={:.3}, entropy={:.3}",
                topology.max_persistence, topology.persistence_entropy
            ));
            collapse.top_hits.clear();
            collapse.average_similarity = 0.0;
            collapse.curator_quality = Some(0.0);
            if let Some(prompt) = reflective_prompt.clone() {
                collapse.aggregated_context = prompt;
            }
        }
        if let Some(prompt) = &reflective_prompt {
            thought_builder.add_reasoning(
                crate::telemetry::ReasoningType::Other("reflection".to_string()),
                vec![format!(
                    "Topology persistence {:.3} below consensus {:.3}",
                    topology.max_persistence, consensus_threshold
                )],
                prompt.clone(),
                0.85,
                None,
            );
        }

        // Topo-Reflection: compute thinking depth and pivot score post O₂ and TCS analysis
        let baseline_sig = baseline_topological_signature(&pad_state, &embedding);
        let reflection =
            TopoReflectionStage::new().run(&topology, &baseline_sig, &pad_state, &collapse);
        let reflection_context = format!(
            "[TopoReflection] thinking_depth={:.4} pivot_score={:.4}",
            reflection.thinking_depth, reflection.pivot_score
        );
        info!(
            thinking_depth = %format!("{:.4}", reflection.thinking_depth),
            pivot_score = %format!("{:.4}", reflection.pivot_score),
            "Topo-Reflection computed"
        );

        // Decision: Compass quadrant selection
        let compass_options = vec![
            crate::telemetry::DecisionOption {
                index: 0,
                description: "Panic".to_string(),
                score: if matches!(compass.quadrant, CompassQuadrant::Panic) {
                    compass.intrinsic_reward as f32
                } else {
                    0.0
                },
                pruned: !matches!(compass.quadrant, CompassQuadrant::Panic),
                pruning_rationale: if matches!(compass.quadrant, CompassQuadrant::Panic) {
                    None
                } else {
                    Some("Not selected".to_string())
                },
            },
            crate::telemetry::DecisionOption {
                index: 1,
                description: "Persist".to_string(),
                score: if matches!(compass.quadrant, CompassQuadrant::Persist) {
                    compass.intrinsic_reward as f32
                } else {
                    0.0
                },
                pruned: !matches!(compass.quadrant, CompassQuadrant::Persist),
                pruning_rationale: if matches!(compass.quadrant, CompassQuadrant::Persist) {
                    None
                } else {
                    Some("Not selected".to_string())
                },
            },
            crate::telemetry::DecisionOption {
                index: 2,
                description: "Discover".to_string(),
                score: if matches!(compass.quadrant, CompassQuadrant::Discover) {
                    compass.intrinsic_reward as f32
                } else {
                    0.0
                },
                pruned: !matches!(compass.quadrant, CompassQuadrant::Discover),
                pruning_rationale: if matches!(compass.quadrant, CompassQuadrant::Discover) {
                    None
                } else {
                    Some("Not selected".to_string())
                },
            },
            crate::telemetry::DecisionOption {
                index: 3,
                description: "Master".to_string(),
                score: if matches!(compass.quadrant, CompassQuadrant::Master) {
                    compass.intrinsic_reward as f32
                } else {
                    0.0
                },
                pruned: !matches!(compass.quadrant, CompassQuadrant::Master),
                pruning_rationale: if matches!(compass.quadrant, CompassQuadrant::Master) {
                    None
                } else {
                    Some("Not selected".to_string())
                },
            },
        ];
        let chosen_quadrant = match compass.quadrant {
            CompassQuadrant::Panic => 0,
            CompassQuadrant::Persist => 1,
            CompassQuadrant::Discover => 2,
            CompassQuadrant::Master => 3,
        };
        thought_builder.add_decision(
            "Compass quadrant selection".to_string(),
            compass_options,
            chosen_quadrant,
            format!(
                "Selected {:?} with confidence {:.3}",
                compass.quadrant, compass.intrinsic_reward
            ),
            compass.intrinsic_reward as f32,
            None,
        );

        // Memory: Retrieved memories
        for (idx, mem) in collapse.top_hits.iter().enumerate() {
            thought_builder.add_memory(
                blake3::hash(format!("{}{}", mem.input, mem.output).as_bytes())
                    .to_hex()
                    .chars()
                    .take(16)
                    .collect(),
                format!(
                    "Memory {}: {} -> {}",
                    idx + 1,
                    mem.input.chars().take(50).collect::<String>(),
                    mem.output.chars().take(50).collect::<String>()
                ),
                collapse.average_similarity,
                1.0 / (idx + 1) as f32,
                None,
            );
        }

        // Refetch nToken features with full context for tokenizer refinement
        // (we already have initial features for compass, but context-aware version is better for tokenizer)
        let ntoken_features = if self.config.n_tokens_bypass {
            info!("nTokens bypass enabled; skipping nToken feature extraction");
            None
        } else if let Ok(endpoint) = std::env::var("NTOKEN_ENDPOINT") {
            match ntoken_client::fetch_features(
                &endpoint,
                prompt,
                Some(&collapse.aggregated_context),
            )
            .await
            {
                Ok(Some(features)) => {
                    info!(
                        "nToken cues (tokenizer): h1_count={:.0}, persistence={:.4}, entropy_norm={:.4}, sheaf_energy={:.4}",
                        features.h1_count,
                        features.h1_total_persistence,
                        features.entropy_norm,
                        features.sheaf_energy
                    );
                    Some(features)
                }
                Ok(None) => ntoken_features_for_compass, // Fall back to compass features if available
                Err(error) => {
                    warn!(%error, "nToken service call failed; using compass features if available");
                    ntoken_features_for_compass // Fall back to compass features
                }
            }
        } else {
            ntoken_features_for_compass // Use compass features if no endpoint
        };

        // RTX 5090 OPTIMIZATION: Parallelize consonance + hyperfocus detection
        // These can run concurrently since hyperfocus only needs consonance score
        let last_compass = self.last_compass_outcome.lock().await.clone();

        // Compute partial consonance (without curator for now)
        // RTX 5090: Use GPU-accelerated consonance if available
        #[cfg(feature = "gpu")]
        let partial_consonance = {
            if let Ok(device) = candle_core::Device::cuda_if_available(0) {
                use crate::gpu_consonance::GpuConsonanceCalculator;
                let gpu_calc = GpuConsonanceCalculator::new(device);

                // Extract PAD state for GPU variance calculation
                let pad_array = [pad_state.pad];
                if let Ok(variances) = gpu_calc.batch_pad_variance(&pad_array).await {
                    if !variances.is_empty() {
                        let pad_variance = variances[0];
                        let emotional_coherence = (1.0 - pad_variance.min(1.0)).max(0.0);

                        // Compute other components (still CPU for now, can be optimized later)
                        let topological_consistency =
                            compute_topological_consistency(&topology, &pad_state);
                        let erag_relevance = compute_erag_relevance(&collapse);
                        let compass_transition =
                            compute_compass_transition(&compass, last_compass.as_ref());

                        // Weighted combination
                        let sources = vec![
                            crate::consonance::ConsonanceSource::EmotionalCoherence(
                                emotional_coherence,
                            ),
                            crate::consonance::ConsonanceSource::TopologicalConsistency(
                                topological_consistency,
                            ),
                            crate::consonance::ConsonanceSource::ERAGRelevance(erag_relevance),
                            crate::consonance::ConsonanceSource::CompassTransition(
                                compass_transition,
                            ),
                            crate::consonance::ConsonanceSource::CuratorQuality(0.5), // No curator yet
                        ];

                        let weights = [0.25, 0.20, 0.25, 0.20, 0.10];
                        let source_scores_array: [f64; 5] = [
                            sources[0].score(),
                            sources[1].score(),
                            sources[2].score(),
                            sources[3].score(),
                            sources[4].score(),
                        ];
                        if let Ok(batch_scores) = gpu_calc
                            .batch_weighted_consonance(&[source_scores_array], &weights)
                            .await
                        {
                            if !batch_scores.is_empty() {
                                let score = batch_scores[0];
                                let confidence = compute_confidence(&sources);
                                crate::consonance::ConsonanceMetrics {
                                    score,
                                    sources,
                                    confidence,
                                    dissonance_score: (1.0 - score).clamp(0.0, 1.0),
                                }
                            } else {
                                compute_consonance(
                                    &pad_state,
                                    &compass,
                                    &collapse,
                                    &topology,
                                    None,
                                    last_compass.as_ref(),
                                )
                            }
                        } else {
                            compute_consonance(
                                &pad_state,
                                &compass,
                                &collapse,
                                &topology,
                                None,
                                last_compass.as_ref(),
                            )
                        }
                    } else {
                        compute_consonance(
                            &pad_state,
                            &compass,
                            &collapse,
                            &topology,
                            None,
                            last_compass.as_ref(),
                        )
                    }
                } else {
                    compute_consonance(
                        &pad_state,
                        &compass,
                        &collapse,
                        &topology,
                        None,
                        last_compass.as_ref(),
                    )
                }
            } else {
                compute_consonance(
                    &pad_state,
                    &compass,
                    &collapse,
                    &topology,
                    None,
                    last_compass.as_ref(),
                )
            }
        };
        #[cfg(not(feature = "gpu"))]
        let partial_consonance = compute_consonance(
            &pad_state,
            &compass,
            &collapse,
            &topology,
            None, // Curator not available yet
            last_compass.as_ref(),
        );

        // Track cascade transition
        let cascade_transition = {
            let mut tracker = self.cascade_tracker.lock().await;
            tracker.detect_transition(&compass, partial_consonance.score)
        };

        // Update compass with cascade stage
        let mut compass_with_cascade = compass.clone();
        if let Some(ref transition) = cascade_transition {
            compass_with_cascade.cascade_stage = Some(transition.to);
        } else {
            // Use current stage if no transition
            let tracker = self.cascade_tracker.lock().await;
            compass_with_cascade.cascade_stage = tracker.current_stage();
        }

        // Update last compass outcome
        *self.last_compass_outcome.lock().await = Some(compass_with_cascade.clone());

        // Detect hyperfocus (need to build signal map)
        use std::collections::HashMap;
        let mut hyperfocus_signals = HashMap::new();
        hyperfocus_signals.insert("compass".to_string(), partial_consonance.clone());
        hyperfocus_signals.insert("erag".to_string(), partial_consonance.clone());
        hyperfocus_signals.insert("topology".to_string(), partial_consonance.clone());

        // Stage 5: Tokenizer
        let tokenizer_start = Instant::now();

        // RTX 5090 OPTIMIZATION: GPU-accelerated RCE scoring for cosine similarity
        let mut top_hits = collapse.top_hits.clone();
        if self.config.rce_erag_lambda > 0.0 && !top_hits.is_empty() {
            let lambda = self.config.rce_erag_lambda;

            // RTX 5090: Use GPU for batch cosine similarity if available
            #[cfg(feature = "gpu")]
            {
                if let Ok(device) = candle_core::Device::cuda_if_available(0) {
                    use crate::gpu_consonance::GpuConsonanceCalculator;
                    let gpu_calc = GpuConsonanceCalculator::new(device);

                    // Prepare vectors for GPU batch processing
                    let pad_vecs: Vec<Vec<f32>> = vec![
                        vec![
                            pad_state.pad[0] as f32,
                            pad_state.pad[1] as f32,
                            pad_state.pad[2] as f32
                        ];
                        top_hits.len()
                    ];
                    let mem_vecs: Vec<Vec<f32>> = top_hits
                        .iter()
                        .map(|m| {
                            vec![
                                m.emotional_vector.joy,
                                m.emotional_vector.anger,
                                m.emotional_vector.surprise,
                            ]
                        })
                        .collect();

                    // Batch compute cosine similarities on GPU
                    if let Ok(cosines) =
                        gpu_calc.batch_cosine_similarity(&pad_vecs, &mem_vecs).await
                    {
                        // Compute entropy scores
                        let ent_after_vec: Vec<f64> =
                            top_hits.iter().map(|m| m.entropy_after).collect();
                        let entropy_scores: Vec<f64> = ent_after_vec
                            .iter()
                            .map(|ent_after| {
                                1.0 - (topology.persistence_entropy - ent_after).abs().min(1.0)
                            })
                            .collect();

                        // Combine scores
                        let scores: Vec<f64> = cosines
                            .iter()
                            .enumerate()
                            .map(|(i, &cosine)| {
                                (self.config.rce_erag_cosine_weight * cosine as f64
                                    + self.config.rce_erag_entropy_weight * entropy_scores[i])
                                    * lambda
                            })
                            .collect();

                        // Sort by score
                        let mut indexed: Vec<(usize, f64)> =
                            scores.into_iter().enumerate().collect();
                        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
                        top_hits = indexed
                            .into_iter()
                            .map(|(i, _)| top_hits[i].clone())
                            .collect();
                    } else {
                        // Fallback to CPU if GPU fails
                        top_hits.sort_by(|a, b| {
                            let score = |m: &crate::erag::EragMemory| {
                                let pad_vec = [
                                    pad_state.pad[0] as f64,
                                    pad_state.pad[1] as f64,
                                    pad_state.pad[2] as f64,
                                ];
                                let mem_vec = [
                                    m.emotional_vector.joy as f64,
                                    m.emotional_vector.anger as f64,
                                    m.emotional_vector.surprise as f64,
                                ];
                                let dot = pad_vec[0] * mem_vec[0]
                                    + pad_vec[1] * mem_vec[1]
                                    + pad_vec[2] * mem_vec[2];
                                let n1 = (pad_vec[0] * pad_vec[0]
                                    + pad_vec[1] * pad_vec[1]
                                    + pad_vec[2] * pad_vec[2])
                                    .sqrt();
                                let n2 = (mem_vec[0] * mem_vec[0]
                                    + mem_vec[1] * mem_vec[1]
                                    + mem_vec[2] * mem_vec[2])
                                    .sqrt();
                                let cosine = if n1 > 0.0 && n2 > 0.0 {
                                    (dot / (n1 * n2)).clamp(-1.0, 1.0)
                                } else {
                                    0.0
                                };
                                let ent_after = m.entropy_after;
                                let ent_score =
                                    1.0 - (topology.persistence_entropy - ent_after).abs().min(1.0);
                                (self.config.rce_erag_cosine_weight * cosine
                                    + self.config.rce_erag_entropy_weight * ent_score)
                                    * lambda
                            };
                            score(b).partial_cmp(&score(a)).unwrap_or(Ordering::Equal)
                        });
                    }
                } else {
                    // CPU fallback
                    top_hits.sort_by(|a, b| {
                        let score = |m: &crate::erag::EragMemory| {
                            let pad_vec = [
                                pad_state.pad[0] as f64,
                                pad_state.pad[1] as f64,
                                pad_state.pad[2] as f64,
                            ];
                            let mem_vec = [
                                m.emotional_vector.joy as f64,
                                m.emotional_vector.anger as f64,
                                m.emotional_vector.surprise as f64,
                            ];
                            let dot = pad_vec[0] * mem_vec[0]
                                + pad_vec[1] * mem_vec[1]
                                + pad_vec[2] * mem_vec[2];
                            let n1 = (pad_vec[0] * pad_vec[0]
                                + pad_vec[1] * pad_vec[1]
                                + pad_vec[2] * pad_vec[2])
                                .sqrt();
                            let n2 = (mem_vec[0] * mem_vec[0]
                                + mem_vec[1] * mem_vec[1]
                                + mem_vec[2] * mem_vec[2])
                                .sqrt();
                            let cosine = if n1 > 0.0 && n2 > 0.0 {
                                (dot / (n1 * n2)).clamp(-1.0, 1.0)
                            } else {
                                0.0
                            };
                            let ent_after = m.entropy_after;
                            let ent_score =
                                1.0 - (topology.persistence_entropy - ent_after).abs().min(1.0);
                            (self.config.rce_erag_cosine_weight * cosine
                                + self.config.rce_erag_entropy_weight * ent_score)
                                * lambda
                        };
                        score(b).partial_cmp(&score(a)).unwrap_or(Ordering::Equal)
                    });
                }
            }
            #[cfg(not(feature = "gpu"))]
            {
                // CPU-only path
                top_hits.sort_by(|a, b| {
                    let score = |m: &crate::erag::EragMemory| {
                        let pad_vec = [
                            pad_state.pad[0] as f64,
                            pad_state.pad[1] as f64,
                            pad_state.pad[2] as f64,
                        ];
                        let mem_vec = [
                            m.emotional_vector.joy as f64,
                            m.emotional_vector.anger as f64,
                            m.emotional_vector.surprise as f64,
                        ];
                        let dot = pad_vec[0] * mem_vec[0]
                            + pad_vec[1] * mem_vec[1]
                            + pad_vec[2] * mem_vec[2];
                        let n1 = (pad_vec[0] * pad_vec[0]
                            + pad_vec[1] * pad_vec[1]
                            + pad_vec[2] * pad_vec[2])
                            .sqrt();
                        let n2 = (mem_vec[0] * mem_vec[0]
                            + mem_vec[1] * mem_vec[1]
                            + mem_vec[2] * mem_vec[2])
                            .sqrt();
                        let cosine = if n1 > 0.0 && n2 > 0.0 {
                            (dot / (n1 * n2)).clamp(-1.0, 1.0)
                        } else {
                            0.0
                        };
                        let ent_after = m.entropy_after;
                        let ent_score =
                            1.0 - (topology.persistence_entropy - ent_after).abs().min(1.0);
                        (self.config.rce_erag_cosine_weight * cosine
                            + self.config.rce_erag_entropy_weight * ent_score)
                            * lambda
                    };
                    score(b).partial_cmp(&score(a)).unwrap_or(Ordering::Equal)
                });
            }
        }

        let mut adapted_context = if self.config.rce_erag_lambda > 0.0 && !top_hits.is_empty() {
            let mut ctx = top_hits
                .iter()
                .flat_map(|m| m.erag_context.clone())
                .collect::<Vec<_>>()
                .join("\n");
            if ctx.len() > self.config.context_truncation_limit {
                ctx.truncate(self.config.context_truncation_limit);
            }
            ctx
        } else {
            collapse.aggregated_context.clone()
        };

        if self.config.rce_actions_enabled && !self.config.rce_shadow_mode {
            if topology.persistence_entropy > self.config.rce_adaptation_entropy_threshold
                || topology.spectral_gap > self.config.rce_adaptation_spectral_gap_threshold
            {
                adapted_context = adapted_context.replace(". ", ".\n");
                adapted_context = adapted_context.replace("; ", ";\n");
                adapted_context = adapted_context.replace(", ", ",\n");
            }
        }

        let collapse_for_tokenizer = crate::erag::CollapseResult {
            top_hits: top_hits.clone(),
            aggregated_context: adapted_context,
            average_similarity: collapse.average_similarity,
            curator_quality: collapse.curator_quality,
        };

        let mut tokenizer_output = self
            .tokenizer
            .process_with_memories(prompt, &collapse_for_tokenizer, &pad_state, top_hits)
            .await?;
        timings.tokenizer_ms = tokenizer_start.elapsed().as_secs_f64() * 1000.0;

        if let Some(stats) = reweight_stats {
            let route_mode = if stats.mean_weight >= self.config.tcs.retrieval_route_convergence {
                "convergent"
            } else {
                "divergent"
            };
            let topology_route = format!(
                "[Topology Route] mode={route_mode} mean_affinity={:.3} best_alignment={:.3} persistence_norm={:.3} entropy_delta={:.3}",
                stats.mean_weight,
                stats.best_betti_alignment,
                stats.best_persistence_norm,
                stats.best_entropy_delta,
            );
            tokenizer_output.augmented_prompt =
                format!("{topology_route}\n{}", tokenizer_output.augmented_prompt);
        }

        if let Some(features) = &ntoken_features {
            let cues = format!(
                "[nToken cues] h1_count={:.0} h1_persistence={:.4} entropy_norm={:.4} sheaf_energy={:.4}",
                features.h1_count,
                features.h1_total_persistence,
                features.entropy_norm,
                features.sheaf_energy
            );
            tokenizer_output.augmented_prompt =
                format!("{cues}\n{}", tokenizer_output.augmented_prompt);
        }
        // Inject Topo-Reflection context to encourage Topo-CoT grounding downstream
        tokenizer_output.augmented_prompt = format!(
            "{reflection_context}\n{}",
            tokenizer_output.augmented_prompt
        );

        // If thinking depth exceeds threshold, elicit or inject TopoCoT guidance
        let cot_stage = TopoReflectionStage::new();
        let cot_trigger = cot_stage.should_trigger_cot(&reflection);
        let mut topocot_telemetry: Option<TopoCotTelemetry> = None;
        let mut topocot_plan_summary: Option<String> = None;
        let mut topocot_payload: Option<TopoCoT> = None;
        if cot_trigger {
            if let Some(plan_eval) = self
                .elicit_topocot_plan(prompt, &topology, &reflection, &pad_state, &collapse)
                .await?
            {
                if let Some(payload) = plan_eval.payload.clone() {
                    topocot_payload = Some(payload.clone());
                    let plan_summary = Self::format_topocot_plan(&payload, &topology, &reflection);
                    tokenizer_output.augmented_prompt =
                        format!("{plan_summary}\n{}", tokenizer_output.augmented_prompt);
                    topocot_plan_summary = Some(plan_summary.clone());
                    topocot_telemetry = Some(TopoCotTelemetry {
                        score_overall: plan_eval.score.overall,
                        score_completeness: plan_eval.score.completeness,
                        score_consistency: plan_eval.score.consistency,
                        score_actionability: plan_eval.score.actionability,
                        issues: plan_eval.issues.clone(),
                        raw_json: plan_eval.raw_json.clone(),
                        thinking_depth: reflection.thinking_depth,
                        pivot_score: reflection.pivot_score,
                        plan_summary: Some(plan_summary),
                    });

                    let topology_mode_label = format!("{:?}", self.config.topology_mode);
                    let response_for_log = plan_eval
                        .raw_json
                        .clone()
                        .unwrap_or_else(|| payload.step_4_final_output_grounding.clone());
                    let plan_log_entry = TopoCotLogEntry::from_evaluation(
                        &plan_eval,
                        prompt,
                        &response_for_log,
                        reflection.thinking_depth,
                        reflection.pivot_score,
                        topology.betti_numbers,
                        &topology_mode_label,
                    );
                    if let Err(error) = append_topocot_log(&plan_log_entry) {
                        warn!(?error, "failed to persist TopoCoT telemetry log");
                    }
                }
            }

            if topocot_plan_summary.is_none() {
                let schema = TopoCoT::json_schema();
                let schema_str = serde_json::to_string_pretty(&schema).unwrap_or_default();
                let cot_instruction = r#"Output ONLY minified JSON whose root keys exactly match the TopoCoT schema (`step_1_analysis`, `step_2_emotional_mapping`, `step_3_causal_bridge`, `step_4_final_output_grounding`, optional `computed_artifacts`). Start with '{' immediately, end with '}'—no prose, no markdown, no arrays like \"steps\" at the root.\nTopoCoT triggered — follow the scaffold to map topology into action:\n- `step_1_analysis` restates the Betti numbers and interprets them for the task.\n- `step_2_emotional_mapping` provides signed PAD shifts plus justification tied to the prompt.\n- `step_3_causal_bridge` links obstacles to resolutions via a reasoning_chain formatted as \"action -> action -> action\".\n- `step_4_final_output_grounding` maps the plan to the requested deliverable.\n\nCRITICAL: respond with JSON only. First character '{', last character '}'. Example shape:\n{"step_1_analysis":{"betti_0_components":2,"betti_1_loops":6,"betti_2_voids":0,"summary":"..."},"step_2_emotional_mapping":{"pad_arousal_shift":0.0,"pad_valence_shift":0.0,"justification":"..."},"step_3_causal_bridge":{"obstacle":"...","resolution_path":"...","reasoning_chain":"step -> step -> step"},"step_4_final_output_grounding":"..."}\n\nNever answer with placeholders such as 'Pull which?' or 'N/A' — compute or derive the needed values."#;
                let schema_with_reminder = format!("{}\n\nRemember: Response is JSON only.", schema_str);
                tokenizer_output.augmented_prompt = format!(
                    "{cot_instruction}\n[TopoCoT_JSON_SCHEMA]\n{schema_with_reminder}\n\n[USER_PROMPT]\n{}",
                    tokenizer_output.augmented_prompt
                );
            }
        }

        // Update generation engine with latest config params (before generation)
        let current_config = self.config_arc.read().clone();
        // Note: apply_runtime_from_config takes CliArgs, not RuntimeConfig
        // Skip for now - generator params are set at initialization
        self.generator
            .update_params(current_config.temperature, current_config.top_p);
        self.config = current_config;

        // Recompute thresholds from updated config and update compass
        self.recompute_thresholds();

        // Stage 6: Generation
        let generation_start = Instant::now();
        // Apply latest runtime parameters before generation
        {
            let cfg = self.config_arc.read().clone();
            // Note: apply_runtime_from_config takes CliArgs, not RuntimeConfig - skip for now
            // self.generator.apply_runtime_from_config(&cfg);
            self.recompute_thresholds();
            self.config = cfg;
        }
        let mut generation = if self.config.enable_consistency_voting {
            let voting = self
                .generator
                .generate_with_consistency(&tokenizer_output, &compass)
                .await?;

            let selected = match voting.winner_index {
                0 => &voting.candidate_1,
                1 => &voting.candidate_2,
                _ => &voting.candidate_3,
            }
            .clone();

            GenerationResult {
                baseline_response: tokenizer_output.augmented_prompt.clone(),
                hybrid_response: selected,
                echoes: Vec::new(),
                rouge_to_baseline: voting.rouge_scores.iter().copied().sum::<f64>()
                    / voting.rouge_scores.len() as f64,
                latency_ms: voting.latency_ms,
                rouge_score: voting.rouge_scores.iter().copied().sum::<f64>()
                    / voting.rouge_scores.len() as f64,
                entropy_delta: 0.0,
                source: "consistency".to_string(),
                ucb1_score: Some(
                    compass
                        .mcts_branches
                        .iter()
                        .map(|b| b.ucb_score)
                        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                        .unwrap_or(0.5),
                ),
                curator_quality: Some(self.config.consistency_voting_quality), // Default quality for consistency voting
                failure_type: None,
                failure_details: None,
            }
        } else {
            self.generator
                .generate_with_topology(&tokenizer_output, &compass, Some(&topology), false)
                .await?
        };
        timings.generation_ms = generation_start.elapsed().as_secs_f64() * 1000.0;
        info!(
            "Pipeline stage: generation completed in {:.2}ms",
            timings.generation_ms
        );

        if cot_trigger {
            let payload_for_fallback = topocot_payload.clone().or_else(|| {
                topocot_telemetry
                    .as_ref()
                    .and_then(|telemetry| telemetry.raw_json.as_ref())
                    .and_then(|raw| serde_json::from_str::<TopoCoT>(raw).ok())
            });

            if let Some(plan) = payload_for_fallback {
                // Check if this is a fallback plan (via issue tags) or if response is low-signal
                let is_fallback_plan = topocot_telemetry
                    .as_ref()
                    .map(|telemetry| {
                        telemetry
                            .issues
                            .iter()
                            .any(|issue| issue.contains("topocot_autofill_fallback"))
                    })
                    .unwrap_or(false);
                let is_low_signal = Self::is_low_signal_response(&generation.hybrid_response);
                
                if is_low_signal || is_fallback_plan {
                    if is_fallback_plan {
                        info!("Executing fallback TopoCoT plan (detected via issue tags)");
                    } else {
                        info!("Replacing low-signal generation with deterministic TopoCoT execution");
                    }
                    match Self::enforced_math_synthesis(
                        &plan,
                        prompt,
                        &topology,
                        &reflection,
                        &pad_state,
                    ) {
                        Ok((executed_plan, exec_result)) => {
                            match Self::validate_deliverables(&exec_result) {
                                Ok(()) => {
                                    info!(
                                        duration_ms = exec_result.execution_duration_ms,
                                        "TopoCoT execution succeeded; injecting deterministic response"
                                    );
                                    generation.hybrid_response = Self::render_topocot_response(
                                        &executed_plan,
                                        &exec_result,
                                        prompt,
                                        &topology,
                                        &reflection,
                                        &pad_state,
                                    );
                                    Self::annotate_topocot_issue(
                                        &mut topocot_telemetry,
                                        &executed_plan,
                                        &topology,
                                        &reflection,
                                        "topocot_autofill_fallback_EXECUTED",
                                    );
                                    if topocot_plan_summary.is_none() {
                                        topocot_plan_summary = Some(Self::format_topocot_plan(
                                            &executed_plan,
                                            &topology,
                                            &reflection,
                                        ));
                                    }
                                }
                                Err(missing) => {
                                    warn!(
                                        missing = %missing.join(", "),
                                        "TopoCoT execution missing required deliverables"
                                    );
                                    generation.hybrid_response =
                                        Self::render_topocot_validation_failure(
                                            &executed_plan,
                                            &exec_result,
                                            &missing,
                                            prompt,
                                            &topology,
                                            &reflection,
                                            &pad_state,
                                        );
                                    Self::annotate_topocot_issue(
                                        &mut topocot_telemetry,
                                        &executed_plan,
                                        &topology,
                                        &reflection,
                                        "topocot_autofill_fallback_VALIDATION_FAILED",
                                    );
                                    if topocot_plan_summary.is_none() {
                                        topocot_plan_summary = Some(Self::format_topocot_plan(
                                            &executed_plan,
                                            &topology,
                                            &reflection,
                                        ));
                                    }
                                }
                            }
                        }
                        Err(error) => {
                            warn!(
                                %error,
                                "Deterministic TopoCoT execution failed; rendering diagnostic scaffold"
                            );
                            generation.hybrid_response = Self::render_topocot_execution_failure(
                                &plan,
                                prompt,
                                &topology,
                                &reflection,
                                &pad_state,
                                &error,
                            );
                            Self::annotate_topocot_issue(
                                &mut topocot_telemetry,
                                &plan,
                                &topology,
                                &reflection,
                                "topocot_autofill_fallback_EXECUTION_ERROR",
                            );
                            if topocot_plan_summary.is_none() {
                                topocot_plan_summary =
                                    Some(Self::format_topocot_plan(&plan, &topology, &reflection));
                            }
                        }
                    }
                }
            }
        }

        if cot_trigger && topocot_telemetry.is_none() {
            let evaluation = TopoCoT::evaluate_response(&generation.hybrid_response);
            let topology_mode_label = format!("{:?}", self.config.topology_mode);
            let log_entry = TopoCotLogEntry::from_evaluation(
                &evaluation,
                prompt,
                &generation.hybrid_response,
                reflection.thinking_depth,
                reflection.pivot_score,
                topology.betti_numbers,
                &topology_mode_label,
            );
            if let Err(error) = append_topocot_log(&log_entry) {
                warn!(?error, "failed to persist TopoCoT telemetry log");
            }

            topocot_telemetry = Some(TopoCotTelemetry {
                score_overall: evaluation.score.overall,
                score_completeness: evaluation.score.completeness,
                score_consistency: evaluation.score.consistency,
                score_actionability: evaluation.score.actionability,
                issues: evaluation.issues.clone(),
                raw_json: evaluation.raw_json.clone(),
                thinking_depth: reflection.thinking_depth,
                pivot_score: reflection.pivot_score,
                plan_summary: None,
            });

            let issues_summary = if evaluation.issues.is_empty() {
                "schema_ok".to_string()
            } else {
                evaluation.issues.join("|")
            };

            thought_builder.add_reasoning(
                crate::telemetry::ReasoningType::Other("topocot_grade".to_string()),
                vec![
                    format!("grade_overall={:.3}", evaluation.score.overall),
                    format!("issues={issues_summary}"),
                ],
                evaluation
                    .raw_json
                    .as_ref()
                    .map(|raw| format!("Captured JSON with {} chars", raw.len()))
                    .unwrap_or_else(|| "No TopoCoT JSON captured".to_string()),
                0.85,
                None,
            );
        }

        // Action: Generation completed
        let mut gen_params = std::collections::HashMap::new();
        gen_params.insert("source".to_string(), generation.source.clone());
        gen_params.insert("latency_ms".to_string(), generation.latency_ms.to_string());
        if let Some(ucb1) = generation.ucb1_score {
            gen_params.insert("ucb1_score".to_string(), ucb1.to_string());
        }
        thought_builder.add_action(
            "generation".to_string(),
            format!(
                "Generated response: {} chars",
                generation.hybrid_response.len()
            ),
            gen_params,
            if generation.failure_type.is_some() {
                0.5
            } else {
                0.9
            },
            None,
        );

        // NEW: Phase 2 Integration - Call curator after generation WITH TOPOLOGY
        let mut curated_experience = self
            .integrate_curator(
                prompt,
                &generation.hybrid_response,
                &pad_state,
                &compass_with_cascade,
                &collapse.aggregated_context,
                &topology,
                &tokenizer_output,
            )
            .await?;

        // Phase 1 — RCE Telemetry (shadow mode): compute β_meta and export metrics
        let mut rce_retry_approved = true; // default allow
        if self.config.rce_enabled {
            // Lazily initialise analyzer on first use
            if self.rce_analyzer.is_none() {
                let w = self.config.rce_beta_meta_weights;
                let weights = tcs_rce::beta_meta::BetaMetaWeights {
                    alpha_betti: w.alpha_betti,
                    alpha_meta: w.alpha_meta,
                    alpha_motif: w.alpha_motif,
                    alpha_sheaf: w.alpha_sheaf,
                };
                let window = self.config.rce_window_seconds as usize;
                let threshold = self.config.rce_breakthrough_threshold;
                self.rce_analyzer = Some(crate::rce::analyzer::RceAnalyzer::new(
                    window.max(2),
                    weights,
                    threshold,
                ));
                tracing::info!("RCE initialized in shadow mode (read-only metrics)");
            }
            if let Some(analyzer) = self.rce_analyzer.as_mut() {
                // Pass prompt timestamp for prompt-to-spike latency tracking
                let beta = analyzer.update_with_prompt_timestamp(
                    &pad_state,
                    &topology,
                    Some(overall_start),
                );
                // Consensus gate (read-only): combine diverse simple votes
                let mut approved = true;
                if self.config.rce_consensus.enabled {
                    let gate = crate::rce::safety::ensemble::ConsensusGate::new(
                        self.config.rce_consensus.clone(),
                    );
                    let vote_beta = beta >= self.config.rce_breakthrough_threshold;
                    let vote_meta =
                        analyzer.current_metastability() * topology.persistence_entropy > 0.0;
                    let vote_spec = topology.spectral_gap > 0.0;
                    approved = gate.approve(&[vote_beta, vote_meta, vote_spec]);
                    if approved {
                        tracing::info!("RCE consensus approved (shadow): beta={:.4}", beta);
                    } else {
                        tracing::info!("RCE consensus rejected (shadow): beta={:.4}", beta);
                    }
                }
                rce_retry_approved = approved;

                // Hyperfocus + Circuit Breaker (config-gated)
                if self.config.rce_actions_enabled && !self.config.rce_shadow_mode {
                    if approved && beta >= self.config.rce_breakthrough_threshold {
                        let streak = self.rce_spike_streak.fetch_add(1, AtomicOrdering::SeqCst) + 1;
                        if streak >= self.config.rce_circuit_breaker_streak {
                            // Circuit breaker: slow mode – avoid further aggressive adjustments
                            tracing::warn!("RCE circuit breaker: sustained β_meta spikes ({}). Entering slow mode.", streak);
                        } else {
                            // Apply focused resource allocation by tightening exploration
                            // Use existing increments from config to avoid magic numbers
                            let temp_delta = -self.config.cot_temp_increment;
                            let top_p_delta = -self.config.phase2_top_p_increment;
                            crate::pipeline::core::Pipeline::adjust_runtime_param(
                                &mut self.config,
                                "temperature",
                                temp_delta,
                            );
                            crate::pipeline::core::Pipeline::adjust_runtime_param(
                                &mut self.config,
                                "top_p",
                                top_p_delta,
                            );
                        }
                    } else {
                        // Reset streak when below threshold or not approved
                        self.rce_spike_streak.store(0, AtomicOrdering::SeqCst);
                    }
                }

                // Feed RCE as a signal to hyperfocus detector (normalized to threshold)
                let rce_score = if self.config.rce_breakthrough_threshold > 0.0 {
                    (beta / self.config.rce_breakthrough_threshold).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                hyperfocus_signals.insert(
                    "rce".to_string(),
                    crate::consonance::ConsonanceMetrics {
                        score: rce_score,
                        sources: vec![crate::consonance::ConsonanceSource::TopologicalConsistency(
                            rce_score,
                        )],
                        confidence: 0.9,
                        dissonance_score: 1.0 - rce_score,
                    },
                );

                // Topology-driven curriculum scheduling
                if self.config.rce_actions_enabled && !self.config.rce_shadow_mode {
                    let mut guard = self.learning.lock().await;
                    guard.rce_schedule(
                        beta,
                        self.config.rce_breakthrough_threshold,
                        topology.persistence_entropy,
                    );
                }
            }
        }

        // Compute full consonance with curator now available
        let full_consonance = if let Some(ref curator) = self.curator {
            // Create a CuratedResponse-like structure for consonance computation
            use crate::curator::CuratedResponse;
            let curator_response = CuratedResponse {
                refined_response: curated_experience.refined_response.clone(),
                learned: curated_experience.learned,
                reason: curated_experience.reason.clone(),
                processing_time_ms: 0.0,
                consonance_score: curated_experience.quality_score as f64,
            };
            compute_consonance(
                &pad_state,
                &compass_with_cascade,
                &collapse,
                &topology,
                Some(&curator_response),
                last_compass.as_ref(),
            )
        } else {
            partial_consonance
        };

        // Update hyperfocus signals with full consonance
        hyperfocus_signals.insert("curator".to_string(), full_consonance.clone());
        let hyperfocus_event = self.hyperfocus_detector.detect(&hyperfocus_signals);

        // Failure evaluation after curator
        let entropy_delta = pad_state.entropy - (self.thresholds.entropy_mean);
        let curator_quality = curated_experience.quality_score as f64;

        // Extract actual UCB1 score from MCTS branches
        let ucb1_score = compass
            .mcts_branches
            .iter()
            .map(|branch| branch.ucb_score)
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(self.thresholds.mcts_c); // Fallback to configured threshold

        let fallback_source = {
            let source = generation.source.to_lowercase();
            source.contains("fallback") || source.contains("mock")
        };
        let failure_signals = FailureSignals::evaluate_with_thresholds(
            generation.rouge_score,
            entropy_delta,
            Some(ucb1_score),
            collapse.average_similarity,
            Some(curator_quality),
            fallback_source,
            tokenizer_output.oov_rate,
            0,
            &self.config.failure_signal_thresholds,
        );
        let mut failure = failure_signals.severity().to_string();
        let mut details = failure_signals.summary();

        let reason_lower = curated_experience.reason.to_lowercase();
        let curator_unavailable = self.curator.is_none()
            || reason_lower.contains("curator_disabled")
            || reason_lower.contains("ollama")
            || reason_lower.contains("curator_error")
            || reason_lower.contains("curator mock mode")
            || reason_lower.contains("mock mode")
            || reason_lower.contains("request_failed");
        let curator_passive = !curated_experience.learned;

        if (curator_unavailable || curator_passive) && failure != "none" {
            info!(reason = %curated_experience.reason, "Curator unavailable or passive; skipping retry escalation");
            failure = "none".to_string();
            details = if curator_unavailable {
                "curator_unavailable".to_string()
            } else {
                "curator_passive".to_string()
            };
        }
        info!("After curator check, failure={}", failure);

        let quality_acceptable = (curated_experience.quality_score as f64)
            >= self.config.curator_minimum_threshold as f64;
        let rouge_acceptable = generation.rouge_score >= self.config.rouge_acceptable_threshold;
        if failure == "soft" && (quality_acceptable || rouge_acceptable) {
            info!(
                rouge = generation.rouge_score,
                quality = curated_experience.quality_score,
                "Soft failure bypassed due to acceptable metrics"
            );
            failure = "none".to_string();
            details = "quality_acceptable".to_string();
        }

        // Phase 2: Handle retries with Reflection and CoT with topology awareness
        let (final_generation, final_failure, threat_cycle_ms) = self
            .handle_retry_with_reflection(
                prompt,
                &failure,
                &details,
                &generation,
                &compass_with_cascade,
                &collapse,
                &curated_experience,
                entropy_delta,
                curator_quality,
                ucb1_score,
                tokenizer_output.oov_rate,
                &topology, // TOPOLOGY INTEGRATION: Pass topology to retry logic
                rce_retry_approved,
            )
            .await?;

        // Update timings with threat cycle timing
        timings.threat_cycle_ms = threat_cycle_ms;

        // Log to ERAG if failure != "none"
        if final_failure != "none" {
            self.erag
                .store_failure(
                    prompt,
                    &final_generation.hybrid_response,
                    Some(details.clone()),
                    &final_failure,
                    self.retry_count.load(AtomicOrdering::Relaxed),
                )
                .await?;
        }

        // FIX: Abort pipeline if TopoCoT evaluation score is zero to prevent wasted learning cycles
        if let Some(ref telemetry) = topocot_telemetry {
            if telemetry.score_overall == 0.0 {
                warn!(
                    score_overall = telemetry.score_overall,
                    issues = ?telemetry.issues,
                    "TopoCoT evaluation score is zero - aborting pipeline to prevent wasted learning cycles"
                );
                return Err(anyhow::anyhow!(
                    "TopoCoT evaluation failed with zero score: {:?}",
                    telemetry.issues
                ));
            }
        }

        // Proceed with learning using curated response (with retry-corrected generation)
        let learning_start = Instant::now();
        info!("About to lock learning mutex");

        // FIX: Instrument mutex wait duration
        let mutex_wait_start = Instant::now();
        let mut learning_lock = self.learning.lock().await;
        let mutex_wait_duration_ms = mutex_wait_start.elapsed().as_secs_f64() * 1000.0;
        info!(
            mutex_wait_ms = mutex_wait_duration_ms,
            "Learning mutex acquired"
        );

        // Wrap learning update in timeout to prevent hanging
        let learning_timeout_secs = self.config.learning_timeout_secs;
        let learning_result =
            tokio::time::timeout(Duration::from_secs(learning_timeout_secs), async {
                learning_lock
                    .update(
                        &pad_state,
                        &compass_with_cascade,
                        &collapse,
                        &final_generation,
                        &topology,
                    )
                    .await
            })
            .await;

        let learning_outcome = match learning_result {
            Ok(Ok(outcome)) => {
                info!("Learning update completed successfully");
                outcome
            }
            Ok(Err(e)) => {
                warn!("Learning update failed: {}", e);
                return Err(anyhow::anyhow!("Learning update failed: {}", e));
            }
            Err(_) => {
                warn!(
                    "Learning update timed out after {}s - using default outcome",
                    learning_timeout_secs
                );
                // Create a default learning outcome
                LearningOutcome {
                    events: vec!["learning_timeout".to_string()],
                    breakthroughs: vec![],
                    qlora_updates: vec![],
                    entropy_delta: 0.0,
                    adjusted_params: std::collections::HashMap::new(),
                    last_replay: None,
                    lora_update: None,
                }
            }
        };

        let pending_gradients = if let Some(stats) = learning_outcome.lora_update.as_ref() {
            let grad_ratio = if stats.grad_b_norm.abs() > f32::EPSILON {
                stats.grad_a_norm / stats.grad_b_norm
            } else {
                0.0
            };
            let timestamp_feature = stats.timestamp.timestamp() as f32
                + (stats.timestamp.timestamp_subsec_millis() as f32 / 1_000.0);
            vec![
                stats.grad_a_norm,
                stats.grad_b_norm,
                stats.weight_delta_norm,
                stats.loss,
                stats.batch_size as f32,
                grad_ratio,
                timestamp_feature,
            ]
        } else {
            Vec::new()
        };

        // Drop the lock before calling set_pending_lora_gradients
        drop(learning_lock);

        self.set_pending_lora_gradients(pending_gradients);

        timings.learning_ms = learning_start.elapsed().as_secs_f64() * 1000.0;

        // Action: Learning update
        let mut learn_params = std::collections::HashMap::new();
        learn_params.insert(
            "entropy_delta".to_string(),
            learning_outcome.entropy_delta.to_string(),
        );
        learn_params.insert(
            "events_count".to_string(),
            learning_outcome.events.len().to_string(),
        );
        learn_params.insert(
            "breakthroughs_count".to_string(),
            learning_outcome.breakthroughs.len().to_string(),
        );
        thought_builder.add_action(
            "learning".to_string(),
            format!(
                "Learning update: {} events, {} breakthroughs",
                learning_outcome.events.len(),
                learning_outcome.breakthroughs.len()
            ),
            learn_params,
            if learning_outcome.breakthroughs.is_empty() {
                0.7
            } else {
                0.9
            },
            None,
        );

        // Remove double-storage: defer storage decision to final gate below

        // Stage 7.5: Curator Quality Gate (single source of truth)
        let response_to_store = curated_experience.refined_response.clone();
        info!(
            "Checking quality gate: score={}, threshold={}",
            curated_experience.quality_score, self.config.curator_minimum_threshold
        );
        let mut topology_reflection_summary: Option<String> = None;
        if curated_experience.quality_score < self.config.curator_minimum_threshold {
            warn!(
                quality = curated_experience.quality_score,
                min = self.config.curator_minimum_threshold,
                "Curated quality below minimum; skipping memory store"
            );
            return Ok(PipelineCycle {
                prompt: prompt.to_string(),
                baseline_response: final_generation.baseline_response.clone(),
                hybrid_response: final_generation.hybrid_response.clone(),
                entropy: pad_state.entropy,
                rouge: final_generation.rouge_to_baseline,
                latency_ms: overall_start.elapsed().as_secs_f64() * 1000.0,
                compass: compass_with_cascade.clone(),
                generation: final_generation,
                tokenizer: tokenizer_output,
                collapse,
                learning: learning_outcome,
                stage_timings: timings,
                last_entropy: pad_state.entropy,
                failure: final_failure,
                pad_state: pad_state.clone(),
                topocot: topocot_telemetry.clone(),
                topology_reflection_summary: topology_reflection_summary.clone(),
                topology: topology.clone(),
                topology_mode: self.config.topology_mode,
                consonance: Some(full_consonance),
                hyperfocus: hyperfocus_event,
                cascade_transition,
            });
        }

        // Create enriched experience record now that curator approved storage
        let mut aggregated_context_lines: Vec<String> = collapse
            .aggregated_context
            .lines()
            .map(|s| s.to_string())
            .collect();
        if !curated_experience.learned {
            let summary =
                TopoReflectionStage::summarize_failure(&topology, &baseline_sig, &reflection);
            aggregated_context_lines.push(format!("[topology_reflection] {summary}"));
            topology_reflection_summary = Some(summary);
        }
        let experience_input = prompt.to_string();
        let experience = Experience::from_pipeline(
            experience_input.clone(),
            response_to_store.clone(),
            embedding.clone(),
            &pad_state,
            &compass_with_cascade,
            aggregated_context_lines.clone(),
        )
        .with_success_score(curated_experience.quality_score)
        .with_task_type("hybrid_generation");
        curated_experience.experience = Some(experience);
        info!("Experience enriched for curator/learning integration");

        // Feed curator output and enriched experience to learning loop if learned=true
        if curated_experience.learned {
            // Phase 4.2: Record curator feedback before applying to learning loop
            if let Some(ref feedback_controller) = self.curator_feedback {
                let mut controller = feedback_controller.lock().await;
                controller
                    .record_feedback(curated_experience.quality_score, curated_experience.learned);
            }

            let reward = generation.rouge_score * self.config.reward_rouge_weight
                + (1.0 - pad_state.entropy) * self.config.reward_entropy_weight;
            if let Err(e) = self
                .learning
                .lock()
                .await
                .apply_curator_learned(
                    &curated_experience.refined_response,
                    true,
                    reward,
                    &topology,
                    prompt,
                    &curated_experience.promoted_tokens,
                    curated_experience.experience.as_ref(),
                )
                .await
            {
                warn!("Failed to apply curator learned data: {}", e);
            }
        } else {
            // Phase 4.2: Record feedback even if not learned
            if let Some(ref feedback_controller) = self.curator_feedback {
                let mut controller = feedback_controller.lock().await;
                controller
                    .record_feedback(curated_experience.quality_score, curated_experience.learned);
            }
            if let Some(summary) = topology_reflection_summary.as_ref() {
                let mut learning_guard = self.learning.lock().await;
                learning_guard.enqueue_topology_reflection(summary, &topology, &reflection);
            }
        }

        // Wrap upsert in timeout to prevent hanging
        info!("About to upsert memory with timeout");
        match tokio::time::timeout(
            Duration::from_secs(self.config.memory_upsert_timeout_secs),
            self.erag.upsert_memory_with_cascade(
                &embedding, // Use embedding directly, not experience_embedding which was moved
                &pad_state,
                &compass_with_cascade,
                &experience_input,
                &response_to_store,
                &aggregated_context_lines,
                pad_state.entropy,
                compass_with_cascade.cascade_stage,
            ),
        )
        .await
        {
            Ok(Ok(_)) => info!("Memory upserted successfully"),
            Ok(Err(e)) => warn!("Failed to upsert memory: {}", e),
            Err(_) => warn!(
                "Upsert memory timed out after {}s - continuing",
                self.config.memory_upsert_timeout_secs
            ),
        }
        info!("After upsert");

        metrics().observe_cycle(
            pad_state.entropy,
            final_generation.latency_ms,
            final_generation.rouge_to_baseline,
            compass_with_cascade.is_threat,
            compass_with_cascade.is_healing,
        );

        // Emit per-cycle WebSocket event (best-effort)
        if let Some(ws_url) = env_value("NIODOO_WS_ENDPOINT") {
            let _ = tokio::spawn({
                let ws_url = ws_url.clone();
                let event = serde_json::json!({
                    "event": "cycle",
                    "entropy": pad_state.entropy,
                    "knot": topology.knot_complexity,
                    "betti": topology.betti_numbers,
                    "ucb1": compass.ucb1_score,
                    "retries": self.retry_count.load(AtomicOrdering::Relaxed),
                    "latency_ms": final_generation.latency_ms,
                });
                async move {
                    let _ = reqwest::Client::new()
                        .post(format!("{}/events", ws_url.trim_end_matches('/')))
                        .json(&event)
                        .send()
                        .await;
                }
            });
        }

        // learning_ms already set above

        // Broadcast telemetry if enabled
        if let Some(ref tx) = self.telemetry_tx {
            let iteration = self.iteration_count.fetch_add(1, AtomicOrdering::Relaxed) + 1;

            // Extract pad_state first 3 dimensions
            let pad_state_3d = [
                pad_state.pad[0] as f32,
                pad_state.pad[1] as f32,
                pad_state.pad[2] as f32,
            ];

            // Compute torus projection using parametric equations
            // Use pad_state values as angles (u, v) for torus mapping
            // Default torus parameters: major_radius=5.0, strip_width=1.0, twists=1
            let major_radius = 5.0f32;
            let strip_width = 1.0f32;
            let twists = 1i32;
            let k = twists as f32;

            // Map pad_state[0] and pad_state[1] to u and v parameters
            // Normalize pad values (which are in [-1, 1] after tanh) to [0, 2π] for u and [-0.5, 0.5] for v
            let u = (pad_state.pad[0] as f32 + 1.0) * std::f32::consts::PI; // Map [-1,1] -> [0, 2π]
            let v_norm = pad_state.pad[1] as f32; // Already in [-1, 1]
            let v = v_norm * strip_width * 0.5; // Scale to [-0.5, 0.5]

            // Apply parametric equations: x(u,v) = (R + v*cos(2ku)) * cos(u)
            let twist_factor = 2.0 * k * u;
            let radius_at_u = major_radius + v * twist_factor.cos();
            let torus_projection = [
                radius_at_u * u.cos(),
                radius_at_u * u.sin(),
                v * twist_factor.sin(),
            ];

            // Extract Betti numbers
            let betti_numbers = (
                topology.betti_numbers[0],
                topology.betti_numbers[1],
                topology.betti_numbers[2],
            );

            // Get persistence entropy
            let persistence_entropy = topology.persistence_entropy;

            // Get compass quadrant and confidence
            let compass_quadrant = format!("{:?}", compass_with_cascade.quadrant);
            let compass_confidence = compass_with_cascade.intrinsic_reward.max(0.0).min(1.0) as f32;

            // Build enhanced telemetry packet with full data
            let timestamp = chrono::Utc::now().to_rfc3339();

            // Build prompt metadata
            let prompt_id = uuid::Uuid::new_v4();
            let prompt_metadata = crate::telemetry::PromptMetadata {
                full_text: prompt.to_string(),
                token_count: tokenizer_output.tokens.len(),
                token_ids: Some(tokenizer_output.tokens.clone()),
                prompt_type: crate::telemetry::PromptType::User, // Could be enhanced to detect type
                prompt_id,
            };

            // Build response metadata
            let response_id = uuid::Uuid::new_v4();
            let response_metadata = crate::telemetry::ResponseMetadata {
                full_text: final_generation.hybrid_response.clone(),
                baseline_text: final_generation.baseline_response.clone(),
                token_count: final_generation.hybrid_response.split_whitespace().count(), // Approximate
                generation_tokens: None, // Token-by-token details not available yet
                finish_reason: if final_generation.failure_type.is_some() {
                    crate::telemetry::FinishReason::Error(
                        final_generation
                            .failure_type
                            .clone()
                            .unwrap_or_else(|| "unknown".to_string()),
                    )
                } else {
                    crate::telemetry::FinishReason::Stop
                },
                response_id,
            };

            // Extract full memory content (not just IDs)
            let memory_entries: Vec<crate::telemetry::MemoryEntry> = collapse
                .top_hits
                .iter()
                .enumerate()
                .map(|(idx, mem)| {
                    let memory_id = blake3::hash(format!("{}{}", mem.input, mem.output).as_bytes())
                        .to_hex()
                        .chars()
                        .take(16)
                        .collect();
                    crate::telemetry::MemoryEntry {
                        memory_id,
                        input: mem.input.clone(),
                        output: mem.output.clone(),
                        retrieval_score: collapse.average_similarity, // Use average for now
                        impact: 1.0 / (idx + 1) as f32,               // Simple impact estimation
                        emotional_vector: Some(vec![
                            mem.emotional_vector.joy,
                            mem.emotional_vector.sadness,
                            mem.emotional_vector.anger,
                            mem.emotional_vector.fear,
                            mem.emotional_vector.surprise,
                        ]),
                        timestamp: Some(mem.timestamp.clone()),
                    }
                })
                .collect();

            let memory_retrieval = crate::telemetry::MemoryRetrieval {
                retrieved_memories: memory_entries,
                retrieval_strategy: "erag_collapse".to_string(),
                average_similarity: collapse.average_similarity,
            };

            // Build pipeline stage execution details
            let pipeline_stages = vec![
                crate::telemetry::StageExecution {
                    stage_name: "embedding".to_string(),
                    timing_ms: timings.embedding_ms,
                    errors: Vec::new(),
                    metrics: std::collections::HashMap::new(),
                },
                crate::telemetry::StageExecution {
                    stage_name: "torus_projection".to_string(),
                    timing_ms: timings.torus_ms,
                    errors: Vec::new(),
                    metrics: std::collections::HashMap::new(),
                },
                crate::telemetry::StageExecution {
                    stage_name: "topology_analysis".to_string(),
                    timing_ms: timings.tcs_ms,
                    errors: Vec::new(),
                    metrics: std::collections::HashMap::new(),
                },
                crate::telemetry::StageExecution {
                    stage_name: "compass".to_string(),
                    timing_ms: timings.compass_ms,
                    errors: Vec::new(),
                    metrics: std::collections::HashMap::new(),
                },
                crate::telemetry::StageExecution {
                    stage_name: "erag".to_string(),
                    timing_ms: timings.erag_ms,
                    errors: Vec::new(),
                    metrics: std::collections::HashMap::new(),
                },
                crate::telemetry::StageExecution {
                    stage_name: "tokenizer".to_string(),
                    timing_ms: timings.tokenizer_ms,
                    errors: Vec::new(),
                    metrics: std::collections::HashMap::new(),
                },
                crate::telemetry::StageExecution {
                    stage_name: "generation".to_string(),
                    timing_ms: timings.generation_ms,
                    errors: if final_generation.failure_type.is_some() {
                        vec![crate::telemetry::StageError {
                            error_type: final_generation
                                .failure_type
                                .clone()
                                .unwrap_or_else(|| "unknown".to_string()),
                            message: final_generation
                                .failure_details
                                .clone()
                                .unwrap_or_else(|| "".to_string()),
                            timestamp: timestamp.clone(),
                        }]
                    } else {
                        Vec::new()
                    },
                    metrics: std::collections::HashMap::new(),
                },
                crate::telemetry::StageExecution {
                    stage_name: "learning".to_string(),
                    timing_ms: timings.learning_ms,
                    errors: Vec::new(),
                    metrics: std::collections::HashMap::new(),
                },
            ];

            // Build performance metrics
            let total_latency = overall_start.elapsed().as_secs_f64() * 1000.0;
            let tokens_per_second = if total_latency > 0.0 {
                Some((response_metadata.token_count as f64 / total_latency) * 1000.0)
            } else {
                None
            };

            let performance = crate::telemetry::PerformanceMetrics {
                latency_ms: total_latency,
                tokens_per_second,
                gpu_utilization: None, // Not available yet
                memory_usage: None,    // Not available yet
                cache_hit_rate: None,  // Could be calculated from cache hits
            };

            // Build thought structure tree
            let thought_tree = thought_builder.build();

            // Build enhanced packet
            let enhanced_packet = crate::telemetry::EnhancedCognitiveStatePacket {
                pad_state: pad_state_3d,
                torus_projection,
                betti_numbers,
                persistence_entropy,
                compass_quadrant,
                compass_confidence,
                iteration,
                timestamp: timestamp.clone(),
                prompt: prompt_metadata,
                response: response_metadata,
                thought_structure: Some(thought_tree),
                memory_retrieval,
                pipeline_stages,
                performance,
                test_run: None, // Will be set if in test run context
            };

            // Send both enhanced and legacy packets for backward compatibility
            let legacy_packet = enhanced_packet.to_legacy();
            let _ = tx.send(legacy_packet); // Non-blocking, ignore errors

            // Log enhanced packet for playback
            if let Some(ref logger) = self.file_logger {
                let logger_clone = logger.clone();
                let packet_clone = enhanced_packet.clone();
                tokio::spawn(async move {
                    if let Err(e) = logger_clone.log(&packet_clone).await {
                        warn!(error = %e, "Failed to log enhanced telemetry packet");
                    }
                });
            }
        }

        info!("About to return PipelineCycle");
        Ok(PipelineCycle {
            prompt: prompt.to_string(),
            baseline_response: final_generation.baseline_response.clone(),
            hybrid_response: final_generation.hybrid_response.clone(),
            entropy: pad_state.entropy,
            rouge: final_generation.rouge_to_baseline,
            latency_ms: overall_start.elapsed().as_secs_f64() * 1000.0,
            compass: compass_with_cascade,
            generation: final_generation,
            tokenizer: tokenizer_output,
            collapse,
            learning: learning_outcome,
            stage_timings: timings,
            last_entropy: pad_state.entropy,
            failure: final_failure,
            pad_state,
            topocot: topocot_telemetry,
            topology_reflection_summary,
            topology,
            topology_mode: self.config.topology_mode,
            consonance: Some(full_consonance),
            hyperfocus: hyperfocus_event,
            cascade_transition,
        })
    }

    async fn handle_retry_with_reflection(
        &self,
        prompt: &str,
        initial_failure: &str,
        details: &str,
        generation: &GenerationResult,
        compass: &CompassOutcome,
        collapse: &CollapseResult,
        curated: &CuratedExperience,
        entropy_delta: f64,
        curator_quality: f64,
        ucb1_score: f64,
        oov_rate: f64,
        topology: &crate::tcs_analysis::TopologicalSignature,
        rce_retry_approved: bool,
    ) -> Result<(GenerationResult, String, f64)> {
        let loop_start = Instant::now();

        // RCE consensus gating: skip retries unless approved
        if !rce_retry_approved {
            tracing::info!("RCE consensus gating: retries skipped");
            return Ok((
                generation.clone(),
                initial_failure.to_string(),
                loop_start.elapsed().as_secs_f64() * 1000.0,
            ));
        }
        // INTEGRATION FIX: Handle healing state specially - enhance instead of retry
        if initial_failure == "none" && compass.is_healing {
            // In healing state with good topology - apply enhancement strategies
            if topology.knot_complexity < self.config.pipeline_healing_knot_threshold
                && topology.spectral_gap > self.config.pipeline_healing_spectral_gap_threshold
            {
                info!("Healing state detected with good topology - applying enhancement");

                // Generate an enhanced version leveraging the good state
                let enhancement_prompt = format!(
                    "{}\n\n[System is in optimal healing state. Enhance clarity and depth.]",
                    prompt
                );

                if let Ok(enhanced_str) = self
                    .generator
                    .generate_with_params(
                        &enhancement_prompt,
                        self.config.enhancement_temperature,
                        self.config.enhancement_top_p,
                    ) // Low temp for stability
                    .await
                {
                    // Wrap String in GenerationResult
                    let enhanced = GenerationResult {
                        baseline_response: generation.baseline_response.clone(),
                        hybrid_response: enhanced_str,
                        echoes: Vec::new(),
                        rouge_to_baseline: generation.rouge_to_baseline,
                        latency_ms: generation.latency_ms,
                        rouge_score: generation.rouge_score,
                        entropy_delta: generation.entropy_delta,
                        source: "enhanced".to_string(),
                        ucb1_score: generation.ucb1_score,
                        curator_quality: generation.curator_quality,
                        failure_type: None,
                        failure_details: None,
                    };
                    return Ok((enhanced, "none".to_string(), 0.0));
                }
            }
            return Ok((generation.clone(), "none".to_string(), 0.0));
        }

        // No failure and not healing, return original
        if initial_failure == "none" {
            return Ok((generation.clone(), "none".to_string(), 0.0));
        }

        let cfg_snapshot = self.config_arc.read().clone();
        let max_retries = cfg_snapshot.phase2_max_retries;
        let base_delay_ms = cfg_snapshot.phase2_retry_base_delay_ms;
        let cot_iterations = cfg_snapshot.phase2_cot_iterations.max(1) as usize;
        let cot_success_rouge = cfg_snapshot.cot_success_rouge_threshold;
        let level3_retry_count = cfg_snapshot.phase2_level3_retry_count;
        let mcts_c_increment = cfg_snapshot.phase2_mcts_c_increment;
        let top_p_increment = cfg_snapshot.phase2_top_p_increment;
        let retrieval_top_k_increment = cfg_snapshot.phase2_retrieval_top_k_increment;
        let backoff_cap_ms = cfg_snapshot.phase2_retry_backoff_cap_ms.max(base_delay_ms);
        let backoff_exponent_cap = cfg_snapshot.retry_backoff_exponent_cap;

        let mut current_gen = generation.clone();
        let mut current_failure = initial_failure.to_string();
        let mut retry_count = 0;

        let loop_start = Instant::now();

        // Retry loop with escalating strategies
        while retry_count < max_retries && current_failure != "none" {
            retry_count += 1;
            info!(retry = retry_count, tier = ?current_failure, detail = ?details, "retry loop attempt");

            // Store failure in ERAG before retry
            if let Err(e) = self
                .erag
                .store_failure(
                    prompt,
                    &current_gen.hybrid_response,
                    Some(format!("Retry {}: {}", retry_count, details)),
                    &current_failure,
                    retry_count,
                )
                .await
            {
                warn!("Failed to store failure: {}", e);
            }

            // Level3+ escalation: Tune MCTS/retrieval params for repeated failures
            let is_level3 = retry_count > level3_retry_count;
            if is_level3 {
                info!(
                    "Level3 escalation: Applying parameter tuning (attempt {})",
                    retry_count
                );
                // Log escalation metrics (actual tuning would require mutable access to compass/thresholds)
                info!(
                    "Suggested tuning: MCTS c += {:.3}, top_p += {:.3}, retrieval_top_k += {}",
                    mcts_c_increment, top_p_increment, retrieval_top_k_increment
                );
            }

            // Determine retry strategy based on failure type
            let retry_response = if current_failure == "hard" {
                // Meso: Reflexion for hard failures, but fallback to baseline if worse
                let reflexion_response = self
                    .generator
                    .reflexion_retry(prompt, current_gen.rouge_score, details)
                    .await?;

                // Compare with baseline and keep the better one
                // Phase 4.1: Parallel ROUGE scoring
                let parallel_rouge = {
                    let config = self.config_arc.read();
                    config.parallel_curator_rouge
                };
                let (baseline_rouge, reflexion_rouge) = if parallel_rouge {
                    let (baseline_result, reflexion_result) = tokio::join!(
                        tokio::task::spawn_blocking({
                            let baseline = current_gen.baseline_response.clone();
                            let prompt = prompt.to_string();
                            move || rouge_l(&baseline, &prompt)
                        }),
                        tokio::task::spawn_blocking({
                            let reflexion = reflexion_response.clone();
                            let prompt = prompt.to_string();
                            move || rouge_l(&reflexion, &prompt)
                        })
                    );
                    (
                        baseline_result.unwrap_or(0.0),
                        reflexion_result.unwrap_or(0.0),
                    )
                } else {
                    (
                        rouge_l(&current_gen.baseline_response, prompt),
                        rouge_l(&reflexion_response, prompt),
                    )
                };

                if reflexion_rouge > baseline_rouge {
                    info!(
                        "Reflexion improved from {:.3} to {:.3}",
                        baseline_rouge, reflexion_rouge
                    );
                    reflexion_response
                } else {
                    info!(
                        "Baseline better than reflexion ({:.3} vs {:.3}), using baseline",
                        baseline_rouge, reflexion_rouge
                    );
                    current_gen.baseline_response.clone()
                }
            } else {
                // Micro: Topology-aware CoT for soft failures (2-3 iterations)
                let mut best_response = current_gen.hybrid_response.clone();
                let mut best_rouge = current_gen.rouge_score;

                for cot_iter in 0..cot_iterations {
                    let cot_result = self
                        .generator
                        .apply_cot_repair_with_topology(
                            prompt,
                            details,
                            cot_iter as u32,
                            Some(topology),
                        )
                        .await?;

                    // Recompute ROUGE
                    let new_rouge = rouge_l(&cot_result.hybrid_response, &best_response);
                    if new_rouge > best_rouge {
                        best_response = cot_result.hybrid_response;
                        best_rouge = new_rouge;
                    }

                    if best_rouge >= cot_success_rouge {
                        info!(
                            "Topology-aware CoT iteration {} achieved target ROUGE {:.3}",
                            cot_iter + 1,
                            best_rouge
                        );
                        break;
                    }
                }
                best_response
            };

            // Create updated generation result with retry
            // Phase 4.1: Parallel ROUGE scoring for rouge_to_baseline and rouge_score
            let parallel_rouge = {
                let config = self.config_arc.read();
                config.parallel_curator_rouge
            };
            let (rouge_to_baseline, rouge_score_val) = if parallel_rouge {
                let (to_baseline_result, score_result) = tokio::join!(
                    tokio::task::spawn_blocking({
                        let retry = retry_response.clone();
                        let baseline = current_gen.baseline_response.clone();
                        move || rouge_l(&retry, &baseline)
                    }),
                    tokio::task::spawn_blocking({
                        let retry = retry_response.clone();
                        let baseline = current_gen.baseline_response.clone();
                        move || rouge_l(&retry, &baseline)
                    })
                );
                (
                    to_baseline_result.unwrap_or(0.0),
                    score_result.unwrap_or(0.0),
                )
            } else {
                let score = rouge_l(&retry_response, &current_gen.baseline_response);
                (score, score)
            };

            let retry_gen = GenerationResult {
                baseline_response: retry_response.clone(),
                hybrid_response: retry_response.clone(),
                echoes: current_gen.echoes.clone(),
                rouge_to_baseline,
                latency_ms: current_gen.latency_ms,
                rouge_score: rouge_score_val,
                entropy_delta: current_gen.entropy_delta,
                source: format!("retry_{}", retry_count),
                ucb1_score: current_gen.ucb1_score,
                curator_quality: current_gen.curator_quality,
                failure_type: None,
                failure_details: None,
            };

            // Re-evaluate failure with new metrics
            // OPTIMIZATION: Adjust ucb1_score based on ROUGE improvement to avoid infinite retry loops
            // If ROUGE improved significantly, boost ucb1 to reflect successful retry
            let adjusted_ucb1 = if retry_gen.rouge_score
                > current_gen.rouge_score + self.config.rouge_improvement_threshold
            {
                // ROUGE improved by configured threshold, boost ucb1 to reflect success
                ucb1_score
                    .max(self.config.ucb1_boost_threshold)
                    .min(self.config.pipeline_ucb1_max_clamp)
            } else if retry_count > self.config.retry_count_for_relaxation {
                // After configured retry count, if we're still here but ROUGE is reasonable, relax ucb1 check
                // This prevents infinite loops from stale ucb1_score
                ucb1_score.max(self.config.ucb1_relaxation_threshold)
            } else {
                ucb1_score
            };

            let retry_curator_quality = retry_gen.curator_quality.or(Some(curator_quality));
            let retry_fallback = {
                let source = retry_gen.source.to_lowercase();
                source.contains("fallback") || source.contains("mock")
            };
            let low_quality_hits = curated.promoted_tokens.len();
            let retry_failure_signals = FailureSignals::evaluate_with_thresholds(
                retry_gen.rouge_score,
                entropy_delta,
                Some(adjusted_ucb1),
                collapse.average_similarity,
                retry_curator_quality,
                retry_fallback,
                oov_rate,
                low_quality_hits,
                &self.config.failure_signal_thresholds,
            );
            let failure = retry_failure_signals.severity().to_string();
            let _new_details = retry_failure_signals.summary();

            current_gen = retry_gen;
            current_failure = failure.clone();

            // Success on retry
            if current_failure == "none" {
                info!(
                    "Retry succeeded on attempt {} (ROUGE: {:.3})",
                    retry_count, current_gen.rouge_score
                );
                self.retry_count.store(retry_count, AtomicOrdering::Relaxed);
                break;
            }

            // Backoff delay before next retry (exponential with jitter, but capped)
            // OPTIMIZATION: Cap exponential backoff to prevent excessive delays
            if retry_count < max_retries {
                let exponent = ((retry_count.saturating_sub(1)) as u32).min(backoff_exponent_cap);
                let multiplier = 1_u64 << exponent;
                let delay_ms = (base_delay_ms * multiplier).min(backoff_cap_ms);
                if delay_ms > self.config.delay_threshold_ms {
                    info!(
                        retry = retry_count,
                        delay_ms, "Backoff delay before next retry"
                    );
                }
                tokio::time::sleep(Duration::from_millis(delay_ms)).await;
            }
        }

        if current_failure != "none" {
            warn!(
                "Failed after {} retry attempts, using degraded response",
                retry_count
            );

            // Graceful degradation: Instead of terminating, mark as degraded but continue
            if retry_count >= max_retries {
                warn!("Circuit breaker triggered: Using degraded response mode");
                // Add degraded marker to generation result
                current_gen.failure_type = Some("degraded".to_string());
                current_gen.failure_details = Some(format!(
                    "Max retries exceeded ({}), using best available response",
                    retry_count
                ));
            }
        }

        let threat_cycle_ms = loop_start.elapsed().as_secs_f64() * 1000.0;

        Ok((current_gen, current_failure, threat_cycle_ms))
    }

    async fn integrate_curator(
        &self,
        input: &str,
        output: &str,
        pad_state: &PadGhostState,
        compass: &CompassOutcome,
        context: &str,
        topology: &crate::tcs_analysis::TopologicalSignature,
        tokenizer_output: &TokenizerOutput,
    ) -> Result<CuratedExperience> {
        // Call curator_executor logic here
        // (either spawn as subprocess or integrate as library)

        // TOPOLOGY INTEGRATION: Analyze quality with topological insights
        // Calculate base quality score based on output length, entropy, and topology
        let base = self.config.quality_base_score;
        let length_factor = (output.len().min(self.config.quality_max_length) as f32
            / self.config.quality_max_length as f32)
            * self.config.quality_length_factor_weight;
        let entropy_factor = if pad_state.entropy < self.config.quality_entropy_threshold {
            self.config.quality_entropy_factor_weight
        } else {
            0.0f32
        };
        let base_quality = base + length_factor + entropy_factor;

        // TOPOLOGY ENHANCEMENT: Adjust quality based on topological features
        let mut adjusted_quality = base_quality;

        // High knot complexity indicates tangled/complex reasoning - slight quality penalty
        if topology.knot_complexity > self.config.knot_complexity_penalty_threshold {
            adjusted_quality *= self.config.knot_complexity_penalty_multiplier;
            info!(
                "High knot complexity {:.3} - reducing quality",
                topology.knot_complexity
            );
        }

        // High spectral gap indicates good exploration - quality bonus
        if topology.spectral_gap > self.config.spectral_gap_bonus_threshold {
            adjusted_quality *= self.config.spectral_gap_bonus_multiplier;
            info!(
                "High spectral gap {:.3} - boosting quality",
                topology.spectral_gap
            );
        }

        // High Betti-1 indicates many loops/cycles - check if intentional
        if topology.betti_numbers[1] > self.config.betti1_quality_threshold {
            // In Discover quadrant, loops are good (exploration)
            if compass.quadrant == CompassQuadrant::Discover {
                adjusted_quality *= self.config.betti1_bonus_multiplier;
            } else {
                // In other quadrants, too many loops might indicate confusion
                adjusted_quality *= self.config.betti1_penalty_multiplier;
            }
            info!(
                "Betti-1={} affecting quality in {:?} quadrant",
                topology.betti_numbers[1], compass.quadrant
            );
        }

        // Persistence entropy indicates structural stability
        if topology.persistence_entropy < self.config.persistence_entropy_quality_threshold {
            // Low entropy = stable structure = good quality
            adjusted_quality *= self.config.persistence_entropy_bonus_multiplier;
        }

        let mut quality_score = adjusted_quality.min(1.0).max(0.0);

        // NEW: Convert to 1-10 gating scale for autonomous curator gating system
        let gating_score = ((quality_score * 9.0) + 1.0).round() as u8;
        info!(
            "🎯 Gating Score: {}/10 (quality: {:.3})",
            gating_score, quality_score
        );

        // NEW: Autonomous Gating Logic - bifurcate data flow based on quality
        match gating_score {
            score if score < 6 => {
                // LEARNING GATE: Process failure
                info!("❌ FAILURE (Score {}) → Learning Gate", score);
                self.process_learning_gate(input, output, score, pad_state, compass)
                    .await?;
            }

            score if score >= 8 => {
                // MEMORY GATE: Process success
                info!("✅ SUCCESS (Score {}) → Memory Gate", score);
                let golden_saved = self
                    .process_memory_gate(
                        input, output, score, pad_state, compass, topology, context,
                    )
                    .await?;

                if !golden_saved {
                    info!("High-quality but not Golden (boring). Using standard ERAG.");
                }
            }

            _ => {
                // INDIFFERENT PATH: Use standard processing
                info!("😐 MEDIOCRE (Score {}) → Standard ERAG", gating_score);
            }
        }

        // TOPOLOGY-AWARE REFINEMENT: Refine if quality is low OR topology indicates issues
        let refinement_threshold = self.config.curator_quality_threshold;
        // Enforce a minimum floor (0.6 quality ≈ 6/10 gating score) so curator always retries low-signal outputs.
        let forced_retry_floor = refinement_threshold.max(0.6);

        // Force refinement if topology shows problematic patterns
        let topology_needs_refinement = topology.knot_complexity > self.config.topology_refinement_knot_threshold  // Too tangled
            || (topology.betti_numbers[1] > self.config.topology_refinement_betti1_threshold && compass.quadrant != CompassQuadrant::Discover)  // Too many loops outside exploration
            || topology.persistence_entropy > self.config.topology_refinement_entropy_threshold; // Too chaotic structure

        let refinement_reason = if quality_score < refinement_threshold && topology_needs_refinement
        {
            "quality_below_threshold+topology_alert"
        } else if quality_score < refinement_threshold {
            "quality_below_threshold"
        } else if topology_needs_refinement {
            "topology_alert"
        } else {
            "stable"
        };

        let quality_below_forced_floor = quality_score < forced_retry_floor;
        if quality_below_forced_floor && !topology_needs_refinement {
            info!(
                quality = quality_score,
                forced_floor = forced_retry_floor,
                "Quality below forced curator floor; triggering refinement retry"
            );
        }

        let mut reason = refinement_reason.to_string();
        if quality_below_forced_floor && refinement_reason == "stable" {
            reason = "forced_low_quality_floor".to_string();
        }

        let mut experience_record: Option<Experience> = None;
        let needs_refinement = quality_score < refinement_threshold
            || topology_needs_refinement
            || quality_below_forced_floor;
        let autonomy_enabled = self.config.curator_autonomous || self.curator.is_none();
        let mut refined = output.to_string();
        let mut learned = false;

        if needs_refinement {
            // First, attempt autonomous refinement if enabled
            if autonomy_enabled {
                let mut auto_improvement = 0.0;
                let autonomy_prompt = format!(
                    "You are NIODOO's autonomous curator. Rewrite the assistant response to be concise, specific, and constitutionally aligned. Remove filler, avoid repeating the prompt, and deliver one high-signal insight in 3-5 sentences.\n\nPrompt:\n{input}\n\nOriginal Response:\n{output}\n\nReturn only the refined response text.",
                    input = input,
                    output = output
                );

                match self
                    .generator
                    .generate_with_params(
                        &autonomy_prompt,
                        self.config.autonomous_refinement_temperature,
                        self.config.autonomous_refinement_top_p,
                    )
                    .await
                {
                    Ok(autonomous_str) => {
                        let candidate = autonomous_str.trim();
                        if !candidate.is_empty() {
                            // Phase 4.1: Parallel ROUGE scoring for auto-improvement
                            let parallel_rouge = {
                                let config = self.config_arc.read();
                                config.parallel_curator_rouge
                            };
                            auto_improvement = if parallel_rouge {
                                tokio::task::spawn_blocking({
                                    let candidate = candidate.to_string();
                                    let output = output.to_string();
                                    move || rouge_l(&candidate, &output)
                                })
                                .await
                                .unwrap_or(0.0)
                            } else {
                                rouge_l(candidate, output)
                            };

                            if auto_improvement.is_finite() {
                                quality_score = (quality_score
                                    + (auto_improvement.clamp(0.0, 1.0)
                                        * self.config.autonomous_refinement_improvement_weight
                                            as f64) as f32)
                                    .min(1.0);
                            }
                            refined = candidate.to_string();
                            learned = auto_improvement
                                > self.config.autonomous_refinement_improvement_threshold;
                            reason = format!(
                                "auto_refine|improvement:{:.3}|mode:{}",
                                auto_improvement,
                                if self.curator.is_some() {
                                    "curator_present"
                                } else {
                                    "curator_absent"
                                }
                            );

                            if auto_improvement < self.config.second_pass_refinement_threshold {
                                let first_improvement = auto_improvement;
                                let second_prompt = format!(
                                    "You are NIODOO's refinement overdrive. Further tighten the assistant reply so it is laser-focused, free of hedging, and emphasizes one actionable takeaway. Maintain constitutional tone and clear structure.\n\nPrompt:\n{input}\n\nCurrent Refinement:\n{refined}\n\nReturn only the upgraded response.",
                                    input = input,
                                    refined = refined
                                );

                                match self
                                    .generator
                                    .generate_with_params(
                                        &second_prompt,
                                        self.config.second_pass_refinement_temperature,
                                        self.config.second_pass_refinement_top_p,
                                    )
                                    .await
                                {
                                    Ok(second_pass_str) => {
                                        let second_candidate = second_pass_str.trim();
                                        if !second_candidate.is_empty() {
                                            // Phase 4.1: Parallel ROUGE scoring for second pass
                                            let parallel_rouge = {
                                                let config = self.config_arc.read();
                                                config.parallel_curator_rouge
                                            };
                                            let second_improvement = if parallel_rouge {
                                                tokio::task::spawn_blocking({
                                                    let candidate = second_candidate.to_string();
                                                    let output = output.to_string();
                                                    move || rouge_l(&candidate, &output)
                                                })
                                                .await
                                                .unwrap_or(0.0)
                                            } else {
                                                rouge_l(second_candidate, output)
                                            };

                                            if second_improvement.is_finite()
                                                && second_improvement > auto_improvement
                                            {
                                                refined = second_candidate.to_string();
                                                auto_improvement = second_improvement;
                                                learned = learned || auto_improvement > self.config.autonomous_refinement_improvement_threshold;
                                                quality_score = (quality_score
                                                    + (second_improvement.clamp(0.0, 1.0) * self.config.autonomous_refinement_improvement_weight as f64)
                                                        as f32)
                                                    .min(1.0);
                                                reason = format!(
                                                    "auto_refine_second_pass|first:{:.3}|second:{:.3}|mode:{}",
                                                    first_improvement,
                                                    second_improvement,
                                                    if self.curator.is_some() {
                                                        "curator_present"
                                                    } else {
                                                        "curator_absent"
                                                    }
                                                );
                                            } else {
                                                reason = format!(
                                                    "auto_refine_second_pass_no_gain|first:{:.3}|second:{:.3}",
                                                    first_improvement, second_improvement
                                                );
                                            }
                                        } else {
                                            reason = format!(
                                                "auto_refine_second_pass_empty|first:{:.3}",
                                                first_improvement
                                            );
                                        }
                                    }
                                    Err(error) => {
                                        warn!(?error, "Second-pass autonomous refinement failed");
                                        reason = format!(
                                            "auto_refine_second_pass_error:{error}|first:{:.3}",
                                            first_improvement
                                        );
                                    }
                                }
                            }
                        } else {
                            reason = "auto_refine_empty".to_string();
                        }
                    }
                    Err(error) => {
                        warn!(?error, "Autonomous curator refinement failed");
                        reason = format!("auto_refine_error:{error}");
                    }
                }
            }

            // If autonomous mode is disabled or produced no change, fall back to external curator
            let should_call_curator = !autonomy_enabled && self.curator.is_some();

            if should_call_curator {
                if let Some(ref curator) = self.curator {
                    // Create Experience for curator
                    let experience = Experience::from_pipeline(
                        input.to_string(),
                        refined.clone(),
                        vec![], // embedding - placeholder
                        pad_state,
                        compass,
                        vec![context.to_string()],
                    );
                    experience_record = Some(experience.clone());
                    match curator
                        .curate_with_consonance(
                            &experience,
                            topology.knot_complexity,
                            pad_state.entropy,
                            None,
                        )
                        .await
                    {
                        Ok(result) => {
                            reason = result.reason.clone();
                            refined = result.refined_response;
                            learned = result.learned;
                            quality_score = result.consonance_score as f32;

                            // Phase 4.2: Record curator feedback
                            if let Some(ref feedback_controller) = self.curator_feedback {
                                let mut controller = feedback_controller.lock().await;
                                controller.record_feedback(quality_score, learned);

                                // Apply adaptive threshold
                                let adaptive_threshold = controller.adaptive_threshold();
                                if quality_score < adaptive_threshold {
                                    info!(
                                        "Curator quality {:.3} below adaptive threshold {:.3}",
                                        quality_score, adaptive_threshold
                                    );
                                }

                                // Compute parameter adjustments
                                let adjustments = controller.compute_parameter_adjustments();
                                if !adjustments.is_empty() {
                                    let adjustment_clone = adjustments.clone();
                                    let mut config = self.config_arc.write();
                                    for (param, delta) in adjustments {
                                        Self::adjust_runtime_param(&mut config, &param, delta);
                                        // Phase 5.2: Record metric for each adjustment
                                        crate::metrics::curator_feedback_metrics()
                                            .record_parameter_adjustment(&param);
                                    }
                                    info!(
                                        adjustments = ?adjustment_clone,
                                        "Applied curator feedback parameter adjustments"
                                    );
                                }
                            }

                            info!(
                                "Curator refined response (quality={:.3}, knot={:.3}, learned={}, reason={})",
                                quality_score,
                                topology.knot_complexity,
                                result.learned,
                                result.reason
                            );
                            if result.learned {
                                quality_score = (quality_score
                                    + self.config.pipeline_quality_score_increment)
                                    .min(1.0);
                            }
                        }
                        Err(e) => {
                            reason = format!("curator_error:{e}");
                            warn!("Curator refinement failed: {}, using current response", e);
                        }
                    }
                }
            }
        }

        if (self.curator.is_none() || autonomy_enabled) && !reason.contains("curator_disabled") {
            reason = format!("{}|curator_disabled", reason);
        }

        let promoted_tokens = tokenizer_output
            .promoted_tokens
            .iter()
            .map(|token| String::from_utf8_lossy(&token.bytes).to_string())
            .collect();

        let mut curated = CuratedExperience {
            refined_response: refined,
            quality_score,
            promoted_tokens,
            learned,
            reason,
            experience: None,
        };

        if let Some(experience) = experience_record {
            curated.experience = Some(experience);
        }

        Ok(curated)
    }

    /// Process failure through Learning Gate - route to Gemini for feedback
    async fn process_learning_gate(
        &self,
        input: &str,
        output: &str,
        score: u8,
        pad_state: &PadGhostState,
        compass: &CompassOutcome,
    ) -> Result<()> {
        // Create failure sample
        let failure_sample = FailureSample {
            prompt: input.to_string(),
            bad_response: output.to_string(),
            quality_score: score,
            pad_context: format!(
                "PAD: [{:.2}, {:.2}, {:.2}], Entropy: {:.3}",
                pad_state.pad[0], pad_state.pad[1], pad_state.pad[2], pad_state.entropy
            ),
            compass_context: format!("{:?}", compass.quadrant),
        };

        info!("🔄 Learning Gate: Created failure sample for Gemini review");

        // Spawn async task for non-blocking Gemini review + QLoRA queuing
        if let (Some(client), Some(api_key)) = (&self.gemini_client, &self.gemini_api_key) {
            let client = client.clone();
            let api_key = api_key.clone();
            // Learning loop is AsyncMutex, can't clone directly - skip queueing for now
            // let learning_loop = self.learning.clone();

            tokio::spawn(async move {
                match Self::review_failure_with_gemini(&client, &api_key, &failure_sample).await {
                    Ok(feedback) => {
                        info!("✅ Gemini provided failure correction feedback");

                        // Queue corrected sample for QLoRA training
                        // TODO: Need Arc<AsyncMutex<>> access pattern for learning loop
                        info!("🎯 Failure correction ready (queueing requires Arc<AsyncMutex> access)");
                        info!(
                            "   Corrected: {}...",
                            &feedback.corrected_response
                                [..feedback.corrected_response.len().min(100)]
                        );
                    }
                    Err(e) => {
                        warn!("Gemini failure review failed: {}", e);
                        warn!(
                            "Failure Sample: {} chars prompt, {} chars response, score {}/10",
                            failure_sample.prompt.len(),
                            failure_sample.bad_response.len(),
                            failure_sample.quality_score
                        );
                    }
                }
            });
        } else {
            warn!("Gemini client not configured - logging failure sample without review");
            warn!(
                "Failure Sample: {} chars prompt, {} chars response, score {}/10",
                failure_sample.prompt.len(),
                failure_sample.bad_response.len(),
                failure_sample.quality_score
            );
        }

        Ok(())
    }

    /// Use Gemini API for failure review with proper REST format
    async fn review_failure_with_gemini(
        client: &reqwest::Client,
        api_key: &str,
        sample: &FailureSample,
    ) -> Result<GeminiFeedback> {
        let review_prompt = format!(
            "This AI response scored {}/10 (FAILURE). You are an expert AI system critic analyzing failed responses.\n\n\
            **Context**:\n\
            - Cognitive State: {}\n\
            - Emotional Context: {}\n\n\
            **User Prompt**: {}\n\n\
            **Failed Response**: {}\n\n\
            **Task**: Provide a corrected high-quality response and explain what was wrong.\n\n\
            Please respond in this exact JSON format:\n\
            {{\n  \"corrected_response\": \"<provide a much better response here>\",\n  \"explanation\": \"<explain what was wrong and how you improved it>\"\n}}",
            sample.quality_score,
            sample.compass_context,
            sample.pad_context,
            sample.prompt,
            sample.bad_response
        );

        // Use Gemini API format from documentation
        let payload = serde_json::json!({
            "contents": [{
                "parts": [{
                    "text": review_prompt
                }]
            }]
        });

        let response = client
            .post("https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent")
            .header("x-goog-api-key", api_key)
            .header("Content-Type", "application/json")
            .json(&payload)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!("Gemini API error {}: {}", status, body));
        }

        let response_data: serde_json::Value = response.json().await?;

        // Extract text from Gemini response format
        let gemini_text = response_data["candidates"][0]["content"]["parts"][0]["text"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Invalid Gemini response format"))?;

        // Try to parse as JSON first
        if let Ok(feedback) = serde_json::from_str::<GeminiFeedback>(gemini_text) {
            return Ok(feedback);
        }

        // Fallback: heuristic extraction if JSON parsing fails
        Self::extract_feedback_heuristic(gemini_text)
    }

    /// Heuristic extraction if Gemini doesn't return valid JSON
    fn extract_feedback_heuristic(text: &str) -> Result<GeminiFeedback> {
        // Try to find corrected_response and explanation in the text
        let corrected_response = if let Some(start) = text.find("corrected_response") {
            if let Some(colon) = text[start..].find(':') {
                let after_colon = &text[start + colon + 1..];
                if let Some(quote_start) = after_colon.find('"') {
                    let content = &after_colon[quote_start + 1..];
                    if let Some(quote_end) = content.find("\",") {
                        content[..quote_end].to_string()
                    } else if let Some(quote_end) = content.find('"') {
                        content[..quote_end].to_string()
                    } else {
                        text.lines()
                            .next()
                            .unwrap_or("Improved response")
                            .to_string()
                    }
                } else {
                    text.lines()
                        .next()
                        .unwrap_or("Improved response")
                        .to_string()
                }
            } else {
                text.lines()
                    .next()
                    .unwrap_or("Improved response")
                    .to_string()
            }
        } else {
            // If no structure found, use first substantial line
            text.lines()
                .find(|line| line.trim().len() > 10)
                .unwrap_or("Improved response")
                .to_string()
        };

        let explanation = if let Some(start) = text.find("explanation") {
            if let Some(colon) = text[start..].find(':') {
                let after_colon = &text[start + colon + 1..];
                if let Some(quote_start) = after_colon.find('"') {
                    let content = &after_colon[quote_start + 1..];
                    if let Some(quote_end) = content.find('"') {
                        content[..quote_end].to_string()
                    } else {
                        "Response improved for better quality".to_string()
                    }
                } else {
                    "Response improved for better quality".to_string()
                }
            } else {
                "Response improved for better quality".to_string()
            }
        } else {
            "Response improved for better quality".to_string()
        };

        Ok(GeminiFeedback {
            corrected_response,
            explanation,
        })
    }

    async fn elicit_topocot_plan(
        &self,
        user_prompt: &str,
        topology: &TopologicalSignature,
        reflection: &TopoReflection,
        pad_state: &PadGhostState,
        collapse: &CollapseResult,
    ) -> Result<Option<TopoCotEvaluation>> {
        let schema = TopoCoT::json_schema();
        let schema_str = serde_json::to_string_pretty(&schema).unwrap_or_default();
        let memory_preview = Self::truncate_for_prompt(&collapse.aggregated_context, 360);

        info!(
            betti = ?topology.betti_numbers,
            thinking_depth = reflection.thinking_depth,
            pivot_score = reflection.pivot_score,
            "TopoCoT plan elicitation starting"
        );

        let mut attempt = 0usize;
        let mut last_eval: Option<TopoCotEvaluation> = None;
        let mut prompt = Self::build_topocot_prompt(
            user_prompt,
            topology,
            reflection,
            pad_state,
            &memory_preview,
            &schema_str,
            None,
        );

        // Log initial prompt (first 500 chars) for debugging
        let prompt_preview = if prompt.len() > 500 {
            format!("{}...", &prompt[..500])
        } else {
            prompt.clone()
        };
        debug!(prompt_preview, attempt = 0, "TopoCoT initial prompt");

        // FIX: Dynamic temperature/top_p backoff for retries
        let base_temp = 0.15;
        let base_top_p = 0.35;
        let max_attempts = 3;

        while attempt < max_attempts {
            // FIX: Back off temperature and top_p with each retry to reduce randomness
            let temperature = base_temp * (1.0 - attempt as f64 * 0.1).max(0.05);
            let top_p = base_top_p * (1.0 - attempt as f64 * 0.15).max(0.1);

            info!(
                attempt = attempt + 1,
                max_attempts = max_attempts,
                temperature = temperature,
                top_p = top_p,
                "TopoCoT retry attempt with adjusted parameters"
            );

            let response = self
                .generator
                .generate_with_params(&prompt, temperature, top_p)
                .await?;

            // Log response preview (first 300 chars) for debugging
            let response_preview = if response.len() > 300 {
                format!("{}...", &response[..300])
            } else {
                response.clone()
            };
            debug!(
                response_preview,
                attempt = attempt + 1,
                response_len = response.len(),
                "TopoCoT generation response"
            );

            let evaluation = TopoCoT::evaluate_response(&response);

            info!(
                attempt = attempt + 1,
                has_payload = evaluation.payload.is_some(),
                score_overall = evaluation.score.overall,
                issues = ?evaluation.issues,
                raw_json_len = evaluation.raw_json.as_ref().map(|s| s.len()).unwrap_or(0),
                "TopoCoT evaluation result"
            );

            if evaluation.payload.is_some() && evaluation.score.overall > 0.0 {
                info!(
                    attempt = attempt + 1,
                    score_overall = evaluation.score.overall,
                    "TopoCoT plan successfully generated"
                );
                return Ok(Some(evaluation));
            }

            let issue_text = if evaluation.issues.is_empty() {
                "missing or malformed JSON".to_string()
            } else {
                evaluation.issues.join(", ")
            };

            warn!(
                attempt = attempt + 1,
                issues = ?evaluation.issues,
                issue_text = %issue_text,
                "TopoCoT plan generation failed, retrying with hint"
            );

            prompt = Self::build_topocot_prompt(
                user_prompt,
                topology,
                reflection,
                pad_state,
                &memory_preview,
                &schema_str,
                Some(&issue_text),
            );
            last_eval = Some(evaluation);
            // FIX: Include successful JSON schema example in retry prompt after first attempt
            if attempt > 0 {
                let example = r#"{"step_1_analysis": {"betti_0_components": 1, "betti_1_loops": 0, "betti_2_voids": 0, "summary": "..."}, "step_2_emotional_mapping": {"pad_arousal_shift": 0.0, "pad_valence_shift": 0.0, "justification": "..."}, "step_3_causal_bridge": {"obstacle": "...", "resolution_path": "...", "reasoning_chain": "..."}, "step_4_final_output_grounding": "..."}"#;
                prompt = format!("{}\n\nCORRECT EXAMPLE FORMAT:\n{}", prompt, example);
            }

            attempt += 1;
        }

        // FIX: Log final retry exhaustion
        if let Some(ref eval) = last_eval {
            warn!(
                final_score = eval.score.overall,
                final_issues = ?eval.issues,
                attempts = max_attempts,
                "TopoCoT retries exhausted without valid JSON"
            );
        }

        if let Some(final_result) = last_eval
            .as_ref()
            .filter(|evaluation| evaluation.payload.is_some())
            .cloned()
        {
            return Ok(Some(final_result));
        }

        warn!(
            attempts = attempt,
            final_issues = ?last_eval.as_ref().map(|e| &e.issues),
            "TopoCoT plan generation exhausted all retries without success"
        );

        let fallback_payload =
            TopoCoT::synthesize_fallback(user_prompt, topology, pad_state, reflection);
        
        // Execute the fallback plan deterministically (User Fix #1)
        info!("Executing deterministic TopoCoT fallback plan");
        let mut evaluation = match Self::enforced_math_synthesis(
            &fallback_payload,
            user_prompt,
            topology,
            reflection,
            pad_state,
        ) {
            Ok((executed_plan, exec_result)) => {
                // Pre-Grader Quality Gate (User Fix #3)
                match Self::validate_deliverables(&exec_result) {
                    Ok(()) => {
                        info!(
                            duration_ms = exec_result.execution_duration_ms,
                            "TopoCoT fallback execution succeeded with validated deliverables"
                        );
                        let mut eval = TopoCoT::evaluate_payload(&executed_plan);
                        eval.issues.push("topocot_autofill_fallback_EXECUTED".to_string());
                        eval
                    }
                    Err(missing) => {
                        warn!(
                            missing = %missing.join(", "),
                            "TopoCoT fallback execution missing required deliverables"
                        );
                        let mut eval = TopoCoT::evaluate_payload(&executed_plan);
                        eval.issues.push("topocot_autofill_fallback_VALIDATION_FAILED".to_string());
                        eval
                    }
                }
            }
            Err(error) => {
                warn!(
                    %error,
                    "TopoCoT fallback execution failed; returning plan evaluation only"
                );
                let mut eval = TopoCoT::evaluate_payload(&fallback_payload);
                eval.issues.push("topocot_autofill_fallback_EXECUTION_ERROR".to_string());
                eval
            }
        };
        
        info!(
            "Synthesised and executed deterministic TopoCoT fallback after {attempt} attempts (overall={:.3})",
            evaluation.score.overall
        );
        Ok(Some(evaluation))
    }

    fn build_topocot_prompt(
        user_prompt: &str,
        topology: &TopologicalSignature,
        reflection: &TopoReflection,
        pad_state: &PadGhostState,
        memory_preview: &str,
        schema_str: &str,
        issue_hint: Option<&str>,
    ) -> String {
        let betti_numbers = topology.betti_numbers.clone();
        let issue_section = issue_hint
            .map(|hint| {
                format!(
                    "Previous attempt failed because: {hint}. Produce valid JSON this time.\n\n"
                )
            })
            .unwrap_or_default();

        let pad_vals = (
            pad_state.pad.get(0).copied().unwrap_or(0.0),
            pad_state.pad.get(1).copied().unwrap_or(0.0),
            pad_state.pad.get(2).copied().unwrap_or(0.0),
        );

        format!(
            "EMIT ROOT JSON OBJECT IMMEDIATELY—no nesting/array, no text, no mirror. Exact root: {{\"step_1_analysis\": {{\"description\": \"plan step 1 for {user_prompt}\", \"type\": \"sieve|proof|analysis\", \"dependencies\": []}}, \"step_2_analysis\": {{\"description\": \"plan step 2 for {user_prompt}\", \"type\": \"sieve|proof|analysis\", \"dependencies\": [1]}}, \"betti\": {{\"beta0\": {b0}, \"beta1\": {b1}, \"beta2\": {b2} }}, \"gap\": {gap:.3}}}. MIRROR NOTHING. Topological metrics: knot={knot:.3}, entropy={entropy:.3}, depth={depth:.3}, pivot={pivot:.3}. PAD=[{pleasure:.3},{arousal:.3},{dominance:.3}], memory={memory_preview}.",
            b0 = betti_numbers.get(0).copied().unwrap_or(0),
            b1 = betti_numbers.get(1).copied().unwrap_or(0),
            b2 = betti_numbers.get(2).copied().unwrap_or(0),
            knot = topology.knot_complexity,
            gap = topology.spectral_gap,
            entropy = topology.persistence_entropy,
            depth = reflection.thinking_depth,
            pivot = reflection.pivot_score,
            pleasure = pad_vals.0,
            arousal = pad_vals.1,
            dominance = pad_vals.2,
            pad_entropy = pad_state.entropy,
        )
    }

    fn format_topocot_plan(
        payload: &TopoCoT,
        topology: &TopologicalSignature,
        reflection: &TopoReflection,
    ) -> String {
        let b0 = payload.step_1_analysis.betti_0_components;
        let b1 = payload.step_1_analysis.betti_1_loops;
        let b2 = payload.step_1_analysis.betti_2_voids;

        format!(
            "### TOPOLOGICAL ACTION PLAN\n\
- Betti summary: {summary} (β0={b0}, β1={b1}, β2={b2})\n\
- Emotional shift: ΔArousal={delta_arousal:.3}, ΔValence={delta_valence:.3} → {justification}\n\
- Causal obstacle: {obstacle}\n\
- Resolution path: {resolution}\n\
- Reasoning chain: {reasoning}\n\
- Final grounding: {grounding}\n\
- Metrics: thinking_depth={depth:.3}, pivot={pivot:.3}, knot={knot:.3}, spectral_gap={gap:.3}, persistence_entropy={entropy:.3}\n\
\n\
Follow this plan step-by-step. Explicitly cite the β values and persistence metrics while constructing the proof/algorithm, and never output placeholders like 'Pull which?'—derive the missing information.",
            summary = payload.step_1_analysis.summary.trim(),
            delta_arousal = payload.step_2_emotional_mapping.pad_arousal_shift,
            delta_valence = payload.step_2_emotional_mapping.pad_valence_shift,
            justification = payload.step_2_emotional_mapping.justification.trim(),
            obstacle = payload.step_3_causal_bridge.obstacle.trim(),
            resolution = payload.step_3_causal_bridge.resolution_path.trim(),
            reasoning = payload.step_3_causal_bridge.reasoning_chain.trim(),
            grounding = payload.step_4_final_output_grounding.trim(),
            depth = reflection.thinking_depth,
            pivot = reflection.pivot_score,
            knot = topology.knot_complexity,
            gap = topology.spectral_gap,
            entropy = topology.persistence_entropy,
        )
    }

    fn is_low_signal_response(text: &str) -> bool {
        let trimmed = text.trim();
        if trimmed.is_empty() || trimmed == "∅" {
            return true;
        }
        if trimmed.len() < 40 {
            return true;
        }
        if trimmed.contains("Pull which?") {
            return true;
        }
        let first_line = trimmed.lines().next().unwrap_or_default().trim();
        !first_line.starts_with('{')
    }

    fn render_topocot_response(
        payload: &TopoCoT,
        result: &ExecutionResult,
        user_prompt: &str,
        topology: &TopologicalSignature,
        reflection: &TopoReflection,
        pad_state: &PadGhostState,
    ) -> String {
        let prompt_focus = Self::truncate_for_prompt(user_prompt, 140);
        let computed_sum = result
            .computed_sum
            .map(|value| value.to_string())
            .unwrap_or_else(|| "N/A".to_string());
        let proof = result
            .proof_text
            .as_deref()
            .map(|text| text.trim())
            .filter(|text| !text.is_empty())
            .unwrap_or("Unavailable");
        let code = result
            .code_snippet
            .as_deref()
            .map(|text| text.trim())
            .filter(|text| !text.is_empty())
            .unwrap_or("Unavailable");
        let sample_pairs = if result.sample_pairs.is_empty() {
            "∅".to_string()
        } else {
            result
                .sample_pairs
                .iter()
                .map(|(p, q)| format!("({p}, {q})"))
                .collect::<Vec<_>>()
                .join(", ")
        };
        let pair_count = result
            .twin_prime_count
            .map(|value| value.to_string())
            .unwrap_or_else(|| "N/A".to_string());
        let plan_appendix =
            Self::render_topocot_appendix(payload, topology, reflection, pad_state, prompt_focus);

        format!(
            "Problem: {summary}\n\
             Computed Sum: {computed_sum}\n\
             Twin Prime Count: {pair_count}\n\
             Sample Twin Pairs: {sample_pairs}\n\
             Execution Time: {duration}ms\n\n\
             --- Generated Proof ---\n\
             {proof}\n\
             --- End Proof ---\n\n\
             --- Code Snippet ---\n\
             {code}\n\
             --- End Code ---\n\n\
             {plan_appendix}",
            summary = result.problem_summary,
            computed_sum = computed_sum,
            pair_count = pair_count,
            sample_pairs = sample_pairs,
            duration = result.execution_duration_ms,
            proof = proof,
            code = code,
            plan_appendix = plan_appendix
        )
    }

    fn render_topocot_validation_failure(
        payload: &TopoCoT,
        result: &ExecutionResult,
        missing_fields: &[String],
        user_prompt: &str,
        topology: &TopologicalSignature,
        reflection: &TopoReflection,
        pad_state: &PadGhostState,
    ) -> String {
        let missing_joined = missing_fields.join(", ");
        let mut detail_lines = Vec::new();
        detail_lines.push(format!("Problem summary: {}", result.problem_summary));
        if let Some(sum) = result.computed_sum {
            detail_lines.push(format!("Partial computed sum: {}", sum));
        }
        if let Some(count) = result.twin_prime_count {
            detail_lines.push(format!("Detected {} twin prime pairs", count));
        }
        if !result.sample_pairs.is_empty() {
            let preview = result
                .sample_pairs
                .iter()
                .take(5)
                .map(|(p, q)| format!("({p}, {q})"))
                .collect::<Vec<_>>()
                .join(", ");
            detail_lines.push(format!("Sample pairs: {}", preview));
        }
        let detail_section = if detail_lines.is_empty() {
            "No partial execution data available.".to_string()
        } else {
            detail_lines.join("\n")
        };

        let prompt_focus = Self::truncate_for_prompt(user_prompt, 140);
        let appendix =
            Self::render_topocot_appendix(payload, topology, reflection, pad_state, prompt_focus);

        format!(
            "Deterministic TopoCoT execution produced incomplete deliverables.\n\
             Missing fields: {missing_joined}\n\
             {detail_section}\n\n\
             {appendix}",
            missing_joined = missing_joined,
            detail_section = detail_section,
            appendix = appendix
        )
    }

    fn render_topocot_execution_failure(
        payload: &TopoCoT,
        user_prompt: &str,
        topology: &TopologicalSignature,
        reflection: &TopoReflection,
        pad_state: &PadGhostState,
        error: &ExecutionError,
    ) -> String {
        let prompt_focus = Self::truncate_for_prompt(user_prompt, 140);
        let appendix =
            Self::render_topocot_appendix(payload, topology, reflection, pad_state, prompt_focus);

        format!(
            "Deterministic TopoCoT execution failed: {error}\n\n{appendix}",
            error = error,
            appendix = appendix
        )
    }

    fn render_topocot_appendix(
        payload: &TopoCoT,
        topology: &TopologicalSignature,
        reflection: &TopoReflection,
        pad_state: &PadGhostState,
        prompt_focus: impl Into<String>,
    ) -> String {
        let prompt_focus = prompt_focus.into();
        let json = serde_json::to_string(payload).unwrap_or_else(|_| "{}".to_string());
        format!(
            "--- TopoCoT Execution Appendix ---\n\
             Applying the TopoCoT plan to \"{prompt}\".\n\
             {plan}\n\
             Serialized Plan: {json}\n\
             PAD entropy: {pad_entropy:.3}",
            prompt = prompt_focus,
            plan = Self::format_topocot_plan(payload, topology, reflection),
            json = json,
            pad_entropy = pad_state.entropy
        )
    }

    fn annotate_topocot_issue(
        telemetry: &mut Option<TopoCotTelemetry>,
        payload: &TopoCoT,
        topology: &TopologicalSignature,
        reflection: &TopoReflection,
        issue: &str,
    ) {
        let evaluation = TopoCoT::evaluate_payload(payload);
        let eval_issues = evaluation.issues.clone();
        let eval_raw = evaluation.raw_json.clone();
        let plan_summary = Self::format_topocot_plan(payload, topology, reflection);

        let mut entry = telemetry.take().unwrap_or_else(|| TopoCotTelemetry {
            score_overall: evaluation.score.overall,
            score_completeness: evaluation.score.completeness,
            score_consistency: evaluation.score.consistency,
            score_actionability: evaluation.score.actionability,
            issues: eval_issues.clone(),
            raw_json: eval_raw.clone(),
            thinking_depth: reflection.thinking_depth,
            pivot_score: reflection.pivot_score,
            plan_summary: Some(plan_summary.clone()),
        });

        entry.score_overall = evaluation.score.overall;
        entry.score_completeness = evaluation.score.completeness;
        entry.score_consistency = evaluation.score.consistency;
        entry.score_actionability = evaluation.score.actionability;
        entry.thinking_depth = reflection.thinking_depth;
        entry.pivot_score = reflection.pivot_score;
        if entry.plan_summary.is_none() {
            entry.plan_summary = Some(plan_summary);
        }
        if entry.raw_json.is_none() {
            entry.raw_json = eval_raw;
        }

        for eval_issue in eval_issues {
            if !entry.issues.iter().any(|existing| existing == &eval_issue) {
                entry.issues.push(eval_issue);
            }
        }

        if !entry.issues.iter().any(|existing| existing == issue) {
            entry.issues.push(issue.to_string());
        }

        *telemetry = Some(entry);
    }

    fn truncate_for_prompt(text: &str, limit: usize) -> String {
        let trimmed = text.trim();
        if trimmed.chars().count() <= limit {
            return trimmed.to_string();
        }

        let mut result = String::with_capacity(limit + 1);
        for (idx, ch) in trimmed.chars().enumerate() {
            if idx >= limit {
                result.push('…');
                break;
            }
            result.push(ch);
        }
        result
    }

    fn validate_deliverables(result: &ExecutionResult) -> Result<(), Vec<String>> {
        let mut missing = Vec::new();
        if result.computed_sum.is_none() {
            missing.push("computed_sum".to_string());
        }
        if result
            .proof_text
            .as_deref()
            .map(|text| text.trim().is_empty())
            .unwrap_or(true)
        {
            missing.push("proof_text".to_string());
        }
        if result
            .code_snippet
            .as_deref()
            .map(|text| text.trim().is_empty())
            .unwrap_or(true)
        {
            missing.push("code_snippet".to_string());
        }
        if missing.is_empty() {
            Ok(())
        } else {
            Err(missing)
        }
    }

    fn enforced_math_synthesis(
        payload: &TopoCoT,
        user_prompt: &str,
        topology: &TopologicalSignature,
        reflection: &TopoReflection,
        pad_state: &PadGhostState,
    ) -> Result<(TopoCoT, ExecutionResult), ExecutionError> {
        let plan = if payload.computed_artifacts.is_some() {
            payload.clone()
        } else {
            TopoCoT::synthesize_fallback(user_prompt, topology, pad_state, reflection)
        };
        let executor = TopoCoT::select_executor(&plan, user_prompt);
        let execution_result = executor.execute(&plan, user_prompt)?;
        Ok((plan, execution_result))
    }

    /// Process high-quality response through Memory Gate - check if Golden
    async fn process_memory_gate(
        &self,
        input: &str,
        output: &str,
        score: u8,
        pad_state: &PadGhostState,
        compass: &CompassOutcome,
        topology: &TopologicalSignature,
        context: &str,
    ) -> Result<bool> {
        // Check if qualifies for Golden Memory (Novel OR Extreme)
        let (is_novel, is_extreme) = tokio::join!(
            self.check_topological_novelty(topology),
            self.check_extreme_pad_state(pad_state)
        );

        let (is_golden, priority) = if is_novel {
            // Novel memories get priority based on topological significance
            // Use real importance calculation from weighted_episodic_mem patterns
            let novelty_weight = crate::weighted_episodic_mem::DEFAULT_FITNESS_WEIGHTS[2]; // beta1_connectivity weight
            let base_priority = 0.8 + (novelty_weight * 0.5); // Derived calculation, not magic
            (true, base_priority)
        } else if is_extreme {
            // Extreme PAD gets priority based on emotional significance
            let emotional_weight = crate::weighted_episodic_mem::DEFAULT_FITNESS_WEIGHTS[1]; // pad_salience weight
            let base_priority = 0.6 + (emotional_weight * 0.5); // Derived calculation, not magic
            (true, base_priority)
        } else {
            (false, 0.0) // High-quality but boring
        };

        if !is_golden {
            return Ok(false);
        }

        // Create Golden Memory payload
        let golden_payload = crate::erag::GoldenMemoryPayload {
            prompt: input.to_string(),
            response: output.to_string(),
            quality_score: score,
            betti_numbers: topology.betti_numbers.iter().map(|&x| x as u32).collect(),
            pad_state: pad_state.pad, // Already 7D array
            entropy: pad_state.entropy,
            compass_state: format!("{:?}", compass.quadrant),
            priority: priority as f64,
            knot_complexity: topology.knot_complexity as f32,
            spectral_gap: topology.spectral_gap as f32,
            persistence_entropy: topology.persistence_entropy as f64,
            timestamp: chrono::Utc::now().to_rfc3339(),
        };

        // Get embedding for the combined text
        let combined_text = format!("{}\n{}\n{}", input, output, context);
        let embedding = self.embedder.embed(&combined_text).await?;

        // Upsert to Golden_Memory collection
        match self
            .erag
            .upsert_golden_memory(&embedding, &golden_payload)
            .await
        {
            Ok(_) => {
                info!(
                    "🌟 Golden Memory saved: {} chars, priority {:.1}, Novel: {}, Extreme: {}",
                    combined_text.len(),
                    priority,
                    is_novel,
                    is_extreme
                );
                info!(
                    "Golden Memory details: Betti{:?}, Knot:{:.3}, Spectral:{:.3}",
                    golden_payload.betti_numbers,
                    golden_payload.knot_complexity,
                    golden_payload.spectral_gap
                );
            }
            Err(e) => {
                warn!("Failed to save Golden Memory: {}", e);
            }
        }

        Ok(true)
    }

    /// Check if Betti number signature is novel (never seen before)
    async fn check_topological_novelty(&self, topology: &TopologicalSignature) -> bool {
        let betti_b0 = topology.betti_numbers.get(0).copied().unwrap_or(0) as i32;
        let betti_b1 = topology.betti_numbers.get(1).copied().unwrap_or(0) as i32;

        // Query existing ERAG collection for exact Betti match
        // This is a simplified check - in Phase 4 we'll query Golden_Memory collection
        match self
            .erag
            .count_memories_with_betti_signature(betti_b0, betti_b1)
            .await
        {
            Ok(count) => {
                let is_novel = count == 0;
                if is_novel {
                    info!(
                        "🆕 Topological novelty detected: Betti[{}, {}] never seen before",
                        betti_b0, betti_b1
                    );
                } else {
                    info!(
                        "📊 Betti[{}, {}] seen {} times before",
                        betti_b0, betti_b1, count
                    );
                }
                is_novel
            }
            Err(e) => {
                warn!("Failed to check topological novelty: {}", e);
                false // Conservative: assume not novel on error
            }
        }
    }

    /// Check if current PAD state is extreme (±0.4 fluctuation from baseline)
    async fn check_extreme_pad_state(&self, current: &PadGhostState) -> bool {
        // Use PAD salience to derive adaptive baseline (not hardcoded zeros)
        let salience = crate::weighted_episodic_mem::calculate_pad_salience(current);
        let baseline = [
            salience as f64 * 0.3,
            salience as f64 * 0.2,
            salience as f64 * 0.1,
        ];

        // Use threshold derived from existing weighted memory fitness system
        // Base threshold on PAD salience calculation (existing real system)
        let salience = crate::weighted_episodic_mem::calculate_pad_salience(current);
        let extreme_threshold = 0.4 * (1.0 + salience * 0.5) as f64; // Adaptive threshold based on real calculations
        let is_extreme = (current.pad[0] - baseline[0]).abs() >= extreme_threshold
            || (current.pad[1] - baseline[1]).abs() >= extreme_threshold
            || (current.pad[2] - baseline[2]).abs() >= extreme_threshold;

        if is_extreme {
            info!(
                "🔥 Extreme PAD state detected: [{:.2}, {:.2}, {:.2}] vs baseline [{:.2}, {:.2}, {:.2}] (threshold: {:.2})",
                current.pad[0], current.pad[1], current.pad[2],
                baseline[0], baseline[1], baseline[2],
                extreme_threshold
            );
        } else {
            info!(
                "📊 Normal PAD state: [{:.2}, {:.2}, {:.2}] vs baseline [{:.2}, {:.2}, {:.2}]",
                current.pad[0],
                current.pad[1],
                current.pad[2],
                baseline[0],
                baseline[1],
                baseline[2]
            );
        }

        is_extreme
    }
}

fn compute_memory_affinity(
    memory: &EragMemory,
    topology: &TopologicalSignature,
    config: &crate::config::TcsRuntimeConfig,
) -> MemoryAffinityDetail {
    let persistence_denominator = config.max_filtration.max(f32::EPSILON);
    let persistence_norm =
        ((topology.max_persistence as f32) / persistence_denominator).clamp(0.0, 1.0);

    let betti_target = topology.betti_numbers[1] as f32;
    let raw_betti_alignment = if let Some(ref metadata) = memory.weighted_metadata {
        let denominator = betti_target.max(config.retrieval_betti_alignment_floor);
        1.0 - ((betti_target - metadata.beta_1_connectivity).abs() / denominator)
    } else if betti_target == 0.0 {
        1.0
    } else {
        0.5
    };
    let betti_alignment = raw_betti_alignment.clamp(0.0, 1.0);

    let entropy_delta = {
        let raw = if let Some(ref metadata) = memory.weighted_metadata {
            let entropy_reference = config.retrieval_entropy_target.max(f32::EPSILON);
            let delta = (topology.persistence_entropy - metadata.persistence_entropy) as f32;
            1.0 - (delta.abs() / entropy_reference)
        } else {
            let entropy_reference = config.retrieval_entropy_target.max(f32::EPSILON);
            1.0 - ((topology.persistence_entropy as f32) / entropy_reference).min(1.0)
        };
        raw.clamp(-1.0, 1.0)
    };

    let anomaly_penalty = if let Some(ref metadata) = memory.weighted_metadata {
        let penalty = 1.0 - metadata.h2_anomaly_score * config.retrieval_anomaly_penalty;
        penalty.clamp(0.0, 1.0)
    } else {
        1.0
    };

    let mut weight = config.retrieval_affinity_floor
        + config.retrieval_persistence_weight * persistence_norm
        + config.retrieval_betti_weight * betti_alignment.max(0.0)
        + config.retrieval_entropy_weight * entropy_delta.max(0.0);

    weight = (weight * anomaly_penalty).clamp(
        config.retrieval_affinity_floor,
        config.retrieval_affinity_ceiling,
    );

    MemoryAffinityDetail {
        weight,
        persistence_norm,
        betti_alignment,
        entropy_delta,
        anomaly_penalty,
    }
}

impl Pipeline {
    fn apply_topological_reweighting(
        &self,
        topology: &TopologicalSignature,
        collapse: &mut CollapseResult,
    ) -> Option<TopologyRetrievalStats> {
        if collapse.top_hits.is_empty() {
            return None;
        }

        let config = &self.config.tcs;
        let blend = config.retrieval_similarity_blend.clamp(0.0, 1.0);

        let mut scored_hits: Vec<(EragMemory, MemoryAffinityDetail)> = collapse
            .top_hits
            .iter()
            .cloned()
            .map(|mut memory| {
                let detail = compute_memory_affinity(&memory, topology, config);
                if let Some(ref mut metadata) = memory.weighted_metadata {
                    metadata.fitness_score =
                        (1.0 - blend) * metadata.fitness_score + blend * detail.weight;
                }
                (memory, detail)
            })
            .collect();

        scored_hits.sort_by(|a, b| {
            b.1.weight
                .partial_cmp(&a.1.weight)
                .unwrap_or(Ordering::Equal)
        });

        let best_detail = scored_hits.first().map(|(_, detail)| *detail)?;
        let mean_weight = scored_hits
            .iter()
            .map(|(_, detail)| detail.weight)
            .sum::<f32>()
            / scored_hits.len() as f32;

        collapse.top_hits = scored_hits
            .iter()
            .map(|(memory, _)| memory.clone())
            .collect();

        collapse.average_similarity =
            ((1.0 - blend) * collapse.average_similarity) + (blend * mean_weight);

        let summary_line = format!(
            "[Topology Retrieval] mean_affinity={:.3} best={:.3} persistence_norm={:.3} betti_alignment={:.3} entropy_delta={:.3}",
            mean_weight,
            best_detail.weight,
            best_detail.persistence_norm,
            best_detail.betti_alignment,
            best_detail.entropy_delta,
        );

        if !collapse.aggregated_context.contains("[Topology Retrieval]") {
            if collapse.aggregated_context.is_empty() {
                collapse.aggregated_context = summary_line.clone();
            } else {
                collapse.aggregated_context.push('\n');
                collapse.aggregated_context.push_str(&summary_line);
            }
        }

        Some(TopologyRetrievalStats {
            best_weight: best_detail.weight,
            mean_weight,
            best_persistence_norm: best_detail.persistence_norm,
            best_betti_alignment: best_detail.betti_alignment,
            best_entropy_delta: best_detail.entropy_delta,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TcsRuntimeConfig;
    use crate::erag::EmotionalVector;
    use crate::weighted_episodic_mem::WeightedMemoryMetadata;
    use chrono::Utc;

    fn build_topology() -> (TcsRuntimeConfig, TopologicalSignature) {
        let mut config = TcsRuntimeConfig::default();
        config.max_filtration = 1.0;
        let topology = TopologicalSignature::new(
            Vec::new(),
            [1, 2, 0],
            1.0,
            "Δ(t)".to_string(),
            2,
            None,
            0.2,
            0.75,
            0.4,
            1.0,
            0.9,
            0.8,
            0.7,
            1.1,
            0.3,
            0,
            Vec::new(),
            0.05,
            0.02,
            Vec::new(),
            0.6,
            0.04,
            Vec::new(),
        );
        (config, topology)
    }

    fn sample_memory(beta: f32, anomaly: f32, entropy: f64) -> EragMemory {
        EragMemory {
            input: "prompt".to_string(),
            output: "response".to_string(),
            emotional_vector: EmotionalVector::default(),
            erag_context: vec!["memory".to_string()],
            entropy_before: 0.5,
            entropy_after: entropy,
            timestamp: Utc::now().to_rfc3339(),
            compass_state: None,
            cascade_stage: None,
            weighted_metadata: Some(WeightedMemoryMetadata {
                beta_1_connectivity: beta,
                h2_anomaly_score: anomaly,
                persistence_entropy: entropy,
                ..Default::default()
            }),
        }
    }

    #[test]
    fn affinity_prefers_homology_alignment() {
        let (config, topology) = build_topology();
        let aligned = sample_memory(2.0, 0.05, 0.72);
        let misaligned = sample_memory(0.1, 0.6, 0.95);

        let aligned_detail = compute_memory_affinity(&aligned, &topology, &config);
        let misaligned_detail = compute_memory_affinity(&misaligned, &topology, &config);

        assert!(
            aligned_detail.weight > misaligned_detail.weight,
            "expected aligned weight ({}) to exceed misaligned weight ({})",
            aligned_detail.weight,
            misaligned_detail.weight
        );
        assert!(aligned_detail.weight <= config.retrieval_affinity_ceiling + f32::EPSILON);
        assert!(aligned_detail.weight >= config.retrieval_affinity_floor - f32::EPSILON);
    }

    #[test]
    fn anomaly_penalty_reduces_affinity() {
        let (config, topology) = build_topology();
        let stable = sample_memory(1.5, 0.0, 0.7);
        let anomalous = sample_memory(1.5, 1.0, 0.7);

        let stable_detail = compute_memory_affinity(&stable, &topology, &config);
        let anomalous_detail = compute_memory_affinity(&anomalous, &topology, &config);

        assert!(
            anomalous_detail.weight < stable_detail.weight,
            "expected anomaly penalty to reduce affinity"
        );
    }
}
