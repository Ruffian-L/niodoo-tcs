use std::cmp::Ordering;
use std::sync::atomic::Ordering as AtomicOrdering;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use tracing::{info, warn};

use crate::compass::{CascadeTransition, CompassOutcome, CompassQuadrant};
use crate::config::{env_value, TopologyMode};
use crate::consonance::{compute_consonance, ConsonanceMetrics};
#[cfg(feature = "gpu")]
use crate::consonance::{compute_topological_consistency, compute_erag_relevance, compute_compass_transition, compute_confidence};
use crate::data::Experience;
use crate::erag::CollapseResult;
use crate::generation::GenerationResult;
use crate::hyperfocus::HyperfocusEvent;
use crate::metrics::metrics;
use crate::signals::FailureSignals;
use crate::tcs_analysis::{baseline_topological_signature, TopologicalSignature};
use crate::token_manager::TokenizerOutput;
use crate::torus::{PadGhostState, TorusPadMapper};
use crate::util::rouge_l;
use crate::ntoken_client;

use super::cache::cache_key;
use super::core::Pipeline;
use super::metrics::StageTimings;
use super::state::{CuratedExperience, CuratorFeedbackController, PipelineCycle};
use crate::learning::LearningOutcome;

impl Pipeline {
    pub async fn process_prompt(&mut self, prompt: &str) -> Result<PipelineCycle> {
        let overall_start = Instant::now();
        let mut timings = StageTimings::default();
        let cache_key = cache_key(prompt);
        let now = Instant::now();

        // Stage 1: Embedding (with cache)
        let embedding_start = Instant::now();
        let embedding_hit = self.embedding_cache.get(&cache_key, now).await;
        let embedding = if let Some(hit) = embedding_hit {
            hit
        } else {
            let emb = self.embedder.embed(prompt).await?;
            self.embedding_cache
                .insert(cache_key, emb.clone(), now)
                .await;
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

        let tcs_start = Instant::now();
        let (topology, analysis_label) = match self.config.topology_mode {
            TopologyMode::Hybrid => match self.tcs_analyzer.as_mut() {
                Some(analyzer) => match analyzer.analyze_state(&pad_state) {
                    Ok(signature) => (signature, "hybrid"),
                    Err(error) => {
                        warn!(%error, "TCS analyzer failed; using analytic baseline signature");
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
                        features.h1_total_persistence,
                        features.sheaf_energy
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
                    let top_k = (self.config.base_retrieval_top_k + retrieval_top_k_increment).clamp(
                        self.config.pipeline_retrieval_top_k_min as i32,
                        self.config.pipeline_retrieval_top_k_max as i32
                    ) as usize;
                    let collapse = erag_client
                        .collapse_with_limit(&embedding_for_collapse, top_k)
                        .await?;
                    collapse_cache
                        .insert(cache_key, collapse.clone(), now)
                        .await;
                    Ok(collapse)
                }
            }
        )?;
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

        // Refetch nToken features with full context for tokenizer refinement
        // (we already have initial features for compass, but context-aware version is better for tokenizer)
        let ntoken_features = if self.config.n_tokens_bypass {
            info!("nTokens bypass enabled; skipping nToken feature extraction");
            None
        } else if let Ok(endpoint) = std::env::var("NTOKEN_ENDPOINT") {
            match ntoken_client::fetch_features(&endpoint, prompt, Some(&collapse.aggregated_context)).await {
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
                        let topological_consistency = compute_topological_consistency(&topology, &pad_state);
                        let erag_relevance = compute_erag_relevance(&collapse);
                        let compass_transition = compute_compass_transition(&compass, last_compass.as_ref());
                        
                        // Weighted combination
                        let sources = vec![
                            crate::consonance::ConsonanceSource::EmotionalCoherence(emotional_coherence),
                            crate::consonance::ConsonanceSource::TopologicalConsistency(topological_consistency),
                            crate::consonance::ConsonanceSource::ERAGRelevance(erag_relevance),
                            crate::consonance::ConsonanceSource::CompassTransition(compass_transition),
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
                        if let Ok(batch_scores) = gpu_calc.batch_weighted_consonance(&[source_scores_array], &weights).await {
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
                                compute_consonance(&pad_state, &compass, &collapse, &topology, None, last_compass.as_ref())
                            }
                        } else {
                            compute_consonance(&pad_state, &compass, &collapse, &topology, None, last_compass.as_ref())
                        }
                    } else {
                        compute_consonance(&pad_state, &compass, &collapse, &topology, None, last_compass.as_ref())
                    }
                } else {
                    compute_consonance(&pad_state, &compass, &collapse, &topology, None, last_compass.as_ref())
                }
            } else {
                compute_consonance(&pad_state, &compass, &collapse, &topology, None, last_compass.as_ref())
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

        let mut hyperfocus_event = self.hyperfocus_detector.detect(&hyperfocus_signals);

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
                    let pad_vecs: Vec<Vec<f32>> = vec![vec![pad_state.pad[0] as f32, pad_state.pad[1] as f32, pad_state.pad[2] as f32]; top_hits.len()];
                    let mem_vecs: Vec<Vec<f32>> = top_hits.iter().map(|m| {
                        vec![m.emotional_vector.joy, m.emotional_vector.anger, m.emotional_vector.surprise]
                    }).collect();
                    
                    // Batch compute cosine similarities on GPU
                    if let Ok(cosines) = gpu_calc.batch_cosine_similarity(&pad_vecs, &mem_vecs).await {
                        // Compute entropy scores
                        let ent_after_vec: Vec<f64> = top_hits.iter().map(|m| m.entropy_after).collect();
                        let entropy_scores: Vec<f64> = ent_after_vec.iter().map(|ent_after| {
                            1.0 - (topology.persistence_entropy - ent_after).abs().min(1.0)
                        }).collect();
                        
                        // Combine scores
                        let scores: Vec<f64> = cosines.iter().enumerate().map(|(i, &cosine)| {
                            (self.config.rce_erag_cosine_weight * cosine as f64 + 
                             self.config.rce_erag_entropy_weight * entropy_scores[i]) * lambda
                        }).collect();
                        
                        // Sort by score
                        let mut indexed: Vec<(usize, f64)> = scores.into_iter().enumerate().collect();
                        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
                        top_hits = indexed.into_iter().map(|(i, _)| top_hits[i].clone()).collect();
                    } else {
                        // Fallback to CPU if GPU fails
                        top_hits.sort_by(|a, b| {
                            let score = |m: &crate::erag::EragMemory| {
                                let pad_vec = [pad_state.pad[0] as f64, pad_state.pad[1] as f64, pad_state.pad[2] as f64];
                                let mem_vec = [
                                    m.emotional_vector.joy as f64,
                                    m.emotional_vector.anger as f64,
                                    m.emotional_vector.surprise as f64,
                                ];
                                let dot = pad_vec[0] * mem_vec[0] + pad_vec[1] * mem_vec[1] + pad_vec[2] * mem_vec[2];
                                let n1 = (pad_vec[0] * pad_vec[0] + pad_vec[1] * pad_vec[1] + pad_vec[2] * pad_vec[2]).sqrt();
                                let n2 = (mem_vec[0] * mem_vec[0] + mem_vec[1] * mem_vec[1] + mem_vec[2] * mem_vec[2]).sqrt();
                                let cosine = if n1 > 0.0 && n2 > 0.0 {
                                    (dot / (n1 * n2)).clamp(-1.0, 1.0)
                                } else {
                                    0.0
                                };
                                let ent_after = m.entropy_after;
                                let ent_score = 1.0 - (topology.persistence_entropy - ent_after).abs().min(1.0);
                                (self.config.rce_erag_cosine_weight * cosine + self.config.rce_erag_entropy_weight * ent_score) * lambda
                            };
                            score(b).partial_cmp(&score(a)).unwrap_or(Ordering::Equal)
                        });
                    }
                } else {
                    // CPU fallback
                    top_hits.sort_by(|a, b| {
                        let score = |m: &crate::erag::EragMemory| {
                            let pad_vec = [pad_state.pad[0] as f64, pad_state.pad[1] as f64, pad_state.pad[2] as f64];
                            let mem_vec = [
                                m.emotional_vector.joy as f64,
                                m.emotional_vector.anger as f64,
                                m.emotional_vector.surprise as f64,
                            ];
                            let dot = pad_vec[0] * mem_vec[0] + pad_vec[1] * mem_vec[1] + pad_vec[2] * mem_vec[2];
                            let n1 = (pad_vec[0] * pad_vec[0] + pad_vec[1] * pad_vec[1] + pad_vec[2] * pad_vec[2]).sqrt();
                            let n2 = (mem_vec[0] * mem_vec[0] + mem_vec[1] * mem_vec[1] + mem_vec[2] * mem_vec[2]).sqrt();
                            let cosine = if n1 > 0.0 && n2 > 0.0 {
                                (dot / (n1 * n2)).clamp(-1.0, 1.0)
                            } else {
                                0.0
                            };
                            let ent_after = m.entropy_after;
                            let ent_score = 1.0 - (topology.persistence_entropy - ent_after).abs().min(1.0);
                            (self.config.rce_erag_cosine_weight * cosine + self.config.rce_erag_entropy_weight * ent_score) * lambda
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
                        let pad_vec = [pad_state.pad[0] as f64, pad_state.pad[1] as f64, pad_state.pad[2] as f64];
                        let mem_vec = [
                            m.emotional_vector.joy as f64,
                            m.emotional_vector.anger as f64,
                            m.emotional_vector.surprise as f64,
                        ];
                        let dot = pad_vec[0] * mem_vec[0] + pad_vec[1] * mem_vec[1] + pad_vec[2] * mem_vec[2];
                        let n1 = (pad_vec[0] * pad_vec[0] + pad_vec[1] * pad_vec[1] + pad_vec[2] * pad_vec[2]).sqrt();
                        let n2 = (mem_vec[0] * mem_vec[0] + mem_vec[1] * mem_vec[1] + mem_vec[2] * mem_vec[2]).sqrt();
                        let cosine = if n1 > 0.0 && n2 > 0.0 {
                            (dot / (n1 * n2)).clamp(-1.0, 1.0)
                        } else {
                            0.0
                        };
                        let ent_after = m.entropy_after;
                        let ent_score = 1.0 - (topology.persistence_entropy - ent_after).abs().min(1.0);
                        (self.config.rce_erag_cosine_weight * cosine + self.config.rce_erag_entropy_weight * ent_score) * lambda
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
            if topology.persistence_entropy > self.config.rce_adaptation_entropy_threshold || topology.spectral_gap > self.config.rce_adaptation_spectral_gap_threshold {
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

        if let Some(features) = &ntoken_features {
            let cues = format!(
                "[nToken cues] h1_count={:.0} h1_persistence={:.4} entropy_norm={:.4} sheaf_energy={:.4}",
                features.h1_count,
                features.h1_total_persistence,
                features.entropy_norm,
                features.sheaf_energy
            );
            tokenizer_output.augmented_prompt = format!("{cues}\n{}", tokenizer_output.augmented_prompt);
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
        let generation = if self.config.enable_consistency_voting {
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
                self.rce_analyzer = Some(crate::rce::analyzer::RceAnalyzer::new(window.max(2), weights, threshold));
                tracing::info!("RCE initialized in shadow mode (read-only metrics)");
            }
            if let Some(analyzer) = self.rce_analyzer.as_mut() {
                // Pass prompt timestamp for prompt-to-spike latency tracking
                let beta = analyzer.update_with_prompt_timestamp(&pad_state, &topology, Some(overall_start));
                // Consensus gate (read-only): combine diverse simple votes
                let mut approved = true;
                if self.config.rce_consensus.enabled {
                    let gate = crate::rce::safety::ensemble::ConsensusGate::new(self.config.rce_consensus.clone());
                    let vote_beta = beta >= self.config.rce_breakthrough_threshold;
                    let vote_meta = analyzer.current_metastability() * topology.persistence_entropy > 0.0;
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
            let streak = self
                .rce_spike_streak
                .fetch_add(1, AtomicOrdering::SeqCst)
                + 1;
                        if streak >= self.config.rce_circuit_breaker_streak {
                            // Circuit breaker: slow mode – avoid further aggressive adjustments
                            tracing::warn!("RCE circuit breaker: sustained β_meta spikes ({}). Entering slow mode.", streak);
                        } else {
                            // Apply focused resource allocation by tightening exploration
                            // Use existing increments from config to avoid magic numbers
                            let temp_delta = -self.config.cot_temp_increment;
                            let top_p_delta = -self.config.phase2_top_p_increment;
                            crate::pipeline::core::Pipeline::adjust_runtime_param(&mut self.config, "temperature", temp_delta);
                            crate::pipeline::core::Pipeline::adjust_runtime_param(&mut self.config, "top_p", top_p_delta);
                        }
                    } else {
                        // Reset streak when below threshold or not approved
                        self.rce_spike_streak
                            .store(0, AtomicOrdering::SeqCst);
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
                        sources: vec![crate::consonance::ConsonanceSource::TopologicalConsistency(rce_score)],
                        confidence: 0.9,
                        dissonance_score: 1.0 - rce_score,
                    },
                );

                // Topology-driven curriculum scheduling
                if self.config.rce_actions_enabled && !self.config.rce_shadow_mode {
                    let mut guard = self.learning.lock().await;
                    guard.rce_schedule(beta, self.config.rce_breakthrough_threshold, topology.persistence_entropy);
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

        // Proceed with learning using curated response (with retry-corrected generation)
        let learning_start = Instant::now();
        info!("About to lock learning mutex");

        // Wrap learning update in timeout to prevent hanging
        let learning_result = tokio::time::timeout(Duration::from_secs(self.config.learning_timeout_secs), async {
            self.learning
                .lock()
                .await
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
                warn!("Learning update timed out after {}s - using default outcome", self.config.learning_timeout_secs);
                // Create a default learning outcome
                LearningOutcome {
                    events: vec!["learning_timeout".to_string()],
                    breakthroughs: vec![],
                    qlora_updates: vec![],
                    entropy_delta: 0.0,
                    adjusted_params: std::collections::HashMap::new(),
                    last_replay: None,
                }
            }
        };

        timings.learning_ms = learning_start.elapsed().as_secs_f64() * 1000.0;

        // Remove double-storage: defer storage decision to final gate below

        // Stage 7.5: Curator Quality Gate (single source of truth)
        let response_to_store = curated_experience.refined_response.clone();
        let final_quality_score = Some(curated_experience.quality_score);
        info!(
            "Checking quality gate: score={}, threshold={}",
            curated_experience.quality_score, self.config.curator_minimum_threshold
        );
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
                topology: topology.clone(),
                topology_mode: self.config.topology_mode,
                consonance: Some(full_consonance),
                hyperfocus: hyperfocus_event,
                cascade_transition,
            });
        }

        // Create enriched experience record now that curator approved storage
        let aggregated_context_lines: Vec<String> = collapse
            .aggregated_context
            .lines()
            .map(|s| s.to_string())
            .collect();
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
                controller.record_feedback(curated_experience.quality_score, curated_experience.learned);
            }
            
            let reward = generation.rouge_score * self.config.reward_rouge_weight + (1.0 - pad_state.entropy) * self.config.reward_entropy_weight;
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
                controller.record_feedback(curated_experience.quality_score, curated_experience.learned);
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
            Err(_) => warn!("Upsert memory timed out after {}s - continuing", self.config.memory_upsert_timeout_secs),
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
            return Ok((generation.clone(), initial_failure.to_string(), loop_start.elapsed().as_secs_f64() * 1000.0));
        }
        // INTEGRATION FIX: Handle healing state specially - enhance instead of retry
        if initial_failure == "none" && compass.is_healing {
            // In healing state with good topology - apply enhancement strategies
            if topology.knot_complexity < self.config.pipeline_healing_knot_threshold 
                && topology.spectral_gap > self.config.pipeline_healing_spectral_gap_threshold {
                info!("Healing state detected with good topology - applying enhancement");

                // Generate an enhanced version leveraging the good state
                let enhancement_prompt = format!(
                    "{}\n\n[System is in optimal healing state. Enhance clarity and depth.]",
                    prompt
                );

                if let Ok(enhanced_str) = self
                    .generator
                    .generate_with_params(&enhancement_prompt, self.config.enhancement_temperature, self.config.enhancement_top_p) // Low temp for stability
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
                    (baseline_result.unwrap_or(0.0), reflexion_result.unwrap_or(0.0))
                } else {
                    (rouge_l(&current_gen.baseline_response, prompt), rouge_l(&reflexion_response, prompt))
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
                (to_baseline_result.unwrap_or(0.0), score_result.unwrap_or(0.0))
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
            let adjusted_ucb1 = if retry_gen.rouge_score > current_gen.rouge_score + self.config.rouge_improvement_threshold {
                // ROUGE improved by configured threshold, boost ucb1 to reflect success
                ucb1_score.max(self.config.ucb1_boost_threshold).min(self.config.pipeline_ucb1_max_clamp)
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
        let length_factor = (output.len().min(self.config.quality_max_length) as f32 / self.config.quality_max_length as f32) * self.config.quality_length_factor_weight;
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

        // TOPOLOGY-AWARE REFINEMENT: Refine if quality is low OR topology indicates issues
        let refinement_threshold = self.config.curator_quality_threshold;

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

        let mut reason = refinement_reason.to_string();
        let mut experience_record: Option<Experience> = None;
        let needs_refinement = quality_score < refinement_threshold || topology_needs_refinement;
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
                    .generate_with_params(&autonomy_prompt, self.config.autonomous_refinement_temperature, self.config.autonomous_refinement_top_p)
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
                            let mut auto_improvement = if parallel_rouge {
                                tokio::task::spawn_blocking({
                                    let candidate = candidate.to_string();
                                    let output = output.to_string();
                                    move || rouge_l(&candidate, &output)
                                }).await.unwrap_or(0.0)
                            } else {
                                rouge_l(candidate, output)
                            };
                            
                            if auto_improvement.is_finite() {
                                quality_score = (quality_score
                                    + (auto_improvement.clamp(0.0, 1.0) * self.config.autonomous_refinement_improvement_weight) as f32)
                                    .min(1.0);
                            }
                            refined = candidate.to_string();
                            learned = auto_improvement > self.config.autonomous_refinement_improvement_threshold;
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
                                    .generate_with_params(&second_prompt, self.config.second_pass_refinement_temperature, self.config.second_pass_refinement_top_p)
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
                                                }).await.unwrap_or(0.0)
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
                                                    + (second_improvement.clamp(0.0, 1.0) * self.config.autonomous_refinement_improvement_weight)
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
                                        crate::metrics::curator_feedback_metrics().record_parameter_adjustment(&param);
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
                                quality_score = (quality_score + self.config.pipeline_quality_score_increment).min(1.0);
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
}
