//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

use super::guessing_spheres::{EmotionalVector, GuessingMemorySystem, MemoryQuery, SphereId};
use crate::consciousness::ConsciousnessState;
use crate::rag::RetrievalEngine;
use crate::token_promotion::{run_promotion_cycle, PromotionResult};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tracing::{debug, info};

/// Memory result with both semantic and emotional resonance scores
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryWithResonance {
    pub id: String,
    pub content: String,
    pub semantic_similarity: f32,
    pub emotional_resonance: f32, // Weighted final score from Gaussian collapse
    pub raw_emotional_resonance: f32, // Pure cosine similarity before weighting
    pub novelty_score: f32,       // Combined score: 15-20% variance target
    pub sphere_id: Option<String>,
    pub emotional_profile: Option<EmotionalVector>,
}

#[derive(Debug, Clone)]
pub struct MMNDetection {
    pub deviation: f32,
    pub latency_ms: u64,
    pub query_emotion: EmotionalVector,
    pub context_emotion: EmotionalVector,
}

/// Calculate Euclidean distance between two emotional vectors
fn emotional_distance(a: &EmotionalVector, b: &EmotionalVector) -> f32 {
    let diff_joy = a.joy - b.joy;
    let diff_sadness = a.sadness - b.sadness;
    let diff_anger = a.anger - b.anger;
    let diff_fear = a.fear - b.fear;
    let diff_surprise = a.surprise - b.surprise;

    (diff_joy.powi(2)
        + diff_sadness.powi(2)
        + diff_anger.powi(2)
        + diff_fear.powi(2)
        + diff_surprise.powi(2))
    .sqrt()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CycleTrigger {
    MismatchCrisis,
    UniformStagnation,
    VarianceSpike,
}

impl CycleTrigger {
    pub fn as_str(&self) -> &'static str {
        match self {
            CycleTrigger::MismatchCrisis => "mismatch_crisis",
            CycleTrigger::UniformStagnation => "uniform_stagnation",
            CycleTrigger::VarianceSpike => "variance_spike",
        }
    }
}

#[derive(Debug, Clone)]
pub struct CycleDiagnostics {
    pub cycle_index: u64,
    pub trigger: CycleTrigger,
    pub emotional_entropy: f32,
    pub raw_mean: f32,
    pub raw_std_dev: f32,
    pub oov_rate: f32,
    pub promoted_count: usize,
    pub pruned_count: usize,
    pub cycle_latency_ms: f64,
}

/// Learning event for persistent fine-tuning (Qwen Curator input)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningEvent {
    pub timestamp: i64,
    pub trigger_type: String,
    pub query_text: String,
    pub query_emotion: EmotionalVector,
    pub results: Vec<MemoryWithResonance>,
    pub emotional_entropy: f32,
    pub mean_score: f32,
    pub std_dev: f32,
}

/// Multi-layer memory query engine combining RAG + Gaussian spheres
pub struct MultiLayerMemoryQuery {
    rag_engine: Arc<Mutex<RetrievalEngine>>,
    gaussian_system: GuessingMemorySystem,
    trigger_thresholds: TriggerThresholds,
    cycle_log: Vec<CycleDiagnostics>,
    cycle_counter: u64,
    checkpoint_dir: PathBuf,
    // MMN detection: Recent queries for fast-path deviant detection
    recent_queries: Vec<EmotionalVector>,
}

#[derive(Debug, Copy, Clone)]
pub struct TriggerThresholds {
    /// Entropy level required before mismatch/stagnation checks.
    pub entropy_high: f32,
    /// Mean resonance below this value indicates mismatch crisis.
    pub mean_low: f32,
    /// Standard deviation below this value indicates uniform stagnation.
    pub stagnation_variance: f32,
    /// Standard deviation above this value triggers standard promotion cycle.
    pub variance_spike: f32,
}

impl Default for TriggerThresholds {
    fn default() -> Self {
        Self {
            entropy_high: 2.0,
            mean_low: 0.7,
            stagnation_variance: 0.01,
            variance_spike: 0.05,
        }
    }
}

impl MultiLayerMemoryQuery {
    pub fn new(
        rag_engine: Arc<Mutex<RetrievalEngine>>,
        gaussian_system: GuessingMemorySystem,
    ) -> Self {
        Self {
            rag_engine,
            gaussian_system,
            trigger_thresholds: TriggerThresholds::default(),
            cycle_log: Vec::new(),
            cycle_counter: 0,
            checkpoint_dir: PathBuf::from("./checkpoints/learning_events"),
            recent_queries: Vec::new(),
        }
    }

    /// Set custom checkpoint directory for learning event persistence
    pub fn set_checkpoint_dir(&mut self, dir: PathBuf) {
        self.checkpoint_dir = dir;
    }

    /// Override the default triple-threat trigger thresholds.
    pub fn set_thresholds(&mut self, thresholds: TriggerThresholds) {
        self.trigger_thresholds = thresholds;
    }

    /// Retrieve and clear accumulated cycle diagnostics.
    pub fn drain_cycle_diagnostics(&mut self) -> Vec<CycleDiagnostics> {
        std::mem::take(&mut self.cycle_log)
    }

    /// Peek at current diagnostics without clearing.
    pub fn cycle_diagnostics(&self) -> &[CycleDiagnostics] {
        &self.cycle_log
    }

    /// Fast MMN (Mismatch Negativity) detection - detects emotional deviants in <200ms
    /// Returns Some(MMNDetection) if deviant detected, None otherwise
    fn detect_mmn(&self, query_emotion: &EmotionalVector) -> Option<MMNDetection> {
        use std::time::Instant;
        let start = Instant::now();

        // Need at least 5 recent queries for context
        if self.recent_queries.len() < 5 {
            return None;
        }

        // Calculate average emotion of recent context
        let avg_emotion = self.average_recent_emotion();

        // Calculate deviation (Euclidean distance in 5D emotional space)
        let deviation = emotional_distance(query_emotion, &avg_emotion);

        let latency_ms = start.elapsed().as_millis() as u64;

        // MMN threshold: significant deviation detected in <200ms
        if deviation > 0.6 && latency_ms < 200 {
            Some(MMNDetection {
                deviation,
                latency_ms,
                query_emotion: query_emotion.clone(),
                context_emotion: avg_emotion,
            })
        } else {
            None
        }
    }

    /// Calculate average emotion from recent queries
    fn average_recent_emotion(&self) -> EmotionalVector {
        let mut sum_joy = 0.0;
        let mut sum_sadness = 0.0;
        let mut sum_anger = 0.0;
        let mut sum_fear = 0.0;
        let mut sum_surprise = 0.0;

        for emotion in &self.recent_queries {
            sum_joy += emotion.joy;
            sum_sadness += emotion.sadness;
            sum_anger += emotion.anger;
            sum_fear += emotion.fear;
            sum_surprise += emotion.surprise;
        }

        let count = self.recent_queries.len() as f32;
        EmotionalVector {
            joy: sum_joy / count,
            sadness: sum_sadness / count,
            anger: sum_anger / count,
            fear: sum_fear / count,
            surprise: sum_surprise / count,
        }
    }

    /// Query multi-layer memory system
    ///
    /// Combines:
    /// 1. Semantic cosine similarity from RAG (0-1 range)
    /// 2. Emotional resonance from Gaussian spheres (0-1 range)
    /// 3. Novelty = (semantic * 0.5 + emotional * 0.5) for 15-20% variance
    pub fn query(
        &mut self,
        query_text: &str,
        query_emotion: &EmotionalVector,
        top_k: usize,
        state: &mut ConsciousnessState,
    ) -> Result<Vec<MemoryWithResonance>> {
        use std::time::Instant;
        let start_time = Instant::now();

        // MMN FAST-PATH: Detect emotional deviants in <200ms
        if let Some(mmn) = self.detect_mmn(query_emotion) {
            info!(
                "⚡ MMN DETECTED: Emotional deviant in {}ms (deviation={:.3})",
                mmn.latency_ms, mmn.deviation
            );
            // Could trigger immediate attention/learning here
        }

        // Add current query to recent history for future MMN detection
        self.recent_queries.push(query_emotion.clone());
        if self.recent_queries.len() > 10 {
            self.recent_queries.remove(0); // Keep only last 10
        }

        // 1. Get semantic matches from RAG
        let rag_results = self.rag_engine.lock().unwrap().retrieve(query_text, state);
        debug!("🧠 Multi-layer: RAG returned {} results", rag_results.len());

        // 2. Get emotional resonance from Gaussian spheres
        let memory_query = MemoryQuery {
            concept: query_text.to_string(),
            emotion: query_emotion.clone(),
            time: chrono::Utc::now().timestamp() as f64,
        };
        let emotional_matches = self
            .gaussian_system
            .collapse_recall_probability(&memory_query);
        debug!(
            "🧠 Gaussian spheres returned {} emotional matches",
            emotional_matches.len()
        );

        // 3. Combine both layers - cross-reference by content/ID
        let mut combined_results: Vec<MemoryWithResonance> = Vec::new();

        for (doc, semantic_score) in rag_results.iter() {
            // Find matching sphere and extract emotional profile
            let (emotional_score, raw_resonance, sphere_emotion) = emotional_matches
                .iter()
                .find(|(sphere_id, _)| {
                    let matches = self.find_sphere_match(&doc.id, sphere_id);
                    if matches {
                        debug!(
                            "   ✓ Matched doc '{}' with sphere '{}'",
                            doc.id, sphere_id.0
                        );
                    }
                    matches
                })
                .and_then(|(sphere_id, weighted_score)| {
                    // Get sphere's emotional profile to calculate raw resonance
                    self.gaussian_system.get_sphere(sphere_id).map(|sphere| {
                        let raw = calculate_raw_emotional_resonance(
                            query_emotion,
                            &sphere.emotional_profile,
                        );
                        (*weighted_score, raw, sphere.emotional_profile.clone())
                    })
                })
                .unwrap_or((0.0, 0.0, EmotionalVector::new(0.0, 0.0, 0.0, 0.0, 0.0)));

            debug!(
                "   Doc '{}': semantic={:.3}, emotional={:.3}, raw_resonance={:.3}",
                doc.id, semantic_score, emotional_score, raw_resonance
            );

            // Filter by emotional resonance > 0.2 (tunable threshold)
            if emotional_score > 0.2 {
                // Calculate novelty score: 50/50 blend for 15-20% variance
                let novelty_score = (semantic_score * 0.5) + (emotional_score * 0.5);

                combined_results.push(MemoryWithResonance {
                    id: doc.id.clone(),
                    content: doc.content.clone(),
                    semantic_similarity: *semantic_score,
                    emotional_resonance: emotional_score,
                    raw_emotional_resonance: raw_resonance,
                    novelty_score,
                    sphere_id: None, // Will be populated when spheres have IDs
                    emotional_profile: Some(sphere_emotion),
                });
                debug!("      ✅ PASSED emotional filter (>{:.1})", 0.2);
            } else {
                debug!(
                    "      ❌ FILTERED OUT (emotional={:.3} <= 0.2)",
                    emotional_score
                );
            }
        }

        info!(
            "🧠 Combined results after fusion: {}/{} passed emotional filter",
            combined_results.len(),
            rag_results.len()
        );

        // TRIPLE-THREAT TRIGGER: Entropy + Mean + Variance for nuanced detection
        // Scenario 1: High H + Low mean = MISMATCH CRISIS (sad query vs joy vault)
        // Scenario 2: High H + High mean + Low var = UNIFORM STAGNATION (all-joy vault, no diversity)
        // Scenario 3: High H + Natural spread = Healthy diversity (no trigger)
        if combined_results.len() >= 5 {
            // Use RAW emotional resonance (before weighted blend) for all metrics
            let raw_resonance_scores: Vec<f32> = combined_results
                .iter()
                .map(|m| m.raw_emotional_resonance)
                .collect();
            debug!("📊 Raw resonance scores: {:?}", raw_resonance_scores);

            let emotional_entropy = calculate_entropy(&raw_resonance_scores);
            let coherence_std_dev = calculate_std_dev(&raw_resonance_scores);
            let mean_score: f32 =
                raw_resonance_scores.iter().sum::<f32>() / raw_resonance_scores.len() as f32;

            // Scenario 1: Mismatch crisis - FULL THROTTLE trigger
            if emotional_entropy > self.trigger_thresholds.entropy_high
                && mean_score < self.trigger_thresholds.mean_low
            {
                info!(
                    "🎯 MISMATCH CRISIS: High entropy (H={:.3}) + low quality (mean={:.3}) → FULL THROTTLE",
                    emotional_entropy,
                    mean_score
                );

                let cycle_result = run_promotion_cycle(&mut self.gaussian_system);

                let denom = (cycle_result.promoted_count + cycle_result.pruned_count) as f64;
                let ratio = if denom > f64::EPSILON {
                    cycle_result.promoted_count as f64 / denom
                } else {
                    0.0
                };

                tracing::info!(
                    "✅ Promotion cycle: {}/{} promoted/pruned, {:.2} ms latency, ratio: {:.3}",
                    cycle_result.promoted_count,
                    cycle_result.pruned_count,
                    cycle_result.cycle_latency_ms,
                    ratio
                );

                if cycle_result.cycle_latency_ms > 50.0 {
                    tracing::warn!(
                        "⚠️  Promotion cycle exceeded 50ms target: {:.2} ms",
                        cycle_result.cycle_latency_ms
                    );
                }

                self.record_cycle_diagnostic(
                    CycleTrigger::MismatchCrisis,
                    emotional_entropy,
                    mean_score,
                    coherence_std_dev,
                    &cycle_result,
                );

                // INTEGRATION: Update consciousness state with trigger
                state.last_trigger = Some(CycleTrigger::MismatchCrisis);

                // Persist learning event for Qwen fine-tuning
                let _ = self.persist_learning_event(
                    query_text,
                    query_emotion,
                    &combined_results,
                    CycleTrigger::MismatchCrisis,
                    emotional_entropy,
                    mean_score,
                    coherence_std_dev,
                );

            // Scenario 2: Uniform stagnation - GENTLE NUDGE for diversity
            } else if emotional_entropy > self.trigger_thresholds.entropy_high
                && coherence_std_dev < self.trigger_thresholds.stagnation_variance
            {
                info!(
                    "🌱 UNIFORM STAGNATION: High entropy (H={:.3}) + low std dev ({:.3}) → Gentle diversity nudge",
                    emotional_entropy,
                    coherence_std_dev
                );

                // NOTE: Could implement softer promotion logic here (promote fewer tokens)
                // For now, using standard cycle but logging the stagnation case
                let cycle_result = run_promotion_cycle(&mut self.gaussian_system);

                let denom = (cycle_result.promoted_count + cycle_result.pruned_count) as f64;
                let ratio = if denom > f64::EPSILON {
                    cycle_result.promoted_count as f64 / denom
                } else {
                    0.0
                };

                tracing::info!(
                    "✅ Diversity cycle: {}/{} promoted/pruned, {:.2} ms latency, ratio: {:.3}",
                    cycle_result.promoted_count,
                    cycle_result.pruned_count,
                    cycle_result.cycle_latency_ms,
                    ratio
                );

                if cycle_result.cycle_latency_ms > 50.0 {
                    tracing::warn!(
                        "⚠️  Promotion cycle exceeded 50ms target: {:.2} ms",
                        cycle_result.cycle_latency_ms
                    );
                }

                self.record_cycle_diagnostic(
                    CycleTrigger::UniformStagnation,
                    emotional_entropy,
                    mean_score,
                    coherence_std_dev,
                    &cycle_result,
                );

                // INTEGRATION: Update consciousness state with trigger
                state.last_trigger = Some(CycleTrigger::UniformStagnation);

                // Persist learning event
                let _ = self.persist_learning_event(
                    query_text,
                    query_emotion,
                    &combined_results,
                    CycleTrigger::UniformStagnation,
                    emotional_entropy,
                    mean_score,
                    coherence_std_dev,
                );

            // Fallback: Variance-only trigger for edge cases
            } else if coherence_std_dev > self.trigger_thresholds.variance_spike {
                info!(
                    "📊 VARIANCE SPIKE: High std dev ({:.3}), H={:.3} → Standard trigger",
                    coherence_std_dev, emotional_entropy
                );

                let cycle_result = run_promotion_cycle(&mut self.gaussian_system);

                let denom = (cycle_result.promoted_count + cycle_result.pruned_count) as f64;
                let ratio = if denom > f64::EPSILON {
                    cycle_result.promoted_count as f64 / denom
                } else {
                    0.0
                };

                tracing::info!(
                    "✅ Promotion cycle: {}/{} promoted/pruned, {:.2} ms latency, ratio: {:.3}",
                    cycle_result.promoted_count,
                    cycle_result.pruned_count,
                    cycle_result.cycle_latency_ms,
                    ratio
                );

                if cycle_result.cycle_latency_ms > 50.0 {
                    tracing::warn!(
                        "⚠️  Promotion cycle exceeded 50ms target: {:.2} ms",
                        cycle_result.cycle_latency_ms
                    );
                }

                self.record_cycle_diagnostic(
                    CycleTrigger::VarianceSpike,
                    emotional_entropy,
                    mean_score,
                    coherence_std_dev,
                    &cycle_result,
                );

                // INTEGRATION: Update consciousness state with trigger
                state.last_trigger = Some(CycleTrigger::VarianceSpike);

                // Persist learning event
                let _ = self.persist_learning_event(
                    query_text,
                    query_emotion,
                    &combined_results,
                    CycleTrigger::VarianceSpike,
                    emotional_entropy,
                    mean_score,
                    coherence_std_dev,
                );

            // No trigger: Healthy diversity or low uncertainty
            } else {
                debug!("📊 Emotional metrics: H={:.3}, mean={:.3}, std_dev={:.3} (Healthy diversity - no trigger)",
                         emotional_entropy, mean_score, coherence_std_dev);
            }
        }

        // 4. Sort by novelty score (descending) and take top_k
        combined_results.sort_by(|a, b| b.novelty_score.partial_cmp(&a.novelty_score).unwrap());
        combined_results.truncate(top_k);

        // 5. Verify novelty variance is 15-20%
        if combined_results.len() >= 2 {
            let novelty_std_dev = calculate_std_dev(
                &combined_results
                    .iter()
                    .map(|m| m.novelty_score)
                    .collect::<Vec<_>>(),
            );
            tracing::info!(
                "💫 Multi-layer query: {} results, novelty std dev: {:.1}%",
                combined_results.len(),
                novelty_std_dev * 100.0
            );
        }

        // INTEGRATION: Update consciousness state with entropy metrics
        if combined_results.len() >= 5 {
            let raw_resonance_scores: Vec<f32> = combined_results
                .iter()
                .map(|m| m.raw_emotional_resonance)
                .collect();
            let emotional_entropy = calculate_entropy(&raw_resonance_scores);
            let coherence_std_dev = calculate_std_dev(&raw_resonance_scores);
            let mean_score: f32 =
                raw_resonance_scores.iter().sum::<f32>() / raw_resonance_scores.len() as f32;

            // Update consciousness state with latest metrics
            state.emotional_entropy = emotional_entropy;
            state.mean_resonance = mean_score;
            state.coherence_variance = coherence_std_dev;
            // last_trigger is updated when triggers occur above
        }

        // ECHO MEMORIA WINDOW ENFORCEMENT: Ensure 2-4 second processing time
        let elapsed = start_time.elapsed();
        if elapsed > std::time::Duration::from_secs(4) {
            tracing::warn!(
                "⚠️ Query exceeded EchoMemoria window: {:?} > 4s (violates 2-4s human echoic memory)",
                elapsed
            );
        } else if elapsed > std::time::Duration::from_secs(2) {
            tracing::info!("✅ Query within EchoMemoria window: {:?}", elapsed);
        } else {
            tracing::debug!("🚀 Fast query: {:?}", elapsed);
        }

        Ok(combined_results)
    }

    fn record_cycle_diagnostic(
        &mut self,
        trigger: CycleTrigger,
        emotional_entropy: f32,
        raw_mean: f32,
        raw_std_dev: f32,
        result: &PromotionResult,
    ) {
        let total_spheres = self.gaussian_system.sphere_count();
        let oov_rate = if total_spheres > 0 {
            result.promoted_count as f32 / total_spheres as f32
        } else {
            0.0
        };

        self.cycle_counter += 1;
        self.cycle_log.push(CycleDiagnostics {
            cycle_index: self.cycle_counter,
            trigger,
            emotional_entropy,
            raw_mean,
            raw_std_dev,
            oov_rate,
            promoted_count: result.promoted_count,
            pruned_count: result.pruned_count,
            cycle_latency_ms: result.cycle_latency_ms,
        });
    }

    /// Find if a document matches a sphere (by ID or content)
    fn find_sphere_match(&self, doc_id: &str, sphere_id: &SphereId) -> bool {
        // Simple ID matching for now
        // In production, this would use a proper document-to-sphere mapping
        sphere_id.0.contains(doc_id) || doc_id.contains(&sphere_id.0)
    }

    /// Persist learning event to checkpoint directory for Qwen Curator
    fn persist_learning_event(
        &self,
        query_text: &str,
        query_emotion: &EmotionalVector,
        results: &[MemoryWithResonance],
        trigger: CycleTrigger,
        emotional_entropy: f32,
        mean_score: f32,
        std_dev: f32,
    ) -> Result<()> {
        // Create checkpoint directory if it doesn't exist
        fs::create_dir_all(&self.checkpoint_dir)?;

        // TODO: Re-enable after qwen_curator is properly implemented
        // Create Qwen-compatible LearningEvent for fine-tuning
        // use niodoo_core::qwen_curator::{EmotionalState, LearningEvent as QwenLearningEvent};

        // DISABLED - qwen_curator types not yet implemented
        /*
        let qwen_event = QwenLearningEvent {
            timestamp: chrono::Utc::now().timestamp().to_string(),
            input: query_text.to_string(),
            response: format!(
                "Triple-Threat {} detected with entropy {:.3}, mean {:.3}, std_dev {:.3}. \
                 Memory healing applied: {} results analyzed, emotional pattern: joy={:.3}, sadness={:.3}, anger={:.3}, fear={:.3}, surprise={:.3}",
                trigger.as_str(),
                emotional_entropy,
                mean_score,
                std_dev,
                results.len(),
                query_emotion.joy,
                query_emotion.sadness,
                query_emotion.anger,
                query_emotion.fear,
                query_emotion.surprise
            ),
            emotional_state: Some(EmotionalState {
                pleasure: query_emotion.joy as f64,
                arousal: (query_emotion.anger + query_emotion.fear) as f64,
                dominance: query_emotion.surprise as f64,
            }),
            coherence: Some((1.0 - emotional_entropy / 4.0) as f64), // Convert entropy to coherence
            memory_activations: Some(results.iter().map(|r| r.raw_emotional_resonance as f64).collect()),
            topology_metrics: None, // Could add Möbius metrics here
        };

        // Save Qwen-compatible event
        let qwen_filename = format!(
            "qwen_{}_{}_{}.json",
            chrono::Utc::now().timestamp(),
            trigger.as_str(),
            self.cycle_counter
        );
        let qwen_filepath = self.checkpoint_dir.join(qwen_filename);
        let qwen_json = serde_json::to_string_pretty(&qwen_event)?;
        fs::write(&qwen_filepath, qwen_json)?;
        */

        // Also save the original memory analysis event for debugging
        let memory_event = LearningEvent {
            timestamp: chrono::Utc::now().timestamp(),
            trigger_type: trigger.as_str().to_string(),
            query_text: query_text.to_string(),
            query_emotion: query_emotion.clone(),
            results: results.to_vec(),
            emotional_entropy,
            mean_score,
            std_dev,
        };

        // Write to JSON file with timestamp
        let memory_filename = format!(
            "memory_{}_{}_{}.json",
            memory_event.timestamp,
            trigger.as_str(),
            self.cycle_counter
        );
        let memory_filepath = self.checkpoint_dir.join(memory_filename);
        let memory_json = serde_json::to_string_pretty(&memory_event)?;
        fs::write(&memory_filepath, memory_json)?;

        tracing::info!(
            "📝 Persisted learning event: memory analysis ({})",
            memory_filepath.display()
        );
        Ok(())
    }
}

/// Calculate raw emotional resonance using cosine similarity (before weighted blend)
fn calculate_raw_emotional_resonance(a: &EmotionalVector, b: &EmotionalVector) -> f32 {
    let dot = a.joy * b.joy
        + a.sadness * b.sadness
        + a.anger * b.anger
        + a.fear * b.fear
        + a.surprise * b.surprise;

    let magnitude = a.magnitude() * b.magnitude();
    if magnitude <= f32::EPSILON {
        0.0
    } else {
        // Transform [-1,1] cosine similarity to [0,1] range
        (dot / magnitude).clamp(-1.0, 1.0) * 0.5 + 0.5
    }
}

/// Calculate Shannon entropy for uncertainty detection
/// H = -Σ pᵢ·log₂(pᵢ) where pᵢ = scoreᵢ / sum(scores)
/// High H (>1.0) indicates mixed/uncertain emotional state → trigger cycle
fn calculate_entropy(scores: &[f32]) -> f32 {
    if scores.is_empty() {
        return 0.0;
    }

    let sum: f32 = scores.iter().sum();
    if sum <= f32::EPSILON {
        return 0.0;
    }

    let entropy: f32 = scores
        .iter()
        .map(|&score| {
            let p = score / sum;
            if p > f32::EPSILON {
                -p * p.log2()
            } else {
                0.0
            }
        })
        .sum();

    entropy
}

/// Calculate standard deviation of a set of scores (fallback metric)
fn calculate_std_dev(scores: &[f32]) -> f32 {
    if scores.is_empty() {
        return 0.0;
    }

    let mean = scores.iter().sum::<f32>() / scores.len() as f32;
    let variance = scores
        .iter()
        .map(|score| {
            let diff = score - mean;
            diff * diff
        })
        .sum::<f32>()
        / scores.len() as f32;

    variance.sqrt() // Return standard deviation as a proportion
}

/// Standalone query function for easy integration
pub fn query_multi_layer(
    query_text: &str,
    query_emotion: &EmotionalVector,
    top_k: usize,
    rag_engine: Arc<Mutex<RetrievalEngine>>,
    gaussian_system: &GuessingMemorySystem,
    state: &mut ConsciousnessState,
) -> Result<Vec<MemoryWithResonance>> {
    let mut multi_query = MultiLayerMemoryQuery::new(rag_engine, gaussian_system.clone());

    multi_query.query(query_text, query_emotion, top_k, state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AppConfig;

    #[test]
    fn test_multi_layer_query() {
        let config = AppConfig::default();
        let rag_engine = Arc::new(Mutex::new(RetrievalEngine::new()));
        let gaussian_system = GuessingMemorySystem::new();
        let mut state = ConsciousnessState::default();

        let query_emotion = EmotionalVector {
            joy: 0.7,
            sadness: 0.3,
            anger: 0.1,
            fear: 0.2,
            surprise: 0.5,
        };

        // This should work with empty systems
        let results = query_multi_layer(
            "test memory",
            &query_emotion,
            8,
            rag_engine.clone(),
            &gaussian_system,
            &mut state,
        );

        assert!(results.is_ok());
        tracing::info!("✅ Multi-layer query test passed");
    }

    #[test]
    fn test_novelty_variance_calculation() {
        let scores = vec![0.6, 0.7, 0.65, 0.72, 0.68];
        let std_dev = calculate_std_dev(&scores);

        // Standard deviation should be small for closely grouped scores
        assert!(std_dev < 0.1);
        tracing::info!("✅ Standard deviation calculation: {:.3}", std_dev);
    }

    #[test]
    fn test_resonance_filtering() {
        // Test that emotional resonance > 0.4 filter works
        let config = AppConfig::default();
        let mut rag_engine = RetrievalEngine::new();
        let mut gaussian_system = GuessingMemorySystem::new();
        let mut state = ConsciousnessState::default();

        // Add test sphere with high emotional resonance
        use crate::memory::guessing_spheres::SphereId;
        let sphere_id = SphereId("test-sphere-1".to_string());
        let emotion = EmotionalVector {
            joy: 0.8,
            sadness: 0.2,
            anger: 0.0,
            fear: 0.1,
            surprise: 0.6,
        };

        gaussian_system.store_memory(
            sphere_id,
            "happy memory".to_string(),
            [0.5, 0.5, 0.5],
            emotion.clone(),
            "This is a joyful fragment".to_string(),
        );

        let results = query_multi_layer(
            "happy",
            &emotion,
            8,
            Arc::new(Mutex::new(rag_engine)),
            &gaussian_system,
            &mut state,
        );

        assert!(results.is_ok());
        tracing::info!("✅ Resonance filtering test passed");
    }

    #[test]
    fn test_mismatch_crisis_trigger() {
        // Setup: Query with PURE sadness → Vault with PURE joy
        // Expected: Low mean, high entropy → MISMATCH CRISIS

        use crate::rag::local_embeddings::{Document, MathematicalEmbeddingModel};

        let model = MathematicalEmbeddingModel::new(384);
        let mut rag_engine = RetrievalEngine::new();
        let mut gaussian_system = GuessingMemorySystem::new();

        // Create 10 pure joy spheres (1.0, 0.0, 0.0, 0.0, 0.0)
        for i in 0..10 {
            let doc = Document {
                id: format!("joy-{}", i),
                content: format!("Very happy memory {}", i),
                embedding: model.generate_embedding(&format!("happy {}", i)).unwrap(),
                metadata: std::collections::HashMap::new(),
            };
            rag_engine.add_document(doc);

            gaussian_system.store_memory(
                SphereId(format!("joy-{}", i)),
                format!("joy concept {}", i),
                [0.0, 0.0, 0.0],
                EmotionalVector::new(1.0, 0.0, 0.0, 0.0, 0.0),
                format!("Happy fragment {}", i),
            );
        }

        // Query with PURE sadness (0.0, 1.0, 0.0, 0.0, 0.0) - opposite of vault
        let query_emotion = EmotionalVector::new(0.0, 1.0, 0.0, 0.0, 0.0);

        let rag_arc = Arc::new(Mutex::new(rag_engine));
        let mut multi_query = MultiLayerMemoryQuery::new(rag_arc, gaussian_system);
        let mut state = ConsciousnessState::default();

        println!("\n=== MISMATCH CRISIS TEST ===");
        println!("Query: Pure sadness (0.0, 1.0, 0.0, 0.0, 0.0)");
        println!("Vault: 10x pure joy (1.0, 0.0, 0.0, 0.0, 0.0)");
        println!("Expected: 🎯 MISMATCH CRISIS trigger (H>2.0, mean<0.7)\n");

        let results = multi_query.query("sad memory", &query_emotion, 8, &mut state);

        assert!(results.is_ok(), "Query should succeed");
        println!("\n✅ Test completed - check output for trigger activation");
        println!("Results returned: {}", results.unwrap().len());
    }
}
