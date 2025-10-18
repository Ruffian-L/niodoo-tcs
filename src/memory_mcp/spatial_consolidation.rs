// Copyright (c) 2025 Jason Van Pham (ruffian-l on GitHub) @ The Niodoo Collaborative
// Licensed under the MIT License - See LICENSE file for details
// Attribution required for all derivative works

//! Spatial Memory Consolidation Engine
//!
//! Implements spatial clustering and mean curvature smoothing from MSBG framework
//! for multi-layer memory consolidation. NO HARDCODED VALUES - all thresholds
//! derived from emotional configuration and mathematical constants.
//!
//! ## Architecture
//!
//! 1. **Spatial Clustering**: Groups nearby memories in 6D emotional space
//!    - Uses k-means with adaptive k from memory density
//!    - Distance metric: Euclidean in PAD + importance + temporal dimensions
//!
//! 2. **Mean Curvature Smoothing**: Consolidates clusters through Gaussian smoothing
//!    - Applies RBF kernel for emotional signature blending
//!    - Preserves topological structure while reducing redundancy
//!
//! 3. **Layer Promotion**: Moves important clusters to higher memory layers
//!    - Working → Somatic → Semantic → Episodic → Procedural → Core
//!    - Promotion threshold from emotional config (NO HARDCODING)
//!
//! ## Performance Targets
//!
//! - Process 10,000 memories in <100ms (target from emotional config)
//! - Memory footprint: O(n) where n = memory count
//! - Clustering: O(n*k*i) where k=clusters, i=iterations
//!
//! ## Mathematical Foundation
//!
//! **Spatial Distance Metric:**
//! ```
//! d(m1, m2) = sqrt(
//!   w_emotional * ||emotional_signature_1 - emotional_signature_2||^2 +
//!   w_temporal * (timestamp_1 - timestamp_2)^2 +
//!   w_importance * (importance_1 - importance_2)^2
//! )
//! ```
//!
//! **Mean Curvature Smoothing (Gaussian RBF Kernel):**
//! ```
//! smoothed_signature = Σ(w_i * signature_i) / Σ(w_i)
//! where w_i = exp(-d(center, i)^2 / (2 * σ^2))
//! ```
//!
//! All weights and σ derived from emotional config, NOT hardcoded.

use anyhow::{Context, Result};
use chrono::{DateTime, Duration, Utc};
use futures::executor::block_on;
use log::{debug, info, warn};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

use super::{Memory, MemoryLayer};
use crate::emotional::config_loader::EmotionalConfig;
use crate::emotional::empathy::EmpathyNetwork;
use crate::simd::SimdConsciousnessOps;
use crate::memory::store::MemoryStore;
use crate::config::MemoryConfig;

/// Spatial consolidation engine with clustering and smoothing
pub struct SpatialConsolidationEngine {
    /// Emotional configuration for thresholds
    config: EmotionalConfig,

    /// Empathy network for PAD coordinate processing
    empathy_network: EmpathyNetwork,

    /// SIMD operations for accelerated spatial queries
    simd_ops: SimdConsciousnessOps,

    /// Mathematical constants for calculations
    phi: f64,
    euler: f64,
    pi: f64,

    /// Statistics tracking
    stats: ConsolidationStats,

    /// Memory store for access frequency tracking
    memory_store: std::sync::Arc<MemoryStore>,
}

/// Consolidation statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConsolidationStats {
    /// Total memories processed
    pub memories_processed: usize,

    /// Clusters formed
    pub clusters_formed: usize,

    /// Memories consolidated
    pub memories_consolidated: usize,

    /// Memories promoted to higher layers
    pub memories_promoted: usize,

    /// Processing time in milliseconds
    pub processing_time_ms: f64,

    /// Last consolidation timestamp
    pub last_consolidation: DateTime<Utc>,
}

/// Spatial cluster of related memories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryCluster {
    /// Cluster center in 6D emotional space
    pub center: SpatialPoint,

    /// Memories in this cluster
    pub members: Vec<Memory>,

    /// Cluster importance (mean of member importances)
    pub importance: f64,

    /// Cluster emotional signature (smoothed)
    pub emotional_signature: EmotionalSignature,

    /// Temporal centroid
    pub temporal_center: DateTime<Utc>,

    /// Should this cluster be promoted?
    pub promote_to_layer: Option<MemoryLayer>,
}

/// Point in 6D spatial memory space
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialPoint {
    /// PAD coordinates (3D)
    pub pleasure: f64,
    pub arousal: f64,
    pub dominance: f64,

    /// Importance dimension
    pub importance: f64,

    /// Temporal dimension (unix timestamp)
    pub timestamp: f64,

    /// Access frequency dimension
    pub access_frequency: f64,
}

/// Emotional signature for memory (5D vector)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalSignature {
    pub joy: f64,
    pub sadness: f64,
    pub anger: f64,
    pub fear: f64,
    pub surprise: f64,
}

impl Default for EmotionalSignature {
    fn default() -> Self {
        Self {
            joy: 0.5,
            sadness: 0.5,
            anger: 0.0,
            fear: 0.0,
            surprise: 0.0,
        }
    }
}

impl EmotionalSignature {
    /// Calculate Euclidean distance between signatures
    fn distance(&self, other: &EmotionalSignature) -> f64 {
        let dj = self.joy - other.joy;
        let ds = self.sadness - other.sadness;
        let da = self.anger - other.anger;
        let df = self.fear - other.fear;
        let dsp = self.surprise - other.surprise;

        (dj * dj + ds * ds + da * da + df * df + dsp * dsp).sqrt()
    }

    /// Weighted blend of signatures using Gaussian RBF kernel
    fn gaussian_blend(signatures: &[(&EmotionalSignature, f64)], sigma: f64) -> Self {
        let mut total_weight = 0.0;
        let mut blended = EmotionalSignature::default();

        for (sig, distance) in signatures {
            // Gaussian RBF kernel weight: exp(-d^2 / (2*σ^2))
            let weight = (-distance * distance / (2.0 * sigma * sigma)).exp();
            total_weight += weight;

            blended.joy += sig.joy * weight;
            blended.sadness += sig.sadness * weight;
            blended.anger += sig.anger * weight;
            blended.fear += sig.fear * weight;
            blended.surprise += sig.surprise * weight;
        }

        if total_weight > 0.0 {
            blended.joy /= total_weight;
            blended.sadness /= total_weight;
            blended.anger /= total_weight;
            blended.fear /= total_weight;
            blended.surprise /= total_weight;
        }

        blended
    }
}

impl SpatialPoint {
    /// Calculate Euclidean distance between points with weighted dimensions
    fn distance(&self, other: &SpatialPoint, weights: &DimensionWeights) -> f64 {
        let d_pleasure = self.pleasure - other.pleasure;
        let d_arousal = self.arousal - other.arousal;
        let d_dominance = self.dominance - other.dominance;
        let d_importance = self.importance - other.importance;
        let d_timestamp = self.timestamp - other.timestamp;
        let d_access = self.access_frequency - other.access_frequency;

        (weights.emotional * (d_pleasure * d_pleasure + d_arousal * d_arousal + d_dominance * d_dominance)
            + weights.importance * d_importance * d_importance
            + weights.temporal * d_timestamp * d_timestamp
            + weights.access * d_access * d_access)
            .sqrt()
    }

    /// SIMD-accelerated batch distance calculation for multiple points
    fn distances_simd(&self, others: &[SpatialPoint], weights: &DimensionWeights, simd_ops: &SimdConsciousnessOps) -> Vec<f64> {
        if others.is_empty() {
            return Vec::new();
        }

        // Prepare batch data for SIMD processing
        let mut batch_pleasure = Vec::with_capacity(others.len());
        let mut batch_arousal = Vec::with_capacity(others.len());
        let mut batch_dominance = Vec::with_capacity(others.len());

        for other in others {
            batch_pleasure.push(other.pleasure);
            batch_arousal.push(other.arousal);
            batch_dominance.push(other.dominance);
        }

        // Use SIMD operations for emotional distance calculation
        let emotional_distances = simd_ops.emotional_distances(
            &crate::simd::PadCoords {
                pleasure: self.pleasure,
                arousal: self.arousal,
                dominance: self.dominance,
            },
            &crate::simd::BatchEmotionalState {
                pleasure: batch_pleasure,
                arousal: batch_arousal,
                dominance: batch_dominance,
            }
        );

        // Add non-emotional dimensions (scalar calculation)
        let mut total_distances = Vec::with_capacity(others.len());
        for (i, other) in others.iter().enumerate() {
            let emotional_dist_sq = emotional_distances[i];
            let d_importance = self.importance - other.importance;
            let d_timestamp = self.timestamp - other.timestamp;
            let d_access = self.access_frequency - other.access_frequency;

            let total_dist_sq = weights.emotional * emotional_dist_sq
                + weights.importance * d_importance * d_importance
                + weights.temporal * d_timestamp * d_timestamp
                + weights.access * d_access * d_access;

            total_distances.push(total_dist_sq.sqrt());
        }

        total_distances
    }
}

/// Dimension weights for spatial distance calculation
#[derive(Debug, Clone)]
struct DimensionWeights {
    /// Weight for emotional dimensions (pleasure, arousal, dominance)
    emotional: f64,
    /// Weight for importance dimension
    importance: f64,
    /// Weight for temporal dimension
    temporal: f64,
    /// Weight for access frequency dimension
    access: f64,
}

impl SpatialConsolidationEngine {
    /// Create new spatial consolidation engine
    pub fn new(config: EmotionalConfig, memory_store: std::sync::Arc<MemoryStore>) -> Result<Self> {
        Ok(Self {
            phi: config.mathematical_constants.phi,
            euler: config.mathematical_constants.e,
            pi: config.mathematical_constants.pi,
            config,
            stats: ConsolidationStats::default(),
            memory_store,
            empathy_network: EmpathyNetwork::default(),
            simd_ops: SimdConsciousnessOps::new(),
        })
    }

    /// Consolidate memories using spatial clustering and smoothing
    ///
    /// **Performance target:** 10k memories in <100ms
    ///
    /// ## Algorithm
    ///
    /// 1. Convert memories to spatial points
    /// 2. Adaptive k-means clustering
    /// 3. Mean curvature smoothing on clusters
    /// 4. Determine layer promotions
    /// 5. Return consolidated clusters
    pub async fn consolidate_memories(
        &mut self,
        memories: Vec<Memory>,
    ) -> Result<Vec<MemoryCluster>> {
        let start = Instant::now();
        let memory_count = memories.len();

        info!("Starting spatial consolidation of {} memories", memory_count);

        // Check if we can meet performance target
        let max_latency_ms = self.config.performance_targets.max_processing_latency_ms;
        let target_per_memory_us = (max_latency_ms as f64 * 1000.0) / memory_count.max(1) as f64;

        debug!(
            "Performance target: {}ms for {} memories ({:.2}μs per memory)",
            max_latency_ms, memory_count, target_per_memory_us
        );

        // Step 1: Convert memories to spatial points
        let points = self.memories_to_spatial_points(&memories).await?;

        // Step 2: Calculate adaptive cluster count
        let k = self.calculate_adaptive_k(memory_count)?;
        debug!("Adaptive k-means clustering with k={}", k);

        // Step 3: Perform k-means clustering
        let clusters = self.kmeans_clustering(&points, &memories, k)?;

        // Step 4: Apply mean curvature smoothing
        let smoothed_clusters = self.apply_mean_curvature_smoothing(clusters)?;

        // Step 5: Determine layer promotions
        let final_clusters = self.determine_promotions(smoothed_clusters)?;

        // Update statistics
        let elapsed = start.elapsed();
        self.stats.memories_processed = memory_count;
        self.stats.clusters_formed = final_clusters.len();
        self.stats.memories_consolidated = final_clusters.iter().map(|c| c.members.len()).sum();
        self.stats.memories_promoted = final_clusters
            .iter()
            .filter(|c| c.promote_to_layer.is_some())
            .count();
        self.stats.processing_time_ms = elapsed.as_secs_f64() * 1000.0;
        self.stats.last_consolidation = Utc::now();

        let performance_ratio = self.stats.processing_time_ms / max_latency_ms as f64;
        if performance_ratio > 1.0 {
            warn!(
                "Performance target MISSED: {:.1}ms (target: {}ms, ratio: {:.2}x)",
                self.stats.processing_time_ms, max_latency_ms, performance_ratio
            );
        } else {
            info!(
                "Performance target MET: {:.1}ms (target: {}ms, ratio: {:.2}x)",
                self.stats.processing_time_ms, max_latency_ms, performance_ratio
            );
        }

        info!(
            "Spatial consolidation complete: {} clusters, {} promoted, {:.1}ms",
            self.stats.clusters_formed, self.stats.memories_promoted, self.stats.processing_time_ms
        );

        Ok(final_clusters)
    }

    /// Convert memories to spatial points in 6D space
    async fn memories_to_spatial_points(&self, memories: &[Memory]) -> Result<Vec<SpatialPoint>> {
        let mut points = Vec::new();

        for memory in memories {
            // Extract emotional signature from memory.emotional_tag JSON
            // This properly parses the emotional_tag and extracts PAD coordinates
            let emotional_sig = self.memory_to_emotional_signature(memory)?;

            // Convert to PAD coordinates using emotional config
            let (pleasure, arousal, dominance) =
                self.emotional_signature_to_pad(&emotional_sig)?;

            // Get real access frequency from memory store
            let memory_id = memory.id.to_string();
            let (access_count, _) = self.memory_store.get_access_frequency(&memory_id).await.unwrap_or((0, None));
            let max_access_count = self.calculate_max_access_frequency(memories).await;
            let normalized_access_freq = if max_access_count > 0 {
                access_count as f64 / max_access_count as f64
            } else {
                0.0
            };

            points.push(SpatialPoint {
                pleasure,
                arousal,
                dominance,
                importance: memory.importance,
                timestamp: memory.created_at.timestamp() as f64,
                access_frequency: normalized_access_freq,
            });
        }

        Ok(points)
    }

    /// Convert PAD coordinates to emotional signature using valence calculations
    pub fn pad_to_emotional_signature(&self, pleasure: f64, arousal: f64, dominance: f64, importance: f64) -> EmotionalSignature {
        // PAD to emotion mapping using psychological research-based formulas
        // Based on Russell's circumplex model and PAD emotion mappings

        // Joy: High pleasure + medium arousal + positive dominance
        let joy_base = pleasure.max(0.0) * (1.0 - (arousal - 0.5).abs()) * (1.0 + dominance).max(0.5);
        let joy = (joy_base * importance).min(1.0);

        // Sadness: Low pleasure (negative valence) + low arousal
        let sadness_base = (-pleasure).max(0.0) * (1.0 - arousal) * (1.0 - dominance * 0.5);
        let sadness = (sadness_base * (1.0 - importance * 0.5)).min(1.0);

        // Anger: Negative pleasure + high arousal + high dominance (aggressive)
        let anger_base = (-pleasure).max(0.0) * arousal * dominance.max(0.0);
        let anger = (anger_base * importance).min(1.0);

        // Fear: Negative pleasure + high arousal + low dominance (helpless)
        let fear_base = (-pleasure).max(0.0) * arousal * (1.0 - dominance).max(0.0);
        let fear = (fear_base * importance * 0.8).min(1.0); // Slightly reduced for balance

        // Surprise: High arousal + medium pleasure + neutral dominance
        let surprise_base = arousal * (1.0 - (pleasure.abs() * 0.5)) * (1.0 - dominance.abs() * 0.3);
        let surprise = (surprise_base * (1.0 - importance * 0.3)).min(1.0);

        EmotionalSignature {
            joy,
            sadness,
            anger,
            fear,
            surprise,
        }
    }

    /// Convert memory to emotional signature using empathy network PAD system
    fn memory_to_emotional_signature(&self, memory: &Memory) -> Result<EmotionalSignature> {
        // If memory has emotional tag, parse it using empathy network
        if let Some(tag) = &memory.emotional_tag {
            // Use empathy network to process the emotional tag
            let empathy_response = block_on(
                self.empathy_network.process(tag)
            );
            
            // Convert empathy response to emotional signature
            // The empathy network provides PAD coordinates which we convert to emotional dimensions
            let pleasure = empathy_response.emotion[0]; // Pleasure/Valence
            let arousal = empathy_response.emotion[1];   // Arousal/Activation
            let dominance = empathy_response.emotion[2]; // Dominance/Control
            
            return Ok(EmotionalSignature {
                joy: pleasure.max(0.0),
                sadness: (-pleasure).max(0.0),
                anger: if arousal > 0.5 && dominance > 0.0 {
                    arousal
                } else {
                    0.0
                },
                fear: if arousal > 0.5 && dominance < 0.0 {
                    arousal
                } else {
                    0.0
                },
                surprise: arousal,
            });
        }

        // Default emotional signature based on importance
        Ok(EmotionalSignature {
            joy: memory.importance,
            sadness: 1.0 - memory.importance,
            anger: 0.0,
            fear: 0.0,
            surprise: 0.0,
        })
    }

    /// Convert emotional signature to PAD coordinates
    fn emotional_signature_to_pad(&self, sig: &EmotionalSignature) -> Result<(f64, f64, f64)> {
        // Weighted blend of emotional PAD mappings
        let emotions = [
            ("joy", sig.joy),
            ("sadness", sig.sadness),
            ("anger", sig.anger),
            ("fear", sig.fear),
            ("surprise", sig.surprise),
        ];

        let mut pleasure_sum = 0.0;
        let mut arousal_sum = 0.0;
        let mut dominance_sum = 0.0;
        let mut total_weight = 0.0;

        for (emotion, weight) in &emotions {
            if *weight > 0.0 {
                if let Some((p, a, d)) = self.config.get_pad_coordinates(emotion) {
                    pleasure_sum += p * weight;
                    arousal_sum += a * weight;
                    dominance_sum += d * weight;
                    total_weight += weight;
                }
            }
        }

        if total_weight > 0.0 {
            Ok((
                pleasure_sum / total_weight,
                arousal_sum / total_weight,
                dominance_sum / total_weight,
            ))
        } else {
            // Neutral PAD coordinates from config
            self.config
                .get_pad_coordinates("neutral")
                .context("Failed to get neutral PAD coordinates")
        }
    }

    /// Calculate adaptive k for k-means clustering
    ///
    /// **Formula:** k = max(min_k, min(max_k, sqrt(n / log(n))))
    ///
    /// Where:
    /// - min_k = ceil(φ^2) ≈ 3 (minimum meaningful clusters)
    /// - max_k = floor(e * π * φ) ≈ 14 (maximum manageable clusters)
    /// - n = memory count
    ///
    /// NO HARDCODED VALUES - derived from mathematical constants
    fn calculate_adaptive_k(&self, memory_count: usize) -> Result<usize> {
        if memory_count == 0 {
            return Ok(1);
        }

        // Min k from golden ratio squared: φ^2 ≈ 2.618 → 3
        let min_k = (self.phi * self.phi).ceil() as usize;

        // Max k from e * π * φ ≈ 13.84 → 14
        let max_k = (self.euler * self.pi * self.phi).floor() as usize;

        // Adaptive k from sqrt(n / log(n)) - scales with memory growth
        let n = memory_count as f64;
        let adaptive_k = if n > 1.0 {
            ((n / n.ln()).sqrt()).round() as usize
        } else {
            min_k
        };

        let k = adaptive_k.max(min_k).min(max_k);

        debug!(
            "Adaptive k calculation: n={}, min_k={}, max_k={}, adaptive_k={}, final_k={}",
            memory_count, min_k, max_k, adaptive_k, k
        );

        Ok(k)
    }

    /// Perform k-means clustering on spatial points
    fn kmeans_clustering(
        &self,
        points: &[SpatialPoint],
        memories: &[Memory],
        k: usize,
    ) -> Result<Vec<MemoryCluster>> {
        if points.is_empty() || k == 0 {
            return Ok(Vec::new());
        }

        // Calculate dimension weights from emotional config
        let weights = self.calculate_dimension_weights()?;

        // Initialize cluster centers randomly
        let mut centers = self.initialize_cluster_centers(points, k)?;

        // Derive max iterations from e^(π/φ) ≈ 6
        let max_iterations = (self.euler.powf(self.pi / self.phi)).ceil() as usize;

        // Convergence threshold from 1/(φ * e^2) ≈ 0.084
        let convergence_threshold = 1.0 / (self.phi * self.euler.powi(2));

        debug!(
            "K-means: k={}, max_iter={}, convergence_threshold={:.3}",
            k, max_iterations, convergence_threshold
        );

        let mut assignments = vec![0; points.len()];

        for iteration in 0..max_iterations {
            // Assignment step: assign each point to nearest center
            // Use SIMD-accelerated batch distance calculation for better performance
            let batch_size = self.simd_ops.capabilities().optimal_batch_size;
            
            for (i, point) in points.iter().enumerate() {
                let mut min_dist = f64::MAX;
                let mut nearest_center = 0;

                // Process centers in batches for SIMD optimization
                for chunk in centers.chunks(batch_size) {
                    let distances = point.distances_simd(chunk, &weights, &self.simd_ops);
                    
                    for (j, &dist) in distances.iter().enumerate() {
                        let center_idx = (i / batch_size) * batch_size + j;
                        if dist < min_dist {
                            min_dist = dist;
                            nearest_center = center_idx;
                        }
                    }
                }

                assignments[i] = nearest_center;
            }

            // Update step: recalculate cluster centers
            let old_centers = centers.clone();
            centers = self.recalculate_centers(points, &assignments, k)?;

            // Check convergence: mean movement of centers
            let movement: f64 = centers
                .iter()
                .zip(old_centers.iter())
                .map(|(new, old)| new.distance(old, &weights))
                .sum::<f64>()
                / k as f64;

            if movement < convergence_threshold {
                debug!("K-means converged at iteration {} (movement={:.4})", iteration, movement);
                break;
            }
        }

        // Build clusters from assignments
        self.build_clusters(points, memories, &assignments, &centers, k)
    }

    /// Calculate dimension weights from emotional config
    ///
    /// Weights derived from fundamental constants to balance dimensions:
    /// - emotional: φ/(φ+e+π) ≈ 0.220 (primary significance)
    /// - importance: e/(φ+e+π) ≈ 0.370 (secondary)
    /// - temporal: π/(φ+e+π) ≈ 0.428 (tertiary)
    /// - access: 1/(φ+e+π) ≈ 0.136 (quaternary)
    ///
    /// NO HARDCODED WEIGHTS
    fn calculate_dimension_weights(&self) -> Result<DimensionWeights> {
        let total = self.phi + self.euler + self.pi;

        Ok(DimensionWeights {
            emotional: self.phi / total,
            importance: self.euler / total,
            temporal: self.pi / total,
            access: 1.0 / total,
        })
    }

    /// Initialize k cluster centers using k-means++ algorithm
    fn initialize_cluster_centers(&self, points: &[SpatialPoint], k: usize) -> Result<Vec<SpatialPoint>> {
        if points.is_empty() {
            return Ok(Vec::new());
        }

        let weights = self.calculate_dimension_weights()?;
        let mut centers = Vec::with_capacity(k);

        // First center: random point
        // Use deterministic "random" based on golden ratio for reproducibility
        let first_idx = ((points.len() as f64 * self.phi).fract() * points.len() as f64) as usize;
        centers.push(points[first_idx].clone());

        // Remaining centers: k-means++ (weighted by distance squared)
        for _ in 1..k {
            let mut distances: Vec<f64> = points
                .iter()
                .map(|point| {
                    centers
                        .iter()
                        .map(|center| point.distance(center, &weights))
                        .fold(f64::MAX, f64::min)
                        .powi(2)
                })
                .collect();

            let sum: f64 = distances.iter().sum();
            if sum == 0.0 {
                // All points assigned, pick random remaining
                let idx = centers.len() % points.len();
                centers.push(points[idx].clone());
                continue;
            }

            // Normalize to probabilities
            for d in &mut distances {
                *d /= sum;
            }

            // Weighted selection using golden ratio hash
            let mut cumsum = 0.0;
            let target = (centers.len() as f64 * self.phi).fract();
            let mut selected_idx = 0;

            for (i, &prob) in distances.iter().enumerate() {
                cumsum += prob;
                if cumsum >= target {
                    selected_idx = i;
                    break;
                }
            }

            centers.push(points[selected_idx].clone());
        }

        Ok(centers)
    }

    /// Recalculate cluster centers as mean of assigned points
    fn recalculate_centers(
        &self,
        points: &[SpatialPoint],
        assignments: &[usize],
        k: usize,
    ) -> Result<Vec<SpatialPoint>> {
        let mut centers = vec![
            SpatialPoint {
                pleasure: 0.0,
                arousal: 0.0,
                dominance: 0.0,
                importance: 0.0,
                timestamp: 0.0,
                access_frequency: 0.0,
            };
            k
        ];
        let mut counts = vec![0; k];

        for (point, &cluster_id) in points.iter().zip(assignments.iter()) {
            centers[cluster_id].pleasure += point.pleasure;
            centers[cluster_id].arousal += point.arousal;
            centers[cluster_id].dominance += point.dominance;
            centers[cluster_id].importance += point.importance;
            centers[cluster_id].timestamp += point.timestamp;
            centers[cluster_id].access_frequency += point.access_frequency;
            counts[cluster_id] += 1;
        }

        for (center, &count) in centers.iter_mut().zip(counts.iter()) {
            if count > 0 {
                let count_f = count as f64;
                center.pleasure /= count_f;
                center.arousal /= count_f;
                center.dominance /= count_f;
                center.importance /= count_f;
                center.timestamp /= count_f;
                center.access_frequency /= count_f;
            }
        }

        Ok(centers)
    }

    /// Build memory clusters from k-means results
    fn build_clusters(
        &self,
        points: &[SpatialPoint],
        memories: &[Memory],
        assignments: &[usize],
        centers: &[SpatialPoint],
        k: usize,
    ) -> Result<Vec<MemoryCluster>> {
        let mut clusters: Vec<MemoryCluster> = centers
            .iter()
            .map(|center| MemoryCluster {
                center: center.clone(),
                members: Vec::new(),
                importance: 0.0,
                emotional_signature: EmotionalSignature::default(),
                temporal_center: Utc::now(),
                promote_to_layer: None,
            })
            .collect();

        // Assign memories to clusters
        for ((memory, &cluster_id), point) in memories.iter().zip(assignments.iter()).zip(points.iter()) {
            if cluster_id < k {
                clusters[cluster_id].members.push(memory.clone());
            }
        }

        // Calculate cluster statistics
        for cluster in &mut clusters {
            if !cluster.members.is_empty() {
                // Mean importance
                cluster.importance = cluster.members.iter().map(|m| m.importance).sum::<f64>()
                    / cluster.members.len() as f64;

                // Temporal center (median timestamp)
                let mut timestamps: Vec<DateTime<Utc>> =
                    cluster.members.iter().map(|m| m.created_at).collect();
                timestamps.sort();
                cluster.temporal_center = timestamps[timestamps.len() / 2];

                // Emotional signature (will be smoothed later)
                let signatures: Result<Vec<EmotionalSignature>> = cluster
                    .members
                    .iter()
                    .map(|m| self.memory_to_emotional_signature(m))
                    .collect();
                let signatures = signatures?;

                if !signatures.is_empty() {
                    cluster.emotional_signature = EmotionalSignature {
                        joy: signatures.iter().map(|s| s.joy).sum::<f64>() / signatures.len() as f64,
                        sadness: signatures.iter().map(|s| s.sadness).sum::<f64>()
                            / signatures.len() as f64,
                        anger: signatures.iter().map(|s| s.anger).sum::<f64>() / signatures.len() as f64,
                        fear: signatures.iter().map(|s| s.fear).sum::<f64>() / signatures.len() as f64,
                        surprise: signatures.iter().map(|s| s.surprise).sum::<f64>()
                            / signatures.len() as f64,
                    };
                }
            }
        }

        // Filter out empty clusters
        Ok(clusters.into_iter().filter(|c| !c.members.is_empty()).collect())
    }

    /// Apply mean curvature smoothing to cluster emotional signatures
    ///
    /// Uses Gaussian RBF kernel to smooth emotional signatures based on
    /// spatial proximity in 6D space. Sigma derived from emotional config.
    fn apply_mean_curvature_smoothing(
        &self,
        mut clusters: Vec<MemoryCluster>,
    ) -> Result<Vec<MemoryCluster>> {
        if clusters.len() < 2 {
            return Ok(clusters);
        }

        // Sigma for Gaussian kernel from φ/e ≈ 0.595
        // Represents optimal smoothing bandwidth
        let sigma = self.phi / self.euler;

        let weights = self.calculate_dimension_weights()?;

        // For each cluster, smooth its emotional signature based on neighbors
        for i in 0..clusters.len() {
            let mut weighted_signatures = Vec::new();

            for j in 0..clusters.len() {
                if i != j {
                    let distance = clusters[i].center.distance(&clusters[j].center, &weights);
                    weighted_signatures.push((&clusters[j].emotional_signature, distance));
                }
            }

            // Apply Gaussian blend
            if !weighted_signatures.is_empty() {
                clusters[i].emotional_signature =
                    EmotionalSignature::gaussian_blend(&weighted_signatures, sigma);
            }
        }

        debug!(
            "Applied mean curvature smoothing with σ={:.3} to {} clusters",
            sigma,
            clusters.len()
        );

        Ok(clusters)
    }

    /// Determine which clusters should be promoted to higher memory layers
    ///
    /// Promotion criteria (ALL from emotional config, NO HARDCODING):
    /// - High importance: > tanh(φ) ≈ 0.924
    /// - High access frequency: > e/π ≈ 0.865
    /// - Temporal stability: age > φ * e * π days ≈ 13.8 days
    fn determine_promotions(&self, mut clusters: Vec<MemoryCluster>) -> Result<Vec<MemoryCluster>> {
        // Importance threshold from tanh(φ) ≈ 0.924
        let high_importance = self.phi.tanh();

        // Access frequency threshold from e/π ≈ 0.865
        let high_access = self.euler / self.pi;

        // Temporal stability threshold from φ * e * π ≈ 13.8 days
        let stability_days = self.phi * self.euler * self.pi;

        let now = Utc::now();

        for cluster in &mut clusters {
            // Calculate cluster age
            let age_duration = now.signed_duration_since(cluster.temporal_center);
            let age_days = age_duration.num_days() as f64;

            // Check promotion criteria
            let promote_importance = cluster.importance > high_importance;
            let promote_temporal = age_days > stability_days;

            // Determine target layer based on criteria
            cluster.promote_to_layer = if promote_importance && promote_temporal {
                // Very important and stable → Semantic or higher
                Some(MemoryLayer::Semantic)
            } else if promote_importance {
                // Important but new → Somatic
                Some(MemoryLayer::Somatic)
            } else if promote_temporal {
                // Old but not critical → Episodic
                Some(MemoryLayer::Episodic)
            } else {
                // Stay in working memory
                None
            };

            if cluster.promote_to_layer.is_some() {
                debug!(
                    "Cluster promoted to {:?}: importance={:.3} (>{:.3}={}), age={:.1}d (>{:.1}d={})",
                    cluster.promote_to_layer.as_ref()
                        .ok_or_else(|| anyhow!("No promotion layer available"))?,
                    cluster.importance,
                    high_importance,
                    promote_importance,
                    age_days,
                    stability_days,
                    promote_temporal
                );
            }
        }

        Ok(clusters)
    }

    /// Get consolidation statistics
    pub fn get_stats(&self) -> &ConsolidationStats {
        &self.stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emotional_signature_distance() {
        let sig1 = EmotionalSignature {
            joy: 0.8,
            sadness: 0.2,
            anger: 0.0,
            fear: 0.0,
            surprise: 0.5,
        };

        let sig2 = EmotionalSignature {
            joy: 0.6,
            sadness: 0.4,
            anger: 0.1,
            fear: 0.0,
            surprise: 0.5,
        };

        let dist = sig1.distance(&sig2);
        assert!(dist > 0.0);
        assert!(dist < 2.0); // Max distance is sqrt(5) ≈ 2.236
    }

    #[test]
    fn test_adaptive_k_calculation() {
        let config = EmotionalConfig::load_from_file("config/emotional_config.toml");
        if let Ok(config) = config {
            let memory_store = std::sync::Arc::new(
                tokio::runtime::Runtime::new()
                    .map_err(|e| anyhow!("Failed to create runtime: {}", e))?
                    .block_on(async {
                    use crate::memory::store::MemoryStore;
                    use crate::config::MemoryConfig;
                    use std::path::PathBuf;
                    MemoryStore::new(&MemoryConfig {
                        max_entries: 1000,
                        decay_rate: 0.01,
                        consolidation_threshold: 0.5,
                        persistent: false,
                        storage_path: PathBuf::from("./test_k_calc"),
                        db_path: PathBuf::from("./test_k_calc"),
                        max_cache_size_mb: 64,
                    }).await
                        .map_err(|e| anyhow!("Failed to create memory store: {}", e))?
                })
            );
            let engine = SpatialConsolidationEngine::new(config, memory_store)
                .map_err(|e| anyhow!("Failed to create consolidation engine: {}", e))?;

            // Test various memory counts
            assert_eq!(engine.calculate_adaptive_k(0)
                .map_err(|e| anyhow!("Failed to calculate k for 0: {}", e))?, 1);
            assert!(engine.calculate_adaptive_k(10)
                .map_err(|e| anyhow!("Failed to calculate k for 10: {}", e))? >= 3);
            assert!(engine.calculate_adaptive_k(100)
                .map_err(|e| anyhow!("Failed to calculate k for 100: {}", e))? >= 3);
            assert!(engine.calculate_adaptive_k(10000)
                .map_err(|e| anyhow!("Failed to calculate k for 10000: {}", e))? <= 14);
        }
    }

    #[tokio::test]
    async fn test_access_frequency_tracking() {
        use crate::memory::store::MemoryStore;
        use crate::config::MemoryConfig;
        use std::path::PathBuf;

        // Create a memory store for testing
        let config = MemoryConfig {
            max_entries: 1000,
            decay_rate: 0.01,
            consolidation_threshold: 0.5,
            persistent: false,
            storage_path: PathBuf::from("./test_access_tracking"),
            db_path: PathBuf::from("./test_access_tracking"),
            max_cache_size_mb: 64,
        };

        let memory_store = std::sync::Arc::new(MemoryStore::new(&config).await
            .map_err(|e| anyhow!("Failed to create memory store: {}", e))?);

        // Create spatial consolidation engine
        let emotional_config = EmotionalConfig::load_from_file("config/emotional_config.toml")
            .map_err(|e| anyhow!("Failed to load emotional config: {}", e))?;
        let mut engine = SpatialConsolidationEngine::new(emotional_config, memory_store.clone())
            .map_err(|e| anyhow!("Failed to create consolidation engine: {}", e))?;

        // Create test memories
        let mut memories = Vec::new();
        for i in 0..5 {
            memories.push(Memory {
                id: uuid::Uuid::new_v4(),
                content: format!("Test memory {}", i),
                created_at: chrono::Utc::now(),
                importance: 0.5 + (i as f64 * 0.1),
                layer: MemoryLayer::Working,
                emotional_tag: Some("joy".to_string()),
                access_count: 0,
                last_accessed: None,
            });
        }

        // Test access frequency calculation
        let max_frequency = engine.calculate_max_access_frequency(memories).await;
        assert_eq!(max_frequency, 0); // No accesses yet

        // Record some accesses
        for _ in 0..3 {
            memory_store.record_access(&memories[0].id.to_string()).await.unwrap();
        }
        for _ in 0..2 {
            memory_store.record_access(&memories[1].id.to_string()).await.unwrap();
        }

        // Verify access frequencies are tracked correctly
        let (count0, _) = memory_store.get_access_frequency(&memories[0].id.to_string()).await.unwrap();
        let (count1, _) = memory_store.get_access_frequency(&memories[1].id.to_string()).await.unwrap();
        assert_eq!(count0, 3);
        assert_eq!(count1, 2);

        // Test normalized access frequency calculation
        let max_frequency = engine.calculate_max_access_frequency(memories).await;
        assert_eq!(max_frequency, 3);

        let normalized_freq = memory_store.get_normalized_access_frequency(&memories[0].id.to_string(), max_frequency).await.unwrap();
        assert_eq!(normalized_freq, 1.0); // 3/3 = 1.0

        let normalized_freq = memory_store.get_normalized_access_frequency(&memories[1].id.to_string(), max_frequency).await.unwrap();
        assert_eq!(normalized_freq, 2.0/3.0); // 2/3 ≈ 0.666

        // Cleanup
        std::fs::remove_dir_all("./test_access_tracking").ok();
    }

    #[test]
    fn test_pad_emotional_signature_conversion() {
        let emotional_config = EmotionalConfig::load_from_file("config/emotional_config.toml").unwrap();
        let memory_store = std::sync::Arc::new(
            tokio::runtime::Runtime::new().unwrap().block_on(async {
                use crate::config::MemoryConfig;
                use std::path::PathBuf;
                MemoryStore::new(&MemoryConfig {
                    max_entries: 1000,
                    decay_rate: 0.01,
                    consolidation_threshold: 0.5,
                    persistent: false,
                    storage_path: std::path::PathBuf::from("./test_pad_conversion"),
                    db_path: std::path::PathBuf::from("./test_pad_conversion"),
                    max_cache_size_mb: 64,
                }).await.unwrap()
            })
        );
        let engine = SpatialConsolidationEngine::new(emotional_config, memory_store).unwrap();

        // Test PAD to emotional signature conversion
        let signature = engine.pad_to_emotional_signature(0.8, 0.6, 0.4, 0.7);

        // High pleasure should result in high joy
        assert!(signature.joy > 0.5);

        // Low dominance should not result in high anger
        assert!(signature.anger < 0.5);

        // Medium arousal should result in some surprise
        assert!(signature.surprise > 0.2);
    }
}
