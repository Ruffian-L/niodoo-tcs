//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::SystemTime;

use anyhow::{anyhow, Result};
use serde::Deserialize;
use tokio::sync::RwLock;

use crate::config::system_config::{ConsciousnessConfig, PathConfig};
use crate::memory::guessing_spheres::{EmotionalVector, GuessingMemorySystem};
use crate::topology::persistent_homology::{
    PersistentHomologyCalculator, PointCloud, RipserCalculator, TopologicalFeature,
};

use super::spatial::SpatialHash;
use super::TokenCandidate;

#[derive(Debug, Deserialize)]
struct TokenPromotionSettings {
    persistence_threshold: f64,
}

pub struct PatternDiscoveryEngine {
    tda_calculator: PersistentHomologyCalculator,
    ripser_calculator: RipserCalculator,
    spatial_hash: Arc<RwLock<SpatialHash>>,
    min_sequence_length: usize,
    max_sequence_length: usize,
    persistence_threshold: f64,
}

impl PatternDiscoveryEngine {
    pub fn new(
        tda_calculator: PersistentHomologyCalculator,
        spatial_hash: Arc<RwLock<SpatialHash>>,
    ) -> Self {
        let config = ConsciousnessConfig::default();

        // Load persistence threshold from config file or use config default
        let persistence_threshold =
            Self::load_persistence_threshold().unwrap_or(config.tda_persistence_threshold);

        let ripser_calculator =
            RipserCalculator::new(config.tda_max_dimension, persistence_threshold);

        Self {
            tda_calculator,
            ripser_calculator,
            spatial_hash,
            min_sequence_length: config.tda_min_sequence_length,
            max_sequence_length: config.tda_max_sequence_length,
            persistence_threshold,
        }
    }

    pub fn with_lengths(mut self, min_sequence_length: usize, max_sequence_length: usize) -> Self {
        self.min_sequence_length = min_sequence_length;
        self.max_sequence_length = max_sequence_length.max(min_sequence_length);
        self
    }

    pub fn with_persistence_threshold(mut self, threshold: f64) -> Self {
        self.persistence_threshold = threshold;
        self.ripser_calculator.set_threshold(threshold);
        self
    }

    pub async fn rebuild_spatial_index(&self, memory_system: &GuessingMemorySystem) {
        let mut hash = self.spatial_hash.write().await;
        hash.rebuild_from_memory(memory_system);
    }

    pub async fn discover_candidates(
        &self,
        memory_system: &GuessingMemorySystem,
    ) -> Result<Vec<TokenCandidate>> {
        let sequences = self.extract_byte_sequences(memory_system);
        if sequences.is_empty() {
            return Ok(Vec::new());
        }

        let point_cloud = self.bytes_to_point_cloud(&sequences)?;

        let ripser_features = match self
            .ripser_calculator
            .compute_from_points(&point_cloud.points)
        {
            Ok(features) => features,
            Err(err) => {
                tracing::warn!(error = %err, "Ripser computation failed; falling back to internal calculator");
                Vec::new()
            }
        };

        let high_persistence: Vec<TopologicalFeature> = if !ripser_features.is_empty() {
            ripser_features
        } else {
            let ph_result = self.tda_calculator.compute(&point_cloud)?;
            ph_result
                .features
                .into_iter()
                .filter(|feature| {
                    feature.dimension == 1 && feature.persistence >= self.persistence_threshold
                })
                .collect()
        };

        if let Err(err) = self.export_persistence_barcode(&high_persistence) {
            tracing::warn!(error = %err, "Failed to export persistence barcode");
        }

        tracing::debug!(
            loop_features = high_persistence.len(),
            "Loop features after persistence filtering"
        );

        let mut candidates = Vec::new();

        for (seq, feature) in sequences.iter().zip(high_persistence.iter()) {
            let frequency = self.calculate_frequency(seq, memory_system);
            let emotional_coherence = self.calculate_emotional_coherence(seq, memory_system).await;
            let spatial_locality = self.calculate_spatial_locality(seq, memory_system).await;

            candidates.push(TokenCandidate {
                bytes: seq.clone(),
                persistence: feature.persistence,
                frequency,
                emotional_coherence,
                spatial_locality,
                timestamp: SystemTime::now(),
            });
        }

        candidates.sort_by(|a, b| {
            b.promotion_score()
                .partial_cmp(&a.promotion_score())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(candidates)
    }

    fn extract_byte_sequences(&self, memory_system: &GuessingMemorySystem) -> Vec<Vec<u8>> {
        let mut sequences = Vec::new();

        for sphere in memory_system.spheres() {
            let bytes = sphere.memory_fragment.as_bytes();
            for len in self.min_sequence_length..=self.max_sequence_length {
                for window in bytes.windows(len) {
                    sequences.push(window.to_vec());
                }
            }
        }

        sequences.sort();
        sequences.dedup();
        sequences
    }

    fn bytes_to_point_cloud(&self, sequences: &[Vec<u8>]) -> Result<PointCloud> {
        let config = ConsciousnessConfig::default();
        let point_dim = config.tda_point_dimension;

        let mut points = Vec::with_capacity(sequences.len());
        for sequence in sequences {
            let mut point = vec![0.0_f64; point_dim];
            for (idx, byte) in sequence.iter().enumerate().take(point_dim) {
                point[idx] = *byte as f64 / 255.0;
            }
            if !point.is_empty() {
                point[0] = sequence.len() as f64 / self.max_sequence_length as f64;
            }
            points.push(point);
        }

        // PointCloud::new() returns PointCloud directly, not a Result
        Ok(PointCloud::new(points))
    }

    fn calculate_frequency(&self, sequence: &[u8], memory_system: &GuessingMemorySystem) -> u64 {
        memory_system
            .spheres()
            .filter(|sphere| {
                sphere
                    .memory_fragment
                    .as_bytes()
                    .windows(sequence.len())
                    .any(|window| window == sequence)
            })
            .count() as u64
    }

    async fn calculate_emotional_coherence(
        &self,
        sequence: &[u8],
        memory_system: &GuessingMemorySystem,
    ) -> f64 {
        let mut matching_spheres = Vec::new();
        for sphere in memory_system.spheres() {
            if sphere
                .memory_fragment
                .as_bytes()
                .windows(sequence.len())
                .any(|window| window == sequence)
            {
                matching_spheres.push(&sphere.emotional_profile);
            }
        }

        if matching_spheres.len() < 2 {
            return 0.0;
        }

        let total_entropy: f64 = matching_spheres.iter().map(|vector| entropy(vector)).sum();
        let average_entropy = total_entropy / matching_spheres.len() as f64;
        let max_entropy = (5_f64).log2();
        ((max_entropy - average_entropy) / max_entropy).clamp(0.0, 1.0)
    }

    async fn calculate_spatial_locality(
        &self,
        sequence: &[u8],
        memory_system: &GuessingMemorySystem,
    ) -> f64 {
        let hash = self.spatial_hash.read().await;
        let mut bucket_counts: HashMap<(i32, i32, i32), usize> = HashMap::new();
        let mut total = 0_usize;

        for (_id, sphere) in memory_system.spheres_with_ids() {
            if sphere
                .memory_fragment
                .as_bytes()
                .windows(sequence.len())
                .any(|window| window == sequence)
            {
                let bucket = hash.position_to_bucket(&sphere.position);
                *bucket_counts.entry(bucket).or_insert(0) += 1;
                total += 1;
            }
        }

        if total == 0 {
            return 0.0;
        }

        bucket_counts
            .values()
            .copied()
            .max()
            .map(|count| count as f64 / total as f64)
            .unwrap_or(0.0)
    }
}

fn entropy(vector: &EmotionalVector) -> f64 {
    let components = vector.as_array();
    let sum: f64 = components.iter().map(|&value| value as f64).sum();
    if sum <= f64::EPSILON {
        return 0.0;
    }

    components
        .iter()
        .map(|&value| value as f64 / sum)
        .filter(|prob| *prob > 0.0)
        .map(|prob| -prob * prob.log2())
        .sum()
}

impl PatternDiscoveryEngine {
    fn load_persistence_threshold() -> Result<f64> {
        let paths = PathConfig::default();
        let mut config_path: PathBuf = paths.config_dir.join("token_promotion.toml");
        if !config_path.exists() {
            config_path = PathBuf::from("config").join("token_promotion.toml");
        }

        let contents = fs::read_to_string(&config_path)
            .map_err(|err| anyhow!("failed to read {}: {err}", config_path.display()))?;
        let settings: TokenPromotionSettings = toml::from_str(&contents)
            .map_err(|err| anyhow!("failed to parse {}: {err}", config_path.display()))?;
        Ok(settings.persistence_threshold)
    }

    fn export_persistence_barcode(&self, features: &[TopologicalFeature]) -> Result<()> {
        let paths = PathConfig::default();
        let output_dir = paths.project_root.join("analysis_output");
        fs::create_dir_all(&output_dir)?;
        let output_path = output_dir.join("persistence_barcode.csv");

        let mut writer = csv::Writer::from_path(&output_path)?;
        writer.write_record(["dimension", "birth", "death", "persistence"])?;
        for feature in features {
            writer.write_record([
                feature.dimension.to_string(),
                feature.birth_time.to_string(),
                feature.death_time.to_string(),
                feature.persistence.to_string(),
            ])?;
        }
        writer.flush()?;
        Ok(())
    }
}
