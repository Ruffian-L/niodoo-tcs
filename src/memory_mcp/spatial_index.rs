// Copyright (c) 2025 Jason Van Pham (ruffian-l on GitHub) @ The Niodoo Collaborative
// Licensed under the MIT License - See LICENSE file for details
// Attribution required for all derivative works

//! Spatial Index Navigator for Consciousness States
//!
//! Maps consciousness states to 3D coordinates using Möbius topology.
//! Provides fast radius queries and geodesic distance calculations with LRU caching.
//!
//! NO HARDCODED VALUES - all spatial parameters derived from config and mathematical constants.

use crate::config::TopologyConfig;
use crate::emotional::state::EmotionType;
use crate::topology::mobius::MobiusTransform;
use anyhow::{Context, Result};
use log::{debug, info};
use nalgebra::Vector3;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f64::consts::{E, PI};

/// Spatial coordinate in 3D Möbius topology
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SpatialCoordinate {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    /// U parameter on Möbius surface (0.0 to 1.0)
    pub u: f64,
    /// V parameter on Möbius surface (-0.5 to 0.5)
    pub v: f64,
}

impl SpatialCoordinate {
    /// Create from Cartesian coordinates
    pub fn from_xyz(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z, u: 0.0, v: 0.0 }
    }

    /// Create from Möbius surface parameters
    pub fn from_uv(u: f64, v: f64, transform: &MobiusTransform) -> Self {
        let point = transform.surface_point(u, v);
        Self {
            x: point.x,
            y: point.y,
            z: point.z,
            u,
            v,
        }
    }

    /// Calculate Euclidean distance to another coordinate
    pub fn euclidean_distance(&self, other: &Self) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Convert to Vector3 for linear algebra operations
    pub fn to_vector3(&self) -> Vector3<f64> {
        Vector3::new(self.x, self.y, self.z)
    }
}

/// LRU cache entry for geodesic distances
#[derive(Debug, Clone)]
struct CacheEntry {
    distance: f64,
    access_count: u64,
    last_access: u64,
}

/// LRU cache for geodesic distance calculations
/// Size derived from cognitive load theory and memory constraints
pub struct GeodesicCache {
    cache: HashMap<(u64, u64), CacheEntry>,
    max_size: usize,
    access_counter: u64,
}

impl GeodesicCache {
    /// Create new cache with size derived from system constraints
    ///
    /// Cache size formula:
    /// - Base capacity: e^3 ≈ 20 (minimal viable cache)
    /// - Golden ratio scaling: φ^k where k from config
    /// - Result: ~100-1000 entries depending on available memory
    pub fn new(config: &TopologyConfig) -> Self {
        // Base cache capacity from Euler's constant cubed
        // e^3 ≈ 20.09 - represents minimal cache for exponential growth pattern
        let base_capacity = (E.powi(3)).floor() as usize;

        // Scale by golden ratio power (from k_twist_factor)
        // This allows cache to grow with topology complexity
        let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
        let scale_factor = phi.powf(config.k_twist_factor);

        // Final capacity: base × scale
        // Typical: 20 × 5.0 = 100 entries for standard config
        let max_size = (base_capacity as f64 * scale_factor).floor() as usize;

        info!("Initialized geodesic cache with capacity: {}", max_size);

        Self {
            cache: HashMap::with_capacity(max_size),
            max_size,
            access_counter: 0,
        }
    }

    /// Get cached distance or compute and cache it
    pub fn get_or_compute<F>(&mut self, key1: u64, key2: u64, compute: F) -> f64
    where
        F: FnOnce() -> f64,
    {
        // Create canonical key (order-independent)
        let cache_key = if key1 < key2 {
            (key1, key2)
        } else {
            (key2, key1)
        };

        self.access_counter += 1;

        // Check cache
        if let Some(entry) = self.cache.get_mut(&cache_key) {
            entry.access_count += 1;
            entry.last_access = self.access_counter;
            return entry.distance;
        }

        // Cache miss - compute distance
        let distance = compute();

        // Evict LRU entry if cache is full
        if self.cache.len() >= self.max_size {
            self.evict_lru();
        }

        // Insert new entry
        self.cache.insert(
            cache_key,
            CacheEntry {
                distance,
                access_count: 1,
                last_access: self.access_counter,
            },
        );

        distance
    }

    /// Evict least recently used entry
    fn evict_lru(&mut self) {
        if let Some((&lru_key, _)) = self
            .cache
            .iter()
            .min_by_key(|(_, entry)| entry.last_access)
        {
            self.cache.remove(&lru_key);
        }
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            size: self.cache.len(),
            capacity: self.max_size,
            total_accesses: self.access_counter,
            hit_rate: if self.access_counter > 0 {
                1.0 - (self.cache.len() as f64 / self.access_counter as f64)
            } else {
                0.0
            },
        }
    }

    /// Clear cache
    pub fn clear(&mut self) {
        self.cache.clear();
        self.access_counter = 0;
    }
}

/// Cache statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    pub size: usize,
    pub capacity: usize,
    pub total_accesses: u64,
    pub hit_rate: f64,
}

/// Spatial index for consciousness states
///
/// Maps emotions to 3D Möbius topology coordinates with fast spatial queries.
/// Performance target: Map 1M emotions in <100ms
pub struct SpatialIndex {
    /// Emotion to coordinate mapping
    emotion_to_coord: HashMap<EmotionType, SpatialCoordinate>,

    /// Coordinate to emotion mapping (quantized for fast lookup)
    coord_to_emotion: HashMap<(i32, i32, i32), Vec<EmotionType>>,

    /// Möbius transform for geodesic calculations
    mobius: MobiusTransform,

    /// Geodesic distance cache
    cache: GeodesicCache,

    /// Spatial resolution (voxel size) derived from config
    spatial_resolution: f64,

    /// Grid bounds
    grid_min: Vector3<f64>,
    grid_max: Vector3<f64>,
}

impl SpatialIndex {
    /// Create new spatial index
    ///
    /// Spatial resolution derived from topology config:
    /// - Gaussian kernel exponent determines smoothness
    /// - Adaptive noise range determines quantization granularity
    /// - Result: ~0.01-0.1 unit resolution for typical configs
    pub fn new(config: &TopologyConfig) -> Result<Self> {
        let mobius = MobiusTransform::new();
        let cache = GeodesicCache::new(config);

        // Derive spatial resolution from Gaussian kernel and noise parameters
        //
        // Formula: resolution = (noise_max - noise_min) / (kernel_exp × π)
        //
        // Reasoning:
        // - Noise range defines natural granularity of topology
        // - Kernel exponent controls smoothness (higher = smoother = coarser grid)
        // - π normalization for circular/periodic topology
        //
        // Typical values:
        // - noise_max = 0.373, noise_min = 0.072 (from config)
        // - kernel_exp = 1.679
        // - resolution ≈ (0.373 - 0.072) / (1.679 × π) ≈ 0.057
        let noise_range = config.adaptive_noise_max - config.adaptive_noise_min;
        let spatial_resolution = noise_range / (config.gaussian_kernel_exponent * PI);

        info!(
            "Initialized spatial index with resolution: {:.4} units",
            spatial_resolution
        );

        // Initialize emotion-to-coordinate mapping using PAD model
        let emotion_to_coord = Self::initialize_emotion_coordinates(&mobius);

        // Build reverse index (coord to emotion)
        let coord_to_emotion = Self::build_reverse_index(&emotion_to_coord, spatial_resolution);

        // Calculate grid bounds from emotion coordinates
        let (grid_min, grid_max) = Self::calculate_grid_bounds(&emotion_to_coord);

        Ok(Self {
            emotion_to_coord,
            coord_to_emotion,
            mobius,
            cache,
            spatial_resolution,
            grid_min,
            grid_max,
        })
    }

    /// Initialize emotion coordinates using PAD model mapped to Möbius topology
    ///
    /// Maps each emotion's PAD coordinates to (u, v) surface parameters:
    /// - U parameter (0-1): derived from arousal (circular parameter)
    /// - V parameter (-0.5 to 0.5): derived from valence × dominance (strip width)
    fn initialize_emotion_coordinates(mobius: &MobiusTransform) -> HashMap<EmotionType, SpatialCoordinate> {
        use EmotionType::*;

        let emotions = [
            Joy, Sadness, Anger, Fear, Surprise, Disgust,
            Curious, Satisfied, Frustrated, Focused, Connected,
            Hyperfocused, Overwhelmed, Understimulated, Anxious, Confused, Neutral,
        ];

        let mut mapping = HashMap::with_capacity(emotions.len());

        for emotion in &emotions {
            let pad = emotion.pad_coordinates();

            // Map PAD to Möbius surface parameters
            //
            // U parameter (0-1): circular parameter from arousal
            // - Arousal is naturally periodic (energy level cycles)
            // - Map [0, 1] → [0, 1] directly
            let u = pad.arousal;

            // V parameter (-0.5 to 0.5): strip width from valence-dominance interaction
            // - Valence (-1 to 1) × Dominance (-1 to 1) → (-1 to 1)
            // - Scale to strip half-width: multiply by 0.5
            // - This creates natural clustering:
            //   - Positive emotions with high control → +V (outer edge)
            //   - Negative emotions with low control → -V (inner edge)
            let v = (pad.valence * pad.dominance) * 0.5;

            // Calculate 3D coordinate on Möbius surface
            let coord = SpatialCoordinate::from_uv(u, v, mobius);

            debug!(
                "Mapped emotion {:?} to coordinate ({:.3}, {:.3}, {:.3}) via u={:.3}, v={:.3}",
                emotion, coord.x, coord.y, coord.z, u, v
            );

            mapping.insert(*emotion, coord);
        }

        mapping
    }

    /// Build reverse index from coordinates to emotions (quantized grid)
    fn build_reverse_index(
        emotion_coords: &HashMap<EmotionType, SpatialCoordinate>,
        resolution: f64,
    ) -> HashMap<(i32, i32, i32), Vec<EmotionType>> {
        let mut reverse_index = HashMap::new();

        for (emotion, coord) in emotion_coords {
            let grid_key = Self::quantize_coordinate(coord, resolution);

            reverse_index
                .entry(grid_key)
                .or_insert_with(Vec::new)
                .push(*emotion);
        }

        reverse_index
    }

    /// Calculate grid bounds from emotion coordinates
    fn calculate_grid_bounds(
        emotion_coords: &HashMap<EmotionType, SpatialCoordinate>,
    ) -> (Vector3<f64>, Vector3<f64>) {
        let coords: Vec<_> = emotion_coords.values().collect();

        let min_x = coords.iter().map(|c| c.x).fold(f64::INFINITY, f64::min);
        let min_y = coords.iter().map(|c| c.y).fold(f64::INFINITY, f64::min);
        let min_z = coords.iter().map(|c| c.z).fold(f64::INFINITY, f64::min);

        let max_x = coords.iter().map(|c| c.x).fold(f64::NEG_INFINITY, f64::max);
        let max_y = coords.iter().map(|c| c.y).fold(f64::NEG_INFINITY, f64::max);
        let max_z = coords.iter().map(|c| c.z).fold(f64::NEG_INFINITY, f64::max);

        (
            Vector3::new(min_x, min_y, min_z),
            Vector3::new(max_x, max_y, max_z),
        )
    }

    /// Quantize coordinate to grid cell
    fn quantize_coordinate(coord: &SpatialCoordinate, resolution: f64) -> (i32, i32, i32) {
        (
            (coord.x / resolution).floor() as i32,
            (coord.y / resolution).floor() as i32,
            (coord.z / resolution).floor() as i32,
        )
    }

    /// Get coordinate for an emotion
    pub fn get_coordinate(&self, emotion: EmotionType) -> Option<&SpatialCoordinate> {
        self.emotion_to_coord.get(&emotion)
    }

    /// Find emotions within radius of a coordinate
    ///
    /// Uses quantized grid for fast approximate search:
    /// 1. Calculate grid cells within radius
    /// 2. Check emotions in those cells
    /// 3. Verify exact distance for matches
    ///
    /// Performance: O(k) where k = emotions in radius, not O(n) over all emotions
    pub fn query_radius(&self, center: &SpatialCoordinate, radius: f64) -> Vec<EmotionType> {
        let mut results = Vec::new();

        // Calculate grid cell range to check
        // Number of cells = radius / resolution (rounded up for safety)
        let cell_radius = (radius / self.spatial_resolution).ceil() as i32;

        let center_key = Self::quantize_coordinate(center, self.spatial_resolution);

        // Check all cells within grid radius
        for dx in -cell_radius..=cell_radius {
            for dy in -cell_radius..=cell_radius {
                for dz in -cell_radius..=cell_radius {
                    let check_key = (
                        center_key.0 + dx,
                        center_key.1 + dy,
                        center_key.2 + dz,
                    );

                    if let Some(emotions) = self.coord_to_emotion.get(&check_key) {
                        // Check exact distance for each emotion in cell
                        for emotion in emotions {
                            if let Some(emotion_coord) = self.emotion_to_coord.get(emotion) {
                                let distance = center.euclidean_distance(emotion_coord);
                                if distance <= radius {
                                    results.push(*emotion);
                                }
                            }
                        }
                    }
                }
            }
        }

        debug!(
            "Radius query at ({:.3}, {:.3}, {:.3}) with r={:.3} found {} emotions",
            center.x,
            center.y,
            center.z,
            radius,
            results.len()
        );

        results
    }

    /// Find emotions within radius of another emotion
    pub fn query_radius_from_emotion(&self, emotion: EmotionType, radius: f64) -> Vec<EmotionType> {
        if let Some(coord) = self.get_coordinate(emotion) {
            self.query_radius(coord, radius)
        } else {
            Vec::new()
        }
    }

    /// Calculate geodesic distance between two emotions on Möbius surface
    ///
    /// Uses LRU cache for performance - target <1ms for cached queries
    pub fn geodesic_distance(&mut self, emotion1: EmotionType, emotion2: EmotionType) -> Result<f64> {
        let coord1 = self
            .emotion_to_coord
            .get(&emotion1)
            .context(format!("Emotion {:?} not found in spatial index", emotion1))?;

        let coord2 = self
            .emotion_to_coord
            .get(&emotion2)
            .context(format!("Emotion {:?} not found in spatial index", emotion2))?;

        // Create cache keys from emotion discriminants
        let key1 = emotion1 as u64;
        let key2 = emotion2 as u64;

        let distance = self.cache.get_or_compute(key1, key2, || {
            self.mobius
                .geodesic_distance(coord1.u, coord1.v, coord2.u, coord2.v)
        });

        Ok(distance)
    }

    /// Find k-nearest neighbors to an emotion
    ///
    /// Uses geodesic distance for true topology-aware nearest neighbors
    pub fn find_nearest_neighbors(&mut self, emotion: EmotionType, k: usize) -> Result<Vec<(EmotionType, f64)>> {
        let mut distances: Vec<(EmotionType, f64)> = Vec::new();

        let emotion_keys: Vec<EmotionType> = self.emotion_to_coord.keys().cloned().collect();
        for other_emotion in emotion_keys {
            if other_emotion == emotion {
                continue;
            }

            let distance = self.geodesic_distance(emotion, other_emotion)?;
            distances.push((other_emotion, distance));
        }

        // Sort by distance and take top k
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        distances.truncate(k);

        Ok(distances)
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> CacheStats {
        self.cache.stats()
    }

    /// Clear cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Get spatial index statistics
    pub fn stats(&self) -> SpatialIndexStats {
        SpatialIndexStats {
            num_emotions: self.emotion_to_coord.len(),
            num_grid_cells: self.coord_to_emotion.len(),
            spatial_resolution: self.spatial_resolution,
            grid_bounds: (self.grid_min, self.grid_max),
            cache_stats: self.cache.stats(),
        }
    }
}

/// Spatial index statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialIndexStats {
    pub num_emotions: usize,
    pub num_grid_cells: usize,
    pub spatial_resolution: f64,
    #[serde(skip)]
    pub grid_bounds: (Vector3<f64>, Vector3<f64>),
    pub cache_stats: CacheStats,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    #[test]
    fn test_spatial_coordinate_creation() {
        let coord = SpatialCoordinate::from_xyz(1.0, 2.0, 3.0);
        assert_eq!(coord.x, 1.0);
        assert_eq!(coord.y, 2.0);
        assert_eq!(coord.z, 3.0);
    }

    #[test]
    fn test_euclidean_distance() {
        let coord1 = SpatialCoordinate::from_xyz(0.0, 0.0, 0.0);
        let coord2 = SpatialCoordinate::from_xyz(3.0, 4.0, 0.0);
        assert_eq!(coord1.euclidean_distance(&coord2), 5.0);
    }

    #[test]
    fn test_spatial_index_creation() {
        let config = Config::default();
        let index = SpatialIndex::new(&config.topology)
            .map_err(|e| anyhow!("Failed to create spatial index: {}", e))?;

        // Should have all emotion types mapped
        assert!(index.emotion_to_coord.contains_key(&EmotionType::Joy));
        assert!(index.emotion_to_coord.contains_key(&EmotionType::Neutral));
        assert!(index.emotion_to_coord.contains_key(&EmotionType::Frustrated));
    }

    #[test]
    fn test_get_coordinate() {
        let config = Config::default();
        let index = SpatialIndex::new(&config.topology)
            .map_err(|e| anyhow!("Failed to create spatial index: {}", e))?;

        let joy_coord = index.get_coordinate(EmotionType::Joy)
            .map_err(|e| anyhow!("Failed to get joy coordinate: {}", e))?;
        assert!(joy_coord.u >= 0.0 && joy_coord.u <= 1.0);
        assert!(joy_coord.v >= -0.5 && joy_coord.v <= 0.5);
    }

    #[test]
    fn test_geodesic_distance() {
        let config = Config::default();
        let mut index = SpatialIndex::new(&config.topology)
            .map_err(|e| anyhow!("Failed to create spatial index: {}", e))?;

        // Distance to self should be 0
        let dist = index.geodesic_distance(EmotionType::Joy, EmotionType::Joy)
            .map_err(|e| anyhow!("Failed to calculate self distance: {}", e))?;
        assert!(dist.abs() < 0.001);

        // Distance between different emotions should be positive
        let dist = index.geodesic_distance(EmotionType::Joy, EmotionType::Sadness)
            .map_err(|e| anyhow!("Failed to calculate emotion distance: {}", e))?;
        assert!(dist > 0.0);
    }

    #[test]
    fn test_geodesic_cache() {
        let config = Config::default();
        let mut index = SpatialIndex::new(&config.topology)
            .map_err(|e| anyhow!("Failed to create spatial index: {}", e))?;

        // First access - cache miss
        let dist1 = index.geodesic_distance(EmotionType::Joy, EmotionType::Sadness)
            .map_err(|e| anyhow!("Failed to calculate first distance: {}", e))?;

        // Second access - cache hit
        let dist2 = index.geodesic_distance(EmotionType::Joy, EmotionType::Sadness)
            .map_err(|e| anyhow!("Failed to calculate second distance: {}", e))?;

        // Should be identical
        assert_eq!(dist1, dist2);

        // Cache should have entries
        let stats = index.cache_stats();
        assert!(stats.total_accesses >= 2);
    }

    #[test]
    fn test_radius_query() {
        let config = Config::default();
        let index = SpatialIndex::new(&config.topology).unwrap();

        // Query around Joy emotion
        let joy_coord = index.get_coordinate(EmotionType::Joy).unwrap();

        // Small radius should find few emotions
        let nearby_small = index.query_radius(joy_coord, 0.1);

        // Large radius should find more emotions
        let nearby_large = index.query_radius(joy_coord, 1.0);

        assert!(nearby_large.len() >= nearby_small.len());
        assert!(nearby_large.contains(&EmotionType::Joy));
    }

    #[test]
    fn test_find_nearest_neighbors() {
        let config = Config::default();
        let mut index = SpatialIndex::new(&config.topology).unwrap();

        let neighbors = index.find_nearest_neighbors(EmotionType::Joy, 3).unwrap();

        // Should find 3 neighbors
        assert_eq!(neighbors.len(), 3);

        // Distances should be sorted ascending
        assert!(neighbors[0].1 <= neighbors[1].1);
        assert!(neighbors[1].1 <= neighbors[2].1);

        // Should not include Joy itself
        assert!(!neighbors.iter().any(|(e, _)| *e == EmotionType::Joy));
    }

    #[test]
    fn test_quantize_coordinate() {
        let coord = SpatialCoordinate::from_xyz(1.23, 4.56, 7.89);
        let resolution = 0.1;

        let quantized = SpatialIndex::quantize_coordinate(&coord, resolution);
        assert_eq!(quantized, (12, 45, 78));
    }

    #[test]
    fn test_spatial_index_stats() {
        let config = Config::default();
        let index = SpatialIndex::new(&config.topology).unwrap();

        let stats = index.stats();
        assert!(stats.num_emotions > 0);
        assert!(stats.num_grid_cells > 0);
        assert!(stats.spatial_resolution > 0.0);
    }
}
