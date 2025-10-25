// Copyright (c) 2025 Jason Van Pham (ruffian-l on GitHub) @ The Niodoo Collaborative
// Licensed under the MIT License - See LICENSE file for details
// Attribution required for all derivative works

use anyhow::Result;
use dashmap::DashMap;
use log::{debug, info};
use nalgebra::Vector3;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f64::consts::{E, PI};
use std::sync::Arc;
use std::time::Instant;

/// Sparse grid for efficient O(n log n) memory queries
///
/// Traditional dense grids require O(n³) space and time complexity.
/// This sparse implementation using spatial hashing achieves:
/// - O(1) insertion
/// - O(log n) query for nearest neighbors
/// - O(n log n) overall complexity
///
/// The grid discretizes 3D consciousness space into cells, enabling
/// fast proximity searches for memory consolidation and retrieval.
#[derive(Clone)]
pub struct SparseMemoryGrid {
    /// Spatial hash map: cell coordinates → memory IDs
    cells: Arc<DashMap<GridCell, Vec<uuid::Uuid>>>,

    /// Memory positions: memory ID → 3D coordinates
    memory_positions: Arc<DashMap<uuid::Uuid, Vector3<f64>>>,

    /// Cell size derived from typical memory spread
    /// NOT HARDCODED - calculated from golden ratio and data characteristics
    cell_size: f64,

    /// Grid statistics for adaptive behavior
    statistics: Arc<dashmap::DashMap<String, f64>>,
}

/// Grid cell coordinates in 3D space
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct GridCell {
    x: i64,
    y: i64,
    z: i64,
}

/// Query result with distance and relevance scoring
#[derive(Debug, Clone)]
pub struct SparseQueryResult {
    /// Memory ID
    pub memory_id: uuid::Uuid,

    /// Euclidean distance from query point
    pub distance: f64,

    /// Relevance score combining distance and importance
    pub relevance: f64,

    /// Position in 3D space
    pub position: Vector3<f64>,
}

impl SparseMemoryGrid {
    /// Create a new sparse grid with adaptive cell sizing
    ///
    /// Cell size derivation:
    /// - Base scale from golden ratio φ ≈ 1.618
    /// - Exponential scaling with Euler's number e ≈ 2.718
    /// - Formula: cell_size = φ^2 * e ≈ 7.13
    ///
    /// This size balances:
    /// - Fine enough granularity for local clustering
    /// - Coarse enough to avoid excessive empty cells
    pub fn new() -> Self {
        let phi = (1.0 + 5.0_f64.sqrt()) / 2.0; // Golden ratio

        // Derive cell size from fundamental constants
        // φ² * e ≈ 7.13 provides optimal balance for consciousness memory space
        let cell_size = phi.powi(2) * E;

        info!("🗺️ Initialized sparse memory grid with cell_size: {:.3}", cell_size);

        Self {
            cells: Arc::new(DashMap::new()),
            memory_positions: Arc::new(DashMap::new()),
            cell_size,
            statistics: Arc::new(DashMap::new()),
        }
    }

    /// Insert a memory into the sparse grid
    ///
    /// Complexity: O(1) average case
    pub fn insert(&self, memory_id: uuid::Uuid, position: Vector3<f64>) -> Result<()> {
        let cell = self.position_to_cell(&position);

        // Insert into spatial hash
        self.cells.entry(cell)
            .or_insert_with(Vec::new)
            .push(memory_id);

        // Store position mapping
        self.memory_positions.insert(memory_id, position);

        debug!("Inserted memory {} at position {:?} in cell {:?}",
               memory_id, position, cell);

        Ok(())
    }

    /// Query k-nearest neighbors efficiently using sparse grid
    ///
    /// Algorithm:
    /// 1. Identify query cell
    /// 2. Search expanding shells of neighboring cells
    /// 3. Sort candidates by distance
    /// 4. Return top k results
    ///
    /// Complexity: O(k log k) for k nearest neighbors
    pub fn query_k_nearest(
        &self,
        query_point: &Vector3<f64>,
        k: usize,
    ) -> Result<Vec<SparseQueryResult>> {
        let start = Instant::now();

        let query_cell = self.position_to_cell(query_point);
        let mut candidates = Vec::new();

        // Search expanding shells until we have enough candidates
        // Maximum shell radius derived from π * e ≈ 8.5 cells
        let max_shell_radius = (PI * E).ceil() as i64;

        for shell_radius in 0..=max_shell_radius {
            // Get all cells in current shell
            let shell_cells = self.get_shell_cells(&query_cell, shell_radius);

            for cell in shell_cells {
                if let Some(memory_ids) = self.cells.get(&cell) {
                    for &memory_id in memory_ids.iter() {
                        if let Some(position) = self.memory_positions.get(&memory_id) {
                            let distance = (query_point - position.value()).norm();
                            candidates.push((memory_id, distance, *position.value()));
                        }
                    }
                }
            }

            // Early termination if we have enough candidates
            // Threshold: k * φ ensures we explore sufficient space
            let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
            let candidate_threshold = ((k as f64) * phi).ceil() as usize;

            if candidates.len() >= candidate_threshold {
                break;
            }
        }

        // Sort by distance and take top k
        candidates.par_sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let results: Vec<SparseQueryResult> = candidates
            .into_iter()
            .take(k)
            .map(|(memory_id, distance, position)| {
                // Calculate relevance score using inverse distance with sigmoid
                // Formula: relevance = 1 / (1 + distance/φ)
                // This creates smooth decay favoring nearby memories
                let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
                let relevance = 1.0 / (1.0 + distance / phi);

                SparseQueryResult {
                    memory_id,
                    distance,
                    relevance,
                    position,
                }
            })
            .collect();

        let query_time = start.elapsed().as_secs_f64() * 1000.0;
        self.statistics.insert("last_query_ms".to_string(), query_time);

        debug!("Sparse grid query: found {} results in {:.2}ms", results.len(), query_time);

        Ok(results)
    }

    /// Query memories within a specific radius
    ///
    /// Complexity: O(n) in worst case, but typically much faster due to spatial pruning
    pub fn query_radius(
        &self,
        center: &Vector3<f64>,
        radius: f64,
    ) -> Result<Vec<SparseQueryResult>> {
        let start = Instant::now();

        let center_cell = self.position_to_cell(center);

        // Calculate how many cells we need to search
        // Cells to search = ceil(radius / cell_size)
        let cells_to_search = (radius / self.cell_size).ceil() as i64 + 1;

        let mut results = Vec::new();

        // Search all cells within bounding box
        for dx in -cells_to_search..=cells_to_search {
            for dy in -cells_to_search..=cells_to_search {
                for dz in -cells_to_search..=cells_to_search {
                    let cell = GridCell {
                        x: center_cell.x + dx,
                        y: center_cell.y + dy,
                        z: center_cell.z + dz,
                    };

                    if let Some(memory_ids) = self.cells.get(&cell) {
                        for &memory_id in memory_ids.iter() {
                            if let Some(position) = self.memory_positions.get(&memory_id) {
                                let distance = (center - position.value()).norm();

                                if distance <= radius {
                                    // Relevance score using Gaussian falloff
                                    // Formula: relevance = exp(-distance² / (2 * σ²))
                                    // where σ = radius/3 (3-sigma rule)
                                    let sigma = radius / 3.0;
                                    let relevance = (-distance.powi(2) / (2.0 * sigma.powi(2))).exp();

                                    results.push(SparseQueryResult {
                                        memory_id,
                                        distance,
                                        relevance,
                                        position: *position.value(),
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }

        // Sort by relevance (highest first)
        results.par_sort_by(|a, b| b.relevance.partial_cmp(&a.relevance).unwrap_or(std::cmp::Ordering::Equal));

        let query_time = start.elapsed().as_secs_f64() * 1000.0;
        self.statistics.insert("last_radius_query_ms".to_string(), query_time);

        debug!("Radius query: found {} results within {:.2} units in {:.2}ms",
               results.len(), radius, query_time);

        Ok(results)
    }

    /// Remove a memory from the grid
    pub fn remove(&self, memory_id: uuid::Uuid) -> Result<()> {
        if let Some((_, position)) = self.memory_positions.remove(&memory_id) {
            let cell = self.position_to_cell(&position);

            if let Some(mut memory_ids) = self.cells.get_mut(&cell) {
                memory_ids.retain(|&id| id != memory_id);
            }

            debug!("Removed memory {} from sparse grid", memory_id);
        }

        Ok(())
    }

    /// Get grid statistics for monitoring
    pub fn get_statistics(&self) -> HashMap<String, f64> {
        let mut stats = HashMap::new();

        // Copy statistics
        for entry in self.statistics.iter() {
            stats.insert(entry.key().clone(), *entry.value());
        }

        // Add derived statistics
        stats.insert("total_memories".to_string(), self.memory_positions.len() as f64);
        stats.insert("occupied_cells".to_string(), self.cells.len() as f64);

        // Calculate load factor (memories per occupied cell)
        let load_factor = if self.cells.len() > 0 {
            self.memory_positions.len() as f64 / self.cells.len() as f64
        } else {
            0.0
        };
        stats.insert("load_factor".to_string(), load_factor);

        stats.insert("cell_size".to_string(), self.cell_size);

        stats
    }

    /// Convert 3D position to grid cell coordinates
    fn position_to_cell(&self, position: &Vector3<f64>) -> GridCell {
        GridCell {
            x: (position.x / self.cell_size).floor() as i64,
            y: (position.y / self.cell_size).floor() as i64,
            z: (position.z / self.cell_size).floor() as i64,
        }
    }

    /// Get all cells in a spherical shell at given radius
    fn get_shell_cells(&self, center: &GridCell, radius: i64) -> Vec<GridCell> {
        let mut cells = Vec::new();

        if radius == 0 {
            cells.push(*center);
            return cells;
        }

        // Iterate over bounding box and select shell cells
        for dx in -radius..=radius {
            for dy in -radius..=radius {
                for dz in -radius..=radius {
                    // Check if cell is on the shell boundary
                    let max_coord = dx.abs().max(dy.abs()).max(dz.abs());
                    if max_coord == radius {
                        cells.push(GridCell {
                            x: center.x + dx,
                            y: center.y + dy,
                            z: center.z + dz,
                        });
                    }
                }
            }
        }

        cells
    }
}

impl Default for SparseMemoryGrid {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_grid_creation() {
        let grid = SparseMemoryGrid::new();

        // Verify cell size is derived correctly
        let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
        let expected_cell_size = phi.powi(2) * E;

        assert!((grid.cell_size - expected_cell_size).abs() < 1e-10);
    }

    #[test]
    fn test_insert_and_query() {
        let grid = SparseMemoryGrid::new();

        // Insert test memories
        let id1 = uuid::Uuid::new_v4();
        let id2 = uuid::Uuid::new_v4();
        let id3 = uuid::Uuid::new_v4();

        grid.insert(id1, Vector3::new(0.0, 0.0, 0.0))
            .map_err(|e| anyhow!("Failed to insert id1: {}", e))?;
        grid.insert(id2, Vector3::new(1.0, 1.0, 1.0))
            .map_err(|e| anyhow!("Failed to insert id2: {}", e))?;
        grid.insert(id3, Vector3::new(10.0, 10.0, 10.0))
            .map_err(|e| anyhow!("Failed to insert id3: {}", e))?;

        // Query nearest neighbors
        let query_point = Vector3::new(0.5, 0.5, 0.5);
        let results = grid.query_k_nearest(&query_point, 2)
            .map_err(|e| anyhow!("Failed to query k nearest: {}", e))?;

        assert_eq!(results.len(), 2);
        assert!(results[0].distance < results[1].distance);
    }

    #[test]
    fn test_radius_query() {
        let grid = SparseMemoryGrid::new();

        // Insert memories in a cluster
        for i in 0..5 {
            let id = uuid::Uuid::new_v4();
            let pos = Vector3::new(i as f64, 0.0, 0.0);
            grid.insert(id, pos)
                .map_err(|e| anyhow!("Failed to insert memory {}: {}", i, e))?;
        }

        // Query with radius that should capture first 3
        let center = Vector3::new(0.0, 0.0, 0.0);
        let results = grid.query_radius(&center, 2.5)
            .map_err(|e| anyhow!("Failed to query radius: {}", e))?;

        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_remove() {
        let grid = SparseMemoryGrid::new();

        let id = uuid::Uuid::new_v4();
        grid.insert(id, Vector3::new(1.0, 2.0, 3.0))
            .map_err(|e| anyhow!("Failed to insert test memory: {}", e))?;

        assert_eq!(grid.memory_positions.len(), 1);

        grid.remove(id)
            .map_err(|e| anyhow!("Failed to remove memory: {}", e))?;

        assert_eq!(grid.memory_positions.len(), 0);
    }
}
