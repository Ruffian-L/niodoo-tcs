// Copyright (c) 2025 Jason Van Pham (ruffian-l on GitHub) @ The Niodoo Collaborative
// Licensed under the MIT License - See LICENSE file for details
// Attribution required for all derivative works

//! Memory module
//!
//! This module provides the complete memory system for consciousness:
//! - `layers`: Six-layer adaptive TTL memory system (Working, Somatic, Semantic, Episodic, Procedural, Core Burned)
//! - `store`: RocksDB persistence layer for durable storage
//! - `spatial_index`: Spatial index navigator for consciousness states using Möbius topology
//! - `block_pool`: Zero-allocation block recycling memory manager
//! - `spatial_consolidation`: Spatial clustering and mean curvature smoothing for memory consolidation
//! - `layered_sparse_grid`: MSBG-inspired multiresolution sparse grids for efficient memory storage
//! - `sparse_block_grid`: HashMap-based O(1) sparse block storage with lazy allocation

pub mod layers;
pub mod store;
pub mod spatial_index;
pub mod block_pool;
pub mod spatial_consolidation;
pub mod layered_sparse_grid;
pub mod sparse_grid;
pub mod sparse_block_grid;
pub mod guessing_spheres;
pub mod vector_index;

pub use layers::{MemorySystem, MemoryLayer, Memory, CoreBelief, BeliefCategory};
pub use guessing_spheres::{GuessingSpheres, MemorySphere, EmotionalVector};
pub use store::{MemoryStore, MemoryStoreStats};
pub use spatial_index::{SpatialIndex, SpatialCoordinate, SpatialIndexStats, CacheStats};
pub use vector_index::{VectorIndex, MemoryEmbedding, SemanticSearchExt};
pub use block_pool::{BlockPool, BlockPoolConfig, BlockPoolStats};
pub use spatial_consolidation::{SpatialConsolidationEngine, MemoryCluster, ConsolidationStats};
pub use layered_sparse_grid::{LayeredSparseGrid, MemoryLayerType, MemoryValue, LayeredGridStats, SparseGridStats};
pub use sparse_grid::{SparseMemoryGrid, SparseQueryResult};
pub use sparse_block_grid::{
    SparseBlockGrid,
    BlockCoord,
    Block,
    BlockMetadata,
    SparseBlockGridConfig,
    GridStats,
    BoundingBox3D,
};
