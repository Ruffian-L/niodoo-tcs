//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

pub mod jones_polynomial;
pub mod mobius_graph;
pub mod mobius_torus_k_twist;
pub mod persistent_homology;
pub mod takens_embedding;

// Re-export commonly used types
pub use jones_polynomial::{Crossing, JonesPolynomial, KnotDiagram, KnotType};
pub use persistent_homology::{
    CognitiveTDA, PersistenceDiagram, PersistencePoint, PersistentHomology,
    PersistentHomologyCalculator, PersistentHomologyResult, PointCloud, RipserCalculator,
    SimplexTree, SimplicialComplex, TopologicalFeature, VietorisRipsComplex,
};
pub use takens_embedding::TakensEmbedding;
