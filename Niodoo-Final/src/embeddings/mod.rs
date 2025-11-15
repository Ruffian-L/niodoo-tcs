//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

pub mod sentence_transformer;

pub use sentence_transformer::{
    // Utility functions
    cosine_similarity,
    // Recommended API (dependency injection)
    EmbeddingService,
    SemanticTransformer,

    SentenceEmbedder,
};
