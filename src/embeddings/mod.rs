pub mod sentence_transformer;

pub use sentence_transformer::{
    // Utility functions
    cosine_similarity,
    // Recommended API (dependency injection)
    EmbeddingService,
    SemanticTransformer,

    SentenceEmbedder,
};
