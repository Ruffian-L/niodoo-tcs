pub mod embedding;
pub mod emotional_mapping;
pub mod compass;
pub mod erag;
pub mod tokenizer;
pub mod generation;
pub mod learning;
pub mod empathy_network;
pub mod types;
pub mod mock_qdrant;
pub mod mock_vllm;

pub use embedding::QwenEmbedder;
pub use emotional_mapping::EmotionalMapper;
pub use compass::{CompassEngine, CompassResult};
pub use erag::ERAGSystem;
pub use tokenizer::TokenizerEngine;
pub use generation::GenerationEngine;
pub use learning::LearningLoop;
pub use empathy_network::CompleteEmpathyNetwork;

// Re-export main types
pub use types::{NiodooIntegrated, PipelineResult, DynamicThresholds, EmotionalSample};
pub use tokenizer::TokenizedResult;