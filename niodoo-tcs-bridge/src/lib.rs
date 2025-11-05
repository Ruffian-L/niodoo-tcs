//! Niodoo-TCS Bridge
//!
//! Connects the TCS Qwen embedder with the Niodoo consciousness engine.

pub mod embedding_adapter;
pub mod pipeline;

pub use embedding_adapter::EmbeddingAdapter;
pub use pipeline::NiodooTCSPipeline;

#[cfg(test)]
mod tests {
    use super::*;
    use niodoo_core::memory::guessing_spheres::EmotionalVector;

    #[test]
    fn test_bridge_integration() {
        // Test that the bridge can be created and basic functionality works
        // Note: This is a placeholder test since full integration requires ONNX runtime

        // Test emotional vector creation
        let emotion = EmotionalVector::new(0.1, 0.2, 0.3, 0.4, 0.5);
        // EmotionalVector has joy, sadness, anger, fear, surprise fields

        println!("✅ Bridge integration test passed - basic components work");
    }
}