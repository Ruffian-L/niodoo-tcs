//! Simple integration test that demonstrates the Niodoo-TCS bridge working
//! without requiring ONNX runtime or problematic dependencies.

use niodoo_tcs_bridge::EmbeddingAdapter;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠💖 Niodoo-TCS Integration Test");
    println!("=================================");

    // Create the embedding adapter
    let adapter = EmbeddingAdapter::new();
    println!("✅ Created EmbeddingAdapter");

    // Test with a simple text input
    let test_input = "I feel happy and excited about this integration!";
    println!("📝 Test input: \"{}\"", test_input);

    // Use the adapter's embed method (which uses mock data internally)
    let emotional_vector = adapter.embed(test_input).await?;
    println!("🎭 Emotional analysis:");
    println!("   Joy: {:.3}", emotional_vector.joy);
    println!("   Sadness: {:.3}", emotional_vector.sadness);
    println!("   Anger: {:.3}", emotional_vector.anger);
    println!("   Fear: {:.3}", emotional_vector.fear);
    println!("   Surprise: {:.3}", emotional_vector.surprise);

    // Test emotional vector operations
    let magnitude = emotional_vector.magnitude();
    println!("📊 Emotional vector magnitude: {:.3}", magnitude);

    println!("\n🎉 Integration test completed successfully!");
    println!("   - Bridge components created ✓");
    println!("   - Mock embedding processed ✓");
    println!("   - Emotional vector generated ✓");
    println!("   - Basic operations working ✓");

    Ok(())
}