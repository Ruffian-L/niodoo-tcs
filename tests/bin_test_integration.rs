//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

//! E2E Integration Test: TCS → Niodoo → vLLM

use niodoo_consciousness::niodoo_tcs_bridge::NiodooTcsBridge;
use niodoo_core::rag_integration::{Document, EmotionalVector};
use anyhow::Result;
use tracing_subscriber;
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<()> {
    println!("DEBUG: Entered main function");
    std::io::Write::flush(&mut std::io::stdout()).unwrap();
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("🚀 Starting TCS → Niodoo E2E Integration Test - DEBUG VERSION");
    std::io::Write::flush(&mut std::io::stdout()).unwrap();

    // Initialize bridge
    let model_path = std::env::var("QWEN_MODEL_PATH")
        .unwrap_or_else(|_| "/home/beelink/models/Qwen2.5-7B-Instruct-AWQ".to_string());

    #[cfg(feature = "onnx")]
    let mut bridge = NiodooTcsBridge::new(&model_path).await?;
    #[cfg(not(feature = "onnx"))]
    let mut bridge = {
        println!("⚠️  Building without ONNX - using mock bridge");
        NiodooTcsBridge::new(&model_path).await?
    };

    println!("✅ Bridge initialized");
    std::io::Write::flush(&mut std::io::stdout()).unwrap();

    println!("🔄 About to load knowledge base...");
    std::io::Write::flush(&mut std::io::stdout()).unwrap();
    // Load sample knowledge base data
    match load_sample_knowledge_base(&mut bridge) {
        Ok(_) => println!("✅ Knowledge base loaded successfully"),
        Err(e) => println!("❌ Failed to load knowledge base: {}", e),
    }
    std::io::Write::flush(&mut std::io::stdout()).unwrap();

    println!("🔄 About to test ERAG retrieval...");
    std::io::Write::flush(&mut std::io::stdout()).unwrap();
    // Test retrieval directly
    match test_erag_retrieval(&mut bridge) {
        Ok(_) => println!("✅ ERAG retrieval tested successfully"),
        Err(e) => println!("❌ Failed to test ERAG retrieval: {}", e),
    }
    std::io::Write::flush(&mut std::io::stdout()).unwrap();

    // Test inputs
    let test_cases = vec![
        "I'm feeling frustrated with this code",
        "This is amazing! Everything is working!",
        "I'm confused about how this topology works",
        "How do I implement a neural network?",
        "What's the best way to handle errors in Rust?",
    ];

    for (i, input) in test_cases.iter().enumerate() {
        println!("\n🧪 Test Case {}: {}", i + 1, input);

        let response = bridge.process(input).await?;
        println!("📤 Response:\n{}", response);
    }

    println!("\n✅ All tests completed!");
    Ok(())
}

fn test_erag_retrieval(bridge: &mut NiodooTcsBridge) -> Result<()> {
    println!("🔍 ENTERING test_erag_retrieval function...");
    std::io::Write::flush(&mut std::io::stdout()).unwrap();

    let rag = bridge.rag_engine();

    // Test with a query that should match our documents
    let test_emotion = EmotionalVector::new(0.1, -0.2, 0.0, 0.1, 0.3); // Similar to rust document
    let results = rag.retrieve(&test_emotion, 3);

    println!("📊 Retrieved {} documents for rust-like emotion", results.len());
    for (doc, score) in results {
        println!("  - {} (score: {:.3})", doc.id, score);
    }

    // Test with another emotion
    let test_emotion2 = EmotionalVector::new(0.4, -0.1, 0.0, 0.2, 0.5); // Similar to neural network document
    let results2 = rag.retrieve(&test_emotion2, 3);

    println!("📊 Retrieved {} documents for ML-like emotion", results2.len());
    for (doc, score) in results2 {
        println!("  - {} (score: {:.3})", doc.id, score);
    }

    Ok(())
}

fn load_sample_knowledge_base(bridge: &mut NiodooTcsBridge) -> Result<()> {
    println!("📚 ENTERING load_sample_knowledge_base function...");
    std::io::Write::flush(&mut std::io::stdout()).unwrap();

    // Get access to the bridge's RAG engine
    let rag_engine = bridge.rag_engine();

    // Sample knowledge base documents
    let sample_docs = vec![
        Document {
            id: "rust_error_handling".to_string(),
            content: "In Rust, errors are handled using Result<T, E> types. Use ? operator for propagation, match statements for handling, or unwrap() for panics. The best practice is to use Result types and handle errors gracefully rather than panicking.".to_string(),
            embedding: EmotionalVector::new(0.1, -0.2, 0.0, 0.1, 0.3), // Slightly positive, surprised
            metadata: {
                let mut m = HashMap::new();
                m.insert("topic".to_string(), "programming".to_string());
                m.insert("language".to_string(), "rust".to_string());
                m.insert("importance".to_string(), "0.8".to_string());
                m
            },
            created_at: chrono::Utc::now(),
        },
        Document {
            id: "neural_network_basics".to_string(),
            content: "Neural networks consist of layers of interconnected nodes called neurons. Each connection has a weight that gets adjusted during training. Use backpropagation to update weights and gradient descent to minimize loss functions.".to_string(),
            embedding: EmotionalVector::new(0.4, -0.1, 0.0, 0.2, 0.5), // Joyful, surprised
            metadata: {
                let mut m = HashMap::new();
                m.insert("topic".to_string(), "machine_learning".to_string());
                m.insert("difficulty".to_string(), "intermediate".to_string());
                m.insert("importance".to_string(), "0.9".to_string());
                m
            },
            created_at: chrono::Utc::now(),
        },
        Document {
            id: "topology_math".to_string(),
            content: "Topology studies properties of spaces that are preserved under continuous deformations. Key concepts include continuity, compactness, connectedness, and homotopy. Möbius strips and Klein bottles are examples of non-orientable surfaces.".to_string(),
            embedding: EmotionalVector::new(0.2, 0.1, -0.1, 0.3, 0.6), // Curious, surprised
            metadata: {
                let mut m = HashMap::new();
                m.insert("topic".to_string(), "mathematics".to_string());
                m.insert("field".to_string(), "topology".to_string());
                m.insert("importance".to_string(), "0.7".to_string());
                m
            },
            created_at: chrono::Utc::now(),
        },
        Document {
            id: "consciousness_theory".to_string(),
            content: "Consciousness involves self-awareness, qualia, and subjective experience. Theories include integrated information theory (IIT), global workspace theory, and quantum consciousness. Emotional processing plays a key role in conscious decision-making.".to_string(),
            embedding: EmotionalVector::new(0.3, 0.0, 0.1, 0.2, 0.4), // Thoughtful, surprised
            metadata: {
                let mut m = HashMap::new();
                m.insert("topic".to_string(), "philosophy".to_string());
                m.insert("field".to_string(), "consciousness".to_string());
                m.insert("importance".to_string(), "0.95".to_string());
                m
            },
            created_at: chrono::Utc::now(),
        },
    ];

    // Add documents to RAG engine
    for doc in sample_docs {
        rag_engine.add_document(doc)?;
    }

    println!("✅ Loaded {} knowledge base documents", rag_engine.memory_stats()?.total_documents);

    // Save the knowledge base
    rag_engine.save()?;

    println!("💾 Knowledge base saved to disk");

    Ok(())
}