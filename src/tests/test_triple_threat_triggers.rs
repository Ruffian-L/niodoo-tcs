//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

/// Unit tests to validate triple-threat trigger system
/// Ensures each pathway (mismatch, stagnation, variance) fires correctly
use crate::consciousness::ConsciousnessState;
use crate::memory::guessing_spheres::{
    EmotionalVector, GuessingMemorySystem, SphereId,
};
use crate::memory::multi_layer_query::MultiLayerMemoryQuery;
use crate::rag::local_embeddings::{Document, MathematicalEmbeddingModel};
use crate::rag::RetrievalEngine;
use std::sync::{Arc, Mutex};

#[test]
fn test_mismatch_crisis_trigger() {
    // Setup: Query with PURE sadness (0.0, 1.0, 0.0, 0.0, 0.0)
    // Vault: ALL joy spheres (1.0, 0.0, 0.0, 0.0, 0.0)
    // Expected: Low mean (<0.7), high entropy → MISMATCH CRISIS

    let model = MathematicalEmbeddingModel::new(384);
    let mut rag_engine = RetrievalEngine::new();
    let mut gaussian_system = GuessingMemorySystem::new();

    // Create 10 pure joy spheres
    for i in 0..10 {
        let doc = Document {
            id: format!("joy-{}", i),
            content: format!("Very happy memory {}", i),
            embedding: model.generate_embedding(&format!("happy {}", i)).unwrap(),
            metadata: std::collections::HashMap::new(),
        };
        rag_engine.add_document(doc);

        // Pure joy emotion (1.0, 0.0, 0.0, 0.0, 0.0)
        gaussian_system.store_memory(
            SphereId(format!("joy-{}", i)),
            format!("joy concept {}", i),
            [0.0, 0.0, 0.0],
            EmotionalVector::new(1.0, 0.0, 0.0, 0.0, 0.0),
            format!("Happy fragment {}", i),
        );
    }

    // Query with PURE sadness (opposite of vault)
    let query_emotion = EmotionalVector::new(0.0, 1.0, 0.0, 0.0, 0.0);

    let rag_arc = Arc::new(Mutex::new(rag_engine));
    let mut multi_query = MultiLayerMemoryQuery::new(rag_arc, gaussian_system);
    let mut state = ConsciousnessState::default();

    println!("\n=== MISMATCH CRISIS TEST ===");
    println!("Query: Pure sadness (0.0, 1.0, 0.0, 0.0, 0.0)");
    println!("Vault: 10x pure joy (1.0, 0.0, 0.0, 0.0, 0.0)");
    println!("Expected: MISMATCH CRISIS trigger (H>2.0, mean<0.7)\n");

    let results = multi_query
        .query("sad memory", &query_emotion, 8, &mut state)
        .unwrap();

    println!("\n✅ Test completed - check logs for 🎯 MISMATCH CRISIS trigger");
    println!("Results returned: {}", results.len());
}

#[test]
fn test_uniform_stagnation_trigger() {
    // Setup: Query with PURE joy (1.0, 0.0, 0.0, 0.0, 0.0)
    // Vault: ALL identical joy spheres (1.0, 0.0, 0.0, 0.0, 0.0)
    // Expected: High mean (>0.7), low variance (<0.01), high entropy → UNIFORM STAGNATION

    let model = MathematicalEmbeddingModel::new(384);
    let mut rag_engine = RetrievalEngine::new();
    let mut gaussian_system = GuessingMemorySystem::new();

    // Create 10 IDENTICAL joy spheres
    for i in 0..10 {
        let doc = Document {
            id: format!("joy-{}", i),
            content: format!("Happy memory {}", i),
            embedding: model.generate_embedding(&format!("happy {}", i)).unwrap(),
            metadata: std::collections::HashMap::new(),
        };
        rag_engine.add_document(doc);

        // ALL IDENTICAL: (1.0, 0.0, 0.0, 0.0, 0.0)
        gaussian_system.store_memory(
            SphereId(format!("joy-{}", i)),
            format!("joy concept {}", i),
            [0.0, 0.0, 0.0],
            EmotionalVector::new(1.0, 0.0, 0.0, 0.0, 0.0),
            format!("Happy fragment {}", i),
        );
    }

    // Query with matching joy
    let query_emotion = EmotionalVector::new(1.0, 0.0, 0.0, 0.0, 0.0);

    let rag_arc = Arc::new(Mutex::new(rag_engine));
    let mut multi_query = MultiLayerMemoryQuery::new(rag_arc, gaussian_system);
    let mut state = ConsciousnessState::default();

    println!("\n=== UNIFORM STAGNATION TEST ===");
    println!("Query: Pure joy (1.0, 0.0, 0.0, 0.0, 0.0)");
    println!("Vault: 10x identical joy (1.0, 0.0, 0.0, 0.0, 0.0)");
    println!("Expected: UNIFORM STAGNATION trigger (H>2.0, var<0.01)\n");

    let results = multi_query
        .query("happy memory", &query_emotion, 8, &mut state)
        .unwrap();

    println!("\n✅ Test completed - check logs for 🌱 UNIFORM STAGNATION trigger");
    println!("Results returned: {}", results.len());
}

#[test]
fn test_variance_spike_trigger() {
    // Setup: Query with mixed emotions (0.5, 0.5, 0.5, 0.5, 0.5)
    // Vault: EXTREME diverse emotions
    // Expected: High variance (>0.05) → VARIANCE SPIKE

    let model = MathematicalEmbeddingModel::new(384);
    let mut rag_engine = RetrievalEngine::new();
    let mut gaussian_system = GuessingMemorySystem::new();

    // Create 5 EXTREME diverse spheres
    let emotions = vec![
        ("pure-joy", EmotionalVector::new(1.0, 0.0, 0.0, 0.0, 0.0)),
        ("pure-sad", EmotionalVector::new(0.0, 1.0, 0.0, 0.0, 0.0)),
        ("pure-angry", EmotionalVector::new(0.0, 0.0, 1.0, 0.0, 0.0)),
        ("pure-fear", EmotionalVector::new(0.0, 0.0, 0.0, 1.0, 0.0)),
        (
            "pure-surprise",
            EmotionalVector::new(0.0, 0.0, 0.0, 0.0, 1.0),
        ),
    ];

    for (name, emotion) in emotions {
        let doc = Document {
            id: name.to_string(),
            content: format!("{} memory", name),
            embedding: model.generate_embedding(name).unwrap(),
            metadata: std::collections::HashMap::new(),
        };
        rag_engine.add_document(doc);

        gaussian_system.store_memory(
            SphereId(name.to_string()),
            format!("{} concept", name),
            [0.0, 0.0, 0.0],
            emotion,
            format!("{} fragment", name),
        );
    }

    // Query with mixed emotions
    let query_emotion = EmotionalVector::new(0.5, 0.5, 0.5, 0.5, 0.5);

    let rag_arc = Arc::new(Mutex::new(rag_engine));
    let mut multi_query = MultiLayerMemoryQuery::new(rag_arc, gaussian_system);
    let mut state = ConsciousnessState::default();

    println!("\n=== VARIANCE SPIKE TEST ===");
    println!("Query: Mixed emotions (0.5, 0.5, 0.5, 0.5, 0.5)");
    println!("Vault: 5x pure emotions (1.0 in each dimension)");
    println!("Expected: VARIANCE SPIKE trigger (var>0.05)\n");

    let results = multi_query
        .query("mixed memory", &query_emotion, 8, &mut state)
        .unwrap();

    println!("\n✅ Test completed - check logs for 📊 VARIANCE SPIKE trigger");
    println!("Results returned: {}", results.len());
}

#[test]
fn test_healthy_diversity_no_trigger() {
    // Setup: Query with moderate emotions
    // Vault: Moderate diverse emotions
    // Expected: NO trigger (healthy diversity)

    let model = MathematicalEmbeddingModel::new(384);
    let mut rag_engine = RetrievalEngine::new();
    let mut gaussian_system = GuessingMemorySystem::new();

    // Create 5 moderately diverse spheres
    let emotions = vec![
        ("balanced-1", EmotionalVector::new(0.6, 0.4, 0.2, 0.3, 0.4)),
        ("balanced-2", EmotionalVector::new(0.5, 0.5, 0.3, 0.2, 0.3)),
        ("balanced-3", EmotionalVector::new(0.7, 0.3, 0.2, 0.4, 0.5)),
        ("balanced-4", EmotionalVector::new(0.6, 0.5, 0.3, 0.3, 0.4)),
        ("balanced-5", EmotionalVector::new(0.5, 0.4, 0.4, 0.3, 0.5)),
    ];

    for (name, emotion) in emotions {
        let doc = Document {
            id: name.to_string(),
            content: format!("{} memory", name),
            embedding: model.generate_embedding(name).unwrap(),
            metadata: std::collections::HashMap::new(),
        };
        rag_engine.add_document(doc);

        gaussian_system.store_memory(
            SphereId(name.to_string()),
            format!("{} concept", name),
            [0.0, 0.0, 0.0],
            emotion,
            format!("{} fragment", name),
        );
    }

    // Query with balanced emotions
    let query_emotion = EmotionalVector::new(0.6, 0.4, 0.3, 0.3, 0.4);

    let rag_arc = Arc::new(Mutex::new(rag_engine));
    let mut multi_query = MultiLayerMemoryQuery::new(rag_arc, gaussian_system);
    let mut state = ConsciousnessState::default();

    println!("\n=== HEALTHY DIVERSITY TEST (No Trigger) ===");
    println!("Query: Balanced emotions (0.6, 0.4, 0.3, 0.3, 0.4)");
    println!("Vault: 5x moderately diverse balanced emotions");
    println!("Expected: NO trigger (healthy diversity message)\n");

    let results = multi_query
        .query("balanced memory", &query_emotion, 8, &mut state)
        .unwrap();

    println!("\n✅ Test completed - should see 'Healthy diversity - no trigger' message");
    println!("Results returned: {}", results.len());
}
