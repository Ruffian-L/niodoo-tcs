//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham
//!
//! End-to-End Test for Phase 2 Integration
//!
//! Run with: cargo test --package niodoo_real_integrated --test phase2_e2e

use anyhow::Result;
use niodoo_real_integrated::conversation_log::{ConversationEntry, ConversationLogStore};
use niodoo_real_integrated::emotional_graph::EmotionalGraphBuilder;
use niodoo_real_integrated::graph_exporter::GraphExporter;
use niodoo_core::memory::EmotionalVector;
use std::path::PathBuf;
use tempfile::TempDir;
use tracing::info;

#[tokio::test]
async fn test_phase2_e2e_integration() -> Result<()> {
    // Initialize logging
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init();

    info!("=== Phase 2 E2E Test: Full Pipeline Integration ===");

    // Create temporary directory for test data
    let temp_dir = TempDir::new()?;
    let conversation_log_path = temp_dir.path().join("conversations.json");
    let graph_export_path = temp_dir.path().join("graph_export.json");

    // Test Phase 2 modules standalone
    test_phase2_modules_standalone(&conversation_log_path, &graph_export_path).await?;

    info!("=== Phase 2 E2E Test Complete ===");
    Ok(())
}

/// Test Phase 2 modules standalone
async fn test_phase2_modules_standalone(
    conversation_log_path: &PathBuf,
    graph_export_path: &PathBuf,
) -> Result<()> {
    info!("=== Testing Phase 2 Modules Standalone ===");

    // Test ConversationLogStore
    info!("Testing ConversationLogStore...");
    let mut log_store = ConversationLogStore::new(conversation_log_path);
    log_store.load()?;

    let test_entries = vec![
        ConversationEntry::new(
            "Hello".to_string(),
            "Hi there!".to_string(),
            EmotionalVector::new(0.8, 0.1, 0.0, 0.0, 0.1),
            "joyful".to_string(),
        ),
        ConversationEntry::new(
            "How are you?".to_string(),
            "I'm doing well!".to_string(),
            EmotionalVector::new(0.7, 0.2, 0.0, 0.0, 0.1),
            "joyful".to_string(),
        ),
        ConversationEntry::new(
            "Tell me about sadness".to_string(),
            "Sadness is an emotion...".to_string(),
            EmotionalVector::new(0.2, 0.8, 0.0, 0.0, 0.0),
            "sad".to_string(),
        ),
    ];

    for entry in test_entries {
        log_store.store(entry)?;
    }
    log_store.save()?;
    assert_eq!(log_store.count(), 3);
    info!("✅ ConversationLogStore test passed");

    // Test EmotionalGraphBuilder
    info!("Testing EmotionalGraphBuilder...");
    let mut graph_builder = EmotionalGraphBuilder::default();
    graph_builder.build_from_conversations(&log_store)?;
    
    assert!(graph_builder.sphere_count() >= 3);
    info!("✅ EmotionalGraphBuilder test passed");

    // Test GraphExporter
    info!("Testing GraphExporter...");
    let graph = graph_builder.graph();
    let export = GraphExporter::export_to_json(graph, graph_export_path)?;
    
    assert_eq!(export.nodes.len(), graph_builder.sphere_count());
    info!("✅ GraphExporter test passed");

    info!("=== Standalone Module Tests Complete ===");
    Ok(())
}

#[tokio::test]
async fn test_phase2_query_capabilities() -> Result<()> {
    info!("=== Phase 2 Query Capabilities Test ===");

    let temp_dir = TempDir::new()?;
    let log_path = temp_dir.path().join("test_conversations.json");
    
    let mut log_store = ConversationLogStore::new(&log_path);
    
    // Add diverse conversations
    let entries = vec![
        ConversationEntry::new(
            "Happy topic".to_string(),
            "Great response".to_string(),
            EmotionalVector::new(0.9, 0.0, 0.0, 0.0, 0.1),
            "joyful".to_string(),
        ),
        ConversationEntry::new(
            "Sad topic".to_string(),
            "Somber response".to_string(),
            EmotionalVector::new(0.0, 0.9, 0.0, 0.0, 0.0),
            "sad".to_string(),
        ),
    ];

    for entry in entries {
        log_store.store(entry)?;
    }

    // Test emotion query
    let query_emotion = EmotionalVector::new(0.8, 0.1, 0.0, 0.0, 0.1);
    let results = log_store.query_by_emotion(&query_emotion, 0.5, 10);
    assert!(!results.is_empty(), "Should find at least one matching conversation");
    info!("✅ Emotion query test passed");

    // Test content query - use lower threshold for better matching
    let results = log_store.query_by_content("Happy", 0.1, 10);
    assert!(!results.is_empty(), "Should find content match");
    info!("✅ Content query test passed");

    info!("=== Query Capabilities Test Complete ===");
    Ok(())
}

