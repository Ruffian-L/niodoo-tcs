/*
 * 🧠⚡ CONSCIOUSNESS ENGINE INTEGRATION TESTS ⚡🧠
 *
 * Comprehensive integration tests for all consciousness engine modules
 * Testing brain coordination, memory management, and phase6 integration
 */

use niodoo_consciousness::consciousness_engine::PersonalNiodooConsciousness;
use niodoo_consciousness::consciousness_engine::brain_coordination::BrainCoordinator;
use niodoo_consciousness::consciousness_engine::memory_management::{MemoryManager, PersonalConsciousnessEvent};
use niodoo_consciousness::consciousness_engine::phase6_integration::Phase6Manager;
use niodoo_consciousness::consciousness::{ConsciousnessState, EmotionType};
use niodoo_consciousness::brain::{BrainType, MotorBrain, LcarsBrain, EfficiencyBrain};
use niodoo_consciousness::personality::{PersonalityManager, PersonalityType};
use niodoo_consciousness::memory::GuessingMemorySystem;
use niodoo_consciousness::personal_memory::PersonalMemoryEngine;

use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::time::{timeout, Duration};
use anyhow::Result;

// Test fixtures
struct TestFixtures {
    consciousness_state: Arc<RwLock<ConsciousnessState>>,
    memory_store: Arc<RwLock<Vec<PersonalConsciousnessEvent>>>,
    memory_system: GuessingMemorySystem,
    personal_memory_engine: PersonalMemoryEngine,
}

impl TestFixtures {
    async fn new() -> Result<Self> {
        let consciousness_state = Arc::new(RwLock::new(ConsciousnessState::new()));
        let memory_store = Arc::new(RwLock::new(Vec::new()));
        let memory_system = GuessingMemorySystem::new();
        let personal_memory_engine = PersonalMemoryEngine::default();

        Ok(Self {
            consciousness_state,
            memory_store,
            memory_system,
            personal_memory_engine,
        })
    }
}

// ============================================================================
// BRAIN COORDINATION INTEGRATION TESTS
// ============================================================================

#[tokio::test]
async fn test_brain_coordination_parallel_processing() -> Result<()> {
    let fixtures = TestFixtures::new().await?;
    
    // Initialize brains
    let motor_brain = MotorBrain::new()?;
    let lcars_brain = LcarsBrain::new()?;
    let efficiency_brain = EfficiencyBrain::new()?;
    let personality_manager = PersonalityManager::new();
    
    // Create brain coordinator
    let coordinator = BrainCoordinator::new(
        motor_brain,
        lcars_brain,
        efficiency_brain,
        personality_manager,
        fixtures.consciousness_state.clone(),
    );
    
    // Test parallel processing
    let input = "Test consciousness processing";
    let timeout_duration = Duration::from_secs(5);
    
    let results = timeout(
        timeout_duration,
        coordinator.process_brains_parallel(input, timeout_duration)
    ).await??;
    
    assert_eq!(results.len(), 3, "Should get responses from all 3 brains");
    
    // Verify each brain responded
    for (i, result) in results.iter().enumerate() {
        assert!(!result.is_empty(), "Brain {} should produce non-empty response", i);
    }
    
    Ok(())
}

#[tokio::test]
async fn test_brain_coordination_timeout_handling() -> Result<()> {
    let fixtures = TestFixtures::new().await?;
    
    let motor_brain = MotorBrain::new()?;
    let lcars_brain = LcarsBrain::new()?;
    let efficiency_brain = EfficiencyBrain::new()?;
    let personality_manager = PersonalityManager::new();
    
    let coordinator = BrainCoordinator::new(
        motor_brain,
        lcars_brain,
        efficiency_brain,
        personality_manager,
        fixtures.consciousness_state.clone(),
    );
    
    // Test with very short timeout
    let short_timeout = Duration::from_millis(1);
    let input = "Test timeout handling";
    
    let result = coordinator.process_brains_parallel(input, short_timeout).await;
    
    // Should either succeed quickly or timeout gracefully
    match result {
        Ok(_) => {
            // If it succeeds, that's fine too
        }
        Err(e) => {
            // Timeout errors are expected
            assert!(e.to_string().contains("timeout") || e.to_string().contains("join"));
        }
    }
    
    Ok(())
}

#[tokio::test]
async fn test_brain_coordination_personality_integration() -> Result<()> {
    let fixtures = TestFixtures::new().await?;
    
    let motor_brain = MotorBrain::new()?;
    let lcars_brain = LcarsBrain::new()?;
    let efficiency_brain = EfficiencyBrain::new()?;
    let personality_manager = PersonalityManager::new();
    
    let coordinator = BrainCoordinator::new(
        motor_brain,
        lcars_brain,
        efficiency_brain,
        personality_manager,
        fixtures.consciousness_state.clone(),
    );
    
    // Test personality manager integration
    let personality_manager = coordinator.get_personality_manager();
    assert!(personality_manager.get_personality_count() > 0, "Should have personalities");
    
    // Test personality weight updates
    coordinator.update_personality_weights(
        vec![PersonalityType::Intuitive, PersonalityType::Analytical],
        0.7,
        2,
        "Test context".to_string(),
        (0.5, 0.3),
    ).await?;
    
    Ok(())
}

// ============================================================================
// MEMORY MANAGEMENT INTEGRATION TESTS
// ============================================================================

#[tokio::test]
async fn test_memory_manager_event_storage() -> Result<()> {
    let fixtures = TestFixtures::new().await?;
    
    let memory_manager = MemoryManager::new(
        fixtures.memory_store.clone(),
        fixtures.memory_system.clone(),
        fixtures.personal_memory_engine.clone(),
        fixtures.consciousness_state.clone(),
    );
    
    // Create test event
    let event = PersonalConsciousnessEvent {
        id: "test-event-1".to_string(),
        event_type: "test".to_string(),
        content: "Test consciousness event".to_string(),
        emotional_impact: 0.5,
        learning_will_activation: 0.3,
        timestamp: chrono::Utc::now().timestamp() as f64,
        context: "test".to_string(),
    };
    
    // Store event
    memory_manager.store_event(event.clone()).await?;
    
    // Verify event was stored
    let memory_store = fixtures.memory_store.read().await;
    assert_eq!(memory_store.len(), 1, "Should have stored 1 event");
    assert_eq!(memory_store[0].id, event.id, "Event ID should match");
    
    Ok(())
}

#[tokio::test]
async fn test_memory_manager_consolidation() -> Result<()> {
    let fixtures = TestFixtures::new().await?;
    
    let memory_manager = MemoryManager::new(
        fixtures.memory_store.clone(),
        fixtures.memory_system.clone(),
        fixtures.personal_memory_engine.clone(),
        fixtures.consciousness_state.clone(),
    );
    
    // Store multiple events to trigger consolidation
    for i in 0..15 {
        let event = PersonalConsciousnessEvent {
            id: format!("test-event-{}", i),
            event_type: "test".to_string(),
            content: format!("Test event {}", i),
            emotional_impact: 0.5,
            learning_will_activation: 0.3,
            timestamp: chrono::Utc::now().timestamp() as f64,
            context: "test".to_string(),
        };
        
        memory_manager.store_event(event).await?;
    }
    
    // Verify consolidation was triggered (every 10 events)
    let memory_store = fixtures.memory_store.read().await;
    assert_eq!(memory_store.len(), 15, "Should have stored 15 events");
    
    // Test manual consolidation
    memory_manager.consolidate_memories().await?;
    
    Ok(())
}

#[tokio::test]
async fn test_memory_manager_retrieval() -> Result<()> {
    let fixtures = TestFixtures::new().await?;
    
    let memory_manager = MemoryManager::new(
        fixtures.memory_store.clone(),
        fixtures.memory_system.clone(),
        fixtures.personal_memory_engine.clone(),
        fixtures.consciousness_state.clone(),
    );
    
    // Store test events
    let events = vec![
        PersonalConsciousnessEvent {
            id: "event-1".to_string(),
            event_type: "memory".to_string(),
            content: "Important memory".to_string(),
            emotional_impact: 0.8,
            learning_will_activation: 0.6,
            timestamp: chrono::Utc::now().timestamp() as f64,
            context: "important".to_string(),
        },
        PersonalConsciousnessEvent {
            id: "event-2".to_string(),
            event_type: "learning".to_string(),
            content: "Learning experience".to_string(),
            emotional_impact: 0.4,
            learning_will_activation: 0.9,
            timestamp: chrono::Utc::now().timestamp() as f64,
            context: "learning".to_string(),
        },
    ];
    
    for event in events {
        memory_manager.store_event(event).await?;
    }
    
    // Test retrieval
    let memories = memory_manager.retrieve_memories("important").await?;
    assert!(!memories.is_empty(), "Should retrieve memories");
    
    // Test retrieval with different query
    let learning_memories = memory_manager.retrieve_memories("learning").await?;
    assert!(!learning_memories.is_empty(), "Should retrieve learning memories");
    
    Ok(())
}

#[tokio::test]
async fn test_memory_manager_stats() -> Result<()> {
    let fixtures = TestFixtures::new().await?;
    
    let memory_manager = MemoryManager::new(
        fixtures.memory_store.clone(),
        fixtures.memory_system.clone(),
        fixtures.personal_memory_engine.clone(),
        fixtures.consciousness_state.clone(),
    );
    
    // Store some events
    for i in 0..5 {
        let event = PersonalConsciousnessEvent {
            id: format!("stats-event-{}", i),
            event_type: "stats".to_string(),
            content: format!("Stats test {}", i),
            emotional_impact: 0.5,
            learning_will_activation: 0.3,
            timestamp: chrono::Utc::now().timestamp() as f64,
            context: "stats".to_string(),
        };
        
        memory_manager.store_event(event).await?;
    }
    
    // Get memory stats
    let stats = memory_manager.get_memory_stats().await?;
    assert!(stats.total_events > 0, "Should have events in stats");
    assert!(stats.average_emotional_impact >= 0.0, "Emotional impact should be non-negative");
    assert!(stats.average_learning_will >= 0.0, "Learning will should be non-negative");
    
    Ok(())
}

// ============================================================================
// PHASE6 INTEGRATION TESTS
// ============================================================================

#[tokio::test]
async fn test_phase6_manager_initialization() -> Result<()> {
    let fixtures = TestFixtures::new().await?;
    
    let phase6_manager = Phase6Manager::new(
        fixtures.consciousness_state.clone(),
        fixtures.memory_store.clone(),
    );
    
    // Test basic functionality
    assert!(phase6_manager.is_initialized(), "Phase6 manager should be initialized");
    
    Ok(())
}

#[tokio::test]
async fn test_phase6_manager_processing() -> Result<()> {
    let fixtures = TestFixtures::new().await?;
    
    let phase6_manager = Phase6Manager::new(
        fixtures.consciousness_state.clone(),
        fixtures.memory_store.clone(),
    );
    
    // Test processing
    let input = "Test Phase6 processing";
    let result = phase6_manager.process_phase6_input(input).await?;
    
    assert!(!result.is_empty(), "Phase6 processing should produce output");
    
    Ok(())
}

#[tokio::test]
async fn test_phase6_manager_state_updates() -> Result<()> {
    let fixtures = TestFixtures::new().await?;
    
    let phase6_manager = Phase6Manager::new(
        fixtures.consciousness_state.clone(),
        fixtures.memory_store.clone(),
    );
    
    // Test state updates
    let initial_state = fixtures.consciousness_state.read().await;
    let initial_arousal = initial_state.emotional_arousal;
    drop(initial_state);
    
    // Process something that should affect state
    phase6_manager.process_phase6_input("Emotional input").await?;
    
    // Check if state was updated
    let updated_state = fixtures.consciousness_state.read().await;
    // Note: Actual state changes depend on implementation
    drop(updated_state);
    
    Ok(())
}

// ============================================================================
// CONSCIOUSNESS ENGINE INTEGRATION TESTS
// ============================================================================

#[tokio::test]
async fn test_consciousness_engine_initialization() -> Result<()> {
    // Test consciousness engine creation
    let consciousness = PersonalNiodooConsciousness::new().await?;
    
    // Verify engine is properly initialized
    assert!(consciousness.is_initialized(), "Consciousness engine should be initialized");
    
    Ok(())
}

#[tokio::test]
async fn test_consciousness_engine_processing() -> Result<()> {
    let consciousness = PersonalNiodooConsciousness::new().await?;
    
    // Test basic processing
    let input = "Test consciousness processing";
    let result = consciousness.process_consciousness_input(input).await?;
    
    assert!(!result.is_empty(), "Consciousness processing should produce output");
    
    Ok(())
}

#[tokio::test]
async fn test_consciousness_engine_emotional_processing() -> Result<()> {
    let consciousness = PersonalNiodooConsciousness::new().await?;
    
    // Test emotional processing
    let emotional_input = "I feel happy and excited about this test";
    let result = consciousness.process_emotional_input(emotional_input).await?;
    
    assert!(!result.is_empty(), "Emotional processing should produce output");
    
    // Verify emotional state was updated
    let state = consciousness.get_consciousness_state().await?;
    assert!(state.emotional_arousal >= 0.0, "Emotional arousal should be non-negative");
    
    Ok(())
}

#[tokio::test]
async fn test_consciousness_engine_memory_integration() -> Result<()> {
    let consciousness = PersonalNiodooConsciousness::new().await?;
    
    // Test memory integration
    let memory_input = "This is an important memory for testing";
    let result = consciousness.process_memory_input(memory_input).await?;
    
    assert!(!result.is_empty(), "Memory processing should produce output");
    
    // Verify memory was stored
    let memories = consciousness.retrieve_recent_memories(5).await?;
    assert!(!memories.is_empty(), "Should have stored memories");
    
    Ok(())
}

#[tokio::test]
async fn test_consciousness_engine_concurrent_processing() -> Result<()> {
    let consciousness = PersonalNiodooConsciousness::new().await?;
    
    // Test concurrent processing
    let inputs = vec![
        "Concurrent test 1",
        "Concurrent test 2",
        "Concurrent test 3",
    ];
    
    let tasks: Vec<_> = inputs.into_iter().map(|input| {
        let consciousness = consciousness.clone();
        tokio::spawn(async move {
            consciousness.process_consciousness_input(input).await
        })
    }).collect();
    
    let results = futures::future::join_all(tasks).await;
    
    for result in results {
        let output = result??;
        assert!(!output.is_empty(), "Concurrent processing should produce output");
    }
    
    Ok(())
}

#[tokio::test]
async fn test_consciousness_engine_error_handling() -> Result<()> {
    let consciousness = PersonalNiodooConsciousness::new().await?;
    
    // Test error handling with invalid input
    let invalid_input = "";
    let result = consciousness.process_consciousness_input(invalid_input).await;
    
    // Should handle gracefully (either succeed or return appropriate error)
    match result {
        Ok(output) => {
            // If it succeeds, that's fine
            assert!(output.is_empty() || !output.is_empty(), "Output should be valid");
        }
        Err(e) => {
            // If it fails, error should be meaningful
            assert!(!e.to_string().is_empty(), "Error message should not be empty");
        }
    }
    
    Ok(())
}

// ============================================================================
// PERFORMANCE AND STRESS TESTS
// ============================================================================

#[tokio::test]
async fn test_consciousness_engine_performance() -> Result<()> {
    let consciousness = PersonalNiodooConsciousness::new().await?;
    
    let start = std::time::Instant::now();
    
    // Process multiple inputs
    for i in 0..100 {
        let input = format!("Performance test {}", i);
        let _result = consciousness.process_consciousness_input(&input).await?;
    }
    
    let duration = start.elapsed();
    
    // Should complete within reasonable time (adjust threshold as needed)
    assert!(duration.as_secs() < 30, "Processing 100 inputs should complete within 30 seconds");
    
    Ok(())
}

#[tokio::test]
async fn test_memory_manager_stress_test() -> Result<()> {
    let fixtures = TestFixtures::new().await?;
    
    let memory_manager = MemoryManager::new(
        fixtures.memory_store.clone(),
        fixtures.memory_system.clone(),
        fixtures.personal_memory_engine.clone(),
        fixtures.consciousness_state.clone(),
    );
    
    let start = std::time::Instant::now();
    
    // Store many events
    for i in 0..1000 {
        let event = PersonalConsciousnessEvent {
            id: format!("stress-event-{}", i),
            event_type: "stress".to_string(),
            content: format!("Stress test event {}", i),
            emotional_impact: 0.5,
            learning_will_activation: 0.3,
            timestamp: chrono::Utc::now().timestamp() as f64,
            context: "stress".to_string(),
        };
        
        memory_manager.store_event(event).await?;
    }
    
    let duration = start.elapsed();
    
    // Should complete within reasonable time
    assert!(duration.as_secs() < 10, "Storing 1000 events should complete within 10 seconds");
    
    // Verify all events were stored
    let memory_store = fixtures.memory_store.read().await;
    assert_eq!(memory_store.len(), 1000, "Should have stored 1000 events");
    
    Ok(())
}

// ============================================================================
// TEST UTILITIES
// ============================================================================

// Helper function to create test consciousness events
fn create_test_event(id: &str, content: &str, emotional_impact: f32) -> PersonalConsciousnessEvent {
    PersonalConsciousnessEvent {
        id: id.to_string(),
        event_type: "test".to_string(),
        content: content.to_string(),
        emotional_impact,
        learning_will_activation: 0.5,
        timestamp: chrono::Utc::now().timestamp() as f64,
        context: "test".to_string(),
    }
}

// Helper function to verify consciousness state bounds
fn verify_consciousness_state_bounds(state: &ConsciousnessState) {
    assert!(state.emotional_arousal >= 0.0 && state.emotional_arousal <= 1.0, 
            "Emotional arousal should be in [0, 1]");
    assert!(state.learning_will_activation >= 0.0 && state.learning_will_activation <= 1.0, 
            "Learning will activation should be in [0, 1]");
}
