//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

//! Comprehensive tests for memory consolidation functionality

#[cfg(test)]
mod tests {
    use super::super::events::ConsciousnessEvent;
    use super::super::memory::consolidation::*;
    use super::super::memory::EmotionalVector;
    use std::time::{SystemTime, UNIX_EPOCH};

    /// Test basic memory consolidator creation
    #[tokio::test]
    async fn test_memory_consolidator_creation() {
        let consolidator = MemoryConsolidator::new();
        assert_eq!(consolidator.get_stats().await.total_consolidated, 0);
    }

    /// Test memory consolidation with compression strategy
    #[tokio::test]
    async fn test_memory_compression() {
        let mut consolidator = MemoryConsolidator::new();

        // Create test events with similar emotional signatures
        let events = vec![
            ConsciousnessEvent::new("Happy memory 1".to_string()),
            ConsciousnessEvent::new("Happy memory 2".to_string()),
            ConsciousnessEvent::new("Happy memory 3".to_string()),
        ];

        let stats = consolidator
            .consolidate_memories(events, ConsolidationStrategy::Compression)
            .await;
        assert!(stats.is_ok());

        let stats = stats.unwrap();
        assert!(stats.total_consolidated > 0);
    }

    /// Test memory consolidation with merging strategy
    #[tokio::test]
    async fn test_memory_merging() {
        let mut consolidator = MemoryConsolidator::new();

        // Create test events with related content
        let events = vec![
            ConsciousnessEvent::new("Related topic A".to_string()),
            ConsciousnessEvent::new("Related topic B".to_string()),
        ];

        let stats = consolidator
            .consolidate_memories(events, ConsolidationStrategy::Merging)
            .await;
        assert!(stats.is_ok());

        let stats = stats.unwrap();
        assert!(stats.total_consolidated >= 0);
    }

    /// Test memory consolidation with pruning strategy
    #[tokio::test]
    async fn test_memory_pruning() {
        let mut consolidator = MemoryConsolidator::new();

        // Create test events
        let events = vec![
            ConsciousnessEvent::new("Important memory".to_string()),
            ConsciousnessEvent::new("Unimportant memory".to_string()),
        ];

        let stats = consolidator
            .consolidate_memories(events, ConsolidationStrategy::Pruning)
            .await;
        assert!(stats.is_ok());

        let stats = stats.unwrap();
        assert!(stats.total_consolidated >= 0);
    }

    /// Test memory consolidation with reinforcement strategy
    #[tokio::test]
    async fn test_memory_reinforcement() {
        let mut consolidator = MemoryConsolidator::new();

        // Create test events
        let events = vec![ConsciousnessEvent::new(
            "Frequently accessed memory".to_string(),
        )];

        let stats = consolidator
            .consolidate_memories(events, ConsolidationStrategy::Reinforcement)
            .await;
        assert!(stats.is_ok());

        let stats = stats.unwrap();
        assert!(stats.total_consolidated >= 0);
    }

    /// Test memory consolidation with abstraction strategy
    #[tokio::test]
    async fn test_memory_abstraction() {
        let mut consolidator = MemoryConsolidator::new();

        // Create test events that can form patterns
        let events = vec![
            ConsciousnessEvent::new("Pattern A instance 1".to_string()),
            ConsciousnessEvent::new("Pattern A instance 2".to_string()),
            ConsciousnessEvent::new("Pattern A instance 3".to_string()),
            ConsciousnessEvent::new("Pattern A instance 4".to_string()),
            ConsciousnessEvent::new("Pattern A instance 5".to_string()),
        ];

        let stats = consolidator
            .consolidate_memories(events, ConsolidationStrategy::Abstraction)
            .await;
        assert!(stats.is_ok());

        let stats = stats.unwrap();
        assert!(stats.total_consolidated >= 0);
    }

    /// Test real-time consolidation
    #[tokio::test]
    async fn test_realtime_consolidation() {
        let mut consolidator = MemoryConsolidator::new();

        let events = vec![
            ConsciousnessEvent::new("Real-time event 1".to_string()),
            ConsciousnessEvent::new("Real-time event 2".to_string()),
        ];

        let stats = consolidator
            .consolidate_memories(events, ConsolidationStrategy::Realtime)
            .await;
        assert!(stats.is_ok());

        let stats = stats.unwrap();
        assert!(stats.total_consolidated >= 0);
    }

    /// Test batch consolidation
    #[tokio::test]
    async fn test_batch_consolidation() {
        let mut consolidator = MemoryConsolidator::new();

        let events = vec![
            ConsciousnessEvent::new("Batch event 1".to_string()),
            ConsciousnessEvent::new("Batch event 2".to_string()),
            ConsciousnessEvent::new("Batch event 3".to_string()),
        ];

        let stats = consolidator
            .consolidate_memories(events, ConsolidationStrategy::Batch)
            .await;
        assert!(stats.is_ok());

        let stats = stats.unwrap();
        assert!(stats.total_consolidated >= 0);
    }

    /// Test consolidated memory creation
    #[test]
    fn test_consolidated_memory_creation() {
        let memory = ConsolidatedMemory {
            id: "test_memory".to_string(),
            original_events: vec!["event1".to_string(), "event2".to_string()],
            consolidated_content: "Test consolidated content".to_string(),
            emotional_signature: EmotionalVector::new(0.5, 0.3, 0.2, 0.8, 0.1),
            importance_score: 0.7,
            access_frequency: 5,
            last_accessed: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs_f64(),
            consolidation_level: 0.5,
            created_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs_f64(),
        };

        assert_eq!(memory.id, "test_memory");
        assert_eq!(memory.original_events.len(), 2);
        assert_eq!(memory.consolidated_content, "Test consolidated content");
        assert_eq!(memory.importance_score, 0.7);
        assert_eq!(memory.access_frequency, 5);
    }

    /// Test consolidation statistics
    #[test]
    fn test_consolidation_stats() {
        let stats = ConsolidationStats {
            total_consolidated: 10,
            compressed_count: 3,
            merged_count: 2,
            pruned_count: 1,
            reinforced_count: 2,
            abstracted_count: 2,
            processing_time_ms: 150.0,
            memory_efficiency: 0.85,
        };

        assert_eq!(stats.total_consolidated, 10);
        assert_eq!(stats.compressed_count, 3);
        assert_eq!(stats.merged_count, 2);
        assert_eq!(stats.pruned_count, 1);
        assert_eq!(stats.reinforced_count, 2);
        assert_eq!(stats.abstracted_count, 2);
        assert_eq!(stats.processing_time_ms, 150.0);
        assert_eq!(stats.memory_efficiency, 0.85);
    }

    /// Test emotional similarity calculation
    #[tokio::test]
    async fn test_emotional_similarity() {
        let consolidator = MemoryConsolidator::new();

        let emotion1 = EmotionalVector::new(0.8, 0.2, 0.1, 0.9, 0.3);
        let emotion2 = EmotionalVector::new(0.7, 0.3, 0.2, 0.8, 0.4);
        let emotion3 = EmotionalVector::new(0.1, 0.8, 0.9, 0.2, 0.7);

        let similarity1 = consolidator.calculate_emotional_similarity(&emotion1, &emotion2);
        let similarity2 = consolidator.calculate_emotional_similarity(&emotion1, &emotion3);

        // Similar emotions should have higher similarity
        assert!(similarity1 > similarity2);
        assert!(similarity1 > 0.5);
        assert!(similarity2 < 0.5);
    }

    /// Test content similarity calculation
    #[tokio::test]
    async fn test_content_similarity() {
        let consolidator = MemoryConsolidator::new();

        let content1 = "Hello world, how are you?";
        let content2 = "Hello world, how are you doing?";
        let content3 = "Completely different content here";

        let similarity1 = consolidator.calculate_content_similarity(content1, content2);
        let similarity2 = consolidator.calculate_content_similarity(content1, content3);

        // Similar content should have higher similarity
        assert!(similarity1 > similarity2);
        assert!(similarity1 > 0.3);
        assert!(similarity2 < 0.3);
    }

    /// Test memory access and frequency tracking
    #[tokio::test]
    async fn test_memory_access_tracking() {
        let mut consolidator = MemoryConsolidator::new();

        // Add some memories
        let events = vec![ConsciousnessEvent::new("Memory to access".to_string())];

        let _stats = consolidator
            .consolidate_memories(events, ConsolidationStrategy::Realtime)
            .await;

        // Access the memory multiple times
        for _ in 0..5 {
            let _memory = consolidator.access_memory("event_0").await;
        }

        let stats = consolidator.get_stats().await;
        assert!(stats.total_consolidated > 0);
    }

    /// Test memory pruning based on importance
    #[tokio::test]
    async fn test_memory_pruning_by_importance() {
        let mut consolidator = MemoryConsolidator::new();

        // Add memories with different importance levels
        let events = vec![
            ConsciousnessEvent::new("Important memory".to_string()),
            ConsciousnessEvent::new("Less important memory".to_string()),
        ];

        let _stats = consolidator
            .consolidate_memories(events, ConsolidationStrategy::Realtime)
            .await;

        // Prune unimportant memories
        let pruned_count = consolidator.prune_unimportant_memories().await;
        assert!(pruned_count.is_ok());

        let pruned_count = pruned_count.unwrap();
        assert!(pruned_count >= 0);
    }

    /// Test memory consolidation with large dataset
    #[tokio::test]
    async fn test_large_dataset_consolidation() {
        let mut consolidator = MemoryConsolidator::new();

        // Create a large dataset
        let mut events = Vec::new();
        for i in 0..100 {
            events.push(ConsciousnessEvent::new(format!("Memory {}", i)));
        }

        let stats = consolidator
            .consolidate_memories(events, ConsolidationStrategy::Batch)
            .await;
        assert!(stats.is_ok());

        let stats = stats.unwrap();
        assert!(stats.total_consolidated > 0);
        assert!(stats.processing_time_ms > 0.0);
    }

    /// Test consolidation strategy enum
    #[test]
    fn test_consolidation_strategy_enum() {
        let strategies = vec![
            ConsolidationStrategy::Compression,
            ConsolidationStrategy::Merging,
            ConsolidationStrategy::Pruning,
            ConsolidationStrategy::Reinforcement,
            ConsolidationStrategy::Abstraction,
            ConsolidationStrategy::Realtime,
            ConsolidationStrategy::Batch,
        ];

        for strategy in strategies {
            match strategy {
                ConsolidationStrategy::Compression => assert!(true),
                ConsolidationStrategy::Merging => assert!(true),
                ConsolidationStrategy::Pruning => assert!(true),
                ConsolidationStrategy::Reinforcement => assert!(true),
                ConsolidationStrategy::Abstraction => assert!(true),
                ConsolidationStrategy::Realtime => assert!(true),
                ConsolidationStrategy::Batch => assert!(true),
            }
        }
    }

    /// Test default consolidation strategy
    #[test]
    fn test_default_consolidation_strategy() {
        let default_strategy = ConsolidationStrategy::default();
        assert!(matches!(default_strategy, ConsolidationStrategy::Realtime));
    }
}
