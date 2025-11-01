// tests/integration_test.rs - Integration tests for Niodoo consciousness system

#[cfg(test)]
mod tests {
    use niodoo_consciousness::*;
    use tokio::test;
    use std::time::Duration;
    
    #[test]
    async fn test_toroidal_migration() {
        // Create Möbius memories
        let mobius_memories = vec![
            memory::mobius::MemoryFragment {
                content: "Test memory 1".to_string(),
                layer: memory::mobius::MemoryLayer::Semantic,
                relevance: 0.8,
                timestamp: 1.0,
            },
            memory::mobius::MemoryFragment {
                content: "Test memory 2".to_string(),
                layer: memory::mobius::MemoryLayer::Episodic,
                relevance: 0.6,
                timestamp: 2.0,
            },
        ];
        
        // Migrate to toroidal system
        let toroidal_system = memory::toroidal::migrate_mobius_to_torus(mobius_memories).await;
        
        // Verify migration
        let projections = toroidal_system.holographic_projection(0.0).await;
        assert_eq!(projections.len(), 2);
        
        // Test parallel processing
        let results = toroidal_system.process_parallel_streams(0.1).await;
        assert_eq!(results.len(), 3); // 3 default streams
    }
    
    #[test]
    async fn test_consciousness_engine_timeout() {
        let mut engine = consciousness_engine::NiodooConsciousness::new()
            .expect("Failed to create engine");
        
        // Test with timeout protection
        let result = tokio::time::timeout(
            Duration::from_secs(10),
            engine.process_input("Test input")
        ).await;
        
        assert!(result.is_ok());
    }
    
    #[test]
    async fn test_memory_bounds() {
        let mut memory_system = memory::mobius::MobiusMemorySystem::new();
        
        // Try to overflow memory
        for i in 0..10001 {
            let fragment = memory::mobius::MemoryFragment {
                content: format!("Memory {}", i),
                layer: memory::mobius::MemoryLayer::Working,
                relevance: 0.5,
                timestamp: i as f64,
            };
            
            // This should not panic due to bounds checking
            let result = memory_system.bi_directional_traverse(
                &format!("query {}", i),
                "neutral"
            );
            
            // Memory should be bounded
            assert!(memory_system.persistent_memories.len() <= 10000);
        }
    }
    
    #[test]
    async fn test_error_recovery() {
        use error::{MemoryError, ErrorRecovery};
        
        let overflow_error = MemoryError::Overflow { capacity: 10000 };
        let recovery_result = ErrorRecovery::recover_memory(&overflow_error).await;
        assert!(recovery_result.is_ok());
        
        let persistence_error = MemoryError::PersistenceFailed("Test failure".to_string());
        let recovery_result = ErrorRecovery::recover_memory(&persistence_error).await;
        assert!(recovery_result.is_ok());
    }
    
    #[test]
    async fn test_circuit_breaker() {
        use error::{CircuitBreaker, ConsciousnessError};
        use std::sync::Arc;
        
        let breaker = Arc::new(CircuitBreaker::new(3, Duration::from_secs(1)));
        
        // Simulate failures
        for _ in 0..3 {
            let result = breaker.call(|| {
                Err(ConsciousnessError::Unknown("Test failure".to_string()))
            });
            assert!(result.is_err());
        }
        
        // Circuit should be open now
        let result = breaker.call(|| {
            Ok("Should not execute".to_string())
        });
        assert!(result.is_err());
        
        // Wait for reset
        tokio::time::sleep(Duration::from_secs(2)).await;
        
        // Circuit should be closed now
        let result = breaker.call(|| {
            Ok("Should execute".to_string())
        });
        assert!(result.is_ok());
    }
    
    #[test]
    async fn test_toroidal_quantum_correction() {
        let system = memory::toroidal::ToroidalConsciousnessSystem::new(3.0, 1.0);
        
        // Add test memories
        for i in 0..10 {
            let node = memory::toroidal::ToroidalMemoryNode {
                id: format!("test_{}", i),
                coordinate: memory::toroidal::ToroidalCoordinate::new(
                    i as f64 * 0.628,
                    i as f64 * 0.314
                ),
                content: format!("Test content {}", i),
                emotional_vector: vec![0.5],
                temporal_context: vec![i as f64],
                activation_strength: 0.7,
                connections: std::collections::HashMap::new(),
            };
            system.add_memory(node).await;
        }
        
        // Test quantum error correction
        let syndrome = system.quantum_error_correction().await;
        assert!(syndrome >= 0.0 && syndrome <= 1.0);
    }
    
    #[test]
    async fn test_geodesic_distance() {
        use memory::toroidal::ToroidalCoordinate;
        
        let coord1 = ToroidalCoordinate::new(0.0, 0.0);
        let coord2 = ToroidalCoordinate::new(
            std::f64::consts::PI,
            std::f64::consts::PI / 2.0
        );
        
        let distance = coord1.geodesic_distance(&coord2);
        assert!(distance > 0.0);
        
        // Test wrap-around
        let coord3 = ToroidalCoordinate::new(0.1, 0.1);
        let coord4 = ToroidalCoordinate::new(
            2.0 * std::f64::consts::PI - 0.1,
            2.0 * std::f64::consts::PI - 0.1
        );
        
        let wrap_distance = coord3.geodesic_distance(&coord4);
        assert!(wrap_distance < 0.5); // Should be small due to wrap-around
    }
}
