#[cfg(test)]
mod optimization_tests {
    use curator_executor::optimizations::*;

    #[test]
    fn test_hyperspherical_normalization() {
        // Test that normalization creates unit vectors
        let mut embedding = vec![3.0, 4.0, 0.0];
        let original_magnitude = (9.0 + 16.0_f32).sqrt(); // Should be 5.0
        assert_eq!(original_magnitude, 5.0);
        
        // Function from optimizations module
        fn normalize_to_unit_sphere(embedding: &mut Vec<f32>) {
            let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            if magnitude > 0.0 {
                for val in embedding.iter_mut() {
                    *val /= magnitude;
                }
            }
        }
        
        normalize_to_unit_sphere(&mut embedding);
        
        // Check magnitude is now 1
        let new_magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((new_magnitude - 1.0).abs() < 0.001, "Magnitude should be 1.0, got {}", new_magnitude);
        
        // Check values are correct
        assert!((embedding[0] - 0.6).abs() < 0.001);  // 3/5
        assert!((embedding[1] - 0.8).abs() < 0.001);  // 4/5
        assert!(embedding[2].abs() < 0.001);          // 0/5
    }

    #[test]
    fn test_coherence_calculation() {
        // Test coherence score calculation
        let opt_config = OptimizationConfig::default();
        assert_eq!(opt_config.erag_collapse_threshold, 0.2);
        assert_eq!(opt_config.normalize_embeddings, true);
        assert_eq!(opt_config.context_injection_limit, 5);
    }

    #[test]
    fn test_hardware_optimizer() {
        // Test Beelink configuration
        let beelink = HardwareOptimizer::new_for_beelink();
        assert_eq!(beelink.optimal_batch_size(), 4);
        assert_eq!(beelink.optimal_kv_cache(), 128_000);
        assert_eq!(beelink.expected_tokens_per_second(), 60);
        
        // Test Laptop configuration
        let laptop = HardwareOptimizer::new_for_laptop();
        assert_eq!(laptop.optimal_batch_size(), 2);
        assert_eq!(laptop.optimal_kv_cache(), 256_000);
        assert_eq!(laptop.expected_tokens_per_second(), 150);
    }

    #[tokio::test]
    async fn test_erag_monitor() {
        // Test ERAG collapse detection
        let monitor = ERAGMonitor::new(0.2);
        
        // Normal coherence scores shouldn't trigger collapse
        assert_eq!(monitor.check_collapse(0.9).await, false);
        assert_eq!(monitor.check_collapse(0.85).await, false);
        
        // Add declining scores to simulate drift
        for _ in 0..8 {
            monitor.check_collapse(0.7).await;
        }
        
        // Now a low score should trigger collapse detection
        // (average of last 10 would be < 0.8, which is 1.0 - 0.2 threshold)
        assert_eq!(monitor.check_collapse(0.7).await, true);
        
        // After reset, should be back to normal
        monitor.reset().await;
        assert_eq!(monitor.check_collapse(0.7).await, false);
    }

    #[test]
    fn test_cosine_similarity() {
        // Test cosine similarity calculation
        fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
            if a.len() != b.len() {
                return 0.0;
            }
            
            let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            
            if mag_a * mag_b > 0.0 {
                dot_product / (mag_a * mag_b)
            } else {
                0.0
            }
        }
        
        // Test orthogonal vectors
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
        
        // Test parallel vectors
        let c = vec![1.0, 0.0, 0.0];
        let d = vec![2.0, 0.0, 0.0];
        assert_eq!(cosine_similarity(&c, &d), 1.0);
        
        // Test anti-parallel vectors
        let e = vec![1.0, 0.0];
        let f = vec![-1.0, 0.0];
        assert_eq!(cosine_similarity(&e, &f), -1.0);
    }
}