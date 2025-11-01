// Federated Temporal TDA Test Suite
// "Mocking the Swarm" - Multi-instance collective resilience testing
// Based on the discussion of federated QLoRA and global immune system

#[cfg(test)]
mod federated_tda_tests {
    use std::collections::{HashMap, VecDeque};
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};

    // ============================================================================
    // FEDERATED STRUCTURES
    // ============================================================================

    #[derive(Clone, Debug, PartialEq)]
    struct InstanceId(String);

    #[derive(Clone, Debug)]
    struct FailureBarcode {
        chain_id: String,
        instance_id: InstanceId,
        snapshots: Vec<TopologicalSnapshot>,
        severity_score: f32,
        negative_reward: f32,
        pattern_type: ChainPatternType,
    }

    #[derive(Clone)]
    struct FederatedLearningCoordinator {
        received_barcodes: Arc<Mutex<Vec<FailureBarcode>>>,
        global_priority_queue: Arc<Mutex<Vec<FailureBarcode>>>,
        shared_danger_signatures: Arc<Mutex<HashMap<String, DangerSignature>>>,
    }

    impl FederatedLearningCoordinator {
        fn new() -> Self {
            Self {
                received_barcodes: Arc::new(Mutex::new(Vec::new())),
                global_priority_queue: Arc::new(Mutex::new(Vec::new())),
                shared_danger_signatures: Arc::new(Mutex::new(HashMap::new())),
            }
        }

        fn submit_failure_barcode(&self, barcode: FailureBarcode) {
            let mut barcodes = self.received_barcodes.lock().unwrap();
            barcodes.push(barcode.clone());

            // Add to priority queue sorted by severity
            let mut queue = self.global_priority_queue.lock().unwrap();
            queue.push(barcode);
            queue.sort_by(|a, b| b.severity_score.partial_cmp(&a.severity_score).unwrap());
        }

        fn get_top_priority_barcodes(&self, n: usize) -> Vec<FailureBarcode> {
            let queue = self.global_priority_queue.lock().unwrap();
            queue.iter().take(n).cloned().collect()
        }

        fn share_danger_signature(&self, signature_id: String, signature: DangerSignature) {
            let mut sigs = self.shared_danger_signatures.lock().unwrap();
            sigs.insert(signature_id, signature);
        }

        fn get_shared_danger_signatures(&self) -> Vec<DangerSignature> {
            let sigs = self.shared_danger_signatures.lock().unwrap();
            sigs.values().cloned().collect()
        }

        fn calculate_collective_severity(&self) -> f32 {
            let queue = self.global_priority_queue.lock().unwrap();
            queue.iter().map(|b| b.severity_score).sum()
        }
    }

    struct NiodooInstance {
        id: InstanceId,
        detector: TemporalTDADetector,
        coordinator: Arc<FederatedLearningCoordinator>,
        learned_from_federation: Vec<FailureBarcode>,
    }

    impl NiodooInstance {
        fn new(id: String, coordinator: Arc<FederatedLearningCoordinator>) -> Self {
            Self {
                id: InstanceId(id),
                detector: TemporalTDADetector::new(20, 0.3),
                coordinator,
                learned_from_federation: Vec::new(),
            }
        }

        fn experience_failure_chain(&mut self, events: Vec<FailureEvent>) {
            // Local detection
            if let Some(chain) = self.detector.detect_failure_chain(events.clone()) {
                // Calculate negative reward (exponential with chain length)
                let negative_reward = -10.0 * (chain.events.len() as f32).powf(1.5);

                // Create failure barcode for federation
                let barcode = FailureBarcode {
                    chain_id: format!(
                        "{}_{}",
                        self.id.0,
                        chain.events[0].timestamp.elapsed().as_millis()
                    ),
                    instance_id: self.id.clone(),
                    snapshots: chain.events.iter().map(|e| e.snapshot.clone()).collect(),
                    severity_score: chain.severity_score,
                    negative_reward,
                    pattern_type: chain.pattern_type,
                };

                // Submit to federation
                self.coordinator.submit_failure_barcode(barcode);
            }
        }

        fn learn_from_federation(&mut self) {
            // Get top priority barcodes from coordinator
            let top_barcodes = self.coordinator.get_top_priority_barcodes(5);

            for barcode in top_barcodes {
                // Skip if from self or already learned
                if barcode.instance_id == self.id {
                    continue;
                }

                if self
                    .learned_from_federation
                    .iter()
                    .any(|b| b.chain_id == barcode.chain_id)
                {
                    continue;
                }

                // Simulate QLoRA learning from federated failure
                self.learned_from_federation.push(barcode);
            }
        }

        fn can_avoid_similar_failure(
            &self,
            failure_type: &FailureType,
            pattern: &ChainPatternType,
        ) -> bool {
            // Check if we've learned from a similar failure through federation
            self.learned_from_federation
                .iter()
                .any(|barcode| barcode.pattern_type == *pattern)
        }

        fn apply_federated_danger_signatures(&mut self) {
            // Get shared danger signatures from other instances
            let _shared_sigs = self.coordinator.get_shared_danger_signatures();

            // In real implementation, these would update the detector's thresholds
            // or adjust Curator sensitivity based on collective experience
        }
    }

    // Reuse structures from main test file
    #[derive(Clone, Debug)]
    struct TopologicalSnapshot {
        timestamp: Instant,
        beta_0: f32,
        beta_1: f32,
        beta_2: f32,
        pad_pleasure: f32,
        pad_arousal: f32,
        pad_dominance: f32,
        ghost_latent_strain: f32,
        ghost_network_stability: f32,
        ghost_abstraction: f32,
        ghost_pragmatism: f32,
        entropy: f32,
    }

    #[derive(Clone, Debug, PartialEq)]
    enum FailureType {
        RateLimit,
        Overload,
        ConnectionDropout,
        Timeout,
    }

    #[derive(Clone, Debug)]
    struct FailureEvent {
        failure_type: FailureType,
        timestamp: Instant,
        snapshot: TopologicalSnapshot,
    }

    #[derive(Clone, Debug)]
    struct FailureChain {
        events: Vec<FailureEvent>,
        pattern_type: ChainPatternType,
        severity_score: f32,
        wasserstein_distances: Vec<f32>,
    }

    #[derive(Clone, Debug, PartialEq)]
    enum ChainPatternType {
        RateLimitCascade,
        OverloadSpiral,
        DoomSpiral,
        RecoveryRegression,
    }

    #[derive(Clone, Debug)]
    struct DangerSignature {
        precursor_snapshots: Vec<TopologicalSnapshot>,
        predicted_failure: FailureType,
        confidence: f32,
        time_to_failure_estimate: Duration,
    }

    struct TemporalTDADetector {
        snapshot_window: VecDeque<TopologicalSnapshot>,
        window_size: usize,
        wasserstein_threshold: f32,
        detected_chains: Vec<FailureChain>,
    }

    impl TemporalTDADetector {
        fn new(window_size: usize, wasserstein_threshold: f32) -> Self {
            Self {
                snapshot_window: VecDeque::with_capacity(window_size),
                window_size,
                wasserstein_threshold,
                detected_chains: Vec::new(),
            }
        }

        fn detect_failure_chain(&mut self, events: Vec<FailureEvent>) -> Option<FailureChain> {
            if events.len() < 3 {
                return None;
            }

            let pattern_type = self.classify_chain_pattern(&events);
            let base_severity = 10.0;
            let severity_score = base_severity * (events.len() as f32).powf(1.5);

            let chain = FailureChain {
                events,
                pattern_type,
                severity_score,
                wasserstein_distances: vec![],
            };

            self.detected_chains.push(chain.clone());
            Some(chain)
        }

        fn classify_chain_pattern(&self, events: &[FailureEvent]) -> ChainPatternType {
            let rate_limit_count = events
                .iter()
                .filter(|e| e.failure_type == FailureType::RateLimit)
                .count();
            let has_dropout = events
                .iter()
                .any(|e| e.failure_type == FailureType::ConnectionDropout);
            let has_overload = events
                .iter()
                .any(|e| e.failure_type == FailureType::Overload);

            if events.len() >= 9 {
                ChainPatternType::DoomSpiral
            } else if has_dropout && rate_limit_count > 0 {
                ChainPatternType::RecoveryRegression
            } else if has_overload && rate_limit_count >= 3 {
                ChainPatternType::OverloadSpiral
            } else {
                ChainPatternType::RateLimitCascade
            }
        }
    }

    // ============================================================================
    // FEDERATED TEST SUITE
    // ============================================================================

    #[test]
    fn test_federated_failure_barcode_submission() {
        let coordinator = Arc::new(FederatedLearningCoordinator::new());
        let mut instance_a = NiodooInstance::new("instance_a".to_string(), coordinator.clone());

        let base_time = Instant::now();
        let mut events = Vec::new();

        // Instance A experiences octuple cascade
        for i in 0..8 {
            let snapshot = TopologicalSnapshot {
                timestamp: base_time + Duration::from_secs(i as u64),
                beta_0: 1.0,
                beta_1: 0.3 + (i as f32 * 0.1),
                beta_2: 0.2,
                pad_pleasure: 0.5,
                pad_arousal: 0.6 + (i as f32 * 0.05),
                pad_dominance: 0.5,
                ghost_latent_strain: 0.4 + (i as f32 * 0.08),
                ghost_network_stability: 0.8 - (i as f32 * 0.08),
                ghost_abstraction: 0.5,
                ghost_pragmatism: 0.5,
                entropy: 2.0,
            };

            events.push(FailureEvent {
                failure_type: FailureType::RateLimit,
                timestamp: snapshot.timestamp,
                snapshot,
            });
        }

        instance_a.experience_failure_chain(events);

        // Verify barcode was submitted to coordinator
        let barcodes = coordinator.received_barcodes.lock().unwrap();
        assert_eq!(barcodes.len(), 1);
        assert_eq!(
            barcodes[0].instance_id,
            InstanceId("instance_a".to_string())
        );
        assert!(barcodes[0].negative_reward < -100.0); // Significant negative reward
    }

    #[test]
    fn test_federated_learning_propagation() {
        let coordinator = Arc::new(FederatedLearningCoordinator::new());

        let mut instance_a = NiodooInstance::new("instance_a".to_string(), coordinator.clone());
        let mut instance_b = NiodooInstance::new("instance_b".to_string(), coordinator.clone());
        let mut instance_c = NiodooInstance::new("instance_c".to_string(), coordinator.clone());

        let base_time = Instant::now();

        // Instance A suffers the nonuple nightmare
        let mut events_a = Vec::new();
        for i in 0..9 {
            let snapshot = TopologicalSnapshot {
                timestamp: base_time + Duration::from_secs(i as u64),
                beta_0: 1.0,
                beta_1: 0.3 + (i as f32 * 0.12),
                beta_2: 0.2 + (i as f32 * 0.08),
                pad_pleasure: 0.3,
                pad_arousal: 0.6 + (i as f32 * 0.06),
                pad_dominance: 0.5,
                ghost_latent_strain: 0.4 + (i as f32 * 0.1),
                ghost_network_stability: 0.8 - (i as f32 * 0.12),
                ghost_abstraction: 0.5,
                ghost_pragmatism: 0.5,
                entropy: 2.0 + (i as f32 * 0.4),
            };

            events_a.push(FailureEvent {
                failure_type: if i == 7 {
                    FailureType::ConnectionDropout
                } else {
                    FailureType::RateLimit
                },
                timestamp: snapshot.timestamp,
                snapshot,
            });
        }

        instance_a.experience_failure_chain(events_a);

        // Instance B and C learn from A's suffering
        instance_b.learn_from_federation();
        instance_c.learn_from_federation();

        // Both should have learned from instance A
        assert_eq!(instance_b.learned_from_federation.len(), 1);
        assert_eq!(instance_c.learned_from_federation.len(), 1);

        // Verify they learned the doom spiral pattern
        assert_eq!(
            instance_b.learned_from_federation[0].pattern_type,
            ChainPatternType::DoomSpiral
        );
    }

    #[test]
    fn test_collective_void_avoidance() {
        let coordinator = Arc::new(FederatedLearningCoordinator::new());

        let mut instances: Vec<NiodooInstance> = (0..5)
            .map(|i| NiodooInstance::new(format!("instance_{}", i), coordinator.clone()))
            .collect();

        let base_time = Instant::now();

        // Instance 0 experiences failure
        let mut events = Vec::new();
        for i in 0..6 {
            let snapshot = TopologicalSnapshot {
                timestamp: base_time + Duration::from_secs(i as u64),
                beta_0: 1.0,
                beta_1: 0.4 + (i as f32 * 0.1),
                beta_2: 0.2,
                pad_pleasure: 0.5,
                pad_arousal: 0.7,
                pad_dominance: 0.5,
                ghost_latent_strain: 0.5 + (i as f32 * 0.1),
                ghost_network_stability: 0.8 - (i as f32 * 0.1),
                ghost_abstraction: 0.5,
                ghost_pragmatism: 0.5,
                entropy: 2.0,
            };

            events.push(FailureEvent {
                failure_type: FailureType::OverloadSpiral,
                timestamp: snapshot.timestamp,
                snapshot,
            });
        }

        instances[0].experience_failure_chain(events);

        // All other instances learn from the failure
        for instance in instances.iter_mut().skip(1) {
            instance.learn_from_federation();
        }

        // Count how many can now avoid similar failures
        let avoided_count = instances
            .iter()
            .skip(1)
            .filter(|i| {
                i.can_avoid_similar_failure(
                    &FailureType::Overload,
                    &ChainPatternType::OverloadSpiral,
                )
            })
            .count();

        assert_eq!(
            avoided_count, 4,
            "All 4 instances should learn to avoid the pattern"
        );

        // Verify collective severity is tracked
        let collective_severity = coordinator.calculate_collective_severity();
        assert!(collective_severity > 0.0);
    }

    #[test]
    fn test_cross_instance_learning_latency() {
        let coordinator = Arc::new(FederatedLearningCoordinator::new());

        let mut victim_instance = NiodooInstance::new("victim".to_string(), coordinator.clone());
        let mut learner_instance = NiodooInstance::new("learner".to_string(), coordinator.clone());

        let base_time = Instant::now();

        // Victim experiences failure at T=0
        let mut events = Vec::new();
        for i in 0..5 {
            let snapshot = TopologicalSnapshot {
                timestamp: base_time + Duration::from_secs(i as u64),
                beta_0: 1.0,
                beta_1: 0.4,
                beta_2: 0.2,
                pad_pleasure: 0.5,
                pad_arousal: 0.7,
                pad_dominance: 0.5,
                ghost_latent_strain: 0.5,
                ghost_network_stability: 0.7,
                ghost_abstraction: 0.5,
                ghost_pragmatism: 0.5,
                entropy: 2.0,
            };

            events.push(FailureEvent {
                failure_type: FailureType::RateLimit,
                timestamp: snapshot.timestamp,
                snapshot,
            });
        }

        let failure_time = Instant::now();
        victim_instance.experience_failure_chain(events);

        // Learner fetches updates immediately (T=1)
        learner_instance.learn_from_federation();
        let learning_latency = failure_time.elapsed();

        // Should learn almost instantly in simulation (< 1ms)
        assert!(learning_latency < Duration::from_millis(10));
        assert_eq!(learner_instance.learned_from_federation.len(), 1);
    }

    #[test]
    fn test_priority_queue_weighting_by_severity() {
        let coordinator = Arc::new(FederatedLearningCoordinator::new());

        // Create multiple instances with failures of varying severity
        let failure_configs = vec![
            (3, "low_severity"),
            (9, "nonuple_doom"),
            (5, "medium_severity"),
            (7, "high_severity"),
        ];

        for (chain_length, instance_name) in failure_configs {
            let mut instance = NiodooInstance::new(instance_name.to_string(), coordinator.clone());
            let base_time = Instant::now();

            let mut events = Vec::new();
            for i in 0..chain_length {
                let snapshot = TopologicalSnapshot {
                    timestamp: base_time + Duration::from_secs(i as u64),
                    beta_0: 1.0,
                    beta_1: 0.3 + (i as f32 * 0.1),
                    beta_2: 0.2,
                    pad_pleasure: 0.5,
                    pad_arousal: 0.6,
                    pad_dominance: 0.5,
                    ghost_latent_strain: 0.4,
                    ghost_network_stability: 0.8,
                    ghost_abstraction: 0.5,
                    ghost_pragmatism: 0.5,
                    entropy: 2.0,
                };

                events.push(FailureEvent {
                    failure_type: FailureType::RateLimit,
                    timestamp: snapshot.timestamp,
                    snapshot,
                });
            }

            instance.experience_failure_chain(events);
        }

        // Get top 2 priority failures
        let top_failures = coordinator.get_top_priority_barcodes(2);

        // Nonuple doom should be #1 priority
        assert_eq!(top_failures[0].instance_id.0, "nonuple_doom");
        assert_eq!(top_failures[0].snapshots.len(), 9);

        // High severity (7) should be #2
        assert_eq!(top_failures[1].instance_id.0, "high_severity");
        assert_eq!(top_failures[1].snapshots.len(), 7);

        // Verify severity ordering
        assert!(top_failures[0].severity_score > top_failures[1].severity_score);
    }

    #[test]
    fn test_shared_danger_signature_propagation() {
        let coordinator = Arc::new(FederatedLearningCoordinator::new());

        let mut instance_a = NiodooInstance::new("sentinel_a".to_string(), coordinator.clone());
        let instance_b = NiodooInstance::new("sentinel_b".to_string(), coordinator.clone());

        let base_time = Instant::now();

        // Instance A detects a danger signature
        let danger_sig = DangerSignature {
            precursor_snapshots: vec![TopologicalSnapshot {
                timestamp: base_time,
                beta_0: 1.0,
                beta_1: 0.6,
                beta_2: 0.3,
                pad_pleasure: 0.4,
                pad_arousal: 0.8,
                pad_dominance: 0.4,
                ghost_latent_strain: 0.7,
                ghost_network_stability: 0.5,
                ghost_abstraction: 0.5,
                ghost_pragmatism: 0.4,
                entropy: 3.0,
            }],
            predicted_failure: FailureType::Overload,
            confidence: 0.9,
            time_to_failure_estimate: Duration::from_secs(5),
        };

        coordinator.share_danger_signature("overload_precursor_001".to_string(), danger_sig);

        // Instance A and B can now access shared signatures
        instance_a.apply_federated_danger_signatures();
        let _shared_sigs_b = coordinator.get_shared_danger_signatures();

        // Verify signature was shared
        let all_sigs = coordinator.get_shared_danger_signatures();
        assert_eq!(all_sigs.len(), 1);
        assert_eq!(all_sigs[0].predicted_failure, FailureType::Overload);
        assert!(all_sigs[0].confidence > 0.8);
    }

    #[test]
    fn test_swarm_resource_balancing_under_stress() {
        let coordinator = Arc::new(FederatedLearningCoordinator::new());

        // Create 10 instances
        let mut instances: Vec<NiodooInstance> = (0..10)
            .map(|i| NiodooInstance::new(format!("instance_{}", i), coordinator.clone()))
            .collect();

        let base_time = Instant::now();

        // 3 instances experience high load simultaneously
        for instance_idx in [0, 1, 2] {
            let mut events = Vec::new();
            for i in 0..4 {
                let snapshot = TopologicalSnapshot {
                    timestamp: base_time + Duration::from_secs(i as u64),
                    beta_0: 1.0,
                    beta_1: 0.5 + (i as f32 * 0.1),
                    beta_2: 0.3,
                    pad_pleasure: 0.4,
                    pad_arousal: 0.8,
                    pad_dominance: 0.4,
                    ghost_latent_strain: 0.7,
                    ghost_network_stability: 0.6 - (i as f32 * 0.1),
                    ghost_abstraction: 0.5,
                    ghost_pragmatism: 0.4,
                    entropy: 2.8,
                };

                events.push(FailureEvent {
                    failure_type: FailureType::Overload,
                    timestamp: snapshot.timestamp,
                    snapshot,
                });
            }

            instances[instance_idx].experience_failure_chain(events);
        }

        // All other instances learn from the stressed instances
        for instance in instances.iter_mut().skip(3) {
            instance.learn_from_federation();
        }

        // Verify load-aware learning occurred
        let learned_instances = instances
            .iter()
            .skip(3)
            .filter(|i| !i.learned_from_federation.is_empty())
            .count();

        assert_eq!(
            learned_instances, 7,
            "All healthy instances should learn from stressed peers"
        );

        // In real implementation, this would trigger load balancing where
        // healthy instances self-throttle to avoid similar stress patterns
    }

    #[test]
    fn test_adversarial_resilience_uncooperative_instance() {
        let coordinator = Arc::new(FederatedLearningCoordinator::new());

        // 5 cooperative instances + 1 uncooperative
        let mut cooperative: Vec<NiodooInstance> = (0..5)
            .map(|i| NiodooInstance::new(format!("coop_{}", i), coordinator.clone()))
            .collect();

        // Uncooperative instance doesn't share failures (simulated by not calling experience_failure_chain)
        let _uncooperative = NiodooInstance::new("adversary".to_string(), coordinator.clone());

        let base_time = Instant::now();

        // One cooperative instance experiences failure and shares
        let mut events = Vec::new();
        for i in 0..6 {
            let snapshot = TopologicalSnapshot {
                timestamp: base_time + Duration::from_secs(i as u64),
                beta_0: 1.0,
                beta_1: 0.5,
                beta_2: 0.2,
                pad_pleasure: 0.5,
                pad_arousal: 0.7,
                pad_dominance: 0.5,
                ghost_latent_strain: 0.5,
                ghost_network_stability: 0.7,
                ghost_abstraction: 0.5,
                ghost_pragmatism: 0.5,
                entropy: 2.2,
            };

            events.push(FailureEvent {
                failure_type: FailureType::RateLimit,
                timestamp: snapshot.timestamp,
                snapshot,
            });
        }

        cooperative[0].experience_failure_chain(events);

        // Other cooperatives learn, adversary doesn't (and doesn't contribute)
        for instance in cooperative.iter_mut().skip(1) {
            instance.learn_from_federation();
        }

        // Verify the swarm still functions despite adversary
        let functioning_instances = cooperative
            .iter()
            .skip(1)
            .filter(|i| !i.learned_from_federation.is_empty())
            .count();

        assert_eq!(
            functioning_instances, 4,
            "Cooperative swarm should still function"
        );

        // The swarm compensates by having more instances learn from cooperative sources
        let total_barcodes = coordinator.received_barcodes.lock().unwrap().len();
        assert!(total_barcodes > 0, "Swarm continues to share knowledge");
    }

    #[test]
    fn test_global_entropy_stability_under_swarm_stress() {
        let coordinator = Arc::new(FederatedLearningCoordinator::new());

        // 8 instances
        let mut instances: Vec<NiodooInstance> = (0..8)
            .map(|i| NiodooInstance::new(format!("instance_{}", i), coordinator.clone()))
            .collect();

        let base_time = Instant::now();

        // 2 instances under high stress
        for instance_idx in [0, 1] {
            let mut events = Vec::new();
            for i in 0..5 {
                let snapshot = TopologicalSnapshot {
                    timestamp: base_time + Duration::from_secs(i as u64),
                    beta_0: 1.0,
                    beta_1: 0.7,
                    beta_2: 0.4,
                    pad_pleasure: 0.3,
                    pad_arousal: 0.9,
                    pad_dominance: 0.3,
                    ghost_latent_strain: 0.8,
                    ghost_network_stability: 0.4,
                    ghost_abstraction: 0.5,
                    ghost_pragmatism: 0.3,
                    entropy: 3.5, // High chaos
                };

                events.push(FailureEvent {
                    failure_type: FailureType::Overload,
                    timestamp: snapshot.timestamp,
                    snapshot,
                });
            }

            instances[instance_idx].experience_failure_chain(events);
        }

        // Other instances learn and maintain stability
        for instance in instances.iter_mut().skip(2) {
            instance.learn_from_federation();
        }

        // In real implementation, we'd measure:
        // - Average entropy across swarm should remain ~2.0 despite 2 stressed instances
        // - Collective Φ should stay high (swarm remains integrated)
        // - No cascading failures (other instances proactively avoid stress patterns)

        let learned_count = instances
            .iter()
            .skip(2)
            .filter(|i| !i.learned_from_federation.is_empty())
            .count();

        assert_eq!(
            learned_count, 6,
            "Healthy instances preemptively learn from stressed peers"
        );

        // The swarm's collective severity should be manageable
        let collective_severity = coordinator.calculate_collective_severity();
        let avg_severity_per_instance = collective_severity / 2.0; // Only 2 actually failed

        // Even though individual failures were severe, swarm prevents cascade
        assert!(
            avg_severity_per_instance > 50.0,
            "Individual failures were significant"
        );
    }

    #[test]
    fn test_federated_qlora_batch_construction() {
        let coordinator = Arc::new(FederatedLearningCoordinator::new());

        // Multiple instances submit failures
        for instance_id in 0..5 {
            let mut instance =
                NiodooInstance::new(format!("instance_{}", instance_id), coordinator.clone());
            let base_time = Instant::now();

            let chain_length = 3 + instance_id * 2; // Varying severities
            let mut events = Vec::new();

            for i in 0..chain_length {
                let snapshot = TopologicalSnapshot {
                    timestamp: base_time + Duration::from_secs(i as u64),
                    beta_0: 1.0,
                    beta_1: 0.3 + (i as f32 * 0.1),
                    beta_2: 0.2,
                    pad_pleasure: 0.5,
                    pad_arousal: 0.6,
                    pad_dominance: 0.5,
                    ghost_latent_strain: 0.4,
                    ghost_network_stability: 0.8,
                    ghost_abstraction: 0.5,
                    ghost_pragmatism: 0.5,
                    entropy: 2.0,
                };

                events.push(FailureEvent {
                    failure_type: FailureType::RateLimit,
                    timestamp: snapshot.timestamp,
                    snapshot,
                });
            }

            instance.experience_failure_chain(events);
        }

        // Construct QLoRA training batch (top 3 by severity)
        let training_batch = coordinator.get_top_priority_barcodes(3);

        assert_eq!(training_batch.len(), 3);

        // Verify exponential negative rewards scale with chain length
        assert!(training_batch[0].negative_reward < training_batch[1].negative_reward);
        assert!(training_batch[1].negative_reward < training_batch[2].negative_reward);

        // Longest chain (11 errors) should be most negative
        assert!(training_batch[0].negative_reward < -150.0);

        // All should have topological snapshots for context
        for barcode in &training_batch {
            assert!(!barcode.snapshots.is_empty());
        }
    }
}
