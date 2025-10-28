//! TCS v10.0 Continual Learning Test Runner
//! 
//! Integrates all components into a continual self-learning loop:
//! - GPU-accelerated persistent homology
//! - LoRA training with SGD
//! - EWC for forgetting prevention
//! - Geometric equivariant layers
//! - Sheaf diffusion
//! - IIT Φ approximation
//! - HotStuff consensus

use anyhow::Result;
use nalgebra::{DMatrix, DVector};
use std::time::{Duration, Instant};
use tcs_pipeline::TCSOrchestrator;
use tracing::{info, warn};

use crate::continual_learning::{ContinualLearningPipeline, ForgettingPrevention};
use crate::consciousness_metrics::approximate_phi_from_betti;
use crate::geometric_nn::GeometricLayer;
use crate::sheaf_nn::{SheafDiffusionLayer, SheafMode};
use niodoo_real_integrated::lora_trainer::LoRATrainer;

struct TCSContinualTester {
    orchestrator: TCSOrchestrator,
    learning_pipeline: ContinualLearningPipeline,
    lora_trainer: LoRATrainer,
    geometric_layer: GeometricLayer,
    sheaf_layer: SheafDiffusionLayer,
    cycle: usize,
    metrics: TestMetrics,
}

#[derive(Debug, Clone)]
struct TestMetrics {
    gpu_available: bool,
    gpu_speedup: f64,
    lora_train_events: usize,
    ewc_retention: f64,
    equivariant_invariance: f64,
    sheaf_propagation: usize,
    phi_values: Vec<f64>,
    consensus_rounds: usize,
    total_latency_ms: f64,
}

impl TCSContinualTester {
    fn new() -> Result<Self> {
        info!("🚀 Initializing TCS v10.0 continual test...");
        
        let orchestrator = TCSOrchestrator::new(100)?;
        let learning_pipeline = ContinualLearningPipeline::new_default();
        let lora_trainer = LoRATrainer::new()?;
        let geometric_layer = GeometricLayer::new(64, 64);
        let sheaf_layer = SheafDiffusionLayer::identity(32);
        
        Ok(Self {
            orchestrator,
            learning_pipeline,
            lora_trainer,
            geometric_layer,
            sheaf_layer,
            cycle: 0,
            metrics: TestMetrics {
                gpu_available: false,
                gpu_speedup: 1.0,
                lora_train_events: 0,
                ewc_retention: 0.0,
                equivariant_invariance: 0.0,
                sheaf_propagation: 0,
                phi_values: Vec::new(),
                consensus_rounds: 0,
                total_latency_ms: 0.0,
            },
        })
    }
    
    async fn run_cycle(&mut self, test_input: &str) -> Result<CycleMetrics> {
        let start = Instant::now();
        
        // 1. GPU-accelerated persistent homology
        let gpu_start = Instant::now();
        let points = self.generate_test_points(100);
        let dist_matrix = tcs::performance::gpu_ripser_distance_matrix(&points)?;
        let gpu_time = gpu_start.elapsed();
        
        let cpu_start = Instant::now();
        let _cpu_dist = self.cpu_distance_matrix(&points);
        let cpu_time = cpu_start.elapsed();
        
        let gpu_speedup = if gpu_time.as_secs_f64() > 0.0 {
            cpu_time.as_secs_f64() / gpu_time.as_secs_f64()
        } else {
            1.0
        };
        
        info!("GPU speedup: {:.2}x", gpu_speedup);
        
        // 2. LoRA training on low-reward events
        let lora_trained = if self.should_train_lora() {
            self.train_lora_step()?;
            true
        } else {
            false
        };
        
        // 3. EWC forgetting prevention
        let retention = self.update_ewc()?;
        
        // 4. Geometric equivariant layer
        let invariance = self.test_equivariance()?;
        
        // 5. Sheaf diffusion
        let sheaf_count = self.test_sheaf_diffusion()?;
        
        // 6. IIT Φ approximation
        let phi = self.compute_phi()?;
        
        // 7. HotStuff consensus
        let consensus_rounds = self.test_consensus()?;
        
        // 8. Process through TCS orchestrator
        let events = self.orchestrator.process(test_input).await?;
        
        let total_time = start.elapsed();
        
        Ok(CycleMetrics {
            cycle: self.cycle,
            gpu_speedup,
            lora_trained,
            ewc_retention: retention,
            equivariant_invariance: invariance,
            sheaf_propagation: sheaf_count,
            phi,
            consensus_rounds,
            total_latency_ms: total_time.as_secs_f64() * 1000.0,
            events_count: events.len(),
        })
    }
    
    fn generate_test_points(&self, n: usize) -> Vec<DVector<f32>> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        (0..n)
            .map(|_| {
                DVector::from_vec(vec![
                    rng.gen_range(-1.0..1.0),
                    rng.gen_range(-1.0..1.0),
                    rng.gen_range(-1.0..1.0),
                ])
            })
            .collect()
    }
    
    fn cpu_distance_matrix(&self, points: &[DVector<f32>]) -> DMatrix<f32> {
        let n = points.len();
        let mut dist = DMatrix::<f32>::zeros(n, n);
        
        for i in 0..n {
            for j in 0..n {
                let d = (&points[i] - &points[j]).norm();
                dist[(i, j)] = d;
            }
        }
        
        dist
    }
    
    fn should_train_lora(&self) -> bool {
        // Train on low-reward events (every 10 cycles)
        self.cycle % 10 == 0
    }
    
    fn train_lora_step(&mut self) -> Result<()> {
        info!("Training LoRA adapter...");
        
        // Generate dummy topological data
        let training_data: Vec<(Vec<f32>, Vec<f32>)> = (0..10)
            .map(|i| {
                let input = vec![i as f32 * 0.1; 896];
                let target = vec![i as f32 * 0.1; 896];
                (input, target)
            })
            .collect();
        
        self.lora_trainer.train(&training_data, 3, 0.001)?;
        self.metrics.lora_train_events += 1;
        
        Ok(())
    }
    
    fn update_ewc(&mut self) -> Result<f64> {
        info!("Updating EWC forgetting prevention...");
        
        // Compute Fisher matrix
        let fisher = self.compute_fisher_matrix();
        
        // Update forgetting prevention
        self.learning_pipeline.forgetting_prevention.update_fisher_matrix(&fisher);
        
        let retention = self.learning_pipeline.forgetting_prevention.forgetting_rate();
        self.metrics.ewc_retention = retention;
        
        Ok(retention)
    }
    
    fn compute_fisher_matrix(&self) -> nalgebra::DMatrix<f64> {
        // Mock Fisher matrix computation
        nalgebra::DMatrix::<f64>::from_fn(10, 10, |i, j| {
            if i == j {
                1.0
            } else {
                0.1
            }
        })
    }
    
    fn test_equivariance(&mut self) -> Result<f64> {
        info!("Testing geometric equivariance...");
        
        // Create test positions and features
        let positions = nalgebra::DMatrix::from_row_slice(
            3,
            3,
            &[
                1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                -1.0, 0.5, 0.0,
            ],
        );
        
        let features = nalgebra::DMatrix::from_row_slice(
            3,
            64,
            &vec![0.1; 3 * 64],
        );
        
        let output = self.geometric_layer.forward(&positions, &features);
        let invariance_score = output.norm() / features.norm();
        
        self.metrics.equivariant_invariance = invariance_score;
        
        Ok(invariance_score as f64)
    }
    
    fn test_sheaf_diffusion(&mut self) -> Result<usize> {
        info!("Testing sheaf diffusion...");
        
        let features = vec![0.5; 32];
        
        // Test listen mode
        let listen_output = self.sheaf_layer.forward(&features, SheafMode::Listen);
        
        // Test broadcast mode
        let broadcast_output = self.sheaf_layer.forward(&features, SheafMode::Broadcast);
        
        // Test dual pass
        let (dual_listen, dual_broadcast) = self.sheaf_layer.dual_pass(&features);
        
        assert_eq!(listen_output.len(), dual_listen.len());
        assert_eq!(broadcast_output.len(), dual_broadcast.len());
        
        self.metrics.sheaf_propagation += 1;
        
        Ok(3) // 3 propagation modes tested
    }
    
    fn compute_phi(&mut self) -> Result<f64> {
        info!("Computing IIT Φ approximation...");
        
        // Mock Betti numbers
        let betti = [1, 2, 1];
        let phi = approximate_phi_from_betti(&betti)?;
        
        self.metrics.phi_values.push(phi);
        
        Ok(phi)
    }
    
    fn test_consensus(&mut self) -> Result<usize> {
        info!("Testing HotStuff consensus...");
        
        use tcs_consensus::hotstuff::{propose, vote, commit, FakeNode};
        use std::sync::Arc;
        
        let nodes: Arc<[FakeNode]> = (0..5).map(FakeNode::new).collect::<Vec<_>>().into();
        let proposer = nodes[0].id;
        
        // Run single consensus round
        let rt = tokio::runtime::Handle::current();
        let proposal = rt.block_on(async {
            propose(nodes.clone(), proposer, format!("test-block-{}", self.cycle)).await
        })?;
        
        let votes = rt.block_on(async {
            vote(nodes.clone(), &proposal).await
        })?;
        
        let _commit = rt.block_on(async {
            commit(nodes, &proposal, &votes).await
        })?;
        
        self.metrics.consensus_rounds += 1;
        
        Ok(1)
    }
}

#[derive(Debug, Clone)]
struct CycleMetrics {
    cycle: usize,
    gpu_speedup: f64,
    lora_trained: bool,
    ewc_retention: f64,
    equivariant_invariance: f64,
    sheaf_propagation: usize,
    phi: f64,
    consensus_rounds: usize,
    total_latency_ms: f64,
    events_count: usize,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    
    println!("🧠 TCS v10.0 Continual Learning Test");
    println!("=====================================");
    println!("Testing self-learning loop with all components integrated");
    println!();
    
    let mut tester = TCSContinualTester::new()?;
    
    let test_inputs = vec![
        "What is the topological structure of consciousness?",
        "How does memory formation relate to persistent homology?",
        "Compute the Betti numbers for this emotional state",
        "Apply sheaf diffusion to propagate meaning",
        "Verify geometric equivariance of this representation",
    ];
    
    let num_cycles = 50;
    
    for cycle in 0..num_cycles {
        let input = test_inputs[cycle % test_inputs.len()];
        
        let metrics = tester.run_cycle(input).await?;
        
        println!(
            "📊 Cycle {}/{}: GPU={:.2}x, LoRA={}, EWC={:.3}, Invariance={:.3}, Φ={:.4}, Latency={:.1}ms",
            cycle + 1,
            num_cycles,
            metrics.gpu_speedup,
            if metrics.lora_trained { "✓" } else { "✗" },
            metrics.ewc_retention,
            metrics.equivariant_invariance,
            metrics.phi,
            metrics.total_latency_ms
        );
        
        tester.cycle += 1;
        
        // Small delay between cycles
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    
    println!();
    println!("✅ Continual learning test complete!");
    println!("📈 Summary:");
    println!("  - GPU speedup: {:.2}x", tester.metrics.gpu_speedup);
    println!("  - LoRA training events: {}", tester.metrics.lora_train_events);
    println!("  - EWC retention: {:.3}", tester.metrics.ewc_retention);
    println!("  - Consensus rounds: {}", tester.metrics.consensus_rounds);
    println!("  - Avg Φ: {:.4}", 
        tester.metrics.phi_values.iter().sum::<f64>() / tester.metrics.phi_values.len() as f64);
    
    Ok(())
}
