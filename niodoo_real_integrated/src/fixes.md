# NIODOO INTEGRATION FIXES - IMPLEMENTATION GUIDE

## 1. Fix Entropy Stuck Issue

### File: `pipeline.rs`
```rust
// Line 274 - Create fresh torus for each request
let mut local_torus = TorusPadMapper::from_entropy();
let pad_state = local_torus.project(&embedding)?;

// Alternative: Add cache busting
let cache_key = format!("{}:{}:{}", 
    cache_key(prompt), 
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_millis(),
    rand::random::<u32>()
);
```

## 2. Connect TCS Predictor

### File: `learning.rs`
```rust
// Remove underscore from line 108
predictor: TcsPredictor,

// In update() method, add after line 261:
// Use predictor for reward prediction
let predicted_reward = self.predictor.predict_reward_delta(topology);
let (tcs_action_param, tcs_action_delta) = self.predictor.predict_action(topology, &self.config.lock().unwrap());

// Combine with DQN action selection
if predicted_reward < -0.5 && rand::random::<f64>() < 0.3 {
    // Override DQN with TCS predictor suggestion
    action = DqnAction {
        param: tcs_action_param,
        delta: tcs_action_delta,
    };
}

// Update predictor with experience (after line 263)
self.predictor.update(topology, reward, generation.rouge_score);
```

## 3. Implement TCS LoRA

### File: `tcs_lora.rs`
```rust
use tch::{nn, Device, Tensor, Kind};

pub struct TcsLoRaPredictor {
    vs: nn::VarStore,
    lora_a: nn::Linear,
    lora_b: nn::Linear,
    rank: i64,
}

impl TcsLoRaPredictor {
    pub fn new(rank: i64, input_dim: i64, output_dim: i64) -> Self {
        let vs = nn::VarStore::new(Device::Cpu);
        let root = vs.root();
        
        let lora_a = nn::linear(&root / "lora_a", input_dim, rank, Default::default());
        let lora_b = nn::linear(&root / "lora_b", rank, output_dim, Default::default());
        
        Self {
            vs,
            lora_a,
            lora_b,
            rank,
        }
    }
    
    pub fn forward(&self, input: &Tensor) -> Tensor {
        let h = self.lora_a.forward(input);
        self.lora_b.forward(&h)
    }
    
    pub fn train_on_tcs(&mut self, features: Vec<Vec<f64>>, labels: Vec<f64>) {
        let device = self.vs.device();
        let mut opt = nn::Adam::default().build(&self.vs, 1e-3).unwrap();
        
        for epoch in 0..10 {
            let x = Tensor::of_slice2(&features).to(device);
            let y = Tensor::of_slice(&labels).to(device);
            
            let pred = self.forward(&x);
            let loss = pred.mse_loss(&y, tch::Reduction::Mean);
            
            opt.backward_step(&loss);
            
            if epoch % 5 == 0 {
                println!("TCS LoRA epoch {}: loss = {}", epoch, loss.double_value(&[]));
            }
        }
    }
}
```

## 4. Fix Curator Integration

### File: `pipeline.rs` 
```rust
async fn integrate_curator(
    &self,
    input: &str,
    output: &str,
    pad_state: &PadGhostState,
    compass: &CompassOutcome,
    context: &str,
) -> Result<CuratedExperience> {
    let experience = Experience::from_pipeline(
        input.to_string(),
        output.to_string(),
        Vec::new(), // embedding will be set by curator
        pad_state,
        compass,
        context.lines().map(|s| s.to_string()).collect(),
    );
    
    // Actually use the curator if available
    if let Some(ref curator) = self.curator {
        match curator.curate_response(experience.clone()).await {
            Ok(curated) => {
                return Ok(CuratedExperience {
                    refined_response: curated.refined_output.unwrap_or(output.to_string()),
                    quality_score: curated.quality_score,
                    processing_time_ms: curated.processing_time_ms.unwrap_or(0.0),
                    metadata: HashMap::from([
                        ("threats_detected".to_string(), curated.threats_detected.to_string()),
                        ("entropy_delta".to_string(), pad_state.entropy.to_string()),
                    ]),
                });
            }
            Err(e) => {
                warn!("Curator failed: {}, using fallback", e);
            }
        }
    }
    
    // Fallback implementation
    Ok(CuratedExperience {
        refined_response: output.to_string(),
        quality_score: 0.5,
        processing_time_ms: 0.0,
        metadata: HashMap::new(),
    })
}
```

## 5. Implement History Distance

### File: `learning.rs`
```rust
// Add method to compute Wasserstein distance
async fn compute_history_distance(&self, current_state: &PadGhostState) -> Result<f64> {
    // Query similar historical states from ERAG
    let similar = self.erag.query_similar_states(current_state, 10).await?;
    
    if similar.is_empty() {
        return Ok(1.0); // Max distance if no history
    }
    
    // Compute Wasserstein distance between current and historical distributions
    let current_dist = &current_state.pad;
    let mut total_distance = 0.0;
    
    for hist_state in similar {
        let hist_dist = &hist_state.pad_values;
        let mut dist = 0.0;
        
        for i in 0..7 {
            dist += (current_dist[i] - hist_dist[i]).powi(2);
        }
        total_distance += dist.sqrt();
    }
    
    Ok(total_distance / similar.len() as f64)
}

// In update() method, replace line 260:
let history_dist = self.compute_history_distance(pad_state).await.unwrap_or(0.0);
```

## 6. Connect Evolution to Topology

### File: `learning.rs`
```rust
// Store latest topology
topology_history: VecDeque<TopologicalSignature>,

// In update() add:
self.topology_history.push_back(topology.clone());
if self.topology_history.len() > 50 {
    self.topology_history.pop_front();
}

// Modify evolution_step:
async fn evolution_step(&mut self) -> Result<()> {
    // Get recent topologies
    let recent_topologies: Vec<_> = self.topology_history.iter().cloned().collect();
    
    // Use topological features to guide evolution
    let avg_knot = recent_topologies.iter()
        .map(|t| t.knot_complexity)
        .sum::<f64>() / recent_topologies.len() as f64;
    
    let avg_pe = recent_topologies.iter()
        .map(|t| t.persistence_entropy)
        .sum::<f64>() / recent_topologies.len() as f64;
    
    // Adjust evolution parameters based on topology
    let mutation_rate = if avg_knot > 0.5 {
        self.evolution.mutation_std * 1.5 // More exploration
    } else {
        self.evolution.mutation_std * 0.8 // More exploitation
    };
    
    // Pass topology-guided params to evolution
    self.evolution.evolve_with_topology(
        &current,
        &recent,
        avg_knot,
        avg_pe,
        mutation_rate,
    ).await?;
    
    Ok(())
}
```

## 7. Wire Up Complete Pipeline

### File: `pipeline.rs`
```rust
pub async fn process_prompt(&mut self, prompt: &str) -> Result<PipelineCycle> {
    // ... existing code ...
    
    // CRITICAL: Break cache for stuck entropy
    let unique_key = format!("{}:{}", prompt, uuid::Uuid::new_v4());
    
    // Use fresh torus mapper
    let mut local_torus = TorusPadMapper::from_entropy();
    let pad_state = local_torus.project(&embedding)?;
    
    // ... existing code ...
    
    // After generation, before learning:
    // Store TCS prediction for learning
    let tcs_prediction = self.tcs_predictor.predict_reward_delta(&topology);
    
    // Pass to learning with prediction
    let learning = self.learning.update_with_prediction(
        &pad_state,
        &compass,
        &collapse,
        &final_generation,
        &topology,
        tcs_prediction,
    ).await?;
    
    // Update TCS predictor with actual outcome
    let actual_reward = learning.compute_reward(
        learning.entropy_delta,
        final_generation.rouge_score
    );
    self.tcs_predictor.update(&topology, actual_reward, final_generation.rouge_score);
    
    // Trigger LoRA if topology indicates high complexity
    if topology.knot_complexity > 0.6 && self.lora_trainer.should_train() {
        self.lora_trainer.train_on_recent_failures(&self.erag).await?;
    }
    
    // ... rest of pipeline ...
}
```

## 8. Add Missing Metrics Collection

### File: `metrics.rs`
```rust
// Add TCS metrics
pub fn record_topology_metrics(
    knot: f64,
    betti: &[usize],
    persistence_entropy: f64,
    spectral_gap: f64,
) {
    KNOT_COMPLEXITY.set(knot);
    PERSISTENCE_ENTROPY.set(persistence_entropy);
    SPECTRAL_GAP.set(spectral_gap);
    
    for (i, &b) in betti.iter().enumerate() {
        BETTI_NUMBERS.with_label_values(&[&i.to_string()]).set(b as f64);
    }
}

// Add predictor accuracy metrics
pub fn record_prediction_accuracy(predicted: f64, actual: f64) {
    let error = (predicted - actual).abs();
    PREDICTION_ERROR.observe(error);
    
    if error < 0.1 {
        PREDICTION_ACCURACY.inc();
    }
}
```

## CRITICAL TESTING SEQUENCE

1. **Test entropy variation**:
```bash
# Should see different entropy values each run
for i in {1..5}; do
    ./target/release/niodoo_real_integrated -p "test $i" | grep entropy
done
```

2. **Test TCS predictor integration**:
```bash
RUST_LOG=debug ./target/release/niodoo_real_integrated -p "complex topology test" 2>&1 | grep -E "(knot|predict|TCS)"
```

3. **Test curator flow**:
```bash
RUST_LOG=debug ./target/release/niodoo_real_integrated -p "curator test" 2>&1 | grep -i curator
```

4. **Test learning loop**:
```bash
RUST_LOG=info ./target/release/niodoo_real_integrated -p "learning test" --iterations 10 2>&1 | grep -E "(reward|Q-table|evolution)"
```

5. **Full integration test**:
```bash
./run_with_metrics.sh -n 10 -p "full integration test" | tee integration_test.log
```

## Expected Outcomes After Fixes:
- ✅ Entropy values change between runs
- ✅ TCS predictor influences action selection
- ✅ Curator refines outputs before storage
- ✅ Evolution uses topological guidance
- ✅ LoRA trains on high-complexity failures
- ✅ History distance affects reward calculation
- ✅ Full end-to-end learning pipeline functions