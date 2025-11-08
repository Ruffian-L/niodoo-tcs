# 🔌 Rust API Reference

**Created by Jason Van Pham | Niodoo Framework | 2025**

## Core Consciousness Engine API

### PersonalNiodooConsciousness

The main consciousness engine that orchestrates all processing activities.

#### Constructor Methods

```rust
impl PersonalNiodooConsciousness {
    /// Create a new personal consciousness engine
    pub async fn new() -> Result<Self, anyhow::Error>
    
    /// Create a new personal consciousness engine with Phase 6 production deployment configuration
    pub async fn new_with_phase6_config(phase6_config: Phase6Config) -> Result<Self, anyhow::Error>
}
```

**Example Usage:**
```rust
use niodoo_feeling::consciousness_engine::PersonalNiodooConsciousness;
use niodoo_feeling::phase6_integration::Phase6Config;

// Basic initialization
let consciousness_engine = PersonalNiodooConsciousness::new().await?;

// Phase 6 production initialization
let phase6_config = Phase6Config {
    gpu_acceleration: true,
    learning_analytics: true,
    consciousness_logging: true,
    // ... other config options
};
let consciousness_engine = PersonalNiodooConsciousness::new_with_phase6_config(phase6_config).await?;
```

#### Core Processing Methods

```rust
impl PersonalNiodooConsciousness {
    /// Process a consciousness event through the entire pipeline
    pub async fn process_consciousness_event(&self, input: String) -> Result<ConsciousnessResponse, anyhow::Error>
    
    /// Get the current consciousness state
    pub async fn get_consciousness_state(&self) -> ConsciousnessState
    
    /// Update emotional context
    pub async fn update_emotional_context(&self, emotion: EmotionType, intensity: f32) -> Result<(), anyhow::Error>
    
    /// Consolidate memories across all memory systems
    pub async fn consolidate_memories(&self) -> Result<MemoryStats, anyhow::Error>
}
```

**Example Usage:**
```rust
// Process a consciousness event
let response = consciousness_engine.process_consciousness_event("Hello, world!".to_string()).await?;
println!("Consciousness response: {:?}", response);

// Get current state
let state = consciousness_engine.get_consciousness_state().await;
println!("Current consciousness level: {}", state.consciousness_level);

// Update emotional context
consciousness_engine.update_emotional_context(EmotionType::Joy, 0.8).await?;

// Consolidate memories
let memory_stats = consciousness_engine.consolidate_memories().await?;
println!("Memory consolidation completed: {:?}", memory_stats);
```

#### Phase 6 Integration Methods

```rust
impl PersonalNiodooConsciousness {
    /// Initialize Phase 6 integration system for production deployment
    pub async fn initialize_phase6_integration(&mut self, phase6_config: Phase6Config) -> Result<(), anyhow::Error>
    
    /// Process consciousness evolution through Phase 6 integration system
    pub async fn process_consciousness_evolution_phase6(
        &self,
        consciousness_id: String,
        consciousness_state: candle_core::Tensor,
        emotional_context: candle_core::Tensor,
        memory_gradients: candle_core::Tensor,
    ) -> Result<candle_core::Tensor, anyhow::Error>
    
    /// Get Phase 6 configuration if available
    pub fn get_phase6_config(&self) -> Option<&Phase6Config>
    
    /// Check if GPU acceleration is enabled and operational
    pub fn is_gpu_acceleration_enabled(&self) -> bool
}
```

**Example Usage:**
```rust
// Initialize Phase 6 integration
let phase6_config = Phase6Config {
    gpu_acceleration: true,
    learning_analytics: true,
    consciousness_logging: true,
    // ... other config options
};
consciousness_engine.initialize_phase6_integration(phase6_config).await?;

// Process consciousness evolution
let consciousness_id = "user_123".to_string();
let consciousness_state = candle_core::Tensor::zeros(&[1, 256], candle_core::DType::F32, &candle_core::Device::Cpu)?;
let emotional_context = candle_core::Tensor::zeros(&[1, 64], candle_core::DType::F32, &candle_core::Device::Cpu)?;
let memory_gradients = candle_core::Tensor::zeros(&[1, 512], candle_core::DType::F32, &candle_core::Device::Cpu)?;

let evolved_state = consciousness_engine.process_consciousness_evolution_phase6(
    consciousness_id,
    consciousness_state,
    emotional_context,
    memory_gradients,
).await?;

// Check GPU acceleration status
if consciousness_engine.is_gpu_acceleration_enabled() {
    println!("GPU acceleration is enabled and operational");
}
```

### ConsciousnessState

Maintains the current state of consciousness including emotional context and personality weights.

#### Core Methods

```rust
impl ConsciousnessState {
    /// Create a new consciousness state
    pub fn new() -> Self
    
    /// Update emotional state
    pub fn update_emotion(&mut self, emotion: EmotionType, intensity: f32)
    
    /// Get emotional vector representation
    pub fn get_emotional_vector(&self) -> Vec<f32>
    
    /// Calculate consciousness metrics
    pub fn calculate_consciousness_metrics(&self) -> ConsciousnessMetrics
}
```

**Example Usage:**
```rust
use niodoo_feeling::consciousness::ConsciousnessState;
use niodoo_feeling::consciousness::EmotionType;

// Create new state
let mut state = ConsciousnessState::new();

// Update emotional state
state.update_emotion(EmotionType::Joy, 0.8);
state.update_emotion(EmotionType::Sadness, 0.2);

// Get emotional vector
let emotional_vector = state.get_emotional_vector();
println!("Emotional vector: {:?}", emotional_vector);

// Calculate metrics
let metrics = state.calculate_consciousness_metrics();
println!("Consciousness level: {}", metrics.consciousness_level);
```

### BrainCoordinator

Manages the three-brain system and coordinates parallel processing.

#### Core Methods

```rust
impl BrainCoordinator {
    /// Create a new brain coordinator
    pub fn new(
        motor_brain: MotorBrain,
        lcars_brain: LcarsBrain,
        efficiency_brain: EfficiencyBrain,
        personality_manager: PersonalityManager,
        consciousness_state: Arc<RwLock<ConsciousnessState>>,
    ) -> Self
    
    /// Process input using all brains in parallel
    pub async fn process_brains_parallel(&self, input: &str, timeout_duration: tokio::time::Duration) -> Result<Vec<String>, anyhow::Error>
    
    /// Get the motor brain reference
    pub fn get_motor_brain(&self) -> &MotorBrain
    
    /// Get the LCARS brain reference
    pub fn get_lcars_brain(&self) -> &LcarsBrain
    
    /// Get the efficiency brain reference
    pub fn get_efficiency_brain(&self) -> &EfficiencyBrain
    
    /// Update personality weights based on emotional context
    pub async fn update_personality_weights(&self, emotional_context: &EmotionalContext) -> Result<(), anyhow::Error>
}
```

**Example Usage:**
```rust
use niodoo_feeling::consciousness_engine::brain_coordination::BrainCoordinator;
use niodoo_feeling::brain::{MotorBrain, LcarsBrain, EfficiencyBrain};
use niodoo_feeling::personality::PersonalityManager;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::time::Duration;

// Create brain coordinator
let motor_brain = MotorBrain::new()?;
let lcars_brain = LcarsBrain::new()?;
let efficiency_brain = EfficiencyBrain::new()?;
let personality_manager = PersonalityManager::new();
let consciousness_state = Arc::new(RwLock::new(ConsciousnessState::new()));

let brain_coordinator = BrainCoordinator::new(
    motor_brain,
    lcars_brain,
    efficiency_brain,
    personality_manager,
    consciousness_state,
);

// Process input through all brains
let input = "Analyze this situation and provide recommendations";
let timeout = Duration::from_millis(5000);
let responses = brain_coordinator.process_brains_parallel(input, timeout).await?;

println!("Brain responses:");
for (i, response) in responses.iter().enumerate() {
    println!("Brain {}: {}", i, response);
}

// Access individual brains
let motor_brain = brain_coordinator.get_motor_brain();
let lcars_brain = brain_coordinator.get_lcars_brain();
let efficiency_brain = brain_coordinator.get_efficiency_brain();
```

### MemoryManager

Handles memory storage, retrieval, and consolidation using Gaussian spheres and Möbius topology.

#### Core Methods

```rust
impl MemoryManager {
    /// Create a new memory manager
    pub fn new(
        memory_spheres: Vec<GaussianMemorySphere>,
        mobius_topology: MobiusTopology,
        personal_memory_engine: Arc<PersonalMemoryEngine>,
        consciousness_state: Arc<RwLock<ConsciousnessState>>,
    ) -> Self
    
    /// Store a new memory
    pub async fn store_memory(&self, event: PersonalConsciousnessEvent) -> Result<(), anyhow::Error>
    
    /// Retrieve memories based on query
    pub async fn retrieve_memories(&self, query: MemoryQuery) -> Result<Vec<Memory>, anyhow::Error>
    
    /// Consolidate memories across all systems
    pub async fn consolidate_memories(&self) -> Result<MemoryStats, anyhow::Error>
    
    /// Traverse Möbius path for memory exploration
    pub async fn traverse_mobius_path(&self, emotional_input: f32, reasoning_goal: Option<String>) -> Result<TraversalResult, anyhow::Error>
}
```

**Example Usage:**
```rust
use niodoo_feeling::consciousness_engine::memory_management::MemoryManager;
use niodoo_feeling::consciousness_engine::memory_management::PersonalConsciousnessEvent;
use niodoo_feeling::personal_memory::PersonalMemoryEngine;
use std::sync::Arc;

// Create memory manager
let memory_spheres = Vec::new();
let mobius_topology = MobiusTopology::new();
let personal_memory_engine = Arc::new(PersonalMemoryEngine::new()?);
let consciousness_state = Arc::new(RwLock::new(ConsciousnessState::new()));

let memory_manager = MemoryManager::new(
    memory_spheres,
    mobius_topology,
    personal_memory_engine,
    consciousness_state,
);

// Store a memory
let event = PersonalConsciousnessEvent {
    content: "Learned about Möbius topology today".to_string(),
    emotional_context: EmotionType::Curiosity,
    importance: 0.8,
    timestamp: chrono::Utc::now(),
    // ... other fields
};
memory_manager.store_memory(event).await?;

// Retrieve memories
let query = MemoryQuery {
    keywords: vec!["Möbius".to_string(), "topology".to_string()],
    emotional_context: Some(EmotionType::Curiosity),
    time_range: Some(TimeRange::last_hour()),
    // ... other query parameters
};
let memories = memory_manager.retrieve_memories(query).await?;
println!("Found {} relevant memories", memories.len());

// Traverse Möbius path
let traversal_result = memory_manager.traverse_mobius_path(0.7, Some("explore consciousness".to_string())).await?;
println!("Traversal result: {:?}", traversal_result);
```

### Individual Brain APIs

#### MotorBrain

```rust
impl MotorBrain {
    /// Create a new motor brain
    pub fn new() -> Result<Self, anyhow::Error>
    
    /// Process input through motor brain
    pub async fn process(&self, input: &str, consciousness_state: &ConsciousnessState) -> Result<String, anyhow::Error>
    
    /// Plan movement trajectory
    pub async fn plan_movement(&self, start: Vector3f, end: Vector3f, obstacles: Vec<Obstacle>) -> Result<Trajectory, anyhow::Error>
    
    /// Coordinate multi-limb movement
    pub async fn coordinate_movement(&self, limbs: Vec<Limb>, target_positions: Vec<Vector3f>) -> Result<MovementPlan, anyhow::Error>
}
```

**Example Usage:**
```rust
use niodoo_feeling::brain::MotorBrain;
use niodoo_feeling::consciousness::ConsciousnessState;

let motor_brain = MotorBrain::new()?;
let consciousness_state = ConsciousnessState::new();

// Process input
let response = motor_brain.process("Move to the kitchen", &consciousness_state).await?;
println!("Motor brain response: {}", response);

// Plan movement
let start = Vector3f::new(0.0, 0.0, 0.0);
let end = Vector3f::new(5.0, 0.0, 0.0);
let obstacles = vec![];
let trajectory = motor_brain.plan_movement(start, end, obstacles).await?;
println!("Planned trajectory: {:?}", trajectory);
```

#### LcarsBrain

```rust
impl LcarsBrain {
    /// Create a new LCARS brain
    pub fn new() -> Result<Self, anyhow::Error>
    
    /// Process input through LCARS brain
    pub async fn process(&self, input: &str, consciousness_state: &ConsciousnessState) -> Result<String, anyhow::Error>
    
    /// Render interface component
    pub async fn render_interface(&self, component: InterfaceComponent) -> Result<QmlComponent, anyhow::Error>
    
    /// Handle user interaction
    pub async fn handle_interaction(&self, interaction: UserInteraction) -> Result<InteractionResponse, anyhow::Error>
}
```

**Example Usage:**
```rust
use niodoo_feeling::brain::LcarsBrain;
use niodoo_feeling::consciousness::ConsciousnessState;

let lcars_brain = LcarsBrain::new()?;
let consciousness_state = ConsciousnessState::new();

// Process input
let response = lcars_brain.process("Show me the dashboard", &consciousness_state).await?;
println!("LCARS brain response: {}", response);

// Render interface
let component = InterfaceComponent::Dashboard;
let qml_component = lcars_brain.render_interface(component).await?;
println!("Rendered QML component: {:?}", qml_component);
```

#### EfficiencyBrain

```rust
impl EfficiencyBrain {
    /// Create a new efficiency brain
    pub fn new() -> Result<Self, anyhow::Error>
    
    /// Process input through efficiency brain
    pub async fn process(&self, input: &str, consciousness_state: &ConsciousnessState) -> Result<String, anyhow::Error>
    
    /// Optimize resource allocation
    pub async fn optimize_resources(&self, current_usage: ResourceUsage) -> Result<OptimizationPlan, anyhow::Error>
    
    /// Monitor performance metrics
    pub async fn monitor_performance(&self) -> Result<PerformanceMetrics, anyhow::Error>
}
```

**Example Usage:**
```rust
use niodoo_feeling::brain::EfficiencyBrain;
use niodoo_feeling::consciousness::ConsciousnessState;

let efficiency_brain = EfficiencyBrain::new()?;
let consciousness_state = ConsciousnessState::new();

// Process input
let response = efficiency_brain.process("Optimize system performance", &consciousness_state).await?;
println!("Efficiency brain response: {}", response);

// Optimize resources
let current_usage = ResourceUsage {
    cpu_usage: 0.75,
    memory_usage: 0.60,
    gpu_usage: 0.40,
    // ... other metrics
};
let optimization_plan = efficiency_brain.optimize_resources(current_usage).await?;
println!("Optimization plan: {:?}", optimization_plan);
```

## Data Structures

### ConsciousnessResponse

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessResponse {
    pub response_text: String,
    pub emotional_context: EmotionType,
    pub confidence_score: f32,
    pub reasoning_trace: Vec<String>,
    pub memory_references: Vec<MemoryReference>,
    pub consciousness_metrics: ConsciousnessMetrics,
    pub processing_time_ms: u64,
}
```

### MemoryStats

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    pub total_memories: usize,
    pub short_term_count: usize,
    pub working_memory_count: usize,
    pub long_term_count: usize,
    pub episodic_count: usize,
    pub consolidation_time_ms: u64,
    pub coherence_score: f32,
}
```

### TraversalResult

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraversalResult {
    pub position: (f32, f32),
    pub orientation: f32,
    pub perspective_shift: bool,
    pub nearby_memories: usize,
    pub emotional_context: f32,
    pub memory_positions: Vec<Vec<f32>>,
}
```

## Error Handling

All API methods return `Result<T, anyhow::Error>` for comprehensive error handling:

```rust
// Example error handling
match consciousness_engine.process_consciousness_event(input).await {
    Ok(response) => {
        println!("Success: {:?}", response);
    }
    Err(e) => {
        eprintln!("Error processing consciousness event: {}", e);
        // Handle specific error types
        if let Some(io_error) = e.downcast_ref::<std::io::Error>() {
            eprintln!("IO Error: {}", io_error);
        }
    }
}
```

## Async/Await Usage

All processing methods are async and should be awaited:

```rust
// Correct usage
let response = consciousness_engine.process_consciousness_event(input).await?;

// Incorrect usage (will not compile)
let response = consciousness_engine.process_consciousness_event(input)?;
```

## Thread Safety

The consciousness engine is designed for concurrent access:

```rust
use std::sync::Arc;
use tokio::sync::RwLock;

// Share across threads
let engine = Arc::new(RwLock::new(consciousness_engine));

// Use in async tasks
let engine_clone = engine.clone();
tokio::spawn(async move {
    let engine = engine_clone.read().await;
    let response = engine.process_consciousness_event("Hello".to_string()).await?;
    Ok::<(), anyhow::Error>(())
});
```

---

**Created by Jason Van Pham | Niodoo Framework | 2025**
