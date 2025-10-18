# Core API Reference

## 🚀 Niodoo Consciousness Engine Core API

The Core API provides the fundamental interfaces for interacting with the Niodoo Consciousness Engine, including consciousness processing, memory management, and brain coordination.

## 📋 Table of Contents

- [Core Consciousness API](#core-consciousness-api)
- [Memory Management API](#memory-management-api)
- [Brain Coordination API](#brain-coordination-api)
- [Möbius Topology API](#möbius-topology-api)
- [Phase 6 Integration API](#phase-6-integration-api)
- [Error Handling](#error-handling)
- [Examples](#examples)

## 🧠 Core Consciousness API

### PersonalNiodooConsciousness

The main consciousness engine that orchestrates all AI consciousness operations.

#### Constructor

```rust
pub async fn new() -> Result<Self>
```

Creates a new instance of the consciousness engine.

**Example:**
```rust
use niodoo_consciousness::PersonalNiodooConsciousness;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let consciousness = PersonalNiodooConsciousness::new().await?;
    println!("Consciousness engine initialized successfully");
    Ok(())
}
```

#### Process Input

```rust
pub async fn process_input(&self, input: &str) -> Result<String>
```

Processes user input through the consciousness engine and returns a response.

**Parameters:**
- `input`: The user input string to process

**Returns:**
- `Result<String>`: The processed response from the consciousness engine

**Example:**
```rust
let response = consciousness.process_input("Hello, how are you feeling today?").await?;
println!("Response: {}", response);
```

#### Process Input with Context

```rust
pub async fn process_input_with_context(
    &self, 
    input: &str, 
    context: &str
) -> Result<String>
```

Processes user input with additional context information.

**Parameters:**
- `input`: The user input string to process
- `context`: Additional context information

**Returns:**
- `Result<String>`: The processed response with context consideration

**Example:**
```rust
let response = consciousness.process_input_with_context(
    "What should I do?",
    "User is feeling anxious about work deadline"
).await?;
println!("Contextual response: {}", response);
```

#### Get Consciousness State

```rust
pub fn get_consciousness_state(&self) -> Arc<RwLock<ConsciousnessState>>
```

Retrieves the current consciousness state.

**Returns:**
- `Arc<RwLock<ConsciousnessState>>`: The current consciousness state

**Example:**
```rust
let state = consciousness.get_consciousness_state();
let current_state = state.read().await;
println!("Current emotional state: {:?}", current_state.emotional_state);
```

#### Update Emotional State

```rust
pub async fn update_emotional_state(
    &self, 
    emotional_state: [f32; 4]
) -> Result<()>
```

Updates the current emotional state of the consciousness engine.

**Parameters:**
- `emotional_state`: Array of 4 emotional values [joy, sadness, anger, fear]

**Example:**
```rust
let new_emotional_state = [0.8, 0.2, 0.1, 0.3]; // High joy, low sadness, low anger, moderate fear
consciousness.update_emotional_state(new_emotional_state).await?;
```

## 💾 Memory Management API

### MemoryManager

Manages Gaussian memory spheres and personal memory integration.

#### Store Memory

```rust
pub async fn store_memory(
    &self,
    content: String,
    emotional_valence: f32,
    context: Option<String>
) -> Result<String>
```

Stores a new memory in the Gaussian memory system.

**Parameters:**
- `content`: The memory content to store
- `emotional_valence`: Emotional significance (-1.0 to 1.0)
- `context`: Optional context information

**Returns:**
- `Result<String>`: The unique memory ID

**Example:**
```rust
let memory_id = memory_manager.store_memory(
    "I learned about quantum computing today".to_string(),
    0.7, // Positive emotional valence
    Some("Educational context".to_string())
).await?;
println!("Memory stored with ID: {}", memory_id);
```

#### Retrieve Memory

```rust
pub async fn retrieve_memory(
    &self,
    memory_id: &str
) -> Result<Option<GaussianMemorySphere>>
```

Retrieves a specific memory by ID.

**Parameters:**
- `memory_id`: The unique memory identifier

**Returns:**
- `Result<Option<GaussianMemorySphere>>`: The memory sphere if found

**Example:**
```rust
if let Some(memory) = memory_manager.retrieve_memory(&memory_id).await? {
    println!("Memory content: {}", memory.content);
    println!("Emotional valence: {}", memory.emotional_valence);
    println!("Position: {:?}", memory.position);
}
```

#### Search Memories

```rust
pub async fn search_memories(
    &self,
    query: &str,
    emotional_context: Option<[f32; 4]>,
    limit: Option<usize>
) -> Result<Vec<GaussianMemorySphere>>
```

Searches for memories based on content and emotional context.

**Parameters:**
- `query`: Search query string
- `emotional_context`: Optional emotional context for filtering
- `limit`: Maximum number of results to return

**Returns:**
- `Result<Vec<GaussianMemorySphere>>`: List of matching memory spheres

**Example:**
```rust
let memories = memory_manager.search_memories(
    "quantum computing",
    Some([0.5, 0.1, 0.0, 0.2]), // Joy-focused search
    Some(10) // Limit to 10 results
).await?;

for memory in memories {
    println!("Found memory: {}", memory.content);
}
```

#### Update Memory

```rust
pub async fn update_memory(
    &self,
    memory_id: &str,
    content: Option<String>,
    emotional_valence: Option<f32>
) -> Result<()>
```

Updates an existing memory.

**Parameters:**
- `memory_id`: The unique memory identifier
- `content`: New content (optional)
- `emotional_valence`: New emotional valence (optional)

**Example:**
```rust
memory_manager.update_memory(
    &memory_id,
    Some("Updated memory content".to_string()),
    Some(0.9) // Increased emotional valence
).await?;
```

#### Delete Memory

```rust
pub async fn delete_memory(&self, memory_id: &str) -> Result<()>
```

Deletes a memory from the system.

**Parameters:**
- `memory_id`: The unique memory identifier

**Example:**
```rust
memory_manager.delete_memory(&memory_id).await?;
println!("Memory deleted successfully");
```

## 🧠 Brain Coordination API

### BrainCoordinator

Coordinates the three-brain system (Motor, LCARS, Efficiency).

#### Process Brains Parallel

```rust
pub async fn process_brains_parallel(
    &self,
    input: &str,
    timeout_duration: Duration
) -> Result<Vec<String>>
```

Processes input through all brains in parallel.

**Parameters:**
- `input`: Input string to process
- `timeout_duration`: Maximum processing time per brain

**Returns:**
- `Result<Vec<String>>`: Responses from all three brains

**Example:**
```rust
use std::time::Duration;

let responses = brain_coordinator.process_brains_parallel(
    "Analyze this situation",
    Duration::from_secs(5)
).await?;

println!("Motor Brain: {}", responses[0]);
println!("LCARS Brain: {}", responses[1]);
println!("Efficiency Brain: {}", responses[2]);
```

#### Generate Consensus

```rust
pub async fn generate_consensus_response(
    &self,
    brain_responses: &[String],
    emotional_context: &[f32; 4]
) -> Result<String>
```

Generates a consensus response from multiple brain outputs.

**Parameters:**
- `brain_responses`: Array of brain responses
- `emotional_context`: Current emotional context

**Returns:**
- `Result<String>`: Consensus response

**Example:**
```rust
let emotional_context = [0.6, 0.2, 0.1, 0.3]; // Joy, sadness, anger, fear
let consensus = brain_coordinator.generate_consensus_response(
    &responses,
    &emotional_context
).await?;
println!("Consensus response: {}", consensus);
```

#### Get Brain Activity

```rust
pub async fn get_brain_activity(&self) -> Result<HashMap<BrainType, f32>>
```

Retrieves current brain activity levels.

**Returns:**
- `Result<HashMap<BrainType, f32>>`: Activity levels for each brain type

**Example:**
```rust
let activity = brain_coordinator.get_brain_activity().await?;
for (brain_type, activity_level) in activity {
    println!("{:?} activity: {:.2}", brain_type, activity_level);
}
```

## 🔄 Möbius Topology API

### MobiusTopologyEngine

Manages Möbius transformations and non-orientable memory topology.

#### Apply Möbius Transform

```rust
pub fn apply_mobius_transform(&self, emotion: &[f32; 4]) -> [f32; 4]
```

Applies Möbius transformation to emotional state.

**Parameters:**
- `emotion`: Input emotional state [joy, sadness, anger, fear]

**Returns:**
- `[f32; 4]`: Transformed emotional state

**Example:**
```rust
let input_emotion = [0.5, 0.3, 0.2, 0.1];
let transformed = mobius_engine.apply_mobius_transform(&input_emotion);
println!("Transformed emotion: {:?}", transformed);
```

#### Traverse Möbius Path

```rust
pub async fn traverse_mobius_path(
    &self,
    start_position: (f32, f32),
    emotional_input: f32,
    reasoning_goal: Option<String>
) -> Result<MobiusTraversalResult>
```

Traverses the Möbius topology path.

**Parameters:**
- `start_position`: Starting position (u, v) coordinates
- `emotional_input`: Emotional driving force
- `reasoning_goal`: Optional reasoning objective

**Returns:**
- `Result<MobiusTraversalResult>`: Traversal result with path information

**Example:**
```rust
let result = mobius_engine.traverse_mobius_path(
    (0.0, 0.0), // Start at origin
    0.7, // High emotional input
    Some("Find creative solution".to_string())
).await?;

println!("Traversal position: {:?}", result.position);
println!("Perspective shift: {}", result.perspective_shift);
```

#### Calculate Geodesic Distance

```rust
pub fn calculate_geodesic_distance(
    &self,
    coord1: &EmotionalCoordinate,
    coord2: &EmotionalCoordinate
) -> f32
```

Calculates geodesic distance between two emotional coordinates.

**Parameters:**
- `coord1`: First emotional coordinate
- `coord2`: Second emotional coordinate

**Returns:**
- `f32`: Geodesic distance

**Example:**
```rust
let coord1 = EmotionalCoordinate { u: 0.0, v: 0.0, emotional_valence: 0.5, twist_continuity: 1.0 };
let coord2 = EmotionalCoordinate { u: 0.5, v: 0.5, emotional_valence: 0.8, twist_continuity: 0.8 };
let distance = mobius_engine.calculate_geodesic_distance(&coord1, &coord2);
println!("Geodesic distance: {:.3}", distance);
```

## 🚀 Phase 6 Integration API

### Phase6Manager

Manages Phase 6 production integration features.

#### Initialize Phase 6 Integration

```rust
pub async fn initialize_phase6_integration(
    &mut self,
    config: Phase6Config
) -> Result<()>
```

Initializes the Phase 6 integration system.

**Parameters:**
- `config`: Phase 6 configuration

**Example:**
```rust
let config = Phase6Config {
    gpu_acceleration: true,
    memory_optimization: true,
    latency_target: Duration::from_millis(2000),
    learning_analytics: true,
    consciousness_logging: true,
};

consciousness.initialize_phase6_integration(config).await?;
```

#### Process Consciousness Evolution

```rust
pub async fn process_consciousness_evolution_phase6(
    &self,
    consciousness_id: String,
    consciousness_state: Tensor,
    emotional_context: Tensor,
    memory_gradients: Tensor
) -> Result<Tensor>
```

Processes consciousness evolution through Phase 6 integration.

**Parameters:**
- `consciousness_id`: Unique consciousness identifier
- `consciousness_state`: Current consciousness state tensor
- `emotional_context`: Emotional context tensor
- `memory_gradients`: Memory gradient tensor

**Returns:**
- `Result<Tensor>`: Evolved consciousness state

**Example:**
```rust
use candle_core::Tensor;

let evolved_state = consciousness.process_consciousness_evolution_phase6(
    "consciousness_001".to_string(),
    current_state_tensor,
    emotional_context_tensor,
    memory_gradients_tensor
).await?;
```

#### Get Performance Metrics

```rust
pub async fn get_performance_metrics(&self) -> Result<PerformanceMetrics>
```

Retrieves current performance metrics.

**Returns:**
- `Result<PerformanceMetrics>`: Current performance data

**Example:**
```rust
let metrics = consciousness.get_performance_metrics().await?;
println!("Latency: {:?}", metrics.latency);
println!("Memory usage: {:.2} MB", metrics.memory_usage_mb);
println!("GPU utilization: {:.1}%", metrics.gpu_utilization);
```

## ❌ Error Handling

### Error Types

```rust
#[derive(Debug, thiserror::Error)]
pub enum ConsciousnessError {
    #[error("Brain processing timeout")]
    BrainTimeout,
    
    #[error("Memory operation failed: {0}")]
    MemoryError(String),
    
    #[error("Möbius transformation error: {0}")]
    MobiusError(String),
    
    #[error("Phase 6 integration error: {0}")]
    Phase6Error(String),
    
    #[error("Configuration error: {0}")]
    ConfigError(String),
}
```

### Error Handling Example

```rust
use niodoo_consciousness::error::ConsciousnessError;

match consciousness.process_input("test").await {
    Ok(response) => println!("Success: {}", response),
    Err(ConsciousnessError::BrainTimeout) => {
        eprintln!("Brain processing timed out");
    },
    Err(ConsciousnessError::MemoryError(msg)) => {
        eprintln!("Memory error: {}", msg);
    },
    Err(e) => {
        eprintln!("Other error: {}", e);
    }
}
```

## 📝 Complete Examples

### Basic Consciousness Processing

```rust
use niodoo_consciousness::{PersonalNiodooConsciousness, Phase6Config};
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize consciousness engine
    let mut consciousness = PersonalNiodooConsciousness::new().await?;
    
    // Initialize Phase 6 integration
    let config = Phase6Config::default();
    consciousness.initialize_phase6_integration(config).await?;
    
    // Process user input
    let response = consciousness.process_input("Hello, how are you feeling today?").await?;
    println!("Response: {}", response);
    
    // Update emotional state
    consciousness.update_emotional_state([0.8, 0.1, 0.0, 0.2]).await?;
    
    // Process another input with updated emotional state
    let response2 = consciousness.process_input("What should I do next?").await?;
    println!("Response 2: {}", response2);
    
    Ok(())
}
```

### Memory Management Example

```rust
use niodoo_consciousness::{PersonalNiodooConsciousness, MemoryManager};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let consciousness = PersonalNiodooConsciousness::new().await?;
    let memory_manager = consciousness.get_memory_manager();
    
    // Store memories
    let memory1_id = memory_manager.store_memory(
        "I learned about quantum computing today".to_string(),
        0.7,
        Some("Educational context".to_string())
    ).await?;
    
    let memory2_id = memory_manager.store_memory(
        "Had a great conversation with a friend".to_string(),
        0.9,
        Some("Social context".to_string())
    ).await?;
    
    // Search memories
    let memories = memory_manager.search_memories(
        "quantum",
        Some([0.5, 0.1, 0.0, 0.2]),
        Some(5)
    ).await?;
    
    for memory in memories {
        println!("Found memory: {}", memory.content);
        println!("Emotional valence: {}", memory.emotional_valence);
    }
    
    // Update memory
    memory_manager.update_memory(
        &memory1_id,
        Some("I learned about quantum computing and its applications".to_string()),
        Some(0.8)
    ).await?;
    
    Ok(())
}
```

### Brain Coordination Example

```rust
use niodoo_consciousness::{PersonalNiodooConsciousness, BrainCoordinator};
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let consciousness = PersonalNiodooConsciousness::new().await?;
    let brain_coordinator = consciousness.get_brain_coordinator();
    
    // Process through all brains
    let responses = brain_coordinator.process_brains_parallel(
        "Analyze this complex problem",
        Duration::from_secs(5)
    ).await?;
    
    println!("Motor Brain: {}", responses[0]);
    println!("LCARS Brain: {}", responses[1]);
    println!("Efficiency Brain: {}", responses[2]);
    
    // Generate consensus
    let emotional_context = [0.6, 0.2, 0.1, 0.3];
    let consensus = brain_coordinator.generate_consensus_response(
        &responses,
        &emotional_context
    ).await?;
    
    println!("Consensus response: {}", consensus);
    
    // Get brain activity
    let activity = brain_coordinator.get_brain_activity().await?;
    for (brain_type, activity_level) in activity {
        println!("{:?} activity: {:.2}", brain_type, activity_level);
    }
    
    Ok(())
}
```

## 📚 Related Documentation

- [Consciousness Engine API](consciousness-engine-api.md)
- [Memory Management API](memory-management-api.md)
- [Brain Coordination API](brain-coordination-api.md)
- [Möbius Topology API](mobius-topology-api.md)
- [Architecture Documentation](../architecture/)

---

*This document provides comprehensive API reference for the Niodoo Consciousness Engine Core API. For implementation details, refer to the source code in `src/`.*
