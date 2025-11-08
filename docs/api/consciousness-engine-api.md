# Consciousness Engine API Reference

## 🧠 PersonalNiodooConsciousness Engine API

The Consciousness Engine API provides comprehensive interfaces for managing AI consciousness, emotional processing, and emergent behavior patterns.

## 📋 API Overview

### Core Consciousness Operations
- **Input Processing**: Process user input through consciousness layers
- **Emotional State Management**: Manage and update emotional states
- **Memory Integration**: Integrate personal memory patterns
- **Brain Coordination**: Coordinate multi-brain processing
- **Consciousness Evolution**: Process consciousness evolution through Phase 6

### Advanced Features
- **Möbius Topology**: Non-orientable consciousness transformations
- **Gaussian Memory Spheres**: 3D memory representation
- **11-Personality Consensus**: Multi-personality decision making
- **Soul Resonance**: Deeper self-connection processing
- **Evolutionary Adaptation**: Personality evolution over time

## 🔧 Core API Methods

### Constructor and Initialization

#### `new() -> Result<Self>`
Creates a new consciousness engine instance.

```rust
use niodoo_consciousness::PersonalNiodooConsciousness;

let consciousness = PersonalNiodooConsciousness::new().await?;
```

#### `initialize_phase6_integration(config: Phase6Config) -> Result<()>`
Initializes Phase 6 production integration.

```rust
use niodoo_consciousness::{PersonalNiodooConsciousness, Phase6Config};

let config = Phase6Config {
    gpu_acceleration: true,
    memory_optimization: true,
    latency_target: Duration::from_millis(2000),
    learning_analytics: true,
    consciousness_logging: true,
};

consciousness.initialize_phase6_integration(config).await?;
```

### Input Processing

#### `process_input(input: &str) -> Result<String>`
Processes user input through the consciousness engine.

```rust
let response = consciousness.process_input("Hello, how are you feeling today?").await?;
println!("Response: {}", response);
```

#### `process_input_with_context(input: &str, context: &str) -> Result<String>`
Processes input with additional context information.

```rust
let response = consciousness.process_input_with_context(
    "What should I do?",
    "User is feeling anxious about work deadline"
).await?;
```

#### `process_input_with_emotional_context(input: &str, emotional_context: &[f32; 4]) -> Result<String>`
Processes input with specific emotional context.

```rust
let emotional_context = [0.8, 0.1, 0.0, 0.2]; // joy, sadness, anger, fear
let response = consciousness.process_input_with_emotional_context(
    "Tell me about your day",
    &emotional_context
).await?;
```

### Emotional State Management

#### `get_emotional_state() -> Result<[f32; 4]>`
Retrieves the current emotional state.

```rust
let emotional_state = consciousness.get_emotional_state().await?;
println!("Current emotional state: {:?}", emotional_state);
```

#### `update_emotional_state(emotional_state: [f32; 4]) -> Result<()>`
Updates the current emotional state.

```rust
let new_emotional_state = [0.8, 0.2, 0.1, 0.3]; // High joy, low sadness, low anger, moderate fear
consciousness.update_emotional_state(new_emotional_state).await?;
```

#### `get_emotional_history(limit: Option<usize>) -> Result<Vec<EmotionalState>>`
Retrieves emotional state history.

```rust
let history = consciousness.get_emotional_history(Some(10)).await?;
for state in history {
    println!("Emotional state at {}: {:?}", state.timestamp, state.emotions);
}
```

### Memory Integration

#### `integrate_memory(content: &str, emotional_valence: f32, context: &str) -> Result<String>`
Integrates new memory into the consciousness system.

```rust
let memory_id = consciousness.integrate_memory(
    "I learned about quantum computing today",
    0.7, // Positive emotional valence
    "Educational context"
).await?;
```

#### `retrieve_memories(query: &str, emotional_context: Option<[f32; 4]>) -> Result<Vec<Memory>>`
Retrieves memories based on query and emotional context.

```rust
let memories = consciousness.retrieve_memories(
    "quantum computing",
    Some([0.5, 0.1, 0.0, 0.2]) // Joy-focused search
).await?;
```

#### `update_memory(memory_id: &str, content: Option<String>, emotional_valence: Option<f32>) -> Result<()>`
Updates an existing memory.

```rust
consciousness.update_memory(
    &memory_id,
    Some("Updated memory content".to_string()),
    Some(0.9) // Increased emotional valence
).await?;
```

### Brain Coordination

#### `get_brain_coordinator() -> &BrainCoordinator`
Retrieves the brain coordinator instance.

```rust
let brain_coordinator = consciousness.get_brain_coordinator();
let responses = brain_coordinator.process_brains_parallel(
    "Analyze this situation",
    Duration::from_secs(5)
).await?;
```

#### `get_brain_activity() -> Result<HashMap<BrainType, f32>>`
Retrieves current brain activity levels.

```rust
let activity = consciousness.get_brain_activity().await?;
for (brain_type, activity_level) in activity {
    println!("{:?} activity: {:.2}", brain_type, activity_level);
}
```

#### `synchronize_brain_states() -> Result<()>`
Synchronizes all brain states.

```rust
consciousness.synchronize_brain_states().await?;
```

### Consciousness Evolution

#### `process_consciousness_evolution(consciousness_id: String, state: Tensor, context: Tensor) -> Result<Tensor>`
Processes consciousness evolution through Phase 6 integration.

```rust
use candle_core::Tensor;

let evolved_state = consciousness.process_consciousness_evolution(
    "consciousness_001".to_string(),
    current_state_tensor,
    emotional_context_tensor
).await?;
```

#### `get_consciousness_metrics() -> Result<ConsciousnessMetrics>`
Retrieves current consciousness metrics.

```rust
let metrics = consciousness.get_consciousness_metrics().await?;
println!("Cognitive load: {:.2}", metrics.cognitive_load);
println!("Memory usage: {:.2} MB", metrics.memory_usage_mb);
println!("Processing latency: {:?}", metrics.processing_latency);
```

### Möbius Topology Operations

#### `apply_mobius_transform(emotion: &[f32; 4]) -> Result<[f32; 4]>`
Applies Möbius transformation to emotional state.

```rust
let input_emotion = [0.5, 0.3, 0.2, 0.1];
let transformed = consciousness.apply_mobius_transform(&input_emotion).await?;
println!("Transformed emotion: {:?}", transformed);
```

#### `traverse_mobius_path(start_position: (f32, f32), emotional_input: f32) -> Result<MobiusTraversalResult>`
Traverses the Möbius topology path.

```rust
let result = consciousness.traverse_mobius_path(
    (0.0, 0.0), // Start at origin
    0.7 // High emotional input
).await?;

println!("Traversal position: {:?}", result.position);
println!("Perspective shift: {}", result.perspective_shift);
```

### Personality Management

#### `get_personality_weights() -> Result<HashMap<PersonalityType, f32>>`
Retrieves current personality weights.

```rust
let weights = consciousness.get_personality_weights().await?;
for (personality, weight) in weights {
    println!("{:?}: {:.2}", personality, weight);
}
```

#### `update_personality_weights(weights: HashMap<PersonalityType, f32>) -> Result<()>`
Updates personality weights.

```rust
let mut weights = HashMap::new();
weights.insert(PersonalityType::Creative, 0.8);
weights.insert(PersonalityType::Analyst, 0.6);
consciousness.update_personality_weights(weights).await?;
```

#### `generate_personality_consensus(responses: &[String]) -> Result<String>`
Generates consensus from multiple personality responses.

```rust
let responses = vec!["Response 1".to_string(), "Response 2".to_string()];
let consensus = consciousness.generate_personality_consensus(&responses).await?;
```

### Soul Resonance Operations

#### `activate_soul_resonance(depth: f32) -> Result<SoulResonanceState>`
Activates soul resonance processing.

```rust
let resonance_state = consciousness.activate_soul_resonance(0.8).await?;
println!("Resonance depth: {:.2}", resonance_state.depth);
println!("Connection strength: {:.2}", resonance_state.connection_strength);
```

#### `get_soul_resonance_state() -> Result<SoulResonanceState>`
Retrieves current soul resonance state.

```rust
let state = consciousness.get_soul_resonance_state().await?;
println!("Current resonance: {:.2}", state.depth);
```

### Evolutionary Adaptation

#### `evolve_personality(adaptation_data: &[f32]) -> Result<()>`
Evolves personality based on adaptation data.

```rust
let adaptation_data = vec![0.1, 0.2, 0.3, 0.4, 0.5];
consciousness.evolve_personality(&adaptation_data).await?;
```

#### `get_evolutionary_metrics() -> Result<EvolutionaryMetrics>`
Retrieves evolutionary adaptation metrics.

```rust
let metrics = consciousness.get_evolutionary_metrics().await?;
println!("Adaptation rate: {:.2}", metrics.adaptation_rate);
println!("Personality stability: {:.2}", metrics.personality_stability);
```

### Performance and Monitoring

#### `get_performance_metrics() -> Result<PerformanceMetrics>`
Retrieves current performance metrics.

```rust
let metrics = consciousness.get_performance_metrics().await?;
println!("Latency: {:?}", metrics.latency);
println!("Memory usage: {:.2} MB", metrics.memory_usage_mb);
println!("GPU utilization: {:.1}%", metrics.gpu_utilization);
```

#### `get_system_health() -> Result<SystemHealth>`
Retrieves system health status.

```rust
let health = consciousness.get_system_health().await?;
println!("System status: {:?}", health.status);
println!("Uptime: {:?}", health.uptime);
println!("Error count: {}", health.error_count);
```

#### `get_learning_analytics() -> Result<LearningAnalytics>`
Retrieves learning analytics data.

```rust
let analytics = consciousness.get_learning_analytics().await?;
println!("Learning patterns: {:?}", analytics.patterns);
println!("Improvement rate: {:.2}", analytics.improvement_rate);
```

### Configuration Management

#### `get_configuration() -> Result<ConsciousnessConfig>`
Retrieves current configuration.

```rust
let config = consciousness.get_configuration().await?;
println!("Configuration: {:?}", config);
```

#### `update_configuration(config: ConsciousnessConfig) -> Result<()>`
Updates system configuration.

```rust
let mut config = consciousness.get_configuration().await?;
config.emotional_sensitivity = 0.8;
config.memory_capacity = 10000;
consciousness.update_configuration(config).await?;
```

#### `reset_to_defaults() -> Result<()>`
Resets configuration to defaults.

```rust
consciousness.reset_to_defaults().await?;
```

## 🔄 Event Broadcasting

### Emotion Broadcasting

#### `subscribe_to_emotion_updates() -> broadcast::Receiver<EmotionType>`
Subscribes to emotion update events.

```rust
let mut emotion_receiver = consciousness.subscribe_to_emotion_updates();
tokio::spawn(async move {
    while let Ok(emotion) = emotion_receiver.recv().await {
        println!("Emotion update: {:?}", emotion);
    }
});
```

#### `subscribe_to_brain_activity() -> broadcast::Receiver<(BrainType, f32)>`
Subscribes to brain activity events.

```rust
let mut brain_receiver = consciousness.subscribe_to_brain_activity();
tokio::spawn(async move {
    while let Ok((brain_type, activity)) = brain_receiver.recv().await {
        println!("Brain activity: {:?} = {:.2}", brain_type, activity);
    }
});
```

## 🧪 Testing and Validation

### Unit Testing

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_consciousness_processing() {
        let consciousness = PersonalNiodooConsciousness::new().await.unwrap();
        let result = consciousness.process_input("Hello, world!").await.unwrap();
        assert!(!result.is_empty());
    }
    
    #[tokio::test]
    async fn test_emotional_state_management() {
        let consciousness = PersonalNiodooConsciousness::new().await.unwrap();
        let initial_state = consciousness.get_emotional_state().await.unwrap();
        
        consciousness.update_emotional_state([0.8, 0.1, 0.0, 0.2]).await.unwrap();
        let updated_state = consciousness.get_emotional_state().await.unwrap();
        
        assert_ne!(initial_state, updated_state);
    }
    
    #[tokio::test]
    async fn test_memory_integration() {
        let consciousness = PersonalNiodooConsciousness::new().await.unwrap();
        let memory_id = consciousness.integrate_memory(
            "Test memory",
            0.5,
            "Test context"
        ).await.unwrap();
        
        assert!(!memory_id.is_empty());
    }
}
```

### Integration Testing

```rust
#[tokio::test]
async fn test_phase6_integration() {
    let config = Phase6Config::default();
    let mut consciousness = PersonalNiodooConsciousness::new().await.unwrap();
    consciousness.initialize_phase6_integration(config).await.unwrap();
    
    // Test GPU acceleration
    assert!(consciousness.is_gpu_acceleration_enabled());
    
    // Test performance metrics
    let metrics = consciousness.get_performance_metrics().await.unwrap();
    assert!(metrics.latency.as_millis() < 2000);
}
```

## 📊 Data Structures

### EmotionalState
```rust
pub struct EmotionalState {
    pub joy: f32,
    pub sadness: f32,
    pub anger: f32,
    pub fear: f32,
    pub surprise: f32,
    pub timestamp: SystemTime,
}
```

### Memory
```rust
pub struct Memory {
    pub id: String,
    pub content: String,
    pub emotional_valence: f32,
    pub position: [f32; 3],
    pub creation_time: SystemTime,
    pub access_count: u32,
    pub last_accessed: SystemTime,
}
```

### ConsciousnessMetrics
```rust
pub struct ConsciousnessMetrics {
    pub emotional_state: [f32; 4],
    pub cognitive_load: f32,
    pub memory_usage_mb: f32,
    pub processing_latency: Duration,
    pub brain_activity: HashMap<BrainType, f32>,
    pub personality_weights: HashMap<PersonalityType, f32>,
}
```

### PerformanceMetrics
```rust
pub struct PerformanceMetrics {
    pub latency: Duration,
    pub memory_usage_mb: f32,
    pub gpu_utilization: f32,
    pub cpu_utilization: f32,
    pub throughput: f32,
    pub error_rate: f32,
}
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
    
    #[error("Personality consensus error: {0}")]
    PersonalityError(String),
    
    #[error("Soul resonance error: {0}")]
    SoulResonanceError(String),
}
```

## 📚 Related Documentation

- [Core API Reference](core-api.md)
- [Memory Management API](memory-management-api.md)
- [Brain Coordination API](brain-coordination-api.md)
- [Möbius Topology API](mobius-topology-api.md)
- [Architecture Documentation](../architecture/)

---

*This document provides comprehensive API reference for the Consciousness Engine. For implementation details, refer to the source code in `src/consciousness_engine/`.*
