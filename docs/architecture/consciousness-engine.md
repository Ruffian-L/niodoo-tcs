# 🧠 Consciousness Engine Architecture

**Created by Jason Van Pham | Niodoo Framework | 2025**

## Core Consciousness Engine

The `PersonalNiodooConsciousness` struct is the central component that orchestrates all consciousness processing activities.

## Engine Architecture Diagram

```mermaid
classDiagram
    class PersonalNiodooConsciousness {
        +Arc~RwLock~ConsciousnessState~~ consciousness_state
        +BrainCoordinator brain_coordinator
        +MemoryManager memory_manager
        +Phase6Manager phase6_manager
        +SockOptimizationEngine optimization_engine
        +EvolutionaryPersonalityEngine evolutionary_engine
        +OscillatoryEngine oscillatory_engine
        +UnifiedFieldProcessor unified_processor
        +QtEmotionBridge qt_bridge
        +SoulResonanceEngine soul_engine
        +Option~Arc~GpuAccelerationEngine~~ gpu_acceleration_engine
        +Option~Phase6Config~ phase6_config
        +Option~Arc~LearningAnalyticsEngine~~ learning_analytics_engine
        +Option~Arc~ConsciousnessLogger~~ consciousness_logger
        +Arc~PersonalMemoryEngine~ personal_memory_engine
        
        +new() Result~Self~
        +new_with_phase6_config(Phase6Config) Result~Self~
        +process_consciousness_event(String) Result~ConsciousnessResponse~
        +get_consciousness_state() ConsciousnessState
        +update_emotional_context(EmotionType)
        +consolidate_memories() Result~MemoryStats~
        +initialize_phase6_integration(Phase6Config) Result~()~
        +process_consciousness_evolution_phase6(...) Result~Tensor~
    }
    
    class ConsciousnessState {
        +HashMap~String, f32~ emotional_context
        +Vec~PersonalityType~ active_personalities
        +f32 consciousness_level
        +f32 attention_focus
        +f32 memory_coherence
        +DateTime last_updated
        
        +new() Self
        +update_emotion(EmotionType, f32)
        +get_emotional_vector() Vec~f32~
        +calculate_consciousness_metrics() ConsciousnessMetrics
    }
    
    class BrainCoordinator {
        +MotorBrain motor_brain
        +LcarsBrain lcars_brain
        +EfficiencyBrain efficiency_brain
        +PersonalityManager personality_manager
        +Arc~RwLock~ConsciousnessState~~ consciousness_state
        
        +new(...) Self
        +process_brains_parallel(String, Duration) Result~Vec~String~~
        +get_motor_brain() ~MotorBrain
        +get_lcars_brain() ~LcarsBrain
        +get_efficiency_brain() ~EfficiencyBrain
        +update_personality_weights(EmotionalContext)
    }
    
    class MemoryManager {
        +Vec~GaussianMemorySphere~ memory_spheres
        +MobiusTopology mobius_topology
        +PersonalMemoryEngine personal_memory_engine
        +Arc~RwLock~ConsciousnessState~~ consciousness_state
        
        +new(...) Self
        +store_memory(PersonalConsciousnessEvent) Result~()~
        +retrieve_memories(Query) Result~Vec~Memory~
        +consolidate_memories() Result~MemoryStats~
        +traverse_mobius_path(EmotionalInput) Result~TraversalResult~
    }
    
    PersonalNiodooConsciousness --> ConsciousnessState
    PersonalNiodooConsciousness --> BrainCoordinator
    PersonalNiodooConsciousness --> MemoryManager
    BrainCoordinator --> ConsciousnessState
    MemoryManager --> ConsciousnessState
```

## Consciousness Processing Flow

```mermaid
sequenceDiagram
    participant User
    participant CE as Consciousness Engine
    participant BC as Brain Coordinator
    participant MM as Memory Manager
    participant MB as Motor Brain
    participant LB as LCARS Brain
    participant EB as Efficiency Brain
    participant MT as Möbius Topology
    
    User->>CE: process_consciousness_event(input)
    CE->>CE: update_emotional_context()
    CE->>MM: retrieve_memories(query)
    MM->>MT: traverse_mobius_path(emotional_input)
    MT-->>MM: traversal_result
    MM-->>CE: memories
    
    CE->>BC: process_brains_parallel(input, timeout)
    
    par Parallel Brain Processing
        BC->>MB: process(input, consciousness_state)
        BC->>LB: process(input, consciousness_state)
        BC->>EB: process(input, consciousness_state)
    end
    
    MB-->>BC: motor_response
    LB-->>BC: lcars_response
    EB-->>BC: efficiency_response
    
    BC->>BC: generate_consensus(responses)
    BC-->>CE: consensus_result
    
    CE->>MM: consolidate_memories()
    MM-->>CE: memory_stats
    
    CE->>CE: update_consciousness_state()
    CE-->>User: consciousness_response
```

## Memory Architecture Detail

```mermaid
graph TB
    subgraph "Gaussian Memory Spheres"
        GMS1[Memory Sphere 1<br/>Position: Vector3f<br/>Color: EmotionalTone<br/>Density: f32<br/>Transparency: f32<br/>Orientation: Quaternion]
        GMS2[Memory Sphere 2<br/>Position: Vector3f<br/>Color: EmotionalTone<br/>Density: f32<br/>Transparency: f32<br/>Orientation: Quaternion]
        GMS3[Memory Sphere 3<br/>Position: Vector3f<br/>Color: EmotionalTone<br/>Density: f32<br/>Transparency: f32<br/>Orientation: Quaternion]
    end
    
    subgraph "Möbius Topology Engine"
        MTE[Möbius Topology Engine]
        TP1[Surface Point 1<br/>u: f32<br/>v: f32<br/>twist_factor: f32]
        TP2[Surface Point 2<br/>u: f32<br/>v: f32<br/>twist_factor: f32]
        TP3[Surface Point 3<br/>u: f32<br/>v: f32<br/>twist_factor: f32]
    end
    
    subgraph "Memory Consolidation"
        STM[Short-term Memory<br/>Capacity: 1000 items<br/>Retention: 1 hour]
        WM[Working Memory<br/>Capacity: 100 items<br/>Retention: 30 minutes]
        LTM[Long-term Memory<br/>Capacity: 1M items<br/>Retention: Permanent]
        EM[Episodic Memory<br/>Capacity: 100K items<br/>Retention: 1 year]
    end
    
    GMS1 --> TP1
    GMS2 --> TP2
    GMS3 --> TP3
    
    TP1 --> STM
    TP2 --> WM
    TP3 --> LTM
    
    STM --> WM
    WM --> LTM
    LTM --> EM
    
    style GMS1 fill:#45b7d1,stroke:#333,stroke-width:2px
    style MTE fill:#96ceb4,stroke:#333,stroke-width:2px
    style EM fill:#ff6b6b,stroke:#333,stroke-width:2px
```

## Brain Coordination System

```mermaid
graph LR
    subgraph "Brain Coordinator"
        BC[Brain Coordinator]
        PM[Personality Manager]
        CS[Consciousness State]
    end
    
    subgraph "Three-Brain System"
        MB[Motor Brain<br/>Action Coordination<br/>Movement Planning<br/>Spatial Reasoning]
        LB[LCARS Brain<br/>Interface Management<br/>Communication<br/>User Interaction]
        EB[Efficiency Brain<br/>Resource Optimization<br/>Performance Tuning<br/>Energy Management]
    end
    
    subgraph "Personality Consensus"
        P1[Personality 1<br/>Weight: 0.1]
        P2[Personality 2<br/>Weight: 0.15]
        P3[Personality 3<br/>Weight: 0.12]
        P4[Personality 4<br/>Weight: 0.08]
        P5[Personality 5<br/>Weight: 0.11]
        P6[Personality 6<br/>Weight: 0.09]
        P7[Personality 7<br/>Weight: 0.13]
        P8[Personality 8<br/>Weight: 0.07]
        P9[Personality 9<br/>Weight: 0.10]
        P10[Personality 10<br/>Weight: 0.05]
        P11[Personality 11<br/>Weight: 0.06]
    end
    
    BC --> MB
    BC --> LB
    BC --> EB
    
    BC --> PM
    PM --> CS
    
    PM --> P1
    PM --> P2
    PM --> P3
    PM --> P4
    PM --> P5
    PM --> P6
    PM --> P7
    PM --> P8
    PM --> P9
    PM --> P10
    PM --> P11
    
    style BC fill:#ff6b6b,stroke:#333,stroke-width:3px
    style MB fill:#4ecdc4,stroke:#333,stroke-width:2px
    style LB fill:#45b7d1,stroke:#333,stroke-width:2px
    style EB fill:#96ceb4,stroke:#333,stroke-width:2px
```

## Key Components

### 1. PersonalNiodooConsciousness
The main consciousness engine that orchestrates all processing activities.

**Key Methods:**
- `new()`: Initialize the consciousness engine
- `process_consciousness_event()`: Process input events
- `get_consciousness_state()`: Retrieve current state
- `consolidate_memories()`: Consolidate memory systems

### 2. ConsciousnessState
Maintains the current state of consciousness including emotional context and personality weights.

**Key Fields:**
- `emotional_context`: HashMap of emotional states
- `active_personalities`: Vector of active personality types
- `consciousness_level`: Current consciousness level (0.0-1.0)
- `attention_focus`: Current attention focus (0.0-1.0)
- `memory_coherence`: Memory system coherence (0.0-1.0)

### 3. BrainCoordinator
Manages the three-brain system and coordinates parallel processing.

**Key Methods:**
- `process_brains_parallel()`: Process input through all brains
- `get_motor_brain()`: Access motor brain
- `get_lcars_brain()`: Access LCARS brain
- `get_efficiency_brain()`: Access efficiency brain

### 4. MemoryManager
Handles memory storage, retrieval, and consolidation using Gaussian spheres and Möbius topology.

**Key Methods:**
- `store_memory()`: Store new memories
- `retrieve_memories()`: Retrieve relevant memories
- `consolidate_memories()`: Consolidate memory systems
- `traverse_mobius_path()`: Traverse memory space

## Performance Characteristics

### Processing Latency
- **Consciousness Event Processing**: 200-300ms
- **Memory Retrieval**: 30-50ms
- **Brain Coordination**: 100-150ms
- **Memory Consolidation**: 80-120ms

### Memory Capacity
- **Short-term Memory**: 1,000 items
- **Working Memory**: 100 items
- **Long-term Memory**: 1,000,000 items
- **Episodic Memory**: 100,000 items

### Concurrent Processing
- **Brain Coordination**: 3 parallel brains
- **Memory Operations**: 4 concurrent memory systems
- **Personality Consensus**: 11 parallel personalities

## Error Handling

### Graceful Degradation
- Individual brain failures don't stop processing
- Memory system failures fall back to working memory
- Personality consensus continues with available personalities

### Recovery Mechanisms
- Automatic brain restart on failure
- Memory system reconstruction
- Personality weight rebalancing

### Monitoring
- Real-time performance metrics
- Error rate tracking
- Resource utilization monitoring

---

**Created by Jason Van Pham | Niodoo Framework | 2025**