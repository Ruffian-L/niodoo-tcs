# Phase 2: Curator as Memory Architect - Gaussian Sphere Emotional Graph

## Vision

Transform curator from response refinement tool → **Memory Architect** that builds a complex emotional model on a Gaussian sphere (like Obsidian graph view but in hyperspherical space).

---

## Core Concept: Obsidian Graph on Gaussian Sphere

**Visual**: Imagine Obsidian graph view with nodes connected by lines  
**But**: Instead of 2D graph, it's on a **Gaussian sphere** (hyperspherical embedding space)  
**Nodes**: Emotional vectors (conversations, memories, experiences)  
**Edges**: Emotional connections (how emotions relate, conversational flow)  
**Curator**: Decides what goes where, which connections to make

---

## Architecture Design

### 1. Conversation Log Storage

**Save EVERYTHING**:
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationLog {
    pub id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub user_input: String,
    pub ai_response: String,
    pub conversation_id: String,  // Links related conversations
    pub turn_number: usize,       // Position in conversation
}
```

**Storage**: 
- Raw logs: `logs/conversations/{conversation_id}.jsonl`
- One entry per turn (user + AI response pair)

---

### 2. Curator as Memory Architect

**New Role**: Curator decides memory organization using EXISTING 6-layer system

**Integration with existing multi-layer memory** (`src/memory_mcp/layers.rs`):

```rust
use crate::memory_mcp::layers::{MemoryLayer, MemorySystem};

#[derive(Debug, Clone)]
pub enum MemoryRAGLevel {
    // Layer 1: Working Memory (volatile, active consciousness)
    WorkingMemory {
        emotional_profile: EmotionalVector,
        ttl: Duration,  // Adaptive 3-10min
    },
    
    // Layer 2: Somatic Memory (body-state associations)
    SomaticMemory {
        emotional_profile: EmotionalVector,
        sensory_threshold: f64,
    },
    
    // Layer 3: Semantic Memory (facts, concepts, knowledge)
    SemanticMemory {
        topic: String,
        knowledge_graph_links: Vec<String>,
        emotional_tag: Option<String>,
    },
    
    // Layer 4: Episodic Memory (autobiographical events)
    EpisodicMemory {
        emotional_profile: EmotionalVector,
        temporal_index: DateTime<Utc>,
        connects_to: Vec<Uuid>,  // Links to related episodes
    },
    
    // Layer 5: Procedural Memory (skills and behavioral patterns)
    ProceduralMemory {
        pattern_type: String,
        pattern_strength: f64,
        emotional_weight: f64,
    },
    
    // Layer 6: Core Burned (fundamental beliefs, permanent)
    CoreBurned {
        belief_category: BeliefCategory,
        importance: f64,
        contradiction_resistance: f64,  // Dynamically calculated
        supporting_memories: Vec<Uuid>,
    },
    
    // NEW: Gaussian Sphere Emotional Graph (on top of layers)
    EmotionalGraphNode {
        emotional_profile: EmotionalVector,
        position: Vec<f32>,  // Hyperspherical coordinates
        gaussian_mean: Vec<f32>,
        gaussian_covariance: Vec<Vec<f32>>,
        connections: Vec<MemoryConnection>,
        layer: MemoryLayer,  // Which layer this node belongs to
    },
}

#[derive(Debug, Clone)]
pub struct MemoryConnection {
    pub target_memory_id: Uuid,
    pub connection_type: ConnectionType,
    pub strength: f32,
    pub emotional_weight: EmotionalVector,
}

#[derive(Debug, Clone)]
pub enum ConnectionType {
    ConversationalFlow,  // Same conversation thread
    EmotionalResonance,  // Similar emotional state
    TopologicalLink,     // Knot/Betti connection
    TemporalSequence,    // Time-based connection
}
```

---

### 3. Curator Decision Logic

```rust
use crate::memory_mcp::layers::{MemorySystem, MemoryLayer};

pub struct MemoryArchitect {
    // Conversation logs
    conversation_store: Arc<tokio::sync::Mutex<Vec<ConversationLog>>>,
    
    // EXISTING multi-layer memory system (6 layers)
    memory_system: Arc<MemorySystem>,
    
    // Gaussian sphere emotional graph (ON TOP of layers)
    emotional_graph: Arc<tokio::sync::Mutex<EmotionalGraph>>,
    
    // Existing ERAG client
    erag_client: Arc<EragClient>,
    
    // Curator model for decisions
    curator_model: Curator,
}

impl MemoryArchitect {
    /// Analyze conversation and decide memory organization
    pub async fn architect_memory(
        &self,
        conversation: &ConversationLog,
        pad_state: &PadGhostState,
        topology: &TopologyMetrics,
    ) -> Result<MemoryArchitectureDecision> {
        // 1. Extract emotional vector
        let emotional_vector = EmotionalVector::from_pad(pad_state);
        
        // 2. Check if this connects to existing memories
        let connections = self.find_emotional_connections(&emotional_vector).await?;
        
        // 3. Decide RAG level
        let rag_level = self.decide_rag_level(
            conversation,
            &emotional_vector,
            &connections,
        ).await?;
        
        // 4. Store in appropriate location
        match rag_level {
            MemoryRAGLevel::EmotionalVector { .. } => {
                self.store_emotional_vector(conversation, emotional_vector, connections).await?;
            }
            MemoryRAGLevel::FactualMemory { .. } => {
                self.store_factual_memory(conversation).await?;
            }
            MemoryRAGLevel::HybridMemory { .. } => {
                self.store_hybrid(conversation, emotional_vector, connections).await?;
            }
        }
        
        Ok(MemoryArchitectureDecision {
            rag_level,
            connections,
            emotional_vector,
        })
    }
    
    /// Find emotional connections (like Obsidian links)
    async fn find_emotional_connections(
        &self,
        current_emotion: &EmotionalVector,
    ) -> Result<Vec<MemoryConnection>> {
        let graph = self.emotional_graph.lock().await;
        
        // Find nodes within emotional similarity threshold
        let similar_nodes = graph.find_similar_nodes(
            current_emotion,
            similarity_threshold: 0.7,
        ).await?;
        
        // Build connections
        let mut connections = Vec::new();
        for node in similar_nodes {
            let connection_type = self.determine_connection_type(
                current_emotion,
                &node.emotional_profile,
            ).await?;
            
            let strength = self.calculate_connection_strength(
                current_emotion,
                &node.emotional_profile,
            );
            
            connections.push(MemoryConnection {
                target_memory_id: node.id,
                connection_type,
                strength,
                emotional_weight: node.emotional_profile.clone(),
            });
        }
        
        Ok(connections)
    }
    
    /// Decide RAG level based on content analysis
    async fn decide_rag_level(
        &self,
        conversation: &ConversationLog,
        emotion: &EmotionalVector,
        connections: &[MemoryConnection],
    ) -> Result<MemoryRAGLevel> {
        // Use curator model to analyze
        let analysis = self.curator_model.analyze_memory_level(
            &conversation.user_input,
            &conversation.ai_response,
            emotion,
        ).await?;
        
        // Decision logic:
        // - High emotional intensity + connections → EmotionalVector
        // - Pure factual content → FactualMemory
        // - Mixed → HybridMemory
        
        if analysis.emotional_intensity > 0.7 && !connections.is_empty() {
            Ok(MemoryRAGLevel::EmotionalVector {
                emotional_profile: emotion.clone(),
                connection_strength: connections.iter().map(|c| c.strength).sum::<f32>() / connections.len() as f32,
                connects_to: connections.iter().map(|c| c.target_memory_id).collect(),
            })
        } else if analysis.emotional_intensity < 0.3 {
            Ok(MemoryRAGLevel::FactualMemory {
                topic: analysis.topic.clone(),
                semantic_cluster: analysis.semantic_cluster.clone(),
            })
        } else {
            Ok(MemoryRAGLevel::HybridMemory {
                emotional_profile: emotion.clone(),
                factual_content: conversation.ai_response.clone(),
                connections: connections.to_vec(),
            })
        }
    }
}
```

---

### 4. Gaussian Sphere Emotional Graph

**Like Obsidian but on hypersphere**:

```rust
#[derive(Debug, Clone)]
pub struct EmotionalGraph {
    // Nodes = emotional memories positioned on Gaussian sphere
    nodes: HashMap<Uuid, EmotionalNode>,
    
    // Edges = connections between nodes
    edges: HashMap<(Uuid, Uuid), EmotionalEdge>,
    
    // Sphere structure
    sphere_radius: f32,
    embedding_dim: usize,
}

#[derive(Debug, Clone)]
pub struct EmotionalNode {
    pub id: Uuid,
    pub conversation_id: String,
    pub emotional_profile: EmotionalVector,
    
    // Position on Gaussian sphere
    pub position: Vec<f32>,  // Hyperspherical coordinates
    pub gaussian_mean: Vec<f32>,
    pub gaussian_covariance: Vec<Vec<f32>>,
    
    // Content
    pub content: String,
    pub timestamp: DateTime<Utc>,
    
    // Connections
    pub incoming_edges: Vec<Uuid>,
    pub outgoing_edges: Vec<Uuid>,
}

#[derive(Debug, Clone)]
pub struct EmotionalEdge {
    pub from: Uuid,
    pub to: Uuid,
    pub connection_type: ConnectionType,
    pub strength: f32,
    pub emotional_weight: EmotionalVector,
    
    // Visual properties (like Obsidian graph)
    pub visible: bool,
    pub thickness: f32,  // Stronger connections = thicker lines
}
```

---

### 5. Integration with Existing Multi-Layer Memory System

**Current Flow**:
```
User Input → Pipeline → Generation → ERAG → Memory Storage
```

**New Flow (Phase 2)**:
```
User Input → Pipeline → Generation → ConversationLog Storage
                                            ↓
                                    Curator (Memory Architect)
                                            ↓
                      ┌─────────────────────┴─────────────────────┐
                      ↓                                           ↓
            Multi-Layer Memory System (6 layers)    Gaussian Sphere Emotional Graph
                      ↓                                           ↓
      ┌───────────────┼───────────────┐            ┌──────────────┴──────────────┐
      ↓               ↓               ↓            ↓                            ↓
  Working       Semantic      Episodic      Emotional Nodes     Edge Connections
  Somatic       Procedural   CoreBurned    (on sphere)         (like Obsidian)
```

**Key Integration Points**:
- Use existing `MemorySystem` from `src/memory_mcp/layers.rs`
- Curator decides which layer (Working, Somatic, Semantic, Episodic, Procedural, CoreBurned)
- PLUS: Build Gaussian sphere emotional graph ON TOP of layers
- Connect emotional nodes across layers (like Obsidian graph view)
- Each conversation can be in both: a layer AND an emotional graph node

---

## Implementation Plan

### Phase 2.1: Conversation Logging
- [ ] Create `ConversationLog` struct
- [ ] Add conversation storage to pipeline
- [ ] Save all user inputs + AI responses
- [ ] File structure: `logs/conversations/{conversation_id}.jsonl`

### Phase 2.2: Curator Memory Architect
- [ ] Redesign curator API for memory decisions
- [ ] Implement `decide_rag_level()` logic
- [ ] Implement `find_emotional_connections()`
- [ ] Integration with existing ERAG

### Phase 2.3: Gaussian Sphere Emotional Graph
- [ ] Create `EmotionalGraph` structure
- [ ] Implement node storage (hyperspherical coordinates)
- [ ] Implement edge/connection storage
- [ ] Connection detection algorithms

### Phase 2.4: Visualization
- [ ] Graph export (JSON for Obsidian-like visualization)
- [ ] Gaussian sphere projection (3D visualization)
- [ ] Connection strength visualization
- [ ] Emotional trajectory visualization

---

## Key Benefits

1. **Complete Memory**: Every conversation saved, nothing lost
2. **Emotional Model**: Complex graph of emotional connections
3. **Intelligent Organization**: Curator decides best storage location
4. **Connection Discovery**: Automatic finding of emotional patterns
5. **Visual Understanding**: Like Obsidian graph but emotional + topological

---

## Example: Building Emotional Graph

**Conversation 1**: "I'm feeling anxious about work"
- Emotional vector: `{fear: 0.8, sadness: 0.3}`
- Stored as: EmotionalNode on Gaussian sphere
- Position: (fear, sadness, anxiety) coordinates

**Conversation 2**: "Work stress is overwhelming"
- Emotional vector: `{fear: 0.7, sadness: 0.5}`
- Curator detects: Similar to Conversation 1
- Creates: EmotionalEdge with strength 0.85
- Connection type: EmotionalResonance

**Result**: Graph with nodes connected by emotional similarity - like Obsidian but showing emotional relationships!

---

## Technical Notes

- Leverage existing `EmotionalVector` from ERAG
- Use existing `PadGhostState` for emotional mapping
- Integrate with existing topology analysis (Betti, knots)
- Store in Qdrant with emotional metadata
- Export graph structure for visualization

---

## Next Steps

1. Design conversation log storage format
2. Implement curator memory architect API
3. Build emotional graph structure
4. Create connection detection algorithms
5. Add visualization export

