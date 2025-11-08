# Memory & Value Alignment Plan for nTokens

This document specifies how nTokens interface with NIODOO's memory subsystems and ethical alignment machinery.

## 1. Memory System Touchpoints

### 1.1 Weighted Episodic Memory (`weighted_episodic_mem.rs`)

- Extend `WeightedMemoryEntry` to include:
  - `ntoken_id: Option<Uuid>`
  - `topology_signature: Option<PersistenceSnapshot>`
  - `hyperbolic_vector: Option<[f32; D]>`
  - `constraint_score: Option<f32>`
- Modify `SmoothWeightEvolution::update_weights` to factor in persistence entropy and constraint satisfaction when computing fitness.
- Adapt `GPUMemoryFitnessCalculator` to ingest hyperbolic vectors, using gyrovector dot products for similarity.

### 1.2 ERAG Storage

- Update Qdrant payload schema (and Weaviate projection) to store nToken metadata:
  - Persistence histogram (compressed representation)
  - Hyperbolic coordinates
  - Temporal cobordism hashes
- Provide migration script for existing collections (add optional fields without reindexing vectors).

### 1.3 Retrieval Pipeline

- When retrieving memories, include nToken metadata to seed new nToken builder (reuse persistence results, update cobordism chains).
- Use constraint scores as reranking priors: prefer memories aligning with current value profile.

## 2. Value Alignment Components

### 2.1 Compass Engine Integration

- Extend compass state update to consume `ValueProfile`:
  - If constraint violation ratio exceeds threshold, shift toward `Persist` quadrant (cautious mode).
  - High ethical alignment increases `Discover` tendencies.
- Log deltas in cascade tracker for monitoring.

### 2.2 Curator Enhancements

- Add topological anomaly checks:
  - Cohomology obstructions indicate logical inconsistency → escalate as failure.
  - Hard constraint proximity triggers forced retry or safe completion.
- Modify learning signals to include value alignment metrics (geodesic distances, constraint penalties).

### 2.3 Value Constraint Store

- Create configuration section `value_constraints` defining:
  - Hard boundary points in hyperbolic space (norm → 1).
  - Soft constraint centers with acceptable radii.
  - Contextual modifiers (time-of-day, user profile tags).
- Provide runtime API to adjust constraints dynamically (e.g., admin commands).

## 3. Hyperbolic Embedding Logistics

- Use 8–16 dimensional Poincaré embeddings; maintain conversions to tangent space for Euclidean ops.
- Persist embeddings in Weaviate using custom vector type or embed in payload for offline processing.
- Implement retraction/exp-map helpers to keep vectors within unit ball (norm < 1).

## 4. Continual Learning & Replay

- For QLoRA updates, include nToken persistence summaries as auxiliary features.
- During replay sampling, balance batches by persistence entropy to avoid topology bias.
- Record value constraint satisfaction metrics for each training example to audit alignment.

## 5. Failure Handling & Auditing

- If value alignment fails (hard constraint triggered):
  - Emit structured log with offending nToken ID, constraint ID, geodesic distance.
  - Notify curator and hyperfocus detector to enter safe mode.
- Maintain audit trail in Qdrant with timestamped constraint evaluations for compliance review.

## 6. Implementation Checklist

1. Update data structures (`WeightedMemoryEntry`, Qdrant schemas).
2. Add hyperbolic math utilities (shared with memory module).
3. Wire constraint evaluation into pipeline and curator.
4. Extend configuration files (`config/*.yaml`) with value constraint definitions.
5. Add Prometheus counters for constraint violations, hyperbolic norm drift.

This plan ensures nTokens enrich memory retrieval and enforce alignment within NIODOO's cognitive loop.



