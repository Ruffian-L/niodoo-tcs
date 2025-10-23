# ✅ Phase 2 Priorities (REVISED)

## Critical Understanding

**Storage feeds into retrieval feeds into generation feeds into storage**

This is a CLOSED LOOP. Without quality gates, the system degrades over time.

## Phase 2 Must Include BOTH

### 1. TCS Topology Layer ⭐ (Week 1 Day 3-4)
**Purpose**: Engine computation
- Compute persistent homology
- Extract knot invariants  
- Compute TQFT signatures
- Store topological features with memories

**Why critical**: This IS the engine - without it, there's no topology analysis.

### 2. Curator Quality Gate ⭐ (Week 1 Day 1-2)
**Purpose**: Prevent feedback loop degradation
- Analyze response quality before storage
- Reject low-quality memories
- Ensure only good memories feed back into system

**Why critical**: Without curator, low-quality memories poison the system over time.

## Revised Integration Order

### Day 1-2: Curator FIRST (Quality Gate)
1. Copy curator from `curator_executor`
2. Integrate into pipeline before `upsert_memory()`
3. Add quality threshold (e.g., 0.7)
4. Replace `.ok()` with proper error handling

**Code Location**: Line 258-273 in `pipeline.rs`

**Before**:
```rust
self.erag.upsert_memory(...).await.ok(); // ❌ No quality check
```

**After**:
```rust
// Curator quality check
let curated = self.curator.analyze(&generation.hybrid_response).await?;

if curated.quality_score >= 0.7 {
    self.erag.upsert_memory(...).await?; // ✅ Only store quality
} else {
    warn!("Low quality memory rejected: {}", curated.quality_score);
}
```

### Day 3-4: TCS Topology Layer (Engine)
1. Create `tcs_analysis.rs` module
2. Compute topology on every state
3. Store topological signatures with memories
4. Integrate into learning loop

**Code Location**: After torus projection (around line 180)

```rust
// After Stage 2: Torus
let pad_state = self.torus.project(&embedding).await?;

// ✨ TCS Topology Analysis ✨
let topology = self.tcs_analyzer.analyze_state(&pad_state).await?;
info!("Topological signature: {:?}", topology);

// Pass topology forward
let compass = self.compass.compute(&pad_state, &topology).await?;
```

### Day 5: Visualization
Connect viz to show both quality metrics and topology

## Why This Order Matters

**If we do TCS first, curator later**:
- Topology computed ✅
- But low-quality memories still stored ❌
- System degrades over time ❌

**If we do curator first, TCS later**:
- Quality gate prevents degradation ✅
- Then topology adds engine features ✅
- System improves over time ✅

## Success Criteria REVISED

Phase 2 complete when:
1. ✅ Curator quality gate working (rejects low-quality)
2. ✅ TCS topology layer computing signatures
3. ✅ Topological features stored with memories
4. ✅ No memory degradation over 100 cycles
5. ✅ Visualization showing both quality and topology

## Key Insight

**Curator ≠ Nice-to-have. Curator = Feedback loop protection.**

Without curator, your beautiful topology engine will be analyzing garbage that accumulates over time.

