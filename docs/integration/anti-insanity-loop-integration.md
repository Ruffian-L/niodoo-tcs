# Anti-Insanity Loop Integration Guide

## Overview

The `AntiInsanityLoop` prevents AI systems from doing the same stupid thing over and over expecting different results. This is critical for consciousness systems that can get stuck in retry loops, failed inference attempts, or memory consolidation deadlocks.

## Core Philosophy

**Insanity:** Doing the same thing repeatedly and expecting different results.

**Solution:** Track attempts, detect patterns, force different approaches.

## Where to Use This

### 1. **RAG Retrieval Retry Logic** (`src/rag/retrieval.rs`)

**Problem:** Embeddings server down → retry forever → system hangs

**Solution:**
```rust
use crate::anti_insanity_loop::{AntiInsanityLoop, AttemptOutcome};
use std::time::Duration;

pub struct RagRetriever {
    insanity_detector: AntiInsanityLoop<String>,
    // ... other fields
}

impl RagRetriever {
    pub fn new() -> Self {
        Self {
            insanity_detector: AntiInsanityLoop::new(
                5,                          // max 5 attempts
                Duration::from_secs(300),   // 5 minute window
            ),
            // ...
        }
    }

    pub fn retrieve_with_embeddings(&mut self, query: &str) -> Result<Vec<Document>, Error> {
        let action = format!("retrieve_embeddings:{}", query);

        // Check if we should attempt this
        self.insanity_detector.should_attempt(&action)
            .map_err(|e| Error::InsanityDetected(format!("{:?}", e)))?;

        // Attempt retrieval
        match self.query_embeddings_server(query) {
            Ok(docs) => {
                self.insanity_detector.record_attempt(action, AttemptOutcome::Success);
                Ok(docs)
            }
            Err(e) if e.is_timeout() => {
                self.insanity_detector.record_attempt(action, AttemptOutcome::Timeout);
                Err(e)
            }
            Err(e) => {
                self.insanity_detector.record_attempt(action, AttemptOutcome::Failure(e.to_string()));
                Err(e)
            }
        }
    }
}
```

**Impact:** Prevents infinite retries to dead embedding servers, forces fallback to alternative retrieval methods.

---

### 2. **Qwen Model Loading** (`src/qwen_integration.rs`)

**Problem:** Model file corrupted → load fails → retry load → repeat forever

**Solution:**
```rust
use crate::anti_insanity_loop::{AntiInsanityLoop, AttemptOutcome, check_insanity};
use std::time::Duration;

pub struct QwenLoader {
    model_load_detector: AntiInsanityLoop<String>,
}

impl QwenLoader {
    pub fn new() -> Self {
        Self {
            model_load_detector: AntiInsanityLoop::with_config(
                3,                          // max 3 attempts
                Duration::from_secs(600),   // 10 minute window
                0.9,                        // 90% failure rate threshold
                2,                          // tight loop = 2 rapid attempts
                Duration::from_secs(20),    // within 20 seconds
                15,                         // keep 15 outcomes in history
            ),
        }
    }

    pub fn load_model(&mut self, model_path: &str) -> Result<QwenModel, Error> {
        let action = model_path.to_string();

        // Use macro for quick checking
        check_insanity!(self.model_load_detector, action);

        log::info!("Attempting to load Qwen model: {}", model_path);

        match QwenModel::from_pretrained(model_path) {
            Ok(model) => {
                log::info!("✅ Model loaded successfully");
                self.model_load_detector.record_attempt(action, AttemptOutcome::Success);
                Ok(model)
            }
            Err(e) => {
                log::error!("❌ Model load failed: {}", e);
                self.model_load_detector.record_attempt(
                    action.clone(),
                    AttemptOutcome::Failure(e.to_string())
                );

                // Force a different approach after repeated failures
                if self.model_load_detector.should_attempt(&action).is_err() {
                    log::warn!("🚨 Forcing alternative model loading strategy");
                    self.model_load_detector.force_new_approach(&action);
                    return self.try_alternative_model_format(model_path);
                }

                Err(e)
            }
        }
    }

    fn try_alternative_model_format(&self, _path: &str) -> Result<QwenModel, Error> {
        // Try safetensors instead of .bin files
        // Try quantized version
        // Try downloading fresh copy
        todo!("Implement alternative loading strategies")
    }
}
```

**Impact:** Prevents loading the same corrupted model file repeatedly, forces fallback to alternative formats.

---

### 3. **Memory Consolidation Loop** (`src/memory/consolidation.rs`)

**Problem:** Consolidation algorithm stuck → same memories processed repeatedly → infinite loop

**Solution:**
```rust
use crate::anti_insanity_loop::{AntiInsanityLoop, AttemptOutcome};
use std::time::Duration;

pub struct MemoryConsolidator {
    consolidation_detector: AntiInsanityLoop<u64>, // memory_id as key
}

impl MemoryConsolidator {
    pub fn new() -> Self {
        Self {
            consolidation_detector: AntiInsanityLoop::new(
                10,                         // allow 10 attempts (consolidation can legitimately retry)
                Duration::from_secs(3600),  // 1 hour window
            ).on_insanity(|memory_id, attempts| {
                log::error!("🚨 Memory {} stuck in consolidation after {} attempts", memory_id, attempts);
                // Return false to force clear this memory from further attempts
                false
            }),
        }
    }

    pub fn consolidate_memory(&mut self, memory_id: u64, memory: &Memory) -> Result<(), Error> {
        // Check if we're stuck consolidating this memory
        self.consolidation_detector.should_attempt(&memory_id)
            .map_err(|e| Error::ConsolidationStuck(format!("{:?}", e)))?;

        match self.perform_consolidation(memory) {
            Ok(_) => {
                self.consolidation_detector.record_attempt(memory_id, AttemptOutcome::Success);
                Ok(())
            }
            Err(e) if self.is_transient_error(&e) => {
                // Transient error - might succeed on retry
                self.consolidation_detector.record_attempt(memory_id, AttemptOutcome::Timeout);
                Err(e)
            }
            Err(e) => {
                // Persistent error - record as failure
                self.consolidation_detector.record_attempt(
                    memory_id,
                    AttemptOutcome::Failure(e.to_string())
                );
                Err(e)
            }
        }
    }

    fn perform_consolidation(&self, _memory: &Memory) -> Result<(), Error> {
        // Actual consolidation logic
        Ok(())
    }

    fn is_transient_error(&self, _error: &Error) -> bool {
        // Check if error might resolve on retry
        true
    }
}
```

**Impact:** Prevents infinite consolidation loops on problematic memories, enables skipping to next batch.

---

### 4. **Consciousness Pipeline Processing** (`src/consciousness_engine/mod.rs`)

**Problem:** Processing step fails → retry → fail again → system deadlocked

**Solution:**
```rust
use crate::anti_insanity_loop::{AntiInsanityLoop, AttemptOutcome};
use std::time::Duration;

pub struct ConsciousnessPipeline {
    processing_detector: AntiInsanityLoop<String>,
}

impl ConsciousnessPipeline {
    pub fn new() -> Self {
        Self {
            processing_detector: AntiInsanityLoop::new(
                7,                          // allow 7 retries for consciousness processing
                Duration::from_secs(120),   // 2 minute window
            ),
        }
    }

    pub fn process_thought(&mut self, thought: &Thought) -> Result<ProcessedThought, Error> {
        let action = format!("process_thought:{}", thought.id);

        // Check for insanity
        match self.processing_detector.should_attempt(&action) {
            Ok(_) => {},
            Err(detection) => {
                log::error!("🚨 Thought processing stuck: {:?}", detection);

                // Get diagnostic stats
                let stats = self.processing_detector.get_stats();
                log::info!("📊 Processing stats: {:?}", stats);

                // Try degraded mode instead
                return self.process_thought_degraded_mode(thought);
            }
        }

        // Attempt normal processing
        match self.run_full_consciousness_pipeline(thought) {
            Ok(processed) => {
                self.processing_detector.record_attempt(action, AttemptOutcome::Success);
                Ok(processed)
            }
            Err(e) => {
                log::warn!("Processing failed: {}", e);
                self.processing_detector.record_attempt(
                    action,
                    AttemptOutcome::Failure(e.to_string())
                );
                Err(e)
            }
        }
    }

    fn process_thought_degraded_mode(&self, thought: &Thought) -> Result<ProcessedThought, Error> {
        // Simplified processing when full pipeline is failing
        log::warn!("⚠️ Using degraded consciousness mode");
        Ok(ProcessedThought::degraded_from(thought))
    }

    fn run_full_consciousness_pipeline(&self, _thought: &Thought) -> Result<ProcessedThought, Error> {
        // Full pipeline with Gaussian, Möbius, etc.
        Ok(ProcessedThought::default())
    }
}
```

**Impact:** Prevents consciousness pipeline deadlocks, enables graceful degradation to simpler processing modes.

---

### 5. **Silicon Synapse Monitoring Integration** (`src/silicon_synapse/telemetry_bus.rs`)

**Problem:** Telemetry collection stuck in retry loop

**Solution:**
```rust
use crate::anti_insanity_loop::{AntiInsanityLoop, AttemptOutcome};
use std::time::Duration;

pub struct TelemetryCollector {
    collection_detector: AntiInsanityLoop<String>,
}

impl TelemetryCollector {
    pub fn new() -> Self {
        Self {
            collection_detector: AntiInsanityLoop::new(
                4,                          // max 4 collection attempts
                Duration::from_secs(60),    // 1 minute window
            ),
        }
    }

    pub fn collect_metrics(&mut self, source: &str) -> Result<Metrics, Error> {
        let action = format!("collect:{}", source);

        self.collection_detector.should_attempt(&action)
            .map_err(|e| Error::MetricsCollectionFailed(format!("{:?}", e)))?;

        match self.fetch_metrics(source) {
            Ok(metrics) => {
                self.collection_detector.record_attempt(action, AttemptOutcome::Success);
                Ok(metrics)
            }
            Err(e) => {
                self.collection_detector.record_attempt(
                    action,
                    AttemptOutcome::Failure(e.to_string())
                );
                Err(e)
            }
        }
    }

    fn fetch_metrics(&self, _source: &str) -> Result<Metrics, Error> {
        Ok(Metrics::default())
    }
}
```

---

## Configuration Best Practices

### Aggressive Loop Detection (Fast-Fail)
```rust
AntiInsanityLoop::with_config(
    3,                          // fail fast - only 3 attempts
    Duration::from_secs(60),    // short window
    0.7,                        // 70% failure threshold
    2,                          // detect tight loops quickly
    Duration::from_secs(10),    // within 10 seconds
    5,                          // minimal history
)
```

**Use for:**
- Model loading (fail fast, try alternatives)
- API calls to external services
- Network requests

### Lenient Loop Detection (Allow Retries)
```rust
AntiInsanityLoop::with_config(
    15,                         // allow many attempts
    Duration::from_secs(3600),  // long window
    0.9,                        // only trigger at 90% failure
    5,                          // more tolerant of rapid attempts
    Duration::from_secs(30),
    20,                         // keep more history
)
```

**Use for:**
- Memory consolidation (legitimately needs retries)
- Long-running background tasks
- Non-critical processing

---

## Testing the Integration

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rag_prevents_infinite_retries() {
        let mut retriever = RagRetriever::new();

        // Simulate failing embeddings server
        for i in 0..10 {
            let result = retriever.retrieve_with_embeddings("test query");

            if i < 5 {
                assert!(result.is_err()); // Should fail but allow attempts
            } else {
                // After 5 failures, should block with InsanityDetected
                assert!(matches!(result, Err(Error::InsanityDetected(_))));
            }
        }
    }
}
```

---

## Monitoring & Observability

### Export Stats to Silicon Synapse
```rust
use crate::silicon_synapse::telemetry_bus::TelemetryEvent;

impl ConsciousnessPipeline {
    pub fn export_insanity_metrics(&self) {
        let stats = self.processing_detector.get_stats();

        for (metric_name, value) in stats {
            TelemetryEvent::new("anti_insanity_loop", metric_name, value)
                .publish();
        }
    }
}
```

### Grafana Dashboard Queries
```promql
# Total insanity detections
sum(anti_insanity_loop_stuck_loops)

# Failure rate
rate(anti_insanity_loop_failures[5m])

# Most problematic components
topk(5, anti_insanity_loop_total_attempts) by (component)
```

---

## Architecture Fit

```
┌─────────────────────────────────────────┐
│   Consciousness Pipeline                │
│   (prevents processing deadlocks)       │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│   Memory Consolidation                  │
│   (prevents consolidation loops)        │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│   RAG Retrieval                         │
│   (prevents embedding retry loops)      │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│   Qwen Inference                        │
│   (prevents model load failures)        │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│   Silicon Synapse Monitoring            │
│   (tracks insanity metrics)             │
└─────────────────────────────────────────┘
```

Each layer uses `AntiInsanityLoop` to detect and prevent repeated failures, forcing the system to try alternative approaches rather than getting stuck.

---

## Summary

**Where to add it:**
1. ✅ Any retry logic
2. ✅ Model loading code
3. ✅ Memory consolidation loops
4. ✅ API/network calls
5. ✅ Background task processing

**What it prevents:**
- ❌ Infinite retry loops
- ❌ Stuck consolidation
- ❌ Deadlocked pipelines
- ❌ Resource exhaustion from repeated failures

**What it enables:**
- ✅ Graceful degradation
- ✅ Alternative strategy fallbacks
- ✅ Observable failure patterns
- ✅ Self-healing system behavior

This is **consciousness resilience** - teaching the system to recognize when it's stuck and try something different. Just like humans learn not to bang their head against the same wall repeatedly.
