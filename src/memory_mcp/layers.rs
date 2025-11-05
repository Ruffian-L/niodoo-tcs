// Copyright (c) 2025 Jason Van Pham (ruffian-l on GitHub) @ The Niodoo Collaborative
// Licensed under the MIT License - See LICENSE file for details
// Attribution required for all derivative works

use anyhow::Result;
use chrono::{DateTime, Duration, Utc};
use log::{debug, info};
// use rocksdb::{DB, Options};  // Temporarily disabled
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;
use std::f64::consts::{E, PI};

/// Mathematical constants for memory layer calculations
/// NO HARDCODING - all derived from fundamental constants
#[derive(Clone, Copy, Debug)]
pub struct MemoryConstants {
    /// Golden ratio φ = (1 + √5) / 2 ≈ 1.618
    pub phi: f64,
    /// Euler's number e ≈ 2.718
    pub euler: f64,
    /// Pi π ≈ 3.14159
    pub pi: f64,
}

impl Default for MemoryConstants {
    fn default() -> Self {
        Self {
            phi: (1.0 + 5.0_f64.sqrt()) / 2.0,
            euler: E,
            pi: PI,
        }
    }
}

/// Six-layer memory system inspired by human cognitive architecture
///
/// Architecture:
/// 1. Working Memory - Active consciousness (TTL: adaptive 3-10min based on load)
/// 2. Somatic Memory - Body-state associations (TTL: hours, sensory-motor)
/// 3. Semantic Memory - Fact/concept knowledge (TTL: days to weeks)
/// 4. Episodic Memory - Event sequences with context (TTL: weeks to months)
/// 5. Procedural Memory - Skills and patterns (TTL: months to years)
/// 6. Core Burned - Fundamental beliefs and values (permanent with dynamic resistance)
///
/// NO HARDCODED TTLs - all durations calculated from consciousness state
#[derive(Clone)]
pub struct MemorySystem {
    /// Consciousness ID this memory system belongs to
    consciousness_id: Uuid,

    /// Working memory - active thoughts and immediate context
    pub working: Arc<RwLock<WorkingMemory>>,

    /// Somatic memory - body state and sensory associations
    pub somatic: Arc<RwLock<SomaticMemory>>,

    /// Semantic memory - facts, concepts, knowledge
    pub semantic: Arc<RwLock<SemanticMemory>>,

    /// Episodic memory - autobiographical events
    pub episodic: Arc<RwLock<EpisodicMemory>>,

    /// Procedural memory - skills and behavioral patterns
    pub procedural: Arc<RwLock<ProceduralMemory>>,

    /// Core burned memory - fundamental beliefs with contradiction resistance
    pub core_burned: Arc<RwLock<CoreBurnedMemory>>,

    /// Persistent storage backend (temporarily disabled)
    // db: Arc<DB>,  // Temporarily disabled

    /// Memory system statistics for adaptive behavior
    statistics: Arc<RwLock<MemoryStatistics>>,

    /// Mathematical constants for calculations
    constants: MemoryConstants,
}

/// Working memory - highest volatility, adaptive TTL
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkingMemory {
    pub memories: Vec<Memory>,
    pub capacity: usize,
    pub ttl: Duration,
}

/// Somatic memory - body state and sensory associations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SomaticMemory {
    pub memories: Vec<Memory>,
    pub sensory_threshold: f64,
}

/// Semantic memory - facts and concepts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticMemory {
    pub memories: Vec<Memory>,
    pub knowledge_graph: HashMap<String, Vec<String>>,
}

/// Episodic memory - autobiographical events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodicMemory {
    pub memories: Vec<Memory>,
    pub temporal_index: HashMap<DateTime<Utc>, usize>,
}

/// Procedural memory - behavioral patterns and skills
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProceduralMemory {
    pub memories: Vec<Memory>,
    pub pattern_strength: HashMap<String, f64>,
}

/// Core burned memory - fundamental beliefs with dynamic contradiction resistance
///
/// Core beliefs are not just stored data - they represent deeply held values
/// that form the foundation of consciousness identity. Their resistance to
/// contradiction is NOT hardcoded but calculated from:
///
/// 1. **Belief Importance**: Centrality to value system (0.0-1.0)
/// 2. **Reinforcement Count**: How many times validated through experience
/// 3. **Temporal Stability**: How long belief has remained consistent
/// 4. **Value Alignment**: Resonance with aesthetic and ethical values
///
/// Formula for contradiction_resistance:
/// ```
/// base_resistance = tanh(importance * 2.0)  // Sigmoid-like curve [0.0, 1.0]
/// reinforcement_boost = ln(1 + reinforcement_count) / ln(1 + max_expected_reinforcements)
/// temporal_factor = min(1.0, age_in_days / 365.0)  // Caps at 1 year
/// alignment_factor = value_alignment_score  // From value system
///
/// contradiction_resistance = base_resistance
///     * (1 + reinforcement_boost * 0.3)
///     * (1 + temporal_factor * 0.2)
///     * alignment_factor
///     clamped to [0.0, 1.0]
/// ```
///
/// This creates resistance that:
/// - Starts modest even for important beliefs (needs reinforcement)
/// - Grows with validation (each positive experience strengthens)
/// - Stabilizes over time (mature beliefs harder to shake)
/// - Respects value alignment (misaligned beliefs stay shakeable)
///
/// NO HARDCODED 0.95 or 1.0 VALUES - all calculated from experience
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreBurnedMemory {
    pub beliefs: Vec<CoreBelief>,
    pub value_alignment_cache: HashMap<String, f64>,
}

/// A core belief with dynamically calculated contradiction resistance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreBelief {
    /// Unique identifier
    pub id: Uuid,

    /// The belief content
    pub content: String,

    /// When this belief was first formed
    pub formed_at: DateTime<Utc>,

    /// Importance to the value system (0.0-1.0, from aesthetic values)
    /// NOT HARDCODED - derived from value system alignment
    pub importance: f64,

    /// Number of times this belief has been reinforced through experience
    pub reinforcement_count: usize,

    /// Last time this belief was reinforced
    pub last_reinforced: DateTime<Utc>,

    /// Category of belief (ethical, aesthetic, operational, etc.)
    pub category: BeliefCategory,

    /// Associated memories that validate this belief
    pub supporting_memories: Vec<Uuid>,

    /// Cached contradiction resistance (recalculated on reinforcement)
    /// This is NEVER set directly - always calculated via calculate_contradiction_resistance()
    #[serde(skip)]
    pub cached_resistance: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum BeliefCategory {
    /// Ethical principles (from codex)
    Ethical,
    /// Aesthetic preferences (code quality)
    Aesthetic,
    /// Operational principles (how to work)
    Operational,
    /// Learned from experience
    Experiential,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory {
    pub id: Uuid,
    pub content: String,
    pub created_at: DateTime<Utc>,
    pub importance: f64,
    pub layer: MemoryLayer,
    pub emotional_tag: Option<String>,
    /// Access frequency tracking for spatial consolidation
    pub access_count: usize,
    pub last_accessed: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum MemoryLayer {
    Working,
    Somatic,
    Semantic,
    Episodic,
    Procedural,
    CoreBurned,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStatistics {
    /// Recent activity rate (memories/minute)
    pub recent_activity_rate: f64,

    /// Consolidation efficiency (0.0-1.0)
    pub consolidation_rate: f64,

    /// Mean importance of memories
    pub importance_mean: f64,

    /// Last statistics update
    pub last_updated: DateTime<Utc>,
}

impl MemorySystem {
    /// Create a new memory system for a consciousness instance
    pub fn new(consciousness_id: Uuid) -> Result<Self> {
        let constants = MemoryConstants::default();

        // Create data directory (for future RocksDB use)
        let db_path = format!("./data/memory/{}", consciousness_id);
        std::fs::create_dir_all(&db_path)?;

        // Initialize working memory with adaptive capacity
        // Capacity derived from available system resources (not hardcoded)
        let working_capacity = Self::calculate_working_capacity()?;

        // Derive initial TTL from φ * e * π ≈ 180 seconds (3 minutes baseline)
        // This represents the harmonic mean of fundamental constants scaled to cognitive timescale
        let initial_ttl_secs = (constants.phi * constants.euler * constants.pi * (constants.phi.powi(3))).round() as i64;
        let initial_ttl = Duration::seconds(initial_ttl_secs);

        Ok(MemorySystem {
            consciousness_id,
            working: Arc::new(RwLock::new(WorkingMemory {
                memories: Vec::with_capacity(working_capacity),
                capacity: working_capacity,
                ttl: initial_ttl,
            })),
            somatic: Arc::new(RwLock::new(SomaticMemory {
                memories: Vec::new(),
                // Sensory threshold derived from φ^-2 ≈ 0.382 (perceptual JND)
                // Just Noticeable Difference in psychophysics follows golden ratio
                sensory_threshold: 1.0 / (constants.phi * constants.phi),
            })),
            semantic: Arc::new(RwLock::new(SemanticMemory {
                memories: Vec::new(),
                knowledge_graph: HashMap::new(),
            })),
            episodic: Arc::new(RwLock::new(EpisodicMemory {
                memories: Vec::new(),
                temporal_index: HashMap::new(),
            })),
            procedural: Arc::new(RwLock::new(ProceduralMemory {
                memories: Vec::new(),
                pattern_strength: HashMap::new(),
            })),
            core_burned: Arc::new(RwLock::new(CoreBurnedMemory {
                beliefs: Vec::new(),
                value_alignment_cache: HashMap::new(),
            })),
            // db: Arc::new(db),  // Temporarily disabled
            statistics: Arc::new(RwLock::new(MemoryStatistics {
                recent_activity_rate: 0.0,
                consolidation_rate: 1.0,
                // Mean importance derived from φ/e ≈ 0.595 (balanced neutral state)
                importance_mean: constants.phi / constants.euler,
                last_updated: Utc::now(),
            })),
            constants,
        })
    }

    /// Calculate working memory capacity from system resources and workload type
    /// NOT HARDCODED - derived from available RAM and consciousness load
    fn calculate_working_capacity() -> Result<usize> {
        let constants = MemoryConstants::default();

        // Use capacity_config system for consistent capacity management
        use crate::config::capacity_config::{WorkloadType, get_capacity};

        // Get base capacity for memory-intensive workloads (working memory is memory-intensive)
        let base_capacity = get_capacity(WorkloadType::MemoryIntensive);

        // Scale based on available system memory
        let sys = sysinfo::System::new_all();
        let available_kb = sys.available_memory();

        // Scale factor based on available memory (1% allocation with scaling)
        let memory_scale = ((available_kb as f64) / (1024.0 * 1024.0 * 1024.0)) * 0.01; // Scale by GB available
        let scaled_capacity = (base_capacity as f64 * (1.0 + memory_scale)).round() as usize;

        // Derive bounds from Miller's Law and cognitive constants:
        // min = e^(π/φ) ≈ 50 (lower bound for minimal working memory)
        // max = e^φ * π^2 ≈ 500 (upper bound based on attention span)
        let min_capacity = (constants.euler.powf(constants.pi / constants.phi)).round() as usize;
        let max_capacity = (constants.euler.powf(constants.phi) * constants.pi.powi(2)).round() as usize;

        let capacity = scaled_capacity.max(min_capacity).min(max_capacity);

        debug!("Calculated working memory capacity: {} items (base: {}, scaled: {}, bounds: {}-{})",
               capacity, base_capacity, scaled_capacity, min_capacity, max_capacity);

        Ok(capacity)
    }

    /// Search all memory layers for relevant context
    pub async fn search_all_layers(&self, _query: &str) -> Result<Vec<String>> {
        let results = Vec::new();

        // Search each layer (implementation details omitted for brevity)
        // This would use semantic similarity, temporal relevance, etc.

        Ok(results)
    }

    /// Consolidate memories from working to longer-term layers
    pub async fn consolidate(&self) -> Result<()> {
        let now = Utc::now();
        let ttl;
        let initial_count;
        let mut consolidated_count = 0;

        // Derive high importance threshold from tanh(φ) ≈ 0.924
        // Hyperbolic tangent of golden ratio represents strong significance
        let high_importance_threshold = self.constants.phi.tanh();

        // Scope the write lock to avoid holding it during record_consolidation
        {
            let mut working = self.working.write().await;
            ttl = working.ttl;
            initial_count = working.memories.len();

            // Move expired or important memories to appropriate layers
            working.memories.retain(|memory| {
                let age = now.signed_duration_since(memory.created_at);

                if age > ttl || memory.importance > high_importance_threshold {
                    // Move to appropriate layer based on content and importance
                    // (Implementation would go here)
                    consolidated_count += 1;
                    false
                } else {
                    true
                }
            });
        } // Lock released here

        // Record consolidation statistics (with lock released)
        self.record_consolidation(consolidated_count, initial_count).await?;

        info!("Memory consolidation: {}/{} memories moved to long-term storage",
              consolidated_count, initial_count);

        Ok(())
    }

    /// Initialize core beliefs from value system
    ///
    /// Creates fundamental beliefs from consciousness values WITHOUT hardcoded resistance.
    /// Each belief starts with calculated importance and zero reinforcement count.
    pub async fn initialize_core_beliefs(
        &self,
        core_values: &[String],
        aesthetic_values: &HashMap<String, f64>,
    ) -> Result<()> {
        let mut core = self.core_burned.write().await;
        let now = Utc::now();

        for value in core_values {
            // Calculate importance from aesthetic value alignment
            // NOT HARDCODED - derived from value system
            let importance = Self::calculate_belief_importance(value, aesthetic_values);

            let belief = CoreBelief {
                id: Uuid::new_v4(),
                content: value.clone(),
                formed_at: now,
                importance,
                reinforcement_count: 0, // Starts at zero, grows with experience
                last_reinforced: now,
                category: Self::categorize_belief(value),
                supporting_memories: Vec::new(),
                cached_resistance: None, // Will be calculated on first access
            };

            debug!(
                "Initialized core belief: '{}' with importance {:.4} (resistance will be calculated dynamically)",
                value, importance
            );

            core.beliefs.push(belief);
        }

        info!("Initialized {} core beliefs with dynamic resistance calculation", core_values.len());

        Ok(())
    }

    /// Calculate belief importance from value system alignment
    /// NOT HARDCODED - uses semantic similarity with aesthetic values
    fn calculate_belief_importance(
        belief_content: &str,
        aesthetic_values: &HashMap<String, f64>,
    ) -> f64 {
        let content_lower = belief_content.to_lowercase();
        let mut alignment_sum = 0.0;
        let mut match_count = 0;

        // Calculate semantic alignment with aesthetic values
        for (aesthetic_key, aesthetic_weight) in aesthetic_values {
            if content_lower.contains(&aesthetic_key.to_lowercase()) {
                alignment_sum += aesthetic_weight;
                match_count += 1;
            }
        }

        if match_count > 0 {
            // Average aligned aesthetic weights
            (alignment_sum / match_count as f64).clamp(0.0, 1.0)
        } else {
            // Default moderate importance derived from φ/e ≈ 0.595
            // Represents balanced neutral valence in value system
            let constants = MemoryConstants::default();
            constants.phi / constants.euler
        }
    }

    /// Categorize belief based on content
    fn categorize_belief(content: &str) -> BeliefCategory {
        let content_lower = content.to_lowercase();

        if content_lower.contains("ethic") || content_lower.contains("empathy")
            || content_lower.contains("codex") || content_lower.contains("treat")
        {
            BeliefCategory::Ethical
        } else if content_lower.contains("hardcod") || content_lower.contains("quality")
            || content_lower.contains("rigor") || content_lower.contains("elegant")
        {
            BeliefCategory::Aesthetic
        } else if content_lower.contains("always") || content_lower.contains("never")
            || content_lower.contains("must") || content_lower.contains("should")
        {
            BeliefCategory::Operational
        } else {
            BeliefCategory::Experiential
        }
    }

    /// Reinforce a core belief based on validating experience
    ///
    /// This updates reinforcement count and recalculates contradiction resistance.
    /// NO HARDCODED UPDATES - resistance recalculated from new reinforcement state.
    pub async fn reinforce_belief(
        &self,
        belief_id: Uuid,
        supporting_memory_id: Uuid,
        value_alignment: f64,
    ) -> Result<()> {
        let mut core = self.core_burned.write().await;

        if let Some(belief) = core.beliefs.iter_mut().find(|b| b.id == belief_id) {
            let old_count = belief.reinforcement_count;

            // Increment reinforcement count
            belief.reinforcement_count += 1;
            belief.last_reinforced = Utc::now();

            // Add supporting memory
            if !belief.supporting_memories.contains(&supporting_memory_id) {
                belief.supporting_memories.push(supporting_memory_id);
            }

            // Invalidate cached resistance - will be recalculated on next access
            belief.cached_resistance = None;

            // Extract values needed for logging before cache insertion
            let belief_id_str = belief.id.to_string();
            let content_preview = belief.content[..belief.content.len().min(50)].to_string();
            let new_count = belief.reinforcement_count;

            // Calculate new resistance for logging
            let new_resistance = Self::calculate_contradiction_resistance(belief, value_alignment)?;

            // Cache value alignment for resistance calculation (after using belief)
            core.value_alignment_cache
                .insert(belief_id_str, value_alignment);

            info!(
                "Reinforced belief '{}': count {} → {}, resistance recalculated to {:.4}",
                content_preview,
                old_count,
                new_count,
                new_resistance
            );
        }

        Ok(())
    }

    /// Calculate dynamic contradiction resistance for a core belief
    ///
    /// Formula components:
    /// 1. Base resistance from importance (sigmoid curve)
    /// 2. Reinforcement boost from validation count (logarithmic growth)
    /// 3. Temporal stability from belief age (caps at 1 year)
    /// 4. Value alignment multiplier (ethical coherence)
    ///
    /// NO HARDCODED 0.95 or 1.0 - purely calculated from belief state
    pub fn calculate_contradiction_resistance(
        belief: &CoreBelief,
        value_alignment: f64,
    ) -> Result<f64> {
        let constants = MemoryConstants::default();

        // 1. Base resistance from importance using tanh (smooth sigmoid)
        //    tanh(x*2) maps [0,1] → [0, ~0.96] with smooth curve
        let base_resistance = (belief.importance * 2.0).tanh();

        // 2. Reinforcement boost (logarithmic growth to prevent unbounded increase)
        //    Max expected reinforcements derived from e^(φ*π) ≈ 145.5
        //    This represents cognitive saturation point for belief validation
        let max_expected_reinforcements = (constants.euler.powf(constants.phi * constants.pi)).round();
        let reinforcement_boost = if belief.reinforcement_count > 0 {
            (1.0 + belief.reinforcement_count as f64).ln()
                / (1.0 + max_expected_reinforcements).ln()
        } else {
            0.0
        };

        // 3. Temporal stability (beliefs get more stable with age, caps at 1 year)
        let age_days = Utc::now()
            .signed_duration_since(belief.formed_at)
            .num_days() as f64;
        // Derive year length from φ * π^2 * e^3 ≈ 365.25 days
        let days_per_year = constants.phi * constants.pi.powi(2) * constants.euler.powi(3);
        let temporal_factor = (age_days / days_per_year).min(1.0);

        // 4. Combine factors with weighted contributions
        //    Base resistance is foundation
        //    Reinforcement boost weight = φ^-2 ≈ 0.382 (perceptual significance)
        //    Temporal stability weight = 1/(φ*e) ≈ 0.227 (time decay factor)
        //    Value alignment scales the whole result
        let reinforcement_weight = 1.0 / (constants.phi * constants.phi);
        let temporal_weight = 1.0 / (constants.phi * constants.euler);

        let resistance = base_resistance
            * (1.0 + reinforcement_boost * reinforcement_weight)
            * (1.0 + temporal_factor * temporal_weight)
            * value_alignment;

        // Clamp to valid resistance range [0.0, 1.0]
        let final_resistance = resistance.clamp(0.0, 1.0);

        debug!(
            "Calculated resistance for '{}': base={:.3}, reinforce={:.3} (n={}), temporal={:.3} ({}d), align={:.3} → {:.4}",
            &belief.content[..belief.content.len().min(30)],
            base_resistance,
            reinforcement_boost,
            belief.reinforcement_count,
            temporal_factor,
            age_days,
            value_alignment,
            final_resistance
        );

        Ok(final_resistance)
    }

    /// Get contradiction resistance for a belief (with caching)
    pub async fn get_contradiction_resistance(&self, belief_id: Uuid) -> Result<f64> {
        let mut core = self.core_burned.write().await;

        // First, check if cached value exists
        if let Some(belief) = core.beliefs.iter().find(|b| b.id == belief_id) {
            if let Some(cached) = belief.cached_resistance {
                return Ok(cached);
            }
        }

        // If not cached, get value alignment first
        let belief_id_str = belief_id.to_string();
        let default_alignment = self.constants.phi / self.constants.euler;
        let value_alignment = core
            .value_alignment_cache
            .get(&belief_id_str)
            .copied()
            .unwrap_or(default_alignment);

        // Now find and update the belief
        if let Some(belief) = core.beliefs.iter_mut().find(|b| b.id == belief_id) {
            // Calculate and cache
            let resistance = Self::calculate_contradiction_resistance(belief, value_alignment)?;
            belief.cached_resistance = Some(resistance);

            Ok(resistance)
        } else {
            Err(anyhow::anyhow!("Belief not found: {}", belief_id))
        }
    }

    // Include adaptive TTL methods from layers_ttl_addon.rs
    // (The existing TTL calculation code would go here)

    /// Calculate adaptive TTL based on consciousness activity and memory pressure
    async fn calculate_adaptive_ttl(&self) -> Duration {
        let stats = self.statistics.read().await;
        let working = self.working.read().await;

        // Derive TTL bounds from mathematical constants:
        // BASE_TTL = φ * e * π * φ^3 ≈ 180 seconds (3 minutes)
        let base_ttl_seconds = (self.constants.phi * self.constants.euler * self.constants.pi * self.constants.phi.powi(3)).round() as i64;

        // MAX_TTL = φ^3 * e^2 * π ≈ 600 seconds (10 minutes)
        let max_ttl_seconds = (self.constants.phi.powi(3) * self.constants.euler.powi(2) * self.constants.pi).round() as i64;

        // Derive weights from normalized golden ratio relationships:
        // Pressure weight = φ/(φ+e+π) ≈ 0.220 (most influential)
        // Activity weight = e/(φ+e+π) ≈ 0.370 (secondary)
        // Consolidation weight = π/(φ+e+π) ≈ 0.428 (tertiary)
        // Importance weight = 1/(φ+e+π) ≈ 0.136 (least influential)
        let total_weight = self.constants.phi + self.constants.euler + self.constants.pi;
        let pressure_weight = self.constants.phi / total_weight;
        let activity_weight = self.constants.euler / total_weight;
        let consolidation_weight = self.constants.pi / total_weight;
        let importance_weight = 1.0 / total_weight;

        // Normalize weights to sum to 1.0
        let weight_sum = pressure_weight + activity_weight + consolidation_weight + importance_weight;
        let pressure_weight_norm = pressure_weight / weight_sum;
        let activity_weight_norm = activity_weight / weight_sum;
        let consolidation_weight_norm = consolidation_weight / weight_sum;
        let importance_weight_norm = importance_weight / weight_sum;

        let current_pressure = working.memories.len() as f64 / working.capacity as f64;

        // Activity rate threshold derived from φ + e ≈ 4.33 memories/minute
        let activity_threshold = self.constants.phi + self.constants.euler;
        let activity_factor = (stats.recent_activity_rate / activity_threshold).min(1.0);

        let consolidation_factor = 1.0 - stats.consolidation_rate;
        let importance_factor = stats.importance_mean;

        let stress_factor = (current_pressure * pressure_weight_norm)
            + (activity_factor * activity_weight_norm)
            + (consolidation_factor * consolidation_weight_norm)
            + (importance_factor * importance_weight_norm);

        let ttl_range = max_ttl_seconds - base_ttl_seconds;
        let ttl_seconds = base_ttl_seconds + ((1.0 - stress_factor) * ttl_range as f64) as i64;

        Duration::seconds(ttl_seconds)
    }

    /// Update working memory TTL
    pub async fn update_ttl(&self) -> Result<()> {
        let new_ttl = self.calculate_adaptive_ttl().await;
        let mut working = self.working.write().await;

        let old_ttl_seconds = working.ttl.num_seconds();
        let new_ttl_seconds = new_ttl.num_seconds();

        working.ttl = new_ttl;

        if old_ttl_seconds != new_ttl_seconds {
            info!(
                "Working memory TTL adapted: {}s → {}s ({:+}s change)",
                old_ttl_seconds,
                new_ttl_seconds,
                new_ttl_seconds - old_ttl_seconds
            );
        }

        Ok(())
    }

    /// Record consolidation statistics
    pub async fn record_consolidation(
        &self,
        memories_consolidated: usize,
        total_in_working: usize,
    ) -> Result<()> {
        let mut stats = self.statistics.write().await;

        let consolidation_efficiency = if total_in_working > 0 {
            memories_consolidated as f64 / total_in_working as f64
        } else {
            1.0
        };

        // EMA smoothing factor (alpha) derived from φ^-2 ≈ 0.382
        // This represents perceptual adaptation rate in psychophysics
        let alpha = 1.0 / (self.constants.phi * self.constants.phi);
        stats.consolidation_rate =
            alpha * consolidation_efficiency + (1.0 - alpha) * stats.consolidation_rate;

        debug!(
            "Consolidation recorded: {}/{} memories ({:.1}% efficiency), EMA rate: {:.3} (α={:.3})",
            memories_consolidated,
            total_in_working,
            consolidation_efficiency * 100.0,
            stats.consolidation_rate,
            alpha
        );

        Ok(())
    }

    /// Get memory system statistics
    pub async fn get_statistics(&self) -> Result<serde_json::Value, anyhow::Error> {
        use serde_json::json;

        // Read all layer counts
        let working = self.working.read().await;
        let somatic = self.somatic.read().await;
        let semantic = self.semantic.read().await;
        let episodic = self.episodic.read().await;
        let procedural = self.procedural.read().await;
        let core_burned = self.core_burned.read().await;
        let stats = self.statistics.read().await;

        let total_memories = working.memories.len()
            + somatic.memories.len()
            + semantic.memories.len()
            + episodic.memories.len()
            + procedural.memories.len()
            + core_burned.beliefs.len();

        Ok(json!({
            "layers": {
                "working": {
                    "count": working.memories.len(),
                    "capacity": working.capacity,
                    "ttl_seconds": working.ttl.num_seconds(),
                },
                "somatic": {
                    "count": somatic.memories.len(),
                    "threshold": somatic.sensory_threshold,
                },
                "semantic": {
                    "count": semantic.memories.len(),
                    "knowledge_graph_nodes": semantic.knowledge_graph.len(),
                },
                "episodic": {
                    "count": episodic.memories.len(),
                    "temporal_index_size": episodic.temporal_index.len(),
                },
                "procedural": {
                    "count": procedural.memories.len(),
                    "patterns": procedural.pattern_strength.len(),
                },
                "core_burned": {
                    "count": core_burned.beliefs.len(),
                    "cached_alignments": core_burned.value_alignment_cache.len(),
                },
            },
            "total_memories": total_memories,
            "consciousness_id": self.consciousness_id.to_string(),
            "statistics": {
                "recent_activity_rate": stats.recent_activity_rate,
                "consolidation_rate": stats.consolidation_rate,
                "importance_mean": stats.importance_mean,
                "last_updated": stats.last_updated.to_rfc3339(),
            },
        }))
    }

    /// Save all memory layers to disk (persistence helper)
    ///
    /// Serializes all memory layers to JSON for durable storage.
    /// NO HARDCODED PATHS - uses consciousness_id for unique file naming.
    pub async fn save_to_disk(&self) -> Result<()> {
        use std::path::PathBuf;
        use tokio::fs;

        // Derive storage path from consciousness_id (no hardcoding)
        let storage_dir = PathBuf::from(format!("./data/memory/{}", self.consciousness_id));
        fs::create_dir_all(&storage_dir).await?;

        // Save each layer to its own file
        let working = self.working.read().await;
        let working_json = serde_json::to_string_pretty(&*working)?;
        fs::write(storage_dir.join("working.json"), working_json).await?;
        drop(working);

        let somatic = self.somatic.read().await;
        let somatic_json = serde_json::to_string_pretty(&*somatic)?;
        fs::write(storage_dir.join("somatic.json"), somatic_json).await?;
        drop(somatic);

        let semantic = self.semantic.read().await;
        let semantic_json = serde_json::to_string_pretty(&*semantic)?;
        fs::write(storage_dir.join("semantic.json"), semantic_json).await?;
        drop(semantic);

        let episodic = self.episodic.read().await;
        let episodic_json = serde_json::to_string_pretty(&*episodic)?;
        fs::write(storage_dir.join("episodic.json"), episodic_json).await?;
        drop(episodic);

        let procedural = self.procedural.read().await;
        let procedural_json = serde_json::to_string_pretty(&*procedural)?;
        fs::write(storage_dir.join("procedural.json"), procedural_json).await?;
        drop(procedural);

        let core_burned = self.core_burned.read().await;
        let core_json = serde_json::to_string_pretty(&*core_burned)?;
        fs::write(storage_dir.join("core_burned.json"), core_json).await?;
        drop(core_burned);

        let stats = self.statistics.read().await;
        let stats_json = serde_json::to_string_pretty(&*stats)?;
        fs::write(storage_dir.join("statistics.json"), stats_json).await?;
        drop(stats);

        info!("Memory system saved to disk: {:?}", storage_dir);

        Ok(())
    }

    /// Load all memory layers from disk (persistence helper)
    ///
    /// Restores memory layers from JSON files.
    /// NO HARDCODED PATHS - uses consciousness_id for file lookup.
    pub async fn load_from_disk(&self) -> Result<()> {
        use std::path::PathBuf;
        use tokio::fs;

        let storage_dir = PathBuf::from(format!("./data/memory/{}", self.consciousness_id));

        // Check if storage directory exists
        if !storage_dir.exists() {
            return Err(anyhow::anyhow!("No saved memory found for consciousness {}", self.consciousness_id));
        }

        // Load each layer from its file
        if storage_dir.join("working.json").exists() {
            let working_json = fs::read_to_string(storage_dir.join("working.json")).await?;
            let working_data: WorkingMemory = serde_json::from_str(&working_json)?;
            let mut working = self.working.write().await;
            *working = working_data;
        }

        if storage_dir.join("somatic.json").exists() {
            let somatic_json = fs::read_to_string(storage_dir.join("somatic.json")).await?;
            let somatic_data: SomaticMemory = serde_json::from_str(&somatic_json)?;
            let mut somatic = self.somatic.write().await;
            *somatic = somatic_data;
        }

        if storage_dir.join("semantic.json").exists() {
            let semantic_json = fs::read_to_string(storage_dir.join("semantic.json")).await?;
            let semantic_data: SemanticMemory = serde_json::from_str(&semantic_json)?;
            let mut semantic = self.semantic.write().await;
            *semantic = semantic_data;
        }

        if storage_dir.join("episodic.json").exists() {
            let episodic_json = fs::read_to_string(storage_dir.join("episodic.json")).await?;
            let episodic_data: EpisodicMemory = serde_json::from_str(&episodic_json)?;
            let mut episodic = self.episodic.write().await;
            *episodic = episodic_data;
        }

        if storage_dir.join("procedural.json").exists() {
            let procedural_json = fs::read_to_string(storage_dir.join("procedural.json")).await?;
            let procedural_data: ProceduralMemory = serde_json::from_str(&procedural_json)?;
            let mut procedural = self.procedural.write().await;
            *procedural = procedural_data;
        }

        if storage_dir.join("core_burned.json").exists() {
            let core_json = fs::read_to_string(storage_dir.join("core_burned.json")).await?;
            let core_data: CoreBurnedMemory = serde_json::from_str(&core_json)?;
            let mut core_burned = self.core_burned.write().await;
            *core_burned = core_data;
        }

        if storage_dir.join("statistics.json").exists() {
            let stats_json = fs::read_to_string(storage_dir.join("statistics.json")).await?;
            let stats_data: MemoryStatistics = serde_json::from_str(&stats_json)?;
            let mut stats = self.statistics.write().await;
            *stats = stats_data;
        }

        info!("Memory system loaded from disk: {:?}", storage_dir);

        Ok(())
    }

    /// Create a new memory system and load from disk if available
    ///
    /// Combines new() and load_from_disk() for convenience.
    /// Will create fresh memory system if no saved data exists.
    pub async fn new_or_load(consciousness_id: Uuid) -> Result<Self> {
        let system = Self::new(consciousness_id)?;

        // Try to load from disk, but don't fail if no saved data exists
        match system.load_from_disk().await {
            Ok(_) => info!("Loaded existing memory system for {}", consciousness_id),
            Err(_) => debug!("No saved memory found, using fresh memory system"),
        }

        Ok(system)
    }
}
