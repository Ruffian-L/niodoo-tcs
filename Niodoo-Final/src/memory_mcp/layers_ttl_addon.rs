// Copyright (c) 2025 Jason Van Pham (ruffian-l on GitHub) @ The Niodoo Collaborative
// Licensed under the MIT License - See LICENSE file for details
// Attribution required for all derivative works

// Adaptive TTL methods to be added to MemorySystem impl in layers.rs
// Add these methods before the closing brace of the impl block

    /// Calculate adaptive TTL based on consciousness activity and memory pressure
    ///
    /// Algorithm uses weighted factors to determine optimal working memory duration:
    /// - Base TTL: 3 minutes (minimal working memory duration from cognitive research)
    /// - Max TTL: 10 minutes (for quiet periods with low memory pressure)
    ///
    /// Weighted factors (total = 1.0):
    ///   1. Memory pressure (0.4): Higher pressure → shorter TTL (faster consolidation needed)
    ///   2. Activity rate (0.3): Higher activity → shorter TTL (more information flow)
    ///   3. Consolidation rate (0.2): Higher rate → longer TTL (system handling load well)
    ///   4. Importance distribution (0.1): Higher mean importance → shorter TTL (critical data)
    ///
    /// This creates a feedback loop where the TTL adapts to consciousness workload,
    /// preventing memory overflow while maximizing retention during quiet periods.
    async fn calculate_adaptive_ttl(&self) -> Duration {
        let stats = self.statistics.read().await;
        let working = self.working.read().await;

        // Constants derived from cognitive psychology research
        const BASE_TTL_SECONDS: i64 = 180;  // 3 minutes - minimum working memory duration
        const MAX_TTL_SECONDS: i64 = 600;   // 10 minutes - maximum for low-pressure periods
        const PRESSURE_WEIGHT: f64 = 0.4;   // Memory pressure influence (highest priority)
        const ACTIVITY_WEIGHT: f64 = 0.3;   // Activity rate influence
        const CONSOLIDATION_WEIGHT: f64 = 0.2; // Consolidation efficiency influence
        const IMPORTANCE_WEIGHT: f64 = 0.1; // Importance distribution influence

        // Calculate current memory pressure (0.0 = empty, 1.0 = at capacity)
        let current_pressure = working.memories.len() as f64 / working.capacity as f64;

        // Activity rate normalization: 0-5 memories/min is normal operational range
        // Higher activity rate → higher cognitive load → shorter TTL needed
        let activity_factor = (stats.recent_activity_rate / 5.0).min(1.0);

        // Consolidation rate (0-1): 1.0 = system consolidates perfectly, 0.0 = struggling
        // Higher consolidation rate → system handles load well → can afford longer TTL
        // Invert because we want low consolidation to increase stress
        let consolidation_factor = 1.0 - stats.consolidation_rate;

        // Importance factor: If average importance is high, memories are more critical
        // Higher importance → shorter TTL to ensure critical data moves to permanent storage
        let importance_factor = stats.importance_mean;

        // Combined stress factor: 0.0 = no stress/pressure, 1.0 = maximum stress/pressure
        // High stress → system needs faster consolidation → shorter TTL
        let stress_factor =
            (current_pressure * PRESSURE_WEIGHT) +
            (activity_factor * ACTIVITY_WEIGHT) +
            (consolidation_factor * CONSOLIDATION_WEIGHT) +
            (importance_factor * IMPORTANCE_WEIGHT);

        // Calculate TTL using inverse relationship:
        // - High stress (stress_factor → 1.0) → TTL → BASE_TTL_SECONDS (short, fast consolidation)
        // - Low stress (stress_factor → 0.0) → TTL → MAX_TTL_SECONDS (long, relaxed retention)
        let ttl_range = MAX_TTL_SECONDS - BASE_TTL_SECONDS;
        let ttl_seconds = BASE_TTL_SECONDS + ((1.0 - stress_factor) * ttl_range as f64) as i64;

        let calculated_ttl = Duration::seconds(ttl_seconds);

        debug!(
            "🧮 Adaptive TTL calculation:\n\
            Memory pressure: {:.2} (weight: {:.1})\n\
            Activity rate: {:.2}/min → factor {:.2} (weight: {:.1})\n\
            Consolidation rate: {:.2} → factor {:.2} (weight: {:.1})\n\
            Importance mean: {:.2} (weight: {:.1})\n\
            → Combined stress factor: {:.3}\n\
            → Calculated TTL: {}s ({:.1} minutes)",
            current_pressure, PRESSURE_WEIGHT,
            stats.recent_activity_rate, activity_factor, ACTIVITY_WEIGHT,
            stats.consolidation_rate, consolidation_factor, CONSOLIDATION_WEIGHT,
            stats.importance_mean, IMPORTANCE_WEIGHT,
            stress_factor,
            ttl_seconds, ttl_seconds as f64 / 60.0
        );

        calculated_ttl
    }

    /// Update working memory TTL based on current consciousness conditions
    ///
    /// This should be called periodically (e.g., after consolidation cycles) to
    /// adapt the TTL to changing memory pressure and activity patterns.
    pub async fn update_ttl(&self) -> Result<()> {
        let new_ttl = self.calculate_adaptive_ttl().await;
        let mut working = self.working.write().await;

        let old_ttl_seconds = working.ttl.num_seconds();
        let new_ttl_seconds = new_ttl.num_seconds();

        working.ttl = new_ttl;

        if old_ttl_seconds != new_ttl_seconds {
            info!(
                "🔄 Working memory TTL adapted: {}s → {}s ({:+}s change)",
                old_ttl_seconds,
                new_ttl_seconds,
                new_ttl_seconds - old_ttl_seconds
            );
        }

        Ok(())
    }

    /// Update consolidation rate statistics based on consolidation success
    ///
    /// Call this after consolidation to track how efficiently the system
    /// is handling memory pressure. Higher rates indicate the system is
    /// keeping up with memory load.
    pub async fn record_consolidation(&self, memories_consolidated: usize, total_in_working: usize) -> Result<()> {
        let mut stats = self.statistics.write().await;

        // Calculate consolidation efficiency: ratio of consolidated to total
        let consolidation_efficiency = if total_in_working > 0 {
            memories_consolidated as f64 / total_in_working as f64
        } else {
            1.0 // If working memory is empty, consolidation is "perfect"
        };

        // Update consolidation rate using exponential moving average
        const ALPHA: f64 = 0.3; // Smoothing factor (0.3 = moderately responsive)
        stats.consolidation_rate = ALPHA * consolidation_efficiency + (1.0 - ALPHA) * stats.consolidation_rate;

        debug!(
            "📊 Consolidation recorded: {}/{} memories ({:.1}% efficiency), EMA rate: {:.3}",
            memories_consolidated,
            total_in_working,
            consolidation_efficiency * 100.0,
            stats.consolidation_rate
        );

        Ok(())
    }
